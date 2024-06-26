import logging
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime as dt
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rdkit
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.nn import GRU, Linear, Sequential
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GINEConv, NNConv, Set2Set
from torch_geometric.utils import from_smiles
from torch_geometric.utils.smiles import e_map, x_map
from torch_scatter import scatter_add, scatter_max, scatter_mean
from tqdm import tqdm

from dl4thermo.extras.utils.metrics import get_torchmetrics
from dl4thermo.extras.utils.optimizer import get_lr_scheduler, get_optimizer
from dl4thermo.extras.utils.pcsaft import is_associating

from .base import LightningValidator, TrainArgs
from .pcsaft_emulator import PcSaftEmulator, PcSaftEmulatorLightningModule


@dataclass
class PyGTrainArgs(TrainArgs):
    """Adapted class for PyG implementation"""

    # Model architecture
    conv_type: Literal["NNConv", "GINEConv"] = "GINEConv"
    num_convs: int = 3
    use_gru: bool = True
    dim_fingerprint: int = 64
    pool_type: Literal["add", "mean", "max", "set2set"] = "add"
    dropout: float = 0.0
    activation: Literal["ReLU", "LeakyReLU"] = "LeakyReLU"
    multi_gru: bool = False
    share_conv: bool = False  # TODO: add multi input case
    model_type: str = "GNN"
    separate_heads: bool = False
    filter_non_associating_inside_loss: bool = True  # Not 100% sure about this
    freeze_encoder: bool = False
    freeze_normalization: bool = False  # Use the mean/std from initialization/loaded in
    balanced_associating_sampling: bool = True
    explicit_hydrogen: bool = False
    num_workers: int = 0  # Seems to be faster!
    wandb_artifact_name: str = "pyg"
    inverse_tr_column: str = "inverse_Tr"

    # Things to help with hyperparameters tuning
    # Not actually used in the code
    use_dropout: bool = False
    use_pretrained_model: bool = False


class PyGDataset(InMemoryDataset):
    r"""PyG dataset with attributed molecular graphs

    Args:
        root (string): Root directory where the dataset should be saved.
    """

    raw_url = ""
    processed_url = ""

    def __init__(
        self,
        root,
        data: pd.DataFrame,
        smiles_columns: List[str],
        target_columns: List[str],
        split_idx: Tuple[np.ndarray, np.ndarray, np.ndarray],
        explicit_hydrogen: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        inverse_tr_column: Optional[str] = None,
    ):
        self._data = data
        self._split_idx = split_idx
        self.smiles_columns, self.target_columns = smiles_columns, target_columns
        self.explicit_hydrogen = explicit_hydrogen
        self.inverse_tr_column = inverse_tr_column
        super(PyGDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_mask, self.valid_mask, self.test_mask = (
            np.load(self.processed_paths[i], allow_pickle=True) for i in range(1, 4)
        )

    @property
    def raw_file_names(self):
        return "raw.pt" if rdkit is None else "raw.csv"

    @property
    def processed_file_names(self):
        return ["data.pt", "train_indices.npy", "valid_indices.npy", "test_indices.npy"]

    def download(self):
        pass

    def process(self):
        p = Path(self.processed_paths[0])
        if not p.exists():
            data_list = self._generate_datalist(
                self._data,
                smiles_columns=self.smiles_columns,
                target_columns=self.target_columns,
            )
            torch.save(self.collate(data_list), self.processed_paths[0])
            for i, mask in enumerate(self._split_idx):
                np.save(self.processed_paths[i + 1], mask)
            self._data = pd.DataFrame()  # free up memory
            self._split_idx = (np.empty((0,)), np.empty((0,)), np.empty((0,)))

    def _generate_datalist(self, df, smiles_columns, target_columns):
        r"""Molecule graph dataset based on PyG Data object

        returns:
        list of PyG data objects that correspond to attributed molecular graphs with target property labels
        """

        if rdkit is None:
            raise NotImplementedError(
                "Rdkit not installed. Preprocessing without rdkit not implemented yet."
            )

        if len(smiles_columns) != 1:
            raise NotImplementedError(
                "PyG MolGraph processing for mixtures (#Smiles per target value != 1) not implemented yet."
            )

        # get dataframe of SMILES and target values
        cols = smiles_columns + target_columns
        df = df[cols]

        # create PyG Molgraphs for each data point
        data_list = []
        bar: Any = tqdm(df.iterrows(), total=df.shape[0])
        for _, row in bar:
            smiles = row[smiles_columns[0]]
            data = from_smiles(smiles, with_hydrogen=self.explicit_hydrogen)
            data.y = torch.tensor(
                [float(row[target_col]) for target_col in target_columns],
                dtype=torch.float,
            ).view(1, -1)
            associating = 1 if is_associating(smiles) else 0
            data.associating = torch.tensor(float(associating), dtype=torch.float).view(
                1, -1
            )
            if self.inverse_tr_column is not None:
                data.inverse_tr = torch.tensor(
                    float(row[self.inverse_tr_column]), dtype=torch.float
                ).view(1, -1)

            data_list.append(data)

        return data_list


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        self.emb_dim = emb_dim
        full_atom_feature_dims = [len(x_feat) for x_feat in x_map.values()]

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = torch.zeros((x.shape[0], self.emb_dim), device=x.device)
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        self.emb_dim = emb_dim
        full_bond_feature_dims = [len(e_feat) for e_feat in e_map.values()]

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = torch.zeros(
            (edge_attr.shape[0], self.emb_dim), device=edge_attr.device
        )
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


class MPNN(nn.Module):
    def __init__(
        self,
        args: PyGTrainArgs,
        num_targets: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        ### Hyperparameters ###
        # architecture
        self.args = args
        self.num_input_smiles = len(self.args.smiles_columns)
        # TODO: implement multi smile
        if self.num_input_smiles != 1:
            raise NotImplementedError(
                "Multi input smiles PyG GNN architecture not implemented yet."
            )
        self.num_targets = num_targets or len(self.args.target_columns)

        self.multi_gru = self.args.multi_gru
        self.use_gru = self.args.use_gru
        if self.use_gru == False and self.multi_gru:
            raise ValueError(
                "Error: you try use multiple GRUs but use of GRU is not active."
            )

        # MLP-prediction
        # Activation function
        if self.args.activation == "LeakyReLU":
            self.act_func = nn.LeakyReLU
        elif self.args.activation == "ReLU":
            self.act_func = nn.ReLU
        elif self.args.activation == "ELU":
            self.act_func = nn.ELU
        else:
            raise NotImplementedError(
                f"Activation function {self.args.activation} not implemented"
            )

        ### Graph convolutions ###
        # node (and edge) feature dimension adjustment
        self.node_encoder = AtomEncoder(self.args.dim_fingerprint)
        self.edge_encoder = BondEncoder(self.args.dim_fingerprint)

        # message passing layers
        self.conv = nn.ModuleList()
        self.gru = nn.ModuleList()
        for _ in range(self.args.num_convs):
            nn_conv = self._get_message_passing_layer()
            self.conv.append(nn_conv)
            if self.args.use_gru:
                gru = GRU(self.args.dim_fingerprint, self.args.dim_fingerprint)
                self.gru.append(gru)
                # check if multiple GRUs are to be used
                if not self.multi_gru:
                    break  # if False: one GRU for multiple processing steps = num_convs

        # special pooling
        dim_channel_in = (
            self.args.dim_fingerprint
        )  # vector dimension after pooling (does not change for add-, mean-, max-pooling)
        if "set2set" in self.args.pool_type:
            # set2set layer
            processing_steps = kwargs.get(
                "set2set_processing_steps", 3
            )  # set2set has optional argument of processing steps
            self.set2set = Set2Set(dim_channel_in, processing_steps=processing_steps)
            # set2set out
            dim_channel_in = (
                2 * dim_channel_in
            )  # vector dimension doubles through set2set pooling

        if self.args.freeze_encoder:
            for module in [self.node_encoder, self.edge_encoder, self.conv, self.gru]:
                for param in module.parameters():
                    param.requires_grad = False

        # MLP: fingerprint -> property
        self.dropout_layer = nn.Dropout(p=self.args.dropout)
        if self.args.separate_heads:
            self.ffns = nn.ModuleList(
                [self.build_ffn(dim_channel_in, 1) for _ in range(self.num_targets)]
            )
        else:
            self.ffn = self.build_ffn(dim_channel_in, self.num_targets)

    def build_ffn(self, dim_channel_in, num_targets):
        return nn.Sequential(
            nn.Linear(dim_channel_in, dim_channel_in),
            self.act_func(),
            self.dropout_layer,
            nn.Linear(dim_channel_in, int(dim_channel_in / 2)),
            self.act_func(),
            self.dropout_layer,
            nn.Linear(int(dim_channel_in / 2), num_targets),
        )

    def forward(self, data):
        """
        Forward pass
        """

        ### Get data ###
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x_batch = data.batch

        ### Message passing & Pooling ###
        # IL message passing -> node embeddings
        out = self.act_func()(self.node_encoder(x))
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
        h = out.unsqueeze(0)
        conv_l_idx = 0
        for _ in range(self.args.num_convs):
            out = self.act_func()(
                self.conv[conv_l_idx](out, edge_index, edge_attr)
            )  # convolutional layer
            if self.use_gru:
                out, h = self.gru[conv_l_idx](out.unsqueeze(0), h)  # GRU
            out = out.squeeze(0)
            if (self.use_gru and self.multi_gru) or (not self.use_gru):
                conv_l_idx += 1

        # pooling -> graph embeddings
        x_forward = out  # node embeddings of i
        if "add" in self.args.pool_type:
            h = scatter_add(x_forward, x_batch, dim=0)
        elif "mean" in self.args.pool_type:
            h = scatter_mean(x_forward, x_batch, dim=0)
        elif "max" in self.args.pool_type:
            h = scatter_max(x_forward, x_batch, dim=0)[0]
        elif "set2set" in self.args.pool_type:
            h = self.set2set(x_forward, x_batch)

        ### MLP-prediction ###
        if self.args.separate_heads:
            y = torch.cat(
                [self.ffns[i](h) for i in range(self.num_targets)], axis=1
            )  # type: ignore
        else:
            y = self.ffn(h)
        return y

    def _get_message_passing_layer(self):
        if self.args.conv_type == "NNConv":
            edge_nn = Sequential(
                Linear(self.args.dim_fingerprint, 128),
                nn.ReLU(),
                Linear(128, self.args.dim_fingerprint * self.args.dim_fingerprint),
            )
            return NNConv(
                self.args.dim_fingerprint,
                self.args.dim_fingerprint,
                edge_nn,
                aggr="add",
            )
        elif self.args.conv_type == "GINEConv":
            gine_nn = Sequential(
                Linear(self.args.dim_fingerprint, int(self.args.dim_fingerprint * 2)),
                nn.ReLU(),
                Linear(int(self.args.dim_fingerprint * 2), self.args.dim_fingerprint),
            )
            return GINEConv(gine_nn, train_eps=True)
        else:
            raise NotImplementedError(f"Conv type {self.conv_type} not implemented")


class PyGDataModule(pl.LightningDataModule):
    _dataset: PyGDataset
    _scaler: StandardScaler

    def __init__(
        self,
        args: PyGTrainArgs,
        data: pd.DataFrame,
        split_idx: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ):
        super().__init__()
        self.args = args
        self._df = data
        self._split_idx = split_idx
        self.logger = logging.getLogger(__name__)

    def setup(self, stage: str) -> None:
        # Set up datasets
        self._dataset = PyGDataset(
            self.args.save_dir,
            self._df,
            smiles_columns=self.args.smiles_columns,
            target_columns=self.args.target_columns,
            split_idx=self._split_idx,
            explicit_hydrogen=self.args.explicit_hydrogen,
        )
        self._df = pd.DataFrame()  # free up memory

        # Set up weighting for inbalance of associating vs no associating
        if self.args.balanced_associating_sampling:
            self.logger.info("Calculating sampler weights")
            associating = np.array(
                [d.associating.squeeze().item() for d in self._dataset]  # type: ignore
            )
            non_associating_counts = len(associating[associating == 0])
            associating_counts = len(associating[associating == 1])
            counts = np.array([non_associating_counts, associating_counts])
            weights = 1.0 / counts
            self._association_weights = np.array(
                [weights[int(a.squeeze().item())] for a in associating]
            )
        else:
            self._association_weights = None

    def train_dataloader(self) -> PyGDataLoader:
        if self._association_weights is not None:
            train_weights: np.ndarray = self._association_weights[
                self._dataset.train_mask
            ]
            sampler = WeightedRandomSampler(
                weights=train_weights.tolist(),
                num_samples=len(train_weights),
                replacement=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return PyGDataLoader(
            dataset=deepcopy(self._dataset[self._dataset.train_mask]),  # type: ignore
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            sampler=sampler,
            shuffle=shuffle,
        )

    def val_dataloader(self) -> PyGDataLoader:
        return PyGDataLoader(
            dataset=self._dataset[self._dataset.valid_mask],  # type: ignore
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> PyGDataLoader:
        return PyGDataLoader(
            dataset=self._dataset[self._dataset.test_mask],  # type: ignore
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        train_no_shuffle = PyGDataLoader(
            dataset=self._dataset[self._dataset.train_mask],  # type: ignore
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

        return [train_no_shuffle, self.val_dataloader(), self.test_dataloader()]


class PyGLightningModule(pl.LightningModule):
    """A directed message-passing neural network base class"""

    means: torch.Tensor
    stds: torch.Tensor

    def __init__(
        self,
        args: PyGTrainArgs,
        model: MPNN,
    ):
        super().__init__()
        self.args = args
        self.model = model

        self.save_hyperparameters(asdict(args), ignore=["model"])
        means = torch.zeros((len(args.target_columns),))
        stds = torch.ones((len(args.target_columns),))
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.metrics = nn.ModuleDict(
            {metric: get_torchmetrics(metric) for metric in self.args.metrics}
        )
        # Setup for filtering non-associating inside loss
        self.associating_target_indices: Optional[List[int]] = None
        if args.filter_non_associating_inside_loss and args.associating_columns:
            self.associating_target_indices = [
                args.target_columns.index(target) for target in args.associating_columns
            ]
            non_associating_values = torch.tensor(
                [0.0] * len(self.associating_target_indices), dtype=torch.float
            )
        else:
            # This is just a placeholder, won't actually be used.
            non_associating_values = torch.tensor(
                [0.0] * len(args.target_columns), dtype=torch.float
            )
        self.register_buffer("non_associating_values", non_associating_values)

    def setup(self, stage: str) -> None:
        dm: PyGDataModule = self.trainer.datamodule  # type: ignore
        if not self.args.freeze_normalization:
            self.means = dm._dataset[dm._dataset.train_mask].data.y.mean(0)  # type: ignore
            self.stds = dm._dataset[dm._dataset.train_mask].data.y.std(0)  # type: ignore

        # Adjust non-associating values for normalization
        if self.associating_target_indices is not None:
            idx = self.associating_target_indices
            values = -1.0 * self.means[idx].clone() / self.stds[idx].clone()
            state_dict = self.state_dict()
            state_dict["non_associating_values"] = values
            self.load_state_dict(state_dict)
        if stage == "fit":
            self.n_train = len(dm._dataset.train_mask)

    def training_step(self, batch: PyGDataset, batch_idx: int):
        loss, preds = self.calc_loss(batch)
        self.log("train_loss", loss)
        self.log_metrics(
            preds,
            self.scale(batch.y),  # type: ignore
            "train",
        )
        return loss

    def validation_step(self, batch: PyGDataset, batch_idx: int):
        loss, preds = self.calc_loss(batch)
        self.log("val_loss", loss)
        self.log_metrics(
            preds,
            self.scale(batch.y),  # type: ignore
            "val",
        )
        return loss

    def test_step(self, batch: PyGDataset, batch_idx: int):
        loss, preds = self.calc_loss(batch)
        self.log("test_loss", loss)
        self.log_metrics(
            preds,
            self.scale(batch.y),  # type: ignore
            "test",
        )
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch, inverse_scale=True)

    def log_metrics(self, preds: torch.Tensor, targets: torch.Tensor, subset: str):
        preds = torch.atleast_2d(preds)
        targets = torch.atleast_2d(targets)
        for i, target in enumerate(self.args.target_columns):
            for metric_name, metric in self.metrics.items():
                score = metric(preds[:, i], targets[:, i])
                self.log(f"{subset}_{target}_{metric_name}", score)

    def on_train_epoch_start(self) -> None:
        self.start_time = dt.now()

    def on_train_epoch_end(self) -> None:
        epoch_time = (dt.now() - self.start_time).total_seconds()
        training_throughput = self.n_train / epoch_time
        self.log("training_throughput", training_throughput)
        self.log("time_per_step", self.args.batch_size / training_throughput)

    def forward(
        self,
        batch: Any,
        dataloader_idx: int = 0,
        inverse_scale: bool = True,
    ):
        associating = batch.associating  # type: ignore
        if self.training:
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()  # type: ignore
                self.log("lr", sch.get_last_lr()[0])  # type: ignore

        parameter_preds = self.model(batch)

        # Set non-associating to zero with offset for normalization
        if self.associating_target_indices:
            idx = self.associating_target_indices
            # offsets for non-associating (hence the -1.0)
            offset = (
                (associating - 1.0)
                * self.non_associating_values
                * torch.ones_like(
                    parameter_preds[:, idx], device=parameter_preds.device
                )
            )
            parameter_preds[:, idx] = associating * parameter_preds[:, idx] - offset

        if inverse_scale:
            parameter_preds = self.inverse_scale(parameter_preds)

        return parameter_preds

    def scale(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.means.unsqueeze(0)) / self.stds.unsqueeze(0)

    def inverse_scale(self, preds: torch.Tensor) -> torch.Tensor:
        return preds * self.stds.unsqueeze(0) + self.means.unsqueeze(0)

    def calc_loss(self, batch: PyGDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        target = self.scale(batch.y)  # type: ignore
        associating = batch.associating  # type: ignore

        if self.args.target_weights is not None:
            target_weights = torch.tensor(self.args.target_weights).unsqueeze(
                0
            )  # shape(1,tasks)
        else:
            target_weights = torch.ones(target.shape[1]).unsqueeze(0)

        # Run model
        preds = self(batch, inverse_scale=False)

        # Move tensors to correct device
        torch_device = preds.device
        target = target.to(torch_device)
        target_weights = target_weights.to(torch_device)

        # Compute loss
        if self.associating_target_indices:
            idx = self.associating_target_indices
            mask = torch.ones_like(target, device=torch_device)
            for idx in self.associating_target_indices:
                mask[~associating.bool().squeeze(), idx] = 0.0
            loss = F.mse_loss(preds, target, reduction="none") * target_weights * mask
        else:
            loss = F.mse_loss(preds, target, reduction="none") * target_weights
        loss = loss.mean()
        return loss, preds

    def configure_optimizers(
        self,
        parameters: Optional[Any] = None,
    ):
        optimizer_kwargs = self.args.optimizer_kwargs or {"lr": self.args.lr}
        parameters = parameters or self.parameters()
        optimizer = get_optimizer(
            self.args.optimizer, parameters, optimizer_kwargs=optimizer_kwargs
        )
        optimizer_dict: Dict[str, Any] = {"optimizer": optimizer}
        if self.args.scheduler:
            scheduler_kwargs = self.args.scheduler_kwargs or {}
            if self.args.scheduler == "noam":
                scheduler_kwargs["steps_per_epoch"] = scheduler_kwargs.get(
                    "steps_per_epoch", self.n_train // self.args.batch_size
                )
                scheduler_kwargs["total_epochs"] = scheduler_kwargs.get(
                    "total_epochs", [self.trainer.max_epochs]
                )
                scheduler_kwargs["init_lr"] = scheduler_kwargs.get(
                    "init_lr", self.args.lr
                )
                scheduler_kwargs["final_lr"] = scheduler_kwargs.get(
                    "final_lr", self.args.lr
                )

            sched = get_lr_scheduler(
                self.args.scheduler, optimizer, scheduler_kwargs=scheduler_kwargs
            )
            optimizer_dict.update(
                {
                    "lr_scheduler": {
                        "scheduler": sched,
                        "interval": "step"
                        if self.args.optimizer == "noam"
                        else "epoch",
                    }
                }
            )
        return optimizer_dict


class E2ELightningModule(PyGLightningModule):
    """End-to-end PCP-SAFT prediction"""

    def __init__(
        self,
        args: PyGTrainArgs,
        phase_equilibria_emulator: PcSaftEmulatorLightningModule,
        critical_point_emulator: PcSaftEmulatorLightningModule,
        pcsaft_emulator_input_columns: List[str],
        pcsaft_emulator_output_columns: List[str],
    ):
        if set(args.target_columns) != set(["rho_l", "Pvap"]):
            raise ValueError(
                "E2E model only supports liquid density and Pvap as targets."
            )

        model = MPNN(args, num_targets=len(pcsaft_emulator_input_columns))

        super().__init__(args, model)

        ### PC-SAFT Emulator ###
        self.phase_head = phase_equilibria_emulator
        self.critical_head = critical_point_emulator
        self.pcsaft_emulator_input_columns = pcsaft_emulator_input_columns
        self.input_idx = [
            self.args.target_columns.index(col)
            for col in self.pcsaft_emulator_input_columns
        ]
        self.pcsaft_emulator_output_columns = pcsaft_emulator_output_columns

    def setup(self, stage: str):
        super().setup(stage)
        # Make sure the PC-SAFT emulator heads are not trained
        for head in [self.phase_head, self.critical_head]:
            for param in head.parameters():
                param.requires_grad = False

    def forward(
        self,
        batch: Any,
        dataloader_idx: int = 0,
    ):
        associating = batch.associating  # type: ignore
        if self.training:
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()  # type: ignore
                self.log("lr", sch.get_last_lr()[0])  # type: ignore

        # Get PCP-SAFT parameter predictions
        parameter_preds = self.model(batch)

        # Set non-associating to zero with offset for normalization
        if self.associating_target_indices:
            idx = self.associating_target_indices
            # offsets for non-associating (hence the -1.0)
            offset = (
                (associating - 1.0)
                * self.non_associating_values
                * torch.ones_like(parameter_preds[:, idx])
            )
            parameter_preds[:, idx] = associating * parameter_preds[:, idx] - offset

        ### PCP-SAFT prediction ###
        # Match column ordering to emulator
        parameter_preds = parameter_preds[:, self.input_idx]

        # Get critical point predictions
        critical_points, failure_prob = self.critical_head(
            parameter_preds, scale_inputs=False
        )
        critical_pressures = torch.exp(critical_points[:, 1])

        # Get phase diagram predictions
        inverse_tr = batch.inverse_tr
        mean = self.phase_head.means[-1]
        std = self.phase_head.stds[-1]
        inverse_tr_scaled = (inverse_tr - mean) / std
        x = torch.cat([parameter_preds, inverse_tr_scaled], dim=1)
        phase_preds = self.phase_head(x, scale_inputs=False)

        # Invert vapor pressure transforms
        if "log_Pr" in self.pcsaft_emulator_output_columns:
            idx = self.pcsaft_emulator_output_columns.index("log_Pr")
            log_pr = phase_preds[:, idx]
            pr = torch.exp(log_pr)
            vapor_pressures = pr * critical_pressures
        elif "Pr" in self.pcsaft_emulator_output_columns:
            idx = self.pcsaft_emulator_output_columns.index("Pr")
            vapor_pressures = phase_preds[:, idx] * critical_pressures
        else:
            idx = self.pcsaft_emulator_output_columns.index("P")
            vapor_pressures = phase_preds[:, idx]

        # Invert density transforms
        m_idx = self.args.target_columns.index("m")
        m = parameter_preds[:, m_idx]
        if "rho_l_s" in self.pcsaft_emulator_output_columns:
            idx = self.pcsaft_emulator_output_columns.index("rho_l_s")
            rho_l_s = phase_preds[:, idx]
            liquid_densities = rho_l_s / m
        else:
            idx = self.pcsaft_emulator_output_columns.index("rho_l")
            liquid_densities = phase_preds[:, idx]
        if "rho_v_s" in self.pcsaft_emulator_output_columns:
            idx = self.pcsaft_emulator_output_columns.index("rho_v_s")
            rho_v_s = phase_preds[:, idx]
            vapor_densities = rho_v_s / m
        else:
            idx = self.pcsaft_emulator_output_columns.index("rho_v")
            vapor_densities = phase_preds[:, idx]

        return (
            parameter_preds,
            vapor_pressures,
            liquid_densities,
            vapor_densities,
            failure_prob,
        )

    def calc_loss(self, batch: PyGDataset):
        target = self.scale(batch.y)  # type: ignore

        # Run model
        (
            parameter_preds,
            vapor_pressures,
            liquid_densities,
            vapor_densities,
            failure_prob,
        ) = self(batch, inverse_scale=False)

        # First version of loss:
        #  on the vapor pressure and liquid density
        # with an extra penality for failed simulations
        pvap_idx = self.args.target_columns.index("Pvap")
        rho_l_idx = self.args.target_columns.index("rho_l")
        loss = (
            F.mse_loss(vapor_pressures, target[:, pvap_idx], reduction="none")
            / target[:, pvap_idx]
            + F.mse_loss(liquid_densities, target[:, rho_l_idx], reduction="none")
            / target[:, rho_l_idx]
            + failure_prob
        )
        loss = loss.mean()

        if self.args.target_columns == ["rho_l", "Pvap"]:
            all_preds = torch.cat(
                [
                    liquid_densities,
                    vapor_pressures,
                    parameter_preds,
                    vapor_densities,
                    failure_prob,
                ],
                dim=1,
            )
        elif self.args.target_columns == ["Pvap", "rho_l"]:
            all_preds = torch.cat(
                [
                    vapor_pressures,
                    liquid_densities,
                    parameter_preds,
                    vapor_densities,
                    failure_prob,
                ],
                dim=1,
            )
        else:
            raise NotImplementedError()
        return loss, all_preds

    def log_metrics(self, preds: torch.Tensor, targets: torch.Tensor, subset: str):
        preds = torch.atleast_2d(preds)
        targets = torch.atleast_2d(targets)
        for i, target in enumerate(self.args.target_columns):
            for metric_name, metric in self.metrics.items():
                score = metric(preds[:, i], targets[:, i])
                self.log(f"{subset}_{target}_{metric_name}", score)

    def configure_optimizers(
        self,
    ):
        return super().configure_optimizers(self.model.parameters)


class PyGValidator(LightningValidator):
    lit_model: PyGLightningModule
    datamodule: PyGDataModule

    def __init__(
        self,
        args: TrainArgs,
        lit_model: PyGLightningModule,
        datamodule: PyGDataModule,
    ):
        super().__init__(args, lit_model, datamodule)

    def get_ground_truth_and_smiles(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
        dataset = self.datamodule._dataset
        y = dataset.data.y
        train_ground_truth = y[dataset.train_mask].numpy()
        val_ground_truth = y[dataset.valid_mask].numpy()
        test_ground_truth = y[dataset.test_mask].numpy()
        train_smiles = [d.smiles for d in dataset[dataset.train_mask]]  # type: ignore
        val_smiles = [d.smiles for d in dataset[dataset.valid_mask]]  # type: ignore
        test_smiles = [d.smiles for d in dataset[dataset.test_mask]]  # type: ignore
        return (
            train_ground_truth,
            val_ground_truth,
            test_ground_truth,
            train_smiles,
            val_smiles,
            test_smiles,
        )

    def _collate(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors)

    def get_target(self, t: torch.Tensor, target: str, idx: int) -> np.ndarray:
        return t[:, idx].cpu().detach().numpy()
