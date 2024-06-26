import logging
from dataclasses import asdict, dataclass
from datetime import datetime as dt
from functools import reduce
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from chemprop.data import (
    MoleculeDatapoint,
    MoleculeDataset,
    set_cache_graph,
    set_cache_mol,
)
from chemprop.data.data import construct_molecule_batch
from chemprop.data.scaler import StandardScaler
from chemprop.features import BatchMolGraph, MolGraph, get_atom_fdim, get_bond_fdim
from chemprop.nn_utils import (
    get_activation_function,
    index_select_ND,
    initialize_weights,
)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from mlsaft.extras.utils.metrics import calculate_metrics
from mlsaft.extras.utils.optimizer import get_lr_scheduler, get_optimizer
from mlsaft.extras.utils.pcsaft import is_associating

from .base import LightningValidator, TrainArgs

Metric = Literal[
    "auc",
    "prc-auc",
    "rmse",
    "mae",
    "mse",
    "r2",
    "accuracy",
    "cross_entropy",
    "binary_cross_entropy",
    "sid",
    "wasserstein",
    "f1",
    "mcc",
    "bounded_rmse",
    "bounded_mae",
    "bounded_mse",
]


@dataclass
class ChempropTrainArgs(TrainArgs):
    # Architecture
    num_convs: int = 3
    hidden_size: int = 300
    dim_fingerprint: int = 300
    activation: str = "ReLU"
    seed: int = 0
    ffn_num_layers: int = 1
    dropout: float = 0.0
    pool_type: Literal["mean", "add", "norm"] = "mean"
    """Aggregation scheme for atomic vectors into molecular vectors"""
    aggregation_norm: int = 100
    mpn_shared: bool = False
    model_type: str = "D-MPNN"
    wandb_artifact_name: str = "chemprop"
    freeze_encoder: bool = False
    filter_non_associating_inside_loss: bool = True
    balanced_associating_sampling: bool = True
    sum_task_losses: bool = False

    @property
    def num_tasks(self) -> int:
        return len(self.target_columns)

    @property
    def num_molecules(self) -> int:
        return len(self.smiles_columns)


SMILES_TO_GRAPH: Dict[str, MolGraph] = {}


class CachedMoleculeDataset(MoleculeDataset):
    cache_graph: bool = True

    def batch_graph(self) -> List[BatchMolGraph]:
        r"""
        Constructs a :class:`~chemprop.features.BatchMolGraph` with the graph featurization of all the molecules.

        .. note::
           The :class:`~chemprop.features.BatchMolGraph` is cached in after the first time it is computed
           and is simply accessed upon subsequent calls to :meth:`batch_graph`. This means that if the underlying
           set of :class:`MoleculeDatapoint`\ s changes, then the returned :class:`~chemprop.features.BatchMolGraph`
           will be incorrect for the underlying data.

        :return: A list of :class:`~chemprop.features.BatchMolGraph` containing the graph featurization of all the
                 molecules in each :class:`MoleculeDatapoint`.
        """
        if self._batch_graph is None:
            self._batch_graph = []
            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                for s, m in zip(d.smiles, d.mol):
                    if s in SMILES_TO_GRAPH:
                        mol_graph = SMILES_TO_GRAPH[s]
                    else:
                        if len(d.smiles) > 1 and (
                            d.atom_features is not None or d.bond_features is not None
                        ):
                            raise NotImplementedError(
                                "Atom descriptors are currently only supported with one molecule "
                                "per input (i.e., number_of_molecules = 1)."
                            )

                        mol_graph = MolGraph(
                            m,
                            d.atom_features,
                            d.bond_features,
                            overwrite_default_atom_features=d.overwrite_default_atom_features,
                            overwrite_default_bond_features=d.overwrite_default_bond_features,
                        )
                        if self.cache_graph:
                            SMILES_TO_GRAPH[s] = mol_graph
                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)

            self._batch_graph = [
                BatchMolGraph([g[i] for g in mol_graphs])
                for i in range(len(mol_graphs[0]))
            ]

        return self._batch_graph


class ChempropDataModule(pl.LightningDataModule):
    _train_dataset: MoleculeDataset
    _val_dataset: MoleculeDataset
    _test_dataset: MoleculeDataset

    def __init__(
        self,
        args: ChempropTrainArgs,
        data: pd.DataFrame,
        split_idx: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ):
        super().__init__()
        self.args = args
        self._data = data
        self._split_idx = split_idx
        self._scaler: StandardScaler
        self.logger = logging.getLogger(__name__)
        set_cache_mol(True)
        set_cache_graph(True)

    def setup(self, stage: str) -> None:
        associating = np.array(
            [
                [1.0 if is_associating(smiles) else 0.0]
                for smiles in self._data[self.args.smiles_columns[0]]
            ]
        )
        self._train_dataset = MoleculeDataset(
            [
                MoleculeDatapoint(
                    smiles=self._data.iloc[i][self.args.smiles_columns].tolist(),
                    targets=self._data.iloc[i][self.args.target_columns].tolist(),
                    features=associating[i],
                )
                for i in self._split_idx[0]
            ]
        )
        self._val_dataset = MoleculeDataset(
            [
                MoleculeDatapoint(
                    smiles=self._data.iloc[i][self.args.smiles_columns].tolist(),
                    targets=self._data.iloc[i][self.args.target_columns].tolist(),
                    features=associating[i],
                )
                for i in self._split_idx[1]
            ]
        )
        self._test_dataset = MoleculeDataset(
            [
                MoleculeDatapoint(
                    smiles=self._data.iloc[i][self.args.smiles_columns].tolist(),
                    targets=self._data.iloc[i][self.args.target_columns].tolist(),
                    features=associating[i],
                )
                for i in self._split_idx[2]
            ]
        )

        # Normalize targets
        self._scaler = self._train_dataset.normalize_targets()

        valid_targets = [d.raw_targets for d in self._val_dataset._data]
        valid_targets = self._scaler.transform(valid_targets).tolist()
        self._val_dataset.set_targets(valid_targets)

        test_targets = [d.raw_targets for d in self._test_dataset._data]
        test_targets = self._scaler.transform(test_targets).tolist()
        self._test_dataset.set_targets(test_targets)

        if self.args.balanced_associating_sampling:
            self.logger.info("Calculating sampler weights")
            non_associating_counts = len(associating[associating == 0])
            associating_counts = len(associating[associating == 1])
            counts = np.array([non_associating_counts, associating_counts])
            weights = 1.0 / counts
            self._association_weights = np.array([weights[int(a)] for a in associating])
        else:
            self._association_weights = None

    def train_dataloader(self) -> DataLoader:
        if self._association_weights is not None:
            train_weights: np.ndarray = self._association_weights[self._split_idx[0]]
            sampler = WeightedRandomSampler(
                weights=train_weights.tolist(),
                num_samples=len(train_weights),
                replacement=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self.args.batch_size,
            collate_fn=construct_molecule_batch,
            num_workers=self.args.num_workers,
            shuffle=shuffle,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._val_dataset,
            collate_fn=construct_molecule_batch,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._test_dataset,
            collate_fn=construct_molecule_batch,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        train_no_shuffle = DataLoader(
            dataset=self._train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=construct_molecule_batch,
            shuffle=False,
        )
        return [train_no_shuffle, self.val_dataloader(), self.test_dataloader()]


class DMPNNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(
        self,
        args: ChempropTrainArgs,
        atom_fdim: int,
        bond_fdim: int,
    ):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        :param hidden_size: Hidden layers dimension
        :param bias: Whether to add bias to linear layers
        :param depth: Number of message passing steps
        """
        super().__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = False
        self.depth = args.num_convs
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = False  # Directed MPNN
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.aggregation = args.pool_type
        self.aggregation_norm = args.aggregation_norm

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(
            torch.zeros(self.hidden_size), requires_grad=False
        )

        # Input
        input_dim = self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    def forward(self, mol_graph: BatchMolGraph) -> torch.Tensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                        a batch of molecular graphs.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components(
            atom_messages=False
        )
        f_atoms, f_bonds, a2b, b2a, b2revb = (
            f_atoms.to(self.device),
            f_bonds.to(self.device),
            a2b.to(self.device),
            b2a.to(self.device),
            b2revb.to(self.device),
        )

        # Input
        input = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
            # message      a_message = sum(nei_a_message)      rev_message
            nei_a_message = index_select_ND(
                message, a2b
            )  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2b
        nei_a_message = index_select_ND(
            message, a2x
        )  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat(
            [f_atoms, a_message], dim=1
        )  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation == "mean":
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == "add":
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == "norm":
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class RegressionMPN(nn.Module):
    def __init__(
        self,
        args: ChempropTrainArgs,
    ):
        super().__init__()
        self.atom_fdim = get_atom_fdim(
            overwrite_default_atom=False,
            is_reaction=False,
        )
        self.bond_fdim = get_bond_fdim(
            overwrite_default_atom=False,
            overwrite_default_bond=False,
            atom_messages=False,
            is_reaction=False,
        )
        self.features_only = False
        self.use_input_features = False
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.atom_descriptors = None
        self.overwrite_default_atom_features = False
        self.overwrite_default_bond_features = False
        self.reaction_solvent = False

        if args.mpn_shared:
            self.encoder = nn.ModuleList(
                [DMPNNEncoder(args, self.atom_fdim, self.bond_fdim)]
                * args.num_molecules
            )
        else:
            self.encoder = nn.ModuleList(
                [
                    DMPNNEncoder(args, self.atom_fdim, self.bond_fdim)
                    for _ in range(args.num_molecules)
                ]
            )

    def forward(
        self,
        batch: List[BatchMolGraph],
    ) -> torch.FloatTensor:
        encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]
        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)
        return output


class RegressionMoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: ChempropTrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super().__init__()
        self.output_size = args.num_tasks
        self.encoder = RegressionMPN(args)
        if args.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.create_ffn(args)
        initialize_weights(self)

    def create_ffn(self, args: ChempropTrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        first_linear_dim = args.hidden_size * args.num_molecules
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [dropout, nn.Linear(first_linear_dim, self.output_size)]
        else:
            ffn = [dropout, nn.Linear(first_linear_dim, args.dim_fingerprint)]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend(
                    [
                        activation,
                        dropout,
                        nn.Linear(args.dim_fingerprint, args.dim_fingerprint),
                    ]
                )
            ffn.extend(
                [
                    activation,
                    dropout,
                    nn.Linear(args.dim_fingerprint, self.output_size),
                ]
            )

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, batch: List[BatchMolGraph]) -> torch.Tensor:
        return self.ffn(self.encoder(batch))


class ChempropLightningModule(pl.LightningModule):
    """A directed message-passing neural network base class"""

    means: torch.Tensor
    stds: torch.Tensor

    def __init__(self, args: ChempropTrainArgs, model: RegressionMoleculeModel):
        super().__init__()
        self.args = args
        self.model = model
        means = torch.zeros((len(args.target_columns),))
        stds = torch.ones((len(args.target_columns),))
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.save_hyperparameters(asdict(args), ignore=["model"])
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
        if stage == "fit":
            dm = self.trainer.datamodule  # type: ignore
            for i, col in enumerate(self.args.target_columns):
                self.means[i] = dm._scaler.means[i]
                self.stds[i] = dm._scaler.stds[i]
            self.n_train = len(dm._train_dataset)

        # Adjust non-associating values for normalization
        if self.associating_target_indices is not None:
            idx = self.associating_target_indices
            values = -1.0 * self.means[idx].clone() / self.stds[idx].clone()
            state_dict = self.state_dict()
            state_dict["non_associating_values"] = values
            self.load_state_dict(state_dict)

    def training_step(self, batch: MoleculeDataset, batch_idx: int):
        loss, preds = self.calc_loss(batch)
        self.log("train_loss", loss, batch_size=len(batch))
        with torch.no_grad():
            self.log_metrics(
                preds.cpu().numpy(),
                np.array(batch.targets()),
                "train",
                batch_size=len(batch),
            )
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()  # type: ignore
        return loss

    def training_step_end(self, outputs):
        sch = self.lr_schedulers()
        if sch is not None:
            self.log("lr", sch.get_last_lr()[0])  # type: ignore

    def validation_step(self, batch: MoleculeDataset, batch_idx: int):
        loss, preds = self.calc_loss(batch)
        self.log("val_loss", loss, batch_size=len(batch))
        self.log_metrics(
            preds.cpu().detach().numpy(),
            np.array(batch.targets()),
            "val",
            batch_size=len(batch),
        )
        return loss

    def test_step(self, batch: MoleculeDataset, batch_idx: int):
        self.model.eval()
        loss, preds = self.calc_loss(batch)
        self.log("test_loss", loss, batch_size=len(batch))
        self.log_metrics(
            preds.cpu().detach().numpy(),
            np.array(batch.targets()),
            "test",
            batch_size=len(batch),
        )
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch, inverse_scale=True)

    def log_metrics(self, preds, targets, subset: str, batch_size: int):
        for i, target in enumerate(self.args.target_columns):
            current_scores = calculate_metrics(
                preds[:, i], targets[:, i], scores=self.args.metrics
            )
            for score_name, score in current_scores.items():
                self.log(
                    f"{subset}_{target}_{score_name}", score, batch_size=batch_size
                )

    def on_train_epoch_start(self) -> None:
        self.start_time = dt.now()

    def on_train_epoch_end(self) -> None:
        epoch_time = (dt.now() - self.start_time).total_seconds()
        training_throughput = self.n_train / epoch_time
        self.log("training_throughput", training_throughput)
        self.log("time_per_step", self.args.batch_size / training_throughput)

    def forward(
        self,
        batch: MoleculeDataset,
        dataloader_idx: int = 0,
        inverse_scale: bool = False,
    ):
        # Unpack batch
        mol_batch = batch.batch_graph()

        # Run model
        m = self.model.to(self.device)
        preds = m(mol_batch)

        # Set non-associating to zero with offset for normalization
        if self.associating_target_indices:
            associating = torch.tensor(batch.features(), dtype=preds.dtype).to(
                self.device
            )
            idx = self.associating_target_indices
            # offsets for non-associating (hence the -1.0)
            offset = (
                (associating - 1.0)
                * self.non_associating_values
                * torch.ones_like(preds[:, idx])
            )
            preds[:, idx] = associating * preds[:, idx] - offset

        # Scale targets
        if inverse_scale:
            preds = self.inverse_scale(preds)

        return preds

    def inverse_scale(self, preds: torch.Tensor) -> torch.Tensor:
        scaler = StandardScaler(self.means.cpu().numpy(), self.stds.cpu().numpy())
        preds_scaled = scaler.inverse_transform(preds.cpu().detach().numpy()).astype(
            float
        )
        return torch.tensor(preds_scaled)

    def calc_loss(self, batch: MoleculeDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unpack batch
        target_batch = batch.targets()

        # Weights and masking
        targets = torch.tensor(
            [[0 if x is None else x for x in tb] for tb in target_batch]
        )  # shape(batch, tasks)

        if self.args.target_weights is not None:
            target_weights = torch.tensor(self.args.target_weights).unsqueeze(
                0
            )  # shape(1,tasks)
        else:
            target_weights = torch.ones(targets.shape[1]).unsqueeze(0)

        # Run model
        preds = self(batch, inverse_scale=False)

        # Move tensors to correct device
        torch_device = preds.device
        targets = targets.to(torch_device)
        target_weights = target_weights.to(torch_device)

        # Compute loss
        if self.associating_target_indices:
            associating = torch.tensor(batch.features(), dtype=preds.dtype).to(
                self.device
            )
            mask = torch.ones_like(targets)
            for idx in self.associating_target_indices:
                mask[~associating.bool().squeeze(), idx] = 0.0
            loss = F.mse_loss(preds, targets, reduction="none") * target_weights * mask
        else:
            loss = F.mse_loss(preds, targets, reduction="none") * target_weights
        if self.args.sum_task_losses:
            loss = loss.mean(0).sum()
        else:
            loss = loss.mean()

        return loss, preds

    def configure_optimizers(
        self,
    ):
        optimizer_kwargs = self.args.optimizer_kwargs or {"lr": self.args.lr}
        optimizer = get_optimizer(
            self.args.optimizer, self.parameters(), optimizer_kwargs=optimizer_kwargs
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


class ChempropValidator(LightningValidator):
    lit_model: ChempropLightningModule
    datamodule: ChempropDataModule

    def __init__(
        self,
        args: ChempropTrainArgs,
        lit_model: ChempropLightningModule,
        datamodule: ChempropDataModule,
    ):
        super().__init__(args, lit_model, datamodule)

    def get_ground_truth_and_smiles(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
        train_ground_truth = np.array(
            [d.raw_targets for d in self.datamodule._train_dataset._data]
        )
        val_ground_truth = np.array(
            [d.raw_targets for d in self.datamodule._val_dataset._data]
        )
        test_ground_truth = np.array(
            [d.raw_targets for d in self.datamodule._test_dataset._data]
        )
        train_smiles = [d.smiles[0] for d in self.datamodule._train_dataset._data]
        val_smiles = [d.smiles[0] for d in self.datamodule._val_dataset._data]
        test_smiles = [d.smiles[0] for d in self.datamodule._test_dataset._data]

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

    def get_target(
        self, t: Union[np.ndarray, torch.Tensor], target: str, idx: int
    ) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            return t[:, idx].cpu().detach().numpy()
        else:
            return t[:, idx]
