import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

from mlsaft.extras.utils.metrics import get_torchmetrics
from mlsaft.extras.utils.optimizer import get_lr_scheduler, get_optimizer
from mlsaft.extras.utils.pcsaft import is_associating

from .base import LightningValidator, TrainArgs


class MoleculeTensorDataset(TensorDataset):
    tensors: Tuple[torch.Tensor, ...]
    smiles: List[str]
    associating: Optional[torch.Tensor]

    def __init__(
        self,
        *tensors: torch.Tensor,
        smiles: List[str],
    ) -> None:
        assert all(
            tensors[0].size(0) == tensor.size(0) for tensor in tensors
        ), "Size mismatch between tensors"
        assert len(tensors[0]) == len(
            smiles
        ), "Size mismatch between tensors and smiles"
        self.tensors = tensors
        self.smiles = smiles


@dataclass
class FFNTrainArgs(TrainArgs):
    """Args feed forward neural network training"""

    # Model architecture
    fp_bits: int = 2048
    activation: str = "ReLU"
    hidden_layer_dims: Optional[List[int]] = None
    num_hidden_layers: int = 1
    model_type: str = "FFN"
    dropout: float = 0.0
    filter_non_associating_inside_loss: bool = True
    balanced_associating_sampling: bool = True
    wandb_artifact_name: str = "ffn"


class FFN(nn.Module):
    def __init__(
        self,
        args: FFNTrainArgs,
    ):
        super().__init__()
        self.args = args

        num_hidden_layers = self.args.num_hidden_layers
        hidden_layer_dims = self.args.hidden_layer_dims
        input_dim = self.args.fp_bits
        output_dim = len(self.args.target_columns)
        if hidden_layer_dims is None:
            hidden_layer_dims = [
                max(int(2 ** (-i) * input_dim), output_dim)
                for i in range(1, num_hidden_layers + 1)
            ]
            hidden_layer_dims.append(output_dim)

        if len(hidden_layer_dims) != num_hidden_layers + 1:
            raise ValueError(
                "Number of hidden layers does not match hidden layer dimensions"
            )

        # Initialize weights
        if self.args.activation == "LeakyReLU":
            act_func = nn.LeakyReLU
        elif self.args.activation == "ReLU":
            act_func = nn.ReLU
        elif self.args.activation == "ELU":
            act_func = nn.ELU
        else:
            raise ValueError(
                f"Activation function {self.args.activation} not supported"
            )

        layers = []
        layers.append(nn.Linear(input_dim, hidden_layer_dims[0]))
        layers.append(act_func())
        if num_hidden_layers > 0:
            for i in range(num_hidden_layers):
                layers.append(nn.Linear(hidden_layer_dims[i], hidden_layer_dims[i + 1]))
                layers.append(act_func())
                self.dropout = nn.Dropout(self.args.dropout)
        layers.append(nn.Linear(hidden_layer_dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return self.layers(x)


class FFNDataModule(pl.LightningDataModule):
    _train_dataset: MoleculeTensorDataset
    _valid_dataset: MoleculeTensorDataset
    _test_dataset: MoleculeTensorDataset
    _scaler: StandardScaler

    def __init__(
        self,
        args: FFNTrainArgs,
        data: pd.DataFrame,
        fps: np.ndarray,
        split_idx: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ):
        super().__init__()
        self.args = args
        self._fps = fps
        self._df = data
        self.split_idx = split_idx
        self.logger = logging.getLogger(__name__)

    def setup(self, stage: str) -> None:
        targets = self._df[self.args.target_columns].to_numpy()
        smiles = self._df[self.args.smiles_columns[0]]
        associating = torch.tensor(
            [[1.0 if is_associating(smi) else 0.0] for smi in smiles]
        )

        # Set up scaler
        self._scaler = StandardScaler()
        self._scaler.fit(targets[self.split_idx[0]])
        targets_scaled = self._scaler.transform(targets)

        # Set up datasets
        self._train_dataset = MoleculeTensorDataset(
            torch.tensor(self._fps[self.split_idx[0]], dtype=torch.float),
            torch.tensor(targets_scaled[self.split_idx[0]], dtype=torch.float),  # type: ignore
            associating[self.split_idx[0]],
            smiles=smiles[self.split_idx[0]].to_list(),
        )
        self._valid_dataset = MoleculeTensorDataset(
            torch.tensor(self._fps[self.split_idx[1]], dtype=torch.float),
            torch.tensor(targets_scaled[self.split_idx[1]], dtype=torch.float),  # type: ignore
            associating[self.split_idx[1]],
            smiles=smiles[self.split_idx[1]].to_list(),
        )
        self._test_dataset = MoleculeTensorDataset(
            torch.tensor(self._fps[self.split_idx[2]], dtype=torch.float),
            torch.tensor(targets_scaled[self.split_idx[2]], dtype=torch.float),  # type: ignore
            associating[self.split_idx[2]],
            smiles=smiles[self.split_idx[2]].to_list(),
        )
        # self._df = pd.DataFrame()  # free up memory
        # self._fps = np.array([])  # free up memory

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
            train_weights: np.ndarray = self._association_weights[self.split_idx[0]]
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
            dataset=self._train_dataset,  # type: ignore
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=shuffle,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._valid_dataset,  # type: ignore
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._test_dataset,  # type: ignore
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        train_no_shuffle = DataLoader(
            dataset=self._train_dataset,  # type: ignore
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

        return [train_no_shuffle, self.val_dataloader(), self.test_dataloader()]


class FFNLightningModule(pl.LightningModule):
    means: torch.Tensor
    stds: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        args: FFNTrainArgs,
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.lr = args.lr
        means = torch.zeros((len(args.target_columns),))
        stds = torch.ones((len(args.target_columns),))
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.save_hyperparameters(asdict(args), ignore=["model"])
        self.associating_target_indices = None
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
        self.metrics = nn.ModuleDict(
            {metric: get_torchmetrics(metric) for metric in self.args.metrics}
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            dm: FFNDataModule = self.trainer.datamodule  # type: ignore
            for i, _ in enumerate(self.args.target_columns):
                self.means[i] = dm._scaler.mean_[i]  # type: ignore
                self.stds[i] = dm._scaler.scale_[i]  # type: ignore

        # Adjust non-associating values for normalization
        if self.associating_target_indices is not None:
            idx = self.associating_target_indices
            values = -1.0 * self.means[idx].clone() / self.stds[idx].clone()
            state_dict = self.state_dict()
            state_dict["non_associating_values"] = values
            self.load_state_dict(state_dict)

        self.n_train = len(self.trainer.datamodule._train_dataset)  # type: ignore

    def training_step(self, batch: Dataset, batch_idx: int):
        _, targets, _ = batch
        loss, preds = self.calc_loss(batch)
        self.log("train_loss", loss)
        self.log_metrics(
            preds,
            targets,
            "train",
        )
        return loss

    def validation_step(self, batch: Dataset, batch_idx: int):
        _, targets, _ = batch
        loss, preds = self.calc_loss(batch)
        self.log("val_loss", loss)
        self.log_metrics(
            preds,
            targets,
            "val",
        )
        return loss

    def test_step(self, batch: Dataset, batch_idx: int):
        _, targets, _ = batch
        loss, preds = self.calc_loss(batch)
        self.log("test_loss", loss)
        self.log_metrics(
            preds,
            targets,
            "test",
        )
        return loss

    def predict_step(
        self, batch: Dataset, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        return self(batch, inverse_scale=True)

    def log_metrics(self, preds: torch.Tensor, targets: torch.Tensor, subset: str):
        preds = torch.atleast_2d(preds)
        targets = torch.atleast_2d(targets)
        for i, target in enumerate(self.args.target_columns):
            for metric_name, metric in self.metrics.items():
                score = metric(preds[:, i], targets[:, i])
                self.log(f"{subset}_{target}_{metric_name}", score)

    def forward(self, batch, inverse_scale=False):
        x, _, associating = batch
        preds = self.model(x)

        # Set non-associating to zero with offset for normalization
        if self.associating_target_indices:
            idx = self.associating_target_indices
            # offsets for non-associating (hence the -1.0)
            offset = (
                (associating - 1.0)
                * self.non_associating_values
                * torch.ones_like(preds[:, idx])
            )
            preds[:, idx] = associating * preds[:, idx] - offset

        if inverse_scale:
            preds = self.inverse_scale(preds)
        return preds

    def calc_loss(self, batch):
        _, targets, associating = batch

        preds = self(batch, inverse_scale=False)

        if self.associating_target_indices:
            idx = self.associating_target_indices
            mask = torch.ones_like(targets)
            for idx in self.associating_target_indices:
                mask[~associating.bool().squeeze(), idx] = 0.0
            loss = F.mse_loss(preds, targets, reduction="none") * mask
        else:
            loss = F.mse_loss(preds, targets, reduction="none")
        loss = loss.mean()
        return loss, preds

    def inverse_scale(self, preds: torch.Tensor) -> torch.Tensor:
        scaler = StandardScaler()
        scaler.mean_ = self.means.cpu().numpy()
        scaler.scale_ = self.stds.cpu().numpy()
        preds_scaled = scaler.inverse_transform(preds.cpu().detach().numpy()).astype(
            float
        )
        return torch.tensor(preds_scaled)

    def configure_optimizers(
        self,
        parameters: Optional[Any] = None,
    ):
        optimizer_kwargs = self.args.optimizer_kwargs or {"lr": self.lr}
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


class FFNValidator(LightningValidator):
    lit_model: FFNLightningModule
    datamodule: FFNDataModule

    def __init__(
        self,
        args: TrainArgs,
        lit_model: FFNLightningModule,
        datamodule: FFNDataModule,
    ):
        super().__init__(args, lit_model, datamodule)

    def get_ground_truth_and_smiles(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
        train_ground_truth = self.lit_model.inverse_scale(
            self.datamodule._train_dataset.tensors[1]
        ).numpy()
        val_ground_truth = self.lit_model.inverse_scale(
            self.datamodule._valid_dataset.tensors[1]
        ).numpy()
        test_ground_truth = self.lit_model.inverse_scale(
            self.datamodule._test_dataset.tensors[1]
        ).numpy()
        train_smiles = self.datamodule._train_dataset.smiles
        val_smiles = self.datamodule._valid_dataset.smiles
        test_smiles = self.datamodule._test_dataset.smiles
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
