import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

from mlsaft.extras.utils.metrics import get_torchmetrics
from mlsaft.extras.utils.optimizer import get_lr_scheduler, get_optimizer

from .base import LightningValidator, TrainArgs


@dataclass
class PcSaftEmulatorTrainArgs(TrainArgs):
    """Args feed forward neural network training"""

    # Model architecture
    input_columns: List[str] = field(
        default_factory=lambda: ["m", "sigma", "epsilon_k", "mu", "epsilonAB", "KAB"]
    )
    failure_status_column: str = "failed"
    activation: str = "ReLU"
    hidden_layer_dims: Optional[List[int]] = None
    num_hidden_layers: int = 1
    dropout: float = 0.0
    predict_failure: bool = True

    # Training
    classification_metrics: List[str] = field(
        default_factory=lambda: ["binary_accuracy", "binary_aucroc", "binary_f1"]
    )
    train_frac: float = 1.0  # Fraction of training data to use
    classification_cutoff: float = 0.5
    balanced_failure_sampling: bool = False

    # Wandb
    model_type: str = "PcSaftEmulator"
    wandb_artifact_name: str = "pcsaft_emulator"
    freeze_normalization: bool = False


class PcSaftEmulator(nn.Module):
    def __init__(
        self,
        args: PcSaftEmulatorTrainArgs,
    ):
        super().__init__()
        self.args = args

        num_hidden_layers = self.args.num_hidden_layers
        hidden_layer_dims = self.args.hidden_layer_dims
        input_dim = len(self.args.input_columns)
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

        regression_layers = []
        regression_layers.append(nn.Linear(input_dim, hidden_layer_dims[0]))
        regression_layers.append(act_func())
        if num_hidden_layers > 0:
            for i in range(num_hidden_layers):
                regression_layers.append(
                    nn.Linear(hidden_layer_dims[i], hidden_layer_dims[i + 1])
                )
                regression_layers.append(act_func())
                self.dropout = nn.Dropout(self.args.dropout)
        regression_layers.append(nn.Linear(hidden_layer_dims[-1], output_dim))
        self.regression_layers = nn.Sequential(*regression_layers)

        if self.args.predict_failure:
            classification_layers = []
            classification_layers.append(nn.Linear(input_dim, hidden_layer_dims[0]))
            classification_layers.append(act_func())
            if num_hidden_layers > 0:
                for i in range(num_hidden_layers):
                    classification_layers.append(
                        nn.Linear(hidden_layer_dims[i], hidden_layer_dims[i + 1])
                    )
                    classification_layers.append(act_func())
                    self.dropout = nn.Dropout(self.args.dropout)
            classification_layers.extend(
                [nn.Linear(hidden_layer_dims[-1], 1), nn.Sigmoid()]
            )
            self.classification_layers = nn.Sequential(*classification_layers)

    def forward(self, x, **kwargs):
        y = self.regression_layers(x)
        if self.args.predict_failure:
            failure_prob = self.classification_layers(x)
        else:
            failure_prob = torch.zeros((len(y), 1), device=y.device)
        return y, failure_prob


class PcSaftEmulatorDataModule(pl.LightningDataModule):
    _train_dataset: TensorDataset
    _valid_dataset: TensorDataset
    _test_dataset: TensorDataset

    def __init__(
        self,
        args: PcSaftEmulatorTrainArgs,
        data: pd.DataFrame,
        split_idx: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ):
        super().__init__()
        self.args = args
        self._df = data
        self.split_idx = split_idx
        self.logger = logging.getLogger(__name__)

    def setup(self, stage: str) -> None:
        failed = torch.tensor(
            self._df[[self.args.failure_status_column]].to_numpy(),
            dtype=torch.float,
        )
        # Set up datasets
        tmp_idx = int(self.args.train_frac * len(self.split_idx[0]))
        train_idx = self.split_idx[0][:tmp_idx]
        self._train_dataset = TensorDataset(
            torch.tensor(
                self._df.iloc[train_idx][self.args.input_columns].to_numpy(),
                dtype=torch.float,
            ),
            torch.tensor(
                self._df.iloc[train_idx][self.args.target_columns].to_numpy(),
                dtype=torch.float,
            ),
            failed[train_idx],
        )
        self._valid_dataset = TensorDataset(
            torch.tensor(
                self._df.iloc[self.split_idx[1]][self.args.input_columns].to_numpy(),
                dtype=torch.float,
            ),
            torch.tensor(
                self._df.iloc[self.split_idx[1]][self.args.target_columns].to_numpy(),
                dtype=torch.float,
            ),
            failed[self.split_idx[1]],
        )
        self._test_dataset = TensorDataset(
            torch.tensor(
                self._df.loc[self.split_idx[2], self.args.input_columns].to_numpy(),
                dtype=torch.float,
            ),
            torch.tensor(
                self._df.loc[self.split_idx[2], self.args.target_columns].to_numpy(),
                dtype=torch.float,
            ),
            failed[self.split_idx[2]],
        )

        if self.args.balanced_failure_sampling:
            failed = self._train_dataset.tensors[2]
            failed_counts = len(failed[failed == 1.0])
            success_counts = 1.0 * len(self._train_dataset) - failed_counts
            counts = np.array([success_counts, failed_counts])
            weights = 1.0 / counts
            self.train_weights = np.array(
                [weights[int(c.squeeze().item())] for c in failed]
            )

    def train_dataloader(self) -> DataLoader:
        if self.args.balanced_failure_sampling:
            sampler = WeightedRandomSampler(
                weights=self.train_weights.tolist(),
                num_samples=len(self.train_weights),
                replacement=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=shuffle,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._valid_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._test_dataset,
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


class PcSaftEmulatorLightningModule(pl.LightningModule):
    means: torch.Tensor
    stds: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        args: PcSaftEmulatorTrainArgs,
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.lr = self.args.lr
        for s in ["input", "output"]:
            means = torch.zeros((len(args.target_columns),))
            stds = torch.ones((len(args.target_columns),))
            self.register_buffer(f"{s}_means", means)
            self.register_buffer(f"{s}_stds", stds)
        self.save_hyperparameters(asdict(args), ignore=["model"])
        self.metrics = nn.ModuleDict(
            {metric: get_torchmetrics(metric) for metric in self.args.metrics}
        )
        self.classification_metrics = nn.ModuleDict(
            {
                metric: get_torchmetrics(metric)
                for metric in self.args.classification_metrics
            }
        )

    def setup(self, stage: str) -> None:
        dm: PcSaftEmulatorDataModule = self.trainer.datamodule  # type: ignore
        if not self.args.freeze_normalization:
            self.input_means = dm._train_dataset.tensors[0].mean(0)
            self.input_stds = dm._train_dataset.tensors[0].std(0)
            self.output_means = dm._train_dataset.tensors[1].mean(0)
            self.output_stds = dm._train_dataset.tensors[1].std(0)
        self.n_train = len(dm._train_dataset)

    def training_step(self, batch: Dataset, batch_idx: int):
        _, targets, failed = batch
        targets = self.scale_outputs(targets)
        regression_loss, classification_loss, preds, pred_failed = self.calc_loss(batch)
        loss = regression_loss + classification_loss
        self.log("train_loss", loss)
        self.log("train_regression_loss", regression_loss)
        self.log("train_classification_loss", classification_loss)
        self.log_metrics(
            preds,
            targets,
            pred_failed,
            failed,
            "train",
        )
        return loss

    def validation_step(self, batch: Dataset, batch_idx: int):
        _, targets, failed = batch
        regression_loss, classification_loss, preds, pred_failed = self.calc_loss(batch)
        targets = self.scale_outputs(targets)
        loss = regression_loss + classification_loss
        self.log("val_loss", loss)
        self.log("val_regression_loss", regression_loss)
        self.log("val_classification_loss", classification_loss)
        self.log_metrics(
            preds,
            targets,
            pred_failed,
            failed,
            "val",
        )
        return loss

    def test_step(self, batch: Dataset, batch_idx: int):
        _, targets, failed = batch
        regression_loss, classification_loss, preds, pred_failed = self.calc_loss(batch)
        targets = self.scale_outputs(targets)
        loss = regression_loss + classification_loss
        self.log("test_loss", loss)
        self.log("test_regression_loss", regression_loss)
        self.log("test_classification_loss", classification_loss)
        self.log_metrics(
            preds,
            targets,
            pred_failed,
            failed,
            "test",
        )
        return loss

    def predict_step(
        self,
        batch: Dataset,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        y, pred_failed = self(batch, inverse_scale_outputs=True)
        if self.args.predict_failure:
            cutoff = torch.tensor(self.args.classification_cutoff).to(
                pred_failed.device
            )
            mask = torch.where(pred_failed > cutoff, 0.0, 1.0)
            y = y * mask
        return y, pred_failed

    def log_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        pred_failed: torch.Tensor,
        failed: torch.Tensor,
        subset: str,
    ):
        preds = torch.atleast_2d(preds)
        pred_failed = torch.atleast_2d(pred_failed)
        cutoff = torch.tensor(self.args.classification_cutoff).to(pred_failed.device)
        preds = torch.where(
            pred_failed > cutoff,
            -self.output_means / self.output_stds,
            preds,
        )
        targets = torch.atleast_2d(targets)
        for i, target in enumerate(self.args.target_columns):
            for metric_name, metric in self.metrics.items():
                score = metric(preds[:, i], targets[:, i])
                self.log(f"{subset}_{target}_{metric_name}", score)
        if self.args.predict_failure:
            for metric_name, metric in self.classification_metrics.items():
                score = metric(pred_failed, failed)
                self.log(f"{subset}_convergence_{metric_name}", score)

    def forward(
        self, batch, scale_inputs: bool = True, inverse_scale_outputs: bool = False
    ):
        x, _, _ = batch
        if scale_inputs:
            x = self.scale_inputs(x)
        preds, pred_failed = self.model(x)

        if inverse_scale_outputs:
            preds = self.inverse_scale_outputs(preds)
        return preds, pred_failed

    def calc_loss(self, batch):
        _, targets, failed = batch
        targets = self.scale_outputs(targets)
        preds, pred_failed = self(batch, inverse_scale_outputs=False)
        # Mask failed simulations from the regression loss
        mask = torch.ones_like(preds, device=preds.device)
        mask[failed.bool().squeeze(), :] = 0.0
        regression_loss = F.mse_loss(preds, targets, reduction="none") * mask
        regression_loss = regression_loss.mean()
        if self.args.predict_failure:
            bce = F.binary_cross_entropy(pred_failed, failed)
        else:
            bce = 0.0
        return regression_loss, bce, preds, pred_failed

    def on_train_epoch_start(self) -> None:
        self.start_time = dt.now()

    def on_train_epoch_end(self) -> None:
        epoch_time = (dt.now() - self.start_time).total_seconds()
        training_throughput = self.n_train / epoch_time
        self.log("training_throughput", training_throughput)
        self.log("time_per_step", self.args.batch_size / training_throughput)

    def scale_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.input_means.unsqueeze(0)) / self.input_stds.unsqueeze(0)

    def inverse_scale_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.output_stds.unsqueeze(0) + self.input_means.unsqueeze(0)

    def scale_outputs(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.output_means.unsqueeze(0)) / self.output_stds.unsqueeze(
            0
        )

    def inverse_scale_outputs(self, preds: torch.Tensor) -> torch.Tensor:
        return preds * self.output_stds.unsqueeze(0) + self.output_means.unsqueeze(0)

    def configure_optimizers(
        self,
    ):
        optimizer_kwargs = self.args.optimizer_kwargs or {"lr": self.lr}
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


class PcSaftEmulatorValidator(LightningValidator):
    lit_model: PcSaftEmulatorLightningModule
    datamodule: PcSaftEmulatorDataModule

    def __init__(
        self,
        args: TrainArgs,
        lit_model: PcSaftEmulatorLightningModule,
        datamodule: PcSaftEmulatorDataModule,
    ):
        super().__init__(args, lit_model, datamodule)

    def get_ground_truth_and_smiles(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, None, None, None]:
        train_ground_truth = self.datamodule._train_dataset.tensors[1].numpy()
        val_ground_truth = self.datamodule._valid_dataset.tensors[1].numpy()
        test_ground_truth = self.datamodule._test_dataset.tensors[1].numpy()
        return (
            train_ground_truth,
            val_ground_truth,
            test_ground_truth,
            None,
            None,
            None,
        )

    def _collate(
        self, tensors: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        preds = [t[0] for t in tensors]
        # failed = [t[1] for t in tensors]
        return torch.cat(preds)

    def get_target(self, t: torch.Tensor, target: str, idx: int) -> np.ndarray:
        return t[:, idx].cpu().detach().numpy()
