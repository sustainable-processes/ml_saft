import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from rdkit import Chem
from rdkit.Chem import Draw, Lipinski
from tqdm import tqdm

import wandb
from dl4thermo.extras.utils import calculate_metrics, parity_plot
from dl4thermo.extras.utils.lightning import load_lightningmodule_from_checkpoint
from dl4thermo.extras.utils.optimizer import OPTIMIZER_OPTIONS, SCHEDULER_OPTIONS
from wandb import plot as wandb_plot


@dataclass
class TrainArgs:
    # Required parameters
    save_dir: str
    smiles_columns: List[str]
    target_columns: List[str]
    model_type: str

    # Training parameters
    lr: float = 0.001
    batch_size: int = 100
    epochs: int = 1000
    optimizer: OPTIMIZER_OPTIONS = "adam"
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    scheduler: SCHEDULER_OPTIONS = None
    scheduler_kwargs: Optional[Dict[str, Any]] = None
    auto_lr_find: bool = False
    early_stopping: bool = False
    metrics: List[str] = field(default_factory=lambda: ["mae", "mse", "r2"])

    # Data and experiment settings
    num_workers: int = 0
    targets_transformation: Optional[List[str]] = None
    target_weights: Optional[List[float]] = None
    data_source: Optional[str] = None
    overwrite_save_dir: bool = True

    # Post processsing
    associating_columns: Optional[List[str]] = None
    filter_non_associating: bool = False

    # Wandb
    wandb_entity: str = "ceb-sre"
    wandb_project: str = "dl4thermo"
    wandb_artifact_name: str = "model_predictions"
    wandb_checkpoint_artifact_id: Optional[str] = None
    checkpoint_name: str = "model.ckpt"
    wandb_tags: Optional[List[str]] = None
    log_all_models: bool = False
    wandb_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Always overwrite the directory
        save_dir = Path(self.save_dir)
        if save_dir.exists() and self.overwrite_save_dir:
            shutil.rmtree(save_dir)
        if self.overwrite_save_dir:
            save_dir.mkdir(parents=True)
        if self.wandb_tags is None:
            self.wandb_tags = []
        if self.wandb_checkpoint_artifact_id and "pretrained" not in self.wandb_tags:
            self.wandb_tags.append("pretrained")


class LightningValidator(ABC):
    def __init__(
        self,
        args: TrainArgs,
        lit_model: LightningModule,
        datamodule: LightningDataModule,
    ):
        self.args = args
        self.lit_model = lit_model
        self.datamodule = datamodule
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_ground_truth_and_smiles(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
        raise NotImplementedError()

    @abstractmethod
    def _collate(self, tensors: List[Any]) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def get_target(self, t: torch.Tensor, target: str, idx: int) -> np.ndarray:
        return np.ones((1,))

    def __call__(
        self,
        wandb_run_id: str,
        target_display_names: Optional[Dict[str, str]] = None,
        update_wandb: bool = True,
        new_wandb_run: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        api = wandb.Api()
        run = api.run(
            f"{self.args.wandb_entity}/{self.args.wandb_project}/{wandb_run_id}"
        )

        # Download best model
        ckpt_path, model_artifact_name = self.download_best_model(run)

        # Make predictions
        (
            val_loss,
            train_predictions,
            val_predictions,
            test_predictions,
        ) = self.make_parameter_predictions(ckpt_path)

        # Make save directory
        save_dir = Path(self.args.save_dir)
        predict_dir = save_dir / "predictions"
        predict_dir.mkdir(exist_ok=True, parents=True)

        (
            train_ground_truth,
            val_ground_truth,
            test_ground_truth,
            train_smiles,
            val_smiles,
            test_smiles,
        ) = self.get_ground_truth_and_smiles()

        # Post-filtering prediction
        if self.args.associating_columns and self.args.filter_non_associating:
            target_idx = [
                self.args.target_columns.index(target)
                for target in self.args.associating_columns
            ]
            for pred, smiles in zip(
                [train_predictions, val_predictions, test_predictions],
                [train_smiles, val_smiles, test_smiles],
            ):
                self.filter_non_associating(
                    pred,
                    target_idx=target_idx,
                    smiles=smiles,
                )

        # Save predictions
        self.save_parameter_predictions(
            predict_dir,
            train_predictions,
            val_predictions,
            test_predictions,
            train_ground_truth,
            val_ground_truth,
            test_ground_truth,
            train_smiles=train_smiles,
            val_smiles=val_smiles,
            test_smiles=test_smiles,
        )

        # Parity plot and scores
        plot_path, plot_test_path, scores = self.parity_plot_and_scores(
            train_predictions,
            val_predictions,
            test_predictions,
            train_ground_truth,
            val_ground_truth,
            test_ground_truth,
            target_display_names=target_display_names,
        )

        # Update wandb scores
        if update_wandb:
            self.logger.info("Updating wandb run")
            score_names = list(scores["val"].values())[0].keys()
            for split in ["train", "val", "test"]:
                sum_scores = {
                    f"total_{split}_{score_name}": 0 for score_name in score_names
                }
                for target in self.args.target_columns:
                    for score_name, score in scores[split][target].items():
                        run.summary[f"{split}_{target}_{score_name}"] = score
                        sum_scores[f"total_{split}_{score_name}"] += score
                run.summary.update(sum_scores)
            run.summary["val_loss_best"] = val_loss
            run.summary.update()

            # Update wandb run
            with wandb.init(
                id=run.id if not new_wandb_run else None,
                entity=self.args.wandb_entity,
                project=self.args.wandb_project,
                resume="allow",
            ) as full_run:  # type: ignore
                full_run.log({"parity_plot": wandb.Image(str(plot_path))})
                full_run.log({"test_parity_plot": wandb.Image(str(plot_test_path))})

                if new_wandb_run:
                    # Say that this run uses the model
                    run.use_artifact(model_artifact_name)

                # Log predictions
                artifact = wandb.Artifact(self.args.wandb_artifact_name, type="dataset")
                artifact.add_dir(str(predict_dir))
                full_run.log_artifact(artifact)

                # Create tables
                if train_smiles is not None:
                    full_run.log(
                        self.make_wandb_table_and_histogram(
                            train_smiles,
                            train_predictions,
                            train_ground_truth,
                            "train",
                        )
                    )
                if val_smiles is not None:
                    full_run.log(
                        self.make_wandb_table_and_histogram(
                            val_smiles, val_predictions, val_ground_truth, "val"
                        )
                    )
                if test_smiles is not None:
                    full_run.log(
                        self.make_wandb_table_and_histogram(
                            test_smiles, test_predictions, test_ground_truth, "test"
                        )
                    )

        return scores

    def download_best_model(self, run) -> Tuple[Path, str]:
        """Download best model from wandb

        Arguments
        ----------


        Returns
        -------
        Path to best model checkpoint

        """
        self.logger.info("Downloading best model from wandb")
        # Get best checkpoint
        artifacts = run.logged_artifacts()  # type: ignore
        ckpt_path = None
        artifact_name = None
        for artifact in artifacts:
            if artifact.type == "model" and "best_k" in artifact.aliases:
                ckpt_path = artifact.download()
                artifact_name = artifact.name
        if ckpt_path is None or artifact_name is None:
            raise ValueError("No best checkpoint found")
        ckpt_path = Path(ckpt_path) / "model.ckpt"

        return ckpt_path, artifact_name

    def make_parameter_predictions(
        self,
        ckpt_path: Path,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        # Create Trainer
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = Trainer(accelerator=accelerator)

        load_lightningmodule_from_checkpoint(
            lit_model=self.lit_model, ckpt_path=str(ckpt_path)
        )

        # Make predictions and get ground truth
        (
            train_predictions,
            val_predictions,
            test_predictions,
        ) = trainer.predict(
            self.lit_model,
            datamodule=self.datamodule,
            return_predictions=True,
        )  # type: ignore

        # Also calculate the final validation loss
        validation_results = trainer.validate(
            self.lit_model,
            datamodule=self.datamodule,
        )
        val_loss = validation_results[0]["val_loss"]

        # Collate
        train_predictions = self._collate(train_predictions)
        val_predictions = self._collate(val_predictions)
        test_predictions = self._collate(test_predictions)

        all_predictions: List[Union[float, np.ndarray]] = [val_loss]
        for preds in [train_predictions, val_predictions, test_predictions]:
            # Group predictions
            preds_target = [
                self.get_target(preds, target, i)
                for i, target in enumerate(self.args.target_columns)
            ]
            preds_target = np.column_stack(preds_target)

            # Make small values zero
            small_values = np.where(preds < 1e-3)[0]
            preds[small_values] = 0.0

            all_predictions.append(preds_target)

        return tuple(all_predictions)  # type: ignore

    def filter_non_associating(
        self, preds: np.ndarray, target_idx: List[int], smiles: List[str]
    ):
        """Set predictions to 0 when there are no acceptors or donor sites"""
        mols = [Chem.MolFromSmiles(s) for s in tqdm(smiles, desc="Generating RDKit mols")]  # type: ignore
        for i, mol in tqdm(enumerate(mols), desc="Checking for hydrogen bond sites"):
            num_acceptors = Lipinski.NumHAcceptors(mol)
            num_donors = Lipinski.NumHDonors(mol)
            if num_acceptors == 0 or num_donors == 0:
                for target_i in target_idx:
                    preds[i, target_i] = 0.0
        return preds

    def save_parameter_predictions(
        self,
        predict_dir: Path,
        train_predictions: np.ndarray,
        val_predictions: np.ndarray,
        test_predictions: np.ndarray,
        train_ground_truth: np.ndarray,
        val_ground_truth: np.ndarray,
        test_ground_truth: np.ndarray,
        train_smiles=None,
        val_smiles=None,
        test_smiles=None,
    ):
        self.logger.info("Saving predictions")

        # Save predictions
        for split, ground_truth, preds, smiles in zip(
            ["train", "val", "test"],
            [
                train_ground_truth,
                val_ground_truth,
                test_ground_truth,
            ],
            [train_predictions, val_predictions, test_predictions],
            [train_smiles, val_smiles, test_smiles],
        ):
            df = pd.DataFrame(ground_truth, columns=self.args.target_columns)

            # Add smiles
            if smiles:
                df["smiles"] = smiles

            # Create predictions dataframe
            df_preds = pd.DataFrame(
                preds,
                columns=[f"{col}_pred" for col in self.args.target_columns],
            )

            # Concatenate and save
            df = pd.concat([df, df_preds], axis=1)
            df.to_csv(predict_dir / f"{split}_predictions.csv", index=False)

    def parity_plot_and_scores(
        self,
        train_predictions: np.ndarray,
        val_predictions: np.ndarray,
        test_predictions: np.ndarray,
        train_ground_truth: np.ndarray,
        val_ground_truth: np.ndarray,
        test_ground_truth: np.ndarray,
        target_display_names: Optional[Dict[str, str]] = None,
    ):
        self.logger.info("Plotting parity plots and calculating scores")
        # Make parity and calculate scores
        n_rows = len(self.args.target_columns) // 2 + len(self.args.target_columns) % 2
        fig = plt.figure(figsize=(10, 5 * n_rows))
        fig_test = plt.figure(figsize=(10, 5 * n_rows))
        fig.subplots_adjust(wspace=0.3)
        fig_test.subplots_adjust(wspace=0.3)
        target_display_names = (
            target_display_names if target_display_names is not None else {}
        )
        scores = {"train": {}, "val": {}, "test": {}}
        for i, target in enumerate(self.args.target_columns):
            ax = fig.add_subplot(n_rows, 2, i + 1)  # type: ignore
            ax_test = fig_test.add_subplot(n_rows, 2, i + 1)  # type: ignore
            for split, ground_truth, preds in zip(
                ["train", "val", "test"],
                [
                    train_ground_truth,
                    val_ground_truth,
                    test_ground_truth,
                ],
                [train_predictions, val_predictions, test_predictions],
            ):
                # Calculate scores
                current_scores = calculate_metrics(
                    ground_truth[:, i], preds[:, i], scores=["mae", "r2"]
                )
                if split in ["train", "val"]:
                    parity_plot(
                        ground_truth[:, i],
                        preds[:, i],
                        ax=ax,
                        label=split,
                        scores=current_scores,
                    )
                else:
                    parity_plot(
                        ground_truth[:, i],
                        preds[:, i],
                        ax=ax_test,
                        label=split,
                        scores=current_scores,
                    )
                other_scores = calculate_metrics(
                    ground_truth[:, i], preds[:, i], scores=["rmse", "mse", "mape"]
                )
                current_scores.update(other_scores)
                scores[split][target] = current_scores
                ax.set_title(target_display_names.get(target, target))

        save_dir = Path(self.args.save_dir)
        plot_path = save_dir / "parity.png"
        fig.savefig(str(plot_path), dpi=300)
        plot_test_path = save_dir / "parity_test.png"
        fig_test.savefig(str(plot_test_path), dpi=300)
        return plot_path, plot_test_path, scores

    def make_wandb_table_and_histogram(
        self,
        smiles: List[str],
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        split: str,
    ) -> Dict:
        columns = ["molecule", "smiles", "split"]
        data = []
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)  # type: ignore
                img = Draw.MolToImage(mol)
                data.append([wandb.Image(img), smi, split])
            except ValueError:
                self.logger.warning(f"Could not parse smiles: {smi}")
                data.append(
                    [wandb.Image(np.zeros((100, 100, 3))), smi, split]
                )  # Just a blank image
        for i, target in enumerate(self.args.target_columns):
            error = np.abs(ground_truth - predictions)  # type: ignore
            for j in range(len(data)):
                data[j] += [ground_truth[j, i], predictions[j, i], error[j, i]]
            columns += [target, f"{target}_pred", f"{target}_error"]
        table = wandb.Table(columns=columns, data=data)
        to_log = {f"{split}_table": table}
        for target in self.args.target_columns:
            to_log[f"{split}_{target}"] = wandb_plot.histogram(
                table, f"{target}_error", title=f"{split}_{target}"
            )
        return to_log
