from typing import Callable, Dict

import numpy as np
import pandas as pd
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from mlsaft.extras.kedro_datasets.checkpoint_dataset import concat_partitioned_dfs
from mlsaft.extras.utils.data_transform import prepare_data, rename_column

from .nodes import (
    calculate_pcsaft_predictions,
    cluster_split,
    combine_splits,
    create_chemprop_cv_modules,
    create_chemprop_modules,
    create_ffn_modules,
    create_ffn_modules_cv,
    create_pcsaft_emulator_modules,
    create_pyg_cv_modules,
    create_pyg_modules,
    create_spk_modules,
    create_spk_modules_qm9,
    cross_validate_sklearn,
    cv_split,
    fingerprints,
    format_extra_data,
    pcsaft_parameter_random_design,
    predefined_split_by_molecule,
    random_split,
    train_pytorch_lightning,
    train_validate_lolopy,
    train_validate_pytorch_lighting_cv,
    train_validate_sklearn,
    validate_pytorch_lightning,
    visualize_cluster_split,
)


def create_base_splits_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        pipe=[
            node(
                func=prepare_data,
                name="splits_prepare_data",
                inputs={
                    "data": "data",
                    "available_conformers": "atoms_dataset",
                    "smiles_columns": "params:smiles_column",
                    "target_columns": "params:target_columns",
                    "outlier_check_columns": "params:outlier_check_columns",
                    "outlier_std_devs_cutoff": "params:outlier_std_devs_cutoff",
                    "conformer_id_lookup_column": "params:conformer_id_lookup_column",
                    "min_n_atoms": "params:min_n_atoms",
                },
                outputs=[
                    "data_filtered",
                    "removed_outliers",
                ],
            ),
            node(
                func=fingerprints,
                inputs={
                    "data": "data_filtered",
                    "smiles_columns": "params:smiles_column",
                    "num_bits": "params:fp_bits",
                },
                outputs="fps",
            ),
            node(
                func=lambda d, smiles_col: d[smiles_col].tolist(),
                inputs={"d": "holdout_data", "smiles_col": "params:smiles_column"},
                outputs="holdout_smiles",
            ),
            node(
                func=predefined_split_by_molecule,
                inputs={
                    "data": "data_filtered",
                    "holdout_molecules": "holdout_smiles",
                    "molecule_column": "params:smiles_column",
                },
                outputs="base_split_idx",
            ),
            node(
                func=cluster_split,
                inputs={
                    "fps": "fps",
                    "valid_size": "params:valid_size",
                    "kmeans_kwargs": "params:kmeans_kwargs",
                    "umap_before_cluster": "params:umap_before_cluster",
                    "umap_kwargs": "params:cluster_split_umap_kwargs",
                },
                outputs=[
                    "cluster_split_idx",
                    "cluster_labels",
                ],
            ),
            node(
                func=combine_splits,
                inputs={
                    "base_split": "base_split_idx",
                    "other_split": "cluster_split_idx",
                },
                outputs="split_idx",
            ),
            node(
                func=visualize_cluster_split,
                name="visualize_cluster_split",
                inputs={
                    "data": "data_filtered",
                    "fps": "fps",
                    "cluster_labels": "cluster_labels",
                    "split_idx": "split_idx",
                    "smiles_column": "params:smiles_column",
                    "target_columns": "params:target_columns",
                    "umap_kwargs": "params:umap_kwargs",
                    "fragments": "fragments",
                    "plot_top_k_functional_groups": "params:plot_top_k_functional_groups",
                    "umap_kwargs_app": "params:umap_kwargs_app",
                    "run_app": "params:run_app",
                    "label_columns": "params:target_columns",
                },
                outputs=[
                    "pairplot",
                    "cluster_umap",
                    "split_umap",
                    "split_functional_groups",
                ],
            ),
        ],
    )


def create_e2e_dataset(data: pd.DataFrame):
    pvap_data = data[data["DEN"].isna()]
    pvap_data = pvap_data.rename(columns={"P": "Pvap"})
    rho_data = data[data["DEN"].notnull()]
    rho_data = rho_data.rename(columns={"DEN": "rho_l"})
    e2e_data = pd.concat([pvap_data, rho_data], axis=0).reset_index(drop=True)
    return e2e_data


def merge_splits(
    data_filtered: pd.DataFrame,
    original_data: pd.DataFrame,
    splits: Dict[str, Callable[[], np.ndarray]],
):
    original_data = original_data.reset_index(drop=True)
    train_idx = splits["train_idx"]()
    valid_idx = splits["valid_idx"]()
    test_idx = splits["test_idx"]()
    # Get smiles of molecules in splits from data_filtered
    train_smiles = data_filtered.iloc[train_idx]["smiles_1"].tolist()
    valid_smiles = data_filtered.iloc[valid_idx]["smiles_1"].tolist()
    test_smiles = data_filtered.iloc[test_idx]["smiles_1"].tolist()
    # Get indices of splits in original data using smiles
    train_idx = original_data[
        original_data["smiles_1"].isin(train_smiles)
    ].index.tolist()
    valid_idx = original_data[
        original_data["smiles_1"].isin(valid_smiles)
    ].index.tolist()
    test_idx = original_data[original_data["smiles_1"].isin(test_smiles)].index.tolist()
    return {"train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx}


def create_e2e_splits(**kwargs) -> Pipeline:
    return pipeline(
        namespace="prepare_e2e_splits",
        inputs={
            "data_input": "dortmund_base_pure_component_filtered",
            "holdout_data": "e2e_holdout_set",
            "fragments": "fragments",
            "atoms_dataset": "dortmund_rdkit_xyz_files",
        },
        outputs={
            "data": "e2e_pure_data",
            "split_idx": "e2e_reduced_split_idx",
            "final_split_idx": "e2e_pure_split_idx",
            "cluster_labels": "e2e_pure_cluster_labels",
            "cluster_umap": "e2e_pure_umap",
            "split_umap": "e2e_pure_split_umap",
            "pairplot": "e2e_pure_pairplot",
            "split_functional_groups": "e2e_split_functional_groups",
        },
        pipe=pipeline(
            [
                node(
                    create_e2e_dataset,
                    inputs="data_input",
                    outputs="data",
                )
            ]
        )
        + create_base_splits_pipeline()
        + pipeline(
            [
                node(
                    func=merge_splits,
                    inputs={
                        "data_filtered": "data_filtered",
                        "original_data": "data",
                        "splits": "split_idx",
                    },
                    outputs="final_split_idx",
                )
            ]
        ),
    )


def create_sepp_splits_pipeline(**kwargs):
    return pipeline(
        namespace="prepare_sepp_splits",
        inputs={
            "data": "pcp_saft_sepp_pure_parameters",
            "holdout_data": "final_holdout_set",
            "atoms_dataset": "rdkit_xyz_files",  # Use rdkit since it is the smallest
            "fragments": "fragments",
        },
        outputs={
            "data_filtered": "pcp_saft_sepp_pure_parameters_filtered",
            "removed_outliers": "pcp_saft_sepp_pure_parameters_removed_outliers",
            "fps": "pcp_saft_sepp_pure_fps",
            "split_idx": "pcp_saft_sepp_pure_parameters_split_idx",
            "cluster_labels": "pcp_saft_sepp_pure_parameters_cluster_labels",
            "cluster_umap": "pcp_saft_sepp_pure_parameters_cluster_umap",
            "split_umap": "pcp_saft_sepp_pure_parameters_split_umap",
            "pairplot": "pcp_saft_sepp_pure_parameters_pairplot",
            "split_functional_groups": "pcp_saft_sepp_pure_parameters_split_functional_groups",
        },
        pipe=create_base_splits_pipeline(),
    )


def create_regressed_splits_pipeline(**kwargs):
    return pipeline(
        namespace="prepare_regressed_splits",
        inputs={
            "data": "pcp_saft_regressed_pure_parameters",
            "holdout_data": "final_holdout_set",
            "atoms_dataset": "dortmund_rdkit_xyz_files",
            "fragments": "fragments",
        },
        outputs={
            "data_filtered": "pcp_saft_regressed_pure_parameters_filtered",
            "removed_outliers": "pcp_saft_regressed_pure_parameters_removed_outliers",
            "fps": "pcp_saft_regressed_pure_fps",
            "split_idx": "pcp_saft_regressed_pure_parameters_split_idx",
            "cv_split_idx": "pcp_saft_regressed_pure_parameters_cv_split_idx",
            "cluster_labels": "pcp_saft_regressed_pure_parameters_cluster_labels",
            "cluster_umap": "pcp_saft_regressed_pure_parameters_cluster_umap",
            "split_umap": "pcp_saft_regressed_pure_parameters_split_umap",
            "pairplot": "pcp_saft_regressed_pure_parameters_pairplot",
            "split_functional_groups": "pcp_saft_regressed_pure_parameters_split_functional_groups",
        },
        pipe=[
            node(
                func=lambda data, manual_check_column, invalid_values: data[
                    ~data[manual_check_column].isin(invalid_values)
                ]
                if manual_check_column in data.columns
                else data,
                name="manual_check",
                inputs={
                    "data": "data",
                    "manual_check_column": "params:manual_check_column",
                    "invalid_values": "params:invalid_values",
                },
                outputs="data_valid",
            ),
            node(
                func=prepare_data,
                name="splits_prepare_data",
                inputs={
                    "data": "data_valid",
                    "available_conformers": "atoms_dataset",
                    "smiles_columns": "params:smiles_column",
                    "target_columns": "params:target_columns",
                    "outlier_check_columns": "params:outlier_check_columns",
                    "outlier_std_devs_cutoff": "params:outlier_std_devs_cutoff",
                    "conformer_id_lookup_column": "params:conformer_id_lookup_column",
                    "min_n_atoms": "params:min_n_atoms",
                },
                outputs=[
                    "data_filtered",
                    "removed_outliers",
                ],
            ),
            node(
                func=fingerprints,
                inputs={
                    "data": "data_filtered",
                    "smiles_columns": "params:smiles_column",
                },
                outputs="fps",
            ),
            node(
                func=lambda d: d["smiles_1"].tolist(),
                inputs={"d": "holdout_data"},
                outputs="holdout_smiles",
            ),
            node(
                func=predefined_split_by_molecule,
                inputs={
                    "data": "data_filtered",
                    "holdout_molecules": "holdout_smiles",
                    "molecule_column": "params:smiles_column",
                },
                outputs="base_split_idx",
            ),
            node(
                func=cluster_split,
                inputs={
                    "fps": "fps",
                    "valid_size": "params:valid_size",
                    "kmeans_kwargs": "params:kmeans_kwargs",
                },
                outputs=[
                    "cluster_split_idx",
                    "cluster_labels",
                ],
            ),
            node(
                func=combine_splits,
                inputs={
                    "base_split": "base_split_idx",
                    "other_split": "cluster_split_idx",
                },
                outputs="split_idx",
            ),
            node(
                func=visualize_cluster_split,
                name="visualize_cluster_split",
                inputs={
                    "data": "data_filtered",
                    "fps": "fps",
                    "cluster_labels": "cluster_labels",
                    "split_idx": "split_idx",
                    "smiles_column": "params:smiles_column",
                    "target_columns": "params:target_columns",
                    "umap_kwargs": "params:umap_kwargs",
                    "fragments": "fragments",
                    "plot_top_k_functional_groups": "params:plot_top_k_functional_groups",
                    "umap_kwargs_app": "params:umap_kwargs_app",
                    "run_app": "params:run_app",
                    "label_columns": "params:target_columns",
                },
                outputs=[
                    "pairplot",
                    "cluster_umap",
                    "split_umap",
                    "split_functional_groups",
                ],
            ),
            node(
                func=cv_split,
                inputs={
                    "data": "data_filtered",
                    "n_folds": "params:n_folds",
                    "valid_size": "params:valid_size",
                },
                outputs="cv_train_val_split_idx",
            ),
            node(
                func=combine_splits,
                inputs={
                    "base_split": "base_split_idx",
                    "other_split": "cv_train_val_split_idx",
                },
                outputs="cv_split_idx",
            ),
        ],
    )


def create_sigma_moments_splits_pipeline(**kwargs):
    """Pipeline for splitting COSMO-RS sigma moment data into training, validation, and test sets."""
    concat_pipeline = pipeline(
        [
            node(
                func=concat_partitioned_dfs,
                inputs={
                    "partitions": "cosmo_rs_results",
                    "keep_columns": "params:keep_columns",
                },
                outputs="data",
            )
        ]
    )
    return pipeline(
        namespace="prepare_sigma_moments_splits",
        inputs={
            "cosmo_rs_results": "cosmo_rs_results",
            "atoms_dataset": "cosmo_rs_xyz_files",
            "fragments": "fragments",
        },
        outputs={
            "data_filtered": "sigma_moments_filtered",
            "removed_outliers": "sigma_moments_removed_outliers",
            "fps": "sigma_moments_fps",
            "split_idx": "sigma_moments_split_idx",
            "cluster_labels": "sigma_moments_cluster_labels",
            "cluster_umap": "sigma_moments_cluster_umap",
            "split_umap": "sigma_moments_split_umap",
            "pairplot": "sigma_moments_pairplot",
            "split_functional_groups": "sigma_moments_split_functional_groups",
        },
        pipe=concat_pipeline + create_base_splits_pipeline(),
    )


def create_pc_saft_data_generation_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="pc_saft_data_generation",
        outputs={
            "pcsaft_emulator_critical_points",
            "pcsaft_emulator_phase_equilibria",
            "pcsaft_emulator_pairplot",
        },
        pipe=[
            node(
                func=pcsaft_parameter_random_design,
                inputs={"n_samples": "params:n_samples"},
                outputs="pc_saft_emulator_design",
            ),
            node(
                func=calculate_pcsaft_predictions,
                inputs={
                    "parameter_df": "pc_saft_emulator_design",
                },
                outputs=[
                    "pcsaft_emulator_critical_points",
                    "pcsaft_emulator_phase_equilibria",
                    "pcsaft_emulator_pairplot",
                ],
            ),
        ],
    )


def create_pc_saft_emulator_critical_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="pc_saft_emulator_critical",
        inputs={"pcsaft_emulator_critical_points"},
        pipe=[
            node(
                func=lambda data: data.fillna(0.0),
                inputs={
                    "data": "pcsaft_emulator_critical_points",
                },
                outputs="critical_points_no_na",
            ),
            node(
                func=random_split,
                inputs={
                    "data": "critical_points_no_na",
                    "valid_size": "params:valid_size",
                    "test_size": "params:test_size",
                },
                outputs="pc_saft_emulator_split_idx",
            ),
            node(
                func=create_pcsaft_emulator_modules,
                inputs={
                    "args": "params:train_args",
                    "data": "critical_points_no_na",
                    "split_idx": "pc_saft_emulator_split_idx",
                },
                outputs=["lit_model", "datamodule", "validator"],
            ),
            node(
                func=train_pytorch_lightning,
                inputs={
                    "args": "params:train_args",
                    "lit_model": "lit_model",
                    "datamodule": "datamodule",
                },
                outputs="pc_saft_emulator_wandb_run_id",
            ),
            node(
                func=validate_pytorch_lightning,
                inputs={
                    "wandb_run_id": "pc_saft_emulator_wandb_run_id",
                    "validator": "validator",
                },
                outputs="scores",
            ),
        ],
    )


def create_pc_saft_emulator_pvap_density_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="pc_saft_emulator_pvap_density",
        inputs={"pcsaft_emulator_phase_equilibria"},
        pipe=[
            node(
                func=random_split,
                inputs={
                    "data": "pcsaft_emulator_phase_equilibria",
                    "valid_size": "params:valid_size",
                    "test_size": "params:test_size",
                },
                outputs="pc_saft_emulator_split_idx",
            ),
            node(
                func=create_pcsaft_emulator_modules,
                inputs={
                    "args": "params:train_args",
                    "data": "pcsaft_emulator_phase_equilibria",
                    "split_idx": "pc_saft_emulator_split_idx",
                },
                outputs=["lit_model", "datamodule", "validator"],
            ),
            node(
                func=train_pytorch_lightning,
                inputs={
                    "args": "params:train_args",
                    "lit_model": "lit_model",
                    "datamodule": "datamodule",
                },
                outputs="pc_saft_emulator_wandb_run_id",
            ),
            node(
                func=validate_pytorch_lightning,
                inputs={
                    "wandb_run_id": "pc_saft_emulator_wandb_run_id",
                    "validator": "validator",
                },
                outputs="scores",
            ),
        ],
    )


def create_pyg_e2e_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="pc_saft_emulator_e2e",
        inputs={"e2e_data"},
        pipe=[
            node(
                func=create_pcsaft_emulator_modules,
                inputs={
                    "args": "params:train_args",
                    "data": "pcsaft_emulator_phase_equilibria",
                    "split_idx": "pc_saft_emulator_split_idx",
                },
                outputs=["lit_model", "datamodule", "validator"],
            ),
            node(
                func=train_pytorch_lightning,
                inputs={
                    "args": "params:train_args",
                    "lit_model": "lit_model",
                    "datamodule": "datamodule",
                },
                outputs="pc_saft_emulator_wandb_run_id",
            ),
            node(
                func=validate_pytorch_lightning,
                inputs={
                    "wandb_run_id": "pc_saft_emulator_wandb_run_id",
                    "validator": "validator",
                },
                outputs="scores",
            ),
        ],
    )


def create_sklearn_sepp_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="sklearn_sepp_pcp_saft_model",
        inputs={
            "pcp_saft_sepp_pure_fps",
            "pcp_saft_sepp_pure_parameters_filtered",
            "pcp_saft_sepp_pure_parameters_split_idx",
        },
        outputs={"pcp_saft_sepp_sklearn_wandb_run_id"},
        pipe=[
            node(
                func=train_validate_sklearn,
                inputs={
                    "args": "params:train_args",
                    "fps": "pcp_saft_sepp_pure_fps",
                    "data": "pcp_saft_sepp_pure_parameters_filtered",
                    "split_idx": "pcp_saft_sepp_pure_parameters_split_idx",
                },
                outputs="pcp_saft_sepp_sklearn_wandb_run_id",
            ),
        ],
    )


def create_sklearn_regressed_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="sklearn_regressed_pcp_saft_model",
        inputs={
            "pcp_saft_regressed_pure_fps",
            "pcp_saft_regressed_pure_parameters_filtered",
            "pcp_saft_regressed_pure_parameters_split_idx",
        },
        outputs={"pcp_saft_regressed_sklearn_wandb_run_id"},
        pipe=[
            node(
                func=train_validate_sklearn,
                inputs={
                    "args": "params:train_args",
                    "fps": "pcp_saft_regressed_pure_fps",
                    "data": "pcp_saft_regressed_pure_parameters_filtered",
                    "split_idx": "pcp_saft_regressed_pure_parameters_split_idx",
                },
                outputs="pcp_saft_regressed_sklearn_wandb_run_id",
            ),
        ],
    )


def create_sklearn_cv_regressed_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="sklearn_cv_regressed_pcp_saft_model",
        inputs={
            "pcp_saft_regressed_pure_fps",
            "pcp_saft_regressed_pure_parameters_filtered",
            "pcp_saft_regressed_pure_parameters_cv_split_idx",
        },
        outputs={"pcp_saft_regressed_sklearn_wandb_group_id"},
        pipe=[
            node(
                func=cross_validate_sklearn,
                inputs={
                    "args": "params:train_args",
                    "fps": "pcp_saft_regressed_pure_fps",
                    "data": "pcp_saft_regressed_pure_parameters_filtered",
                    "cv_split_idx": "pcp_saft_regressed_pure_parameters_cv_split_idx",
                },
                outputs="pcp_saft_regressed_sklearn_wandb_group_id",
            ),
        ],
    )


def create_lolo_regressed_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="lolo_regressed_pcp_saft_model",
        inputs={
            "pcp_saft_regressed_pure_fps",
            "pcp_saft_regressed_pure_parameters_filtered",
            "pcp_saft_regressed_pure_parameters_split_idx",
        },
        outputs={"pcp_saft_regressed_sklearn_wandb_run_id"},
        pipe=[
            node(
                func=train_validate_lolopy,
                inputs={
                    "args": "params:train_args",
                    "fps": "pcp_saft_regressed_pure_fps",
                    "data": "pcp_saft_regressed_pure_parameters_filtered",
                    "split_idx": "pcp_saft_regressed_pure_parameters_split_idx",
                },
                outputs="pcp_saft_regressed_sklearn_wandb_run_id",
            ),
        ],
    )


def create_base_ffn_train_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=lambda data, args: fingerprints(
                    data, args.smiles_columns, num_bits=args.fp_bits
                ),
                inputs={
                    "data": "data",
                    "args": "params:train_args",
                },
                outputs="fps",
            ),
            node(
                func=create_ffn_modules,
                inputs={
                    "args": "params:train_args",
                    "fps": "fps",
                    "data": "data",
                    "split_idx": "split_idx",
                },
                outputs=["lit_model", "datamodule", "validator"],
            ),
            node(
                func=train_pytorch_lightning,
                inputs={
                    "args": "params:train_args",
                    "lit_model": "lit_model",
                    "datamodule": "datamodule",
                },
                outputs="ffn_wandb_run_id",
            ),
            node(
                func=validate_pytorch_lightning,
                inputs={
                    "wandb_run_id": "ffn_wandb_run_id",
                    "validator": "validator",
                },
                outputs="scores",
            ),
        ],
    )


def create_base_ffn_cv_train_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=lambda data, args: fingerprints(
                    data, args.smiles_columns, num_bits=args.fp_bits
                ),
                inputs={
                    "data": "data",
                    "args": "params:train_args",
                },
                outputs="fps",
            ),
            node(
                func=create_ffn_modules_cv,
                inputs={
                    "args": "params:train_args",
                    "fps": "fps",
                    "data": "data",
                    "cv_split_idx": "cv_split_idx",
                },
                outputs="split_items",
            ),
            node(
                func=train_validate_pytorch_lighting_cv,
                inputs={
                    "args": "params:train_args",
                    "split_items": "split_items",
                },
                outputs="ffn_wandb_run_ids",
            ),
        ],
    )


def create_ffn_sepp_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="ffn_sepp_pcp_saft_model",
        inputs={
            # "fps": "pcp_saft_sepp_pure_fps",
            "data": "pcp_saft_sepp_pure_parameters_filtered",
            "split_idx": "pcp_saft_sepp_pure_parameters_split_idx",
            # "ffn_wandb_run_id": "pcp_saft_sepp_ffn_wandb_run_id",
        },
        outputs={"ffn_wandb_run_id": "pcp_saft_sepp_ffn_wandb_run_id"},
        pipe=create_base_ffn_train_pipeline(),
    )


def create_ffn_regressed_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="ffn_regressed_pcp_saft_model",
        inputs={
            "data": "pcp_saft_regressed_pure_parameters_filtered",
            "split_idx": "pcp_saft_regressed_pure_parameters_split_idx",
        },
        outputs={"ffn_wandb_run_id": "pcp_saft_regressed_ffn_wandb_run_id"},
        pipe=create_base_ffn_train_pipeline(),
    )


def create_ffn_cv_regressed_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="ffn_cv_regressed_pcp_saft_model",
        inputs={
            "data": "pcp_saft_regressed_pure_parameters_filtered",
            "cv_split_idx": "pcp_saft_regressed_pure_parameters_cv_split_idx",
        },
        outputs={"ffn_wandb_run_ids": "pcp_saft_regressed_ffn_wandb_run_ids"},
        pipe=create_base_ffn_cv_train_pipeline(),
    )


def create_base_chemprop_train_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=create_chemprop_modules,
                inputs={
                    "args": "params:train_args",
                    "data": "data",
                    "split_idx": "split_idx",
                },
                outputs=["lit_model", "datamodule", "validator"],
            ),
            node(
                func=train_pytorch_lightning,
                inputs={
                    "args": "params:train_args",
                    "lit_model": "lit_model",
                    "datamodule": "datamodule",
                },
                outputs="chemprop_wandb_run_id",
            ),
            node(
                func=validate_pytorch_lightning,
                inputs={
                    "wandb_run_id": "chemprop_wandb_run_id",
                    "validator": "validator",
                },
                outputs="scores",
            ),
        ],
    )


def create_base_chemprop_cv_train_pipeline(**kwargs):
    return pipeline(
        [
            node(
                func=create_chemprop_cv_modules,
                inputs={
                    "args": "params:train_args",
                    "data": "data",
                    "cv_split_idx": "cv_split_idx",
                },
                outputs="split_items",
            ),
            node(
                func=train_validate_pytorch_lighting_cv,
                inputs={
                    "args": "params:train_args",
                    "split_items": "split_items",
                },
                outputs="chemprop_wandb_run_ids",
            ),
        ],
    )


def create_chemprop_sepp_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="chemprop_sepp_pcp_saft_model",
        inputs={
            "data": "pcp_saft_sepp_pure_parameters_filtered",
            "split_idx": "pcp_saft_sepp_pure_parameters_split_idx",
            # "chemprop_wandb_run_id": "pcp_saft_sepp_chemprop_wandb_run_id",
        },
        outputs={"chemprop_wandb_run_id": "pcp_saft_sepp_chemprop_wandb_run_id"},
        pipe=create_base_chemprop_train_pipeline(),
    )


def create_chemprop_regressed_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="chemprop_regressed_pcp_saft_model",
        inputs={
            "data": "pcp_saft_regressed_pure_parameters_filtered",
            "split_idx": "pcp_saft_regressed_pure_parameters_split_idx",
            # "chemprop_wandb_run_id": "pcp_saft_regressed_chemprop_wandb_run_id",
        },
        outputs={"chemprop_wandb_run_id": "pcp_saft_regressed_chemprop_wandb_run_id"},
        pipe=create_base_chemprop_train_pipeline(),
    )


def create_chemprop_cv_regressed_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="chemprop_cv_regressed_pcp_saft_model",
        inputs={
            "data": "pcp_saft_regressed_pure_parameters_filtered",
            "cv_split_idx": "pcp_saft_regressed_pure_parameters_cv_split_idx",
        },
        outputs={"chemprop_wandb_run_ids": "pcp_saft_regressed_chemprop_wandb_run_ids"},
        pipe=create_base_chemprop_cv_train_pipeline(),
    )


def create_base_pyg_train_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_pyg_modules,
                name="create_pyg_modules",
                inputs={
                    "args": "params:train_args",
                    "data": "data",
                    "split_idx": "split_idx",
                },
                outputs=["lit_model", "datamodule", "validator"],
            ),
            node(
                func=train_pytorch_lightning,
                inputs={
                    "args": "params:train_args",
                    "lit_model": "lit_model",
                    "datamodule": "datamodule",
                },
                outputs="pyg_wandb_run_id",
            ),
            node(
                func=validate_pytorch_lightning,
                inputs={
                    "wandb_run_id": "pyg_wandb_run_id",
                    "validator": "validator",
                },
                outputs="scores",
            ),
        ],
    )


def create_base_pyg_cv_train_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_pyg_cv_modules,
                name="create_pyg_modules",
                inputs={
                    "args": "params:train_args",
                    "data": "data",
                    "cv_split_idx": "cv_split_idx",
                },
                outputs="split_items",
            ),
            node(
                func=train_validate_pytorch_lighting_cv,
                inputs={
                    "args": "params:train_args",
                    "split_items": "split_items",
                },
                outputs="pyg_wandb_run_ids",
            ),
        ],
    )


def create_pyg_sepp_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="pyg_sepp_pcp_saft_model",
        inputs={
            "data": "pcp_saft_sepp_pure_parameters_filtered",
            "split_idx": "pcp_saft_sepp_pure_parameters_split_idx",
            # "pyg_wandb_run_id": "pcp_saft_sepp_pyg_wandb_run_id",
        },
        outputs={"pyg_wandb_run_id": "pcp_saft_sepp_pyg_wandb_run_id"},
        pipe=create_base_pyg_train_pipeline(),
    )


def create_pyg_regressed_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="pyg_regressed_pcp_saft_model",
        inputs={
            "data": "pcp_saft_regressed_pure_parameters_filtered",
            "split_idx": "pcp_saft_regressed_pure_parameters_split_idx",
            # "pyg_wandb_run_id": "pcp_saft_regressed_pyg_wandb_run_id",
        },
        outputs={"pyg_wandb_run_id": "pcp_saft_regressed_pyg_wandb_run_id"},
        pipe=create_base_pyg_train_pipeline(),
    )


def create_pyg_cv_regressed_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="pyg_cv_regressed_pcp_saft_model",
        inputs={
            "data": "pcp_saft_regressed_pure_parameters_filtered",
            "cv_split_idx": "pcp_saft_regressed_pure_parameters_cv_split_idx",
        },
        outputs={"pyg_wandb_run_ids": "pcp_saft_regressed_pyg_wandb_run_ids"},
        pipe=create_base_pyg_cv_train_pipeline(),
    )


def create_pyg_pretrain_cosmo_model_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="pyg_pretrain_cosmo_model",
        inputs={
            "data_init": "pcp_saft_cosmo_regressed_pure_parameters",
            # "regressed_data": "pcp_saft_regressed_pure_parameters_filtered",
            # "regressed_split_idx": "pcp_saft_regressed_pure_parameters_split_idx",
            # "pyg_wandb_run_id": "pyg_pretrain_cosmo_wandb_run_id",
        },
        outputs={
            "pyg_wandb_run_id": "pyg_pretrain_cosmo_wandb_run_id",
            "data_removed": "pyg_pretrain_cosmo_data_removed",
            "split_idx": "pyg_pretrain_cosmo_split_idx",
        },
        pipe=pipeline(
            [
                node(
                    func=lambda data, args, std_cutoff: prepare_data(
                        data,
                        smiles_columns=args.smiles_columns,
                        target_columns=args.target_columns,
                        outlier_check_columns=args.target_columns,
                        outlier_std_devs_cutoff=std_cutoff,
                    ),
                    inputs={
                        "data": "data_init",
                        "args": "params:train_args",
                        "std_cutoff": "params:outlier_std_devs_cutoff",
                    },
                    outputs=["data", "data_removed"],
                ),
                node(
                    func=random_split,
                    inputs={
                        "data": "data",
                        "valid_size": "params:val_size",
                        "test_size": "params:test_size",
                    },
                    outputs="split_idx",
                ),
            ]
        )
        + create_base_pyg_train_pipeline(),
        # + pipeline(
        #     [
        #         node(
        #             func=lambda df: rename_column(df, [("smiles_1", "smiles")]),
        #             inputs={
        #                 "df": "regressed_data",
        #             },
        #             outputs="regressed_data_new",
        #         ),
        #         node(
        #             func=create_pyg_modules,
        #             name="create_pyg_experiment_modules",
        #             inputs={
        #                 "args": "params:train_args",
        #                 "data": "regressed_data_new",
        #                 "split_idx": "regressed_split_idx",
        #             },
        #             outputs=["_tmp1", "_tmp2", "validator_new"],
        #         ),
        #         node(
        #             func=validate_pytorch_lightning,
        #             name="validate_on_experiment",
        #             inputs={
        #                 "wandb_run_id": "pyg_wandb_run_id",
        #                 "validator": "validator_new",
        #                 "new_wandb_run": "params:new_wandb_run_for_zero_shot_eval",
        #             },
        #             outputs="scores_final",
        #         ),
        #     ]
        # ),
    )


def create_base_spk_train_pipeline():
    return pipeline(
        [
            node(
                func=create_spk_modules,
                inputs={
                    "args": "params:train_args",
                    "molecules": "atoms_dataset",
                    "target_property_data": "data",
                    "split_idx": "split_idx",
                },
                outputs=["lit_model", "datamodule", "validator"],
            ),
            node(
                func=train_pytorch_lightning,
                inputs={
                    "args": "params:train_args",
                    "lit_model": "lit_model",
                    "datamodule": "datamodule",
                },
                outputs="spk_wandb_run_id",
            ),
            node(
                func=validate_pytorch_lightning,
                inputs={
                    "wandb_run_id": "spk_wandb_run_id",
                    "validator": "validator",
                },
                outputs="scores",
            ),
        ]
    )


def create_spk_sepp_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    """Train PaiNN model from SchnetPack trained on SEPP data with DFT generated conformers"""
    return pipeline(
        namespace="spk_sepp_pcp_saft_model",
        inputs={
            "atoms_dataset": "xyz_files",
            "data": "pcp_saft_sepp_pure_parameters_filtered",
            "split_idx": "pcp_saft_sepp_pure_parameters_split_idx",
        },
        outputs={"spk_wandb_run_id": "pcp_saft_sepp_spk_wandb_run_id"},
        pipe=create_base_spk_train_pipeline(),
    )


def create_spk_rdkit_sepp_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    """Train PaiNN model from SchnetPack trained on SEPP data with RDKit generated conformers"""
    return pipeline(
        namespace="spk_rdkit_sepp_pcp_saft_model",
        inputs={
            "atoms_dataset": "rdkit_xyz_files",
            "data": "pcp_saft_sepp_pure_parameters_filtered",
            "split_idx": "pcp_saft_sepp_pure_parameters_split_idx",
        },
        outputs={"spk_wandb_run_id": "pcp_saft_sepp_spk_rdkit_wandb_run_id"},
        pipe=create_base_spk_train_pipeline(),
    )


# def create_spk_regressed_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
#     """Train PaiNN model from SchnetPack trained on regressed data with DFT generated conformers"""
#     return pipeline(
#         namespace="spk_regressed_pcp_saft_model",
#         inputs={
#             "atoms_dataset": "dortmund_rdkit_xyz_files",
#             "data": "pcp_saft_sepp_pure_parameters_filtered",
#             "split_idx": "pcp_saft_sepp_pure_parameters_split_idx",
#         },
#         outputs={"spk_wandb_run_id": "pcp_saft_regressed_spk_wandb_run_id"},
#         pipe=create_base_spk_train_pipeline(),
#     )


def create_spk_rdkit_regressed_pcp_saft_model_pipeline(**kwargs) -> Pipeline:
    """Train PaiNN model from SchnetPack trained on regressed data with RDKit generated conformers"""
    return pipeline(
        namespace="spk_rdkit_regressed_pcp_saft",
        inputs={
            "atoms_dataset": "dortmund_rdkit_xyz_files",
            "data": "pcp_saft_regressed_pure_parameters_filtered",
            "split_idx": "pcp_saft_regressed_pure_parameters_split_idx",
        },
        outputs={"spk_wandb_run_id": "pcp_saft_regressed_spk_rdkit_wandb_run_id"},
        pipe=create_base_spk_train_pipeline(),
    )


def create_spk_qm9_pretrained_mu_model_pipeline(**kwargs) -> Pipeline:
    """Train PaiNN model from SchnetPack on QM9 dipole moment data"""
    return pipeline(
        namespace="spk_qm9",
        outputs={"spk_qm9_wandb_run_id"},
        pipe=[
            node(
                func=create_spk_modules_qm9,
                inputs={
                    "args": "params:train_args",
                },
                outputs=["lit_model", "datamodule", "validator"],
            ),
            node(
                func=train_pytorch_lightning,
                inputs={
                    "args": "params:train_args",
                    "lit_model": "lit_model",
                    "datamodule": "datamodule",
                },
                outputs="spk_qm9_wandb_run_id",
            ),
        ],
    )


def create_spk_sepp_mu_model_pipeline(**kwargs) -> Pipeline:
    """Train PaiNN model from SchnetPack  on regressed data with RDKit generated conformers"""
    return pipeline(
        namespace="spk_mu_sepp",
        inputs={
            "atoms_dataset": "xyz_files",
            "data": "pcp_saft_sepp_pure_parameters_filtered",
            "split_idx": "pcp_saft_sepp_pure_parameters_split_idx",
        },
        outputs={"spk_wandb_run_id": "spk_mu_sepp_wandb_run_id"},
        pipe=create_base_spk_train_pipeline(),
    )


def create_spk_qm9_mixed_mu_model_pipeline(**kwargs) -> Pipeline:
    """Train PaiNN model from SchnetPack on QM9, SEPP, and CRC experimental dipole moment data"""
    return pipeline(
        namespace="spk_qm9_mu_mixed",
        inputs={
            "pcp_saft_sepp_pure_parameters_filtered",
            "xyz_files",
            "crc_filtered",
            "crc_xyz_files",
        },
        outputs={"spk_qm9_mu_wandb_run_id"},
        pipe=[
            node(
                func=rename_column,
                inputs={
                    "df": "pcp_saft_sepp_pure_parameters_filtered",
                    "name_chages": "params:sepp_name_changes",
                },
                outputs="pcp_saft_sepp_pure_parameters_filtered_renamed",
            ),
            node(
                func=format_extra_data,
                inputs={
                    "sepp_data": "pcp_saft_sepp_pure_parameters_filtered_renamed",
                    "sepp_molecules": "xyz_files",
                    "experimental_data": "crc_filtered",
                    "experimental_molecules": "crc_xyz_files",
                },
                outputs="extra_data",
            ),
            node(
                func=create_spk_modules_qm9,
                inputs={
                    "args": "params:train_args",
                    "extra_data": "extra_data",
                },
                outputs=["lit_model", "datamodule", "validator"],
            ),
            node(
                func=train_pytorch_lightning,
                inputs={
                    "args": "params:train_args",
                    "lit_model": "lit_model",
                    "datamodule": "datamodule",
                },
                outputs="spk_qm9_mu_wandb_run_id",
            ),
        ],
    )


def create_spk_sigma_moments_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="spk_model_sigma_moments",
        inputs={
            "cosmo_rs_xyz_files",
            "sigma_moments_filtered",
            "sigma_moments_split_idx",
        },
        outputs={"spk_sigma_moments_wandb_run_id"},
        pipe=[
            node(
                func=create_spk_modules,
                inputs={
                    "args": "params:train_args",
                    "molecules": "cosmo_rs_xyz_files",
                    "target_property_data": "sigma_moments_filtered",
                    "split_idx": "sigma_moments_split_idx",
                },
                outputs=["lit_model", "data", "validator"],
            ),
            node(
                func=train_pytorch_lightning,
                inputs={
                    "args": "params:train_args",
                    "lit_model": "lit_model",
                    "datamodule": "data",
                },
                outputs="spk_sigma_moments_wandb_run_id",
            ),
            node(
                func=validate_pytorch_lightning,
                inputs={
                    "wandb_run_id": "spk_sigma_moments_wandb_run_id",
                    "validator": "validator",
                },
                outputs="scores",
            ),
        ],
    )
