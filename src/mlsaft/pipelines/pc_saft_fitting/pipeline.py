# """
# This is a boilerplate pipeline 'pc_saft_fitting'
# generated using Kedro 0.17.7
# """

from functools import partial

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.modular_pipeline import pipeline

from mlsaft.extras.kedro_datasets.checkpoint_dataset import concat_partitioned_dfs
from mlsaft.extras.utils.data_transform import prepare_data
from mlsaft.extras.utils.plotting import sensitivity_boxplot

from .nodes import (
    create_cosmo_dfs,
    make_dipole_moment_predictions,
    parameter_dicts_to_dataframes,
    regress_pc_saft_pure,
    sensitivity_analysis,
)


def create_sensitivity_analysis_pipeline(**kwargs) -> Pipeline:
    """PC-SAFT sensitivity analysis pipeline."""
    return pipeline(
        namespace="sensitivity_analysis",
        inputs={"pcp_saft_sepp_pure_parameters", "dortmund_base_pure_component"},
        outputs={
            "pvap_sensitivity_analysis_results",
            "rho_sensitivity_analysis_results",
            "sensitivity_boxplot",
        },
        pipe=[
            node(
                sensitivity_analysis,
                inputs={
                    "pcsaft_data": "pcp_saft_sepp_pure_parameters",
                    "pcsaft_data_smiles_column": "params:pcsaft_data_smiles_column",
                    "experimental_data": "dortmund_base_pure_component",
                    "experimental_data_smiles_column": "params:experimental_data_smiles_column",
                    "n_samples": "params:n_samples",
                    "batch_size": "params:batch_size",
                },
                outputs={
                    "pvap": "pvap_sensitivity_analysis_results",
                    "rho": "rho_sensitivity_analysis_results",
                },
            ),
            node(
                sensitivity_boxplot,
                name="make_sensitivity_boxplot",
                inputs={
                    "pvap_df": "pvap_sensitivity_analysis_results",
                    "rho_df": "rho_sensitivity_analysis_results",
                },
                outputs="sensitivity_boxplot",
            ),
        ],
    )


def create_pure_component_regression_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="regress_pcsaft",
        inputs={
            "dortmund_base_pure_component_filtered",
            "dortmund_rdkit_xyz_files",
        },
        outputs={
            "dortmund_dipole_moment_predictions",
            "pcp_saft_regressed_pure_parameters",
            "pcp_saft_initial_pure_parameters",
            "pcp_saft_regressed_pure_parameters_removed",
        },
        pipe=[
            node(
                func=make_dipole_moment_predictions,
                name="dipole_predictions",
                inputs={
                    "data": "dortmund_base_pure_component_filtered",
                    "molecules": "dortmund_rdkit_xyz_files",
                    "smiles_column": "params:smiles_column",
                    "wandb_run_id": "params:dipole_model_wandb_run_id",
                    "wandb_entity": "params:wandb_entity",
                    "wandb_project": "params:wandb_project",
                    "molecule_lookup_column_name": "params:molecule_lookup_column_name",
                    "batch_size": "params:dipole_moment_prediction_batch_size",
                    "dipole_moment_column": "params:dipole_moment_column",
                    "smiles_column": "params:smiles_column",
                },
                outputs="dortmund_dipole_moment_predictions",
            ),
            node(
                func=regress_pc_saft_pure,
                inputs={
                    "experimental_data": "dortmund_base_pure_component_filtered",
                    "dipole_moment_data": "dortmund_dipole_moment_predictions",
                    "dipole_moment_data_column": "params:dipole_moment_column",
                    "smiles_column": "params:smiles_column",
                    "dipole_moment_data_smiles_column": "params:smiles_column",
                    "density_column": "params:density_column",
                    "temperature_column": "params:temperature_column",
                    "pressure_column": "params:pressure_column",
                    "id_column": "params:ddb_id_column",
                    "fix_kab": "params:fix_kab",
                    "fix_dipole_moment": "params:fix_dipole_moment",
                    "batch_size": "params:batch_size",
                    "density_weight": "params:density_weight",
                    "min_num_density_data": "params:min_num_density_data",
                    "min_num_pvap_data": "params:min_num_pvap_data",
                    "start_batch": "params:start_batch",
                    "fit_log_pressure": "params:fit_log_pressure",
                },
                outputs="pcp_saft_pure_regression_results",
            ),
            node(
                func=parameter_dicts_to_dataframes,
                name="regression_dfs",
                inputs={
                    "batches": "pcp_saft_pure_regression_results",
                    "smiles_column": "params:smiles_column",
                    "id_column": "params:ddb_id_column",
                    "max_total_residual": "params:max_total_residual",
                    "max_residual": "params:max_residual",
                },
                outputs=[
                    "pcp_saft_initial_pure_parameters",
                    "pcp_saft_regressed_pure_parameters",
                    "pcp_saft_regressed_pure_parameters_removed",
                ],
            ),
        ],
    )


def create_pure_component_cosmo_regression_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="regress_cosmo_pcsaft",
        inputs={
            "cosmo_rs_results",
            "cosmo_rs_xyz_files",
        },
        outputs={
            "cosmo_pure_component",
            "cosmo_pure_component_removed",
            "cosmo_dipole_moment_predictions",
            "cosmo_painn_dipole_moment_predictions",
            "pcp_saft_cosmo_initial_pure_parameters",
            "pcp_saft_cosmo_regressed_pure_parameters",
            "pcp_saft_cosmo_regressed_pure_parameters_removed",
        },
        pipe=[
            node(
                func=concat_partitioned_dfs,
                inputs={"partitions": "cosmo_rs_results"},
                outputs="cosmo_virtual_pure_component_data",
            ),
            node(
                func=create_cosmo_dfs,
                inputs={
                    "cosmo_df": "cosmo_virtual_pure_component_data",
                    "density_column": "params:density_column",
                    "smiles_column": "params:smiles_column",
                    "pressure_column": "params:pressure_column",
                    "temperature_column": "params:temperature_column",
                    "dipole_moment_column": "params:dipole_moment_column",
                },
                outputs=["cosmo_df", "cosmo_dipole_moment_predictions"],
            ),
            node(
                func=partial(prepare_data, drop_duplicates=False),
                inputs={
                    "data": "cosmo_df",
                    "smiles_columns": "params:smiles_column",
                    "target_columns": "params:pressure_column",
                    "available_conformers": "cosmo_rs_xyz_files",
                    "conformer_id_lookup_column": "params:name_column",
                },
                outputs=["cosmo_pure_component", "cosmo_pure_component_removed"],
            ),
            node(
                func=make_dipole_moment_predictions,
                name="dipole_predictions",
                inputs={
                    "data": "cosmo_pure_component",
                    "molecules": "cosmo_rs_xyz_files",
                    "smiles_column": "params:smiles_column",
                    "wandb_run_id": "params:dipole_model_wandb_run_id",
                    "wandb_entity": "params:wandb_entity",
                    "wandb_project": "params:wandb_project",
                    "molecule_lookup_column_name": "params:name_column",
                    "batch_size": "params:dipole_moment_prediction_batch_size",
                    "dipole_moment_column": "params:dipole_moment_column",
                    "smiles_column": "params:smiles_column",
                },
                outputs="cosmo_painn_dipole_moment_predictions",
            ),
            node(
                func=regress_pc_saft_pure,
                inputs={
                    "experimental_data": "cosmo_pure_component",
                    "dipole_moment_data": "cosmo_painn_dipole_moment_predictions",
                    "dipole_moment_data_column": "params:dipole_moment_column",
                    "smiles_column": "params:smiles_column",
                    "dipole_moment_data_smiles_column": "params:smiles_column",
                    "density_column": "params:density_column",
                    "temperature_column": "params:temperature_column",
                    "pressure_column": "params:pressure_column",
                    "id_column": "params:name_column",
                    "fix_kab": "params:fix_kab",
                    "fix_dipole_moment": "params:fix_dipole_moment",
                    "batch_size": "params:batch_size",
                    "density_weight": "params:density_weight",
                    "min_num_density_data": "params:min_num_density_data",
                    "min_num_pvap_data": "params:min_num_pvap_data",
                    "start_batch": "params:start_batch",
                    "fit_log_pressure": "params:fit_log_pressure",
                },
                outputs="pcp_saft_pure_cosmo_regression_results",
            ),
            node(
                func=parameter_dicts_to_dataframes,
                name="cosmo_regression_dfs",
                inputs={
                    "batches": "pcp_saft_pure_cosmo_regression_results",
                    "smiles_column": "params:smiles_column",
                    "id_column": "params:name_column",
                    "max_total_residual": "params:max_total_residual",
                },
                outputs=[
                    "pcp_saft_cosmo_initial_pure_parameters",
                    "pcp_saft_cosmo_regressed_pure_parameters",
                    "pcp_saft_cosmo_regressed_pure_parameters_removed",
                ],
            ),
        ],
    )
