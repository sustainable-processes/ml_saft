"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from functools import partial

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from dl4thermo.extras.utils.data_transform import rename_column

from .nodes import (
    get_counterfactuals,
    get_pcsaft_thermo_scores,
    make_parity_plots,
    make_pcsaft_parameters_predictions_results_table,
    make_regresion_histograms,
    make_thermo_results_table,
    reformat_regression_parameter_dfs,
)


def create_base_pcsaft_regression_results_table_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        pipe=[
            node(
                func=reformat_regression_parameter_dfs,
                inputs={
                    "Regressed": "regressed_parameters",
                    "initial_guess": "inital_parameters",
                },
                outputs="regression_dfs",
            ),
            node(
                func=get_pcsaft_thermo_scores,
                inputs={
                    "experimental_data": "experimental_data",
                    "parameters_dfs": "regression_dfs",
                    "plot_figures": "params:plot_figures",
                    "parameter_dfs_smiles_column": "params:parameter_dfs_smiles_column",
                    "experimental_data_smiles_column": "params:experimental_data_smiles_column",
                    "experimental_data_name_column": "params:experimental_data_name_column",
                    # "experimental_temperature_column": "params:experimental_temperature_column",
                },
                outputs=["figures", "thermo_scores", "_all_data", "groups"],
            ),
            node(
                func=make_regresion_histograms,
                inputs={
                    "regression_dfs": "thermo_scores",
                    "cutoffs": "params:cutoffs",
                },
                outputs="regression_histograms",
            ),
            node(
                func=make_thermo_results_table,
                inputs={"thermo_scores": "thermo_scores"},
                outputs="thermo_scores_table",
            ),
        ],
    )


def create_pcsaft_regression_results_table_pipeline(**kwargs) -> Pipeline:
    """Create a LaTeX table of the results from PC-SAFT regresion"""
    return pipeline(
        namespace="pcsaft_regression_results_table",
        inputs={
            "experimental_data": "dortmund_base_pure_component_filtered",
            "inital_parameters": "pcp_saft_initial_pure_parameters",
            "regressed_parameters": "pcp_saft_regressed_pure_parameters",
        },
        outputs={
            "figures": "pcsaft_regression_pure_component_figures",
            "regression_histograms": "pcsaft_regression_pure_component_histograms",
            "thermo_scores": "pcsaft_regression_pure_component_thermo_scores",
            "thermo_scores_table": "pcsaft_regression_pure_component_thermo_scores_table",
        },
        pipe=create_base_pcsaft_regression_results_table_pipeline(),
    )


def create_cosmo_pcsaft_regression_results_table_pipeline(**kwargs) -> Pipeline:
    """Create a LaTeX table of the results from the models."""
    return pipeline(
        namespace="cosmo_pcsaft_regression_results_table",
        inputs={
            "experimental_data": "cosmo_pure_component",
            "inital_parameters": "pcp_saft_cosmo_initial_pure_parameters",
            "regressed_parameters": "pcp_saft_cosmo_regressed_pure_parameters",
        },
        outputs={
            "figures": "cosmo_pcsaft_regression_pure_component_figures",
            "regression_histograms": "cosmo_pcsaft_regression_pure_component_histograms",
            "thermo_scores": "cosmo_pcsaft_regression_pure_component_thermo_scores",
            "thermo_scores_table": "cosmo_pcsaft_regression_pure_component_thermo_scores_table",
            "regression_dfs": "none",
        },
        pipe=create_base_pcsaft_regression_results_table_pipeline(),
    )


def create_results_table_pipeline(**kwargs) -> Pipeline:
    """Create a LaTeX table of the results from the models."""
    return pipeline(
        namespace="results_table",
        inputs={
            "dortmund_base_pure_component_filtered",
            "pcp_saft_sepp_pure_parameters",
            "pure_gc_data",
            "sauer2014_homo_segments_gc_data",
            "pcp_saft_regressed_pure_parameters",
            "dortmund_dipole_moment_predictions",
        },
        outputs={
            "thermo_scores",
            "thermo_scores_sepp",
            "thermo_scores_gc",
            "feos_predictions_figures",
            "parity_plot_figures",
            "thermo_scores_table",
            "thermo_scores_table_cv",
            "thermo_scores_table_gc",
            "thermo_scores_table_sepp",
            # "thermo_scores_table_balanced",
            "parameters_scores_table",
            "parameters_scores_table_cv",
            "model_parameter_predictions",
            "all_data",
        },
        pipe=[
            # PCP-SAFT parameters
            node(
                func=partial(
                    make_pcsaft_parameters_predictions_results_table, only_tabular=True
                ),
                name="thermo_results_table_cv",
                inputs={
                    "wandb_groups": "params:thermo_table_wandb_groups",
                    "targets": "params:target_columns",
                    "wandb_entity": "params:wandb_entity",
                    "wandb_project": "params:wandb_project",
                    "split": "params:split",
                    "metric": "params:model_table_score",
                    # "rounding": "params:rounding",
                },
                outputs="parameters_scores_table_cv",
            ),
            node(
                func=partial(
                    make_pcsaft_parameters_predictions_results_table,
                    only_tabular=True,
                ),
                name="thermo_results_table",
                inputs={
                    "wandb_runs": "params:thermo_table_wandb_runs",
                    "targets": "params:target_columns",
                    "wandb_entity": "params:wandb_entity",
                    "wandb_project": "params:wandb_project",
                    "split": "params:split",
                    "metric": "params:model_table_score",
                    # "rounding": "params:rounding",
                },
                outputs="parameters_scores_table",
            ),
            node(
                func=make_parity_plots,
                name="make_parity_plots",
                inputs={
                    "wandb_runs": "params:thermo_table_wandb_runs",
                    "wandb_entity": "params:wandb_entity",
                    "wandb_project": "params:wandb_project",
                    "split": "params:split",
                },
                outputs=["parity_plot_figures", "model_parameter_predictions"],
            ),
            # Balanced association sampling
            # node(
            #     func=get_pcsaft_thermo_scores,
            #     name="get_pcsaft_thermo_scores_balanced",
            #     inputs={
            #         "wandb_groups": "params:balanced_association_wandb_groups",
            #         "parameters_dfs": "regressed_only",
            #         "dipole_moment_data": "dortmund_dipole_moment_predictions",
            #         "experimental_data": "dortmund_base_pure_component_filtered",
            #         "wandb_entity": "params:wandb_entity",
            #         "wandb_project": "params:wandb_project",
            #         "split": "params:split",
            #         "parameter_dfs_smiles_column": "params:parameter_dfs_smiles_column",
            #         "dipole_moment_data_smiles_column": "params:dipole_moment_data_smiles_column",
            #         "plot_figures": "params:plot_figures",
            #     },
            #     outputs=[
            #         "_tmp_figures_1",
            #         "thermo_scores_balanced",
            #         "all_data_balanced",
            #         "_tmp_groups_balanced",
            #     ],
            # ),
            # node(
            #     func=make_thermo_results_table,
            #     inputs={"thermo_scores": "thermo_scores_balanced"},
            #     outputs="thermo_scores_table_balanced",
            # ),
            # SEPP scores
            node(
                func=reformat_regression_parameter_dfs,
                inputs={
                    "SEPP": "pcp_saft_sepp_pure_parameters",
                    "regressed": "pcp_saft_regressed_pure_parameters",
                },
                outputs="regressed_and_sepp",
            ),
            node(
                func=get_pcsaft_thermo_scores,
                name="get_pcsaft_thermo_scores_with_sepp",
                inputs={
                    "wandb_runs": "params:thermo_table_wandb_runs",
                    "parameters_dfs": "regressed_and_sepp",
                    "experimental_data": "dortmund_base_pure_component_filtered",
                    "dipole_moment_data": "dortmund_dipole_moment_predictions",
                    "wandb_entity": "params:wandb_entity",
                    "wandb_project": "params:wandb_project",
                    "split": "params:split",
                    "parameter_dfs_smiles_column": "params:parameter_dfs_smiles_column",
                    "dipole_moment_data_smiles_column": "params:dipole_moment_data_smiles_column",
                },
                outputs=[
                    "_tmp",
                    "thermo_scores_sepp",
                    "all_data_sepp",
                    "_tmp_groups_sepp",
                ],
            ),
            node(
                func=partial(make_thermo_results_table, only_tabular=True),
                name="make_thermo_results_table_sepp",
                inputs={"thermo_scores": "thermo_scores_sepp"},
                outputs="thermo_scores_table_sepp",
            ),
            # Cross-validation scores
            node(
                func=reformat_regression_parameter_dfs,
                inputs={
                    "regressed": "pcp_saft_regressed_pure_parameters",
                },
                outputs="regressed_only",
            ),
            node(
                func=get_pcsaft_thermo_scores,
                name="get_pcsaft_thermo_scores_cv",
                inputs={
                    "wandb_groups": "params:thermo_table_wandb_groups",
                    "parameters_dfs": "regressed_only",
                    "dipole_moment_data": "dortmund_dipole_moment_predictions",
                    "experimental_data": "dortmund_base_pure_component_filtered",
                    "wandb_entity": "params:wandb_entity",
                    "wandb_project": "params:wandb_project",
                    "split": "params:split",
                    "parameter_dfs_smiles_column": "params:parameter_dfs_smiles_column",
                    "dipole_moment_data_smiles_column": "params:dipole_moment_data_smiles_column",
                },
                outputs=[
                    "_tmp_feos_figures",
                    "thermo_scores_cv",
                    "all_data_cv",
                    "thermo_scores_cv_groups",
                ],
            ),
            node(
                func=partial(make_thermo_results_table, only_tabular=True),
                inputs={
                    "thermo_scores": "thermo_scores_cv",
                    "groups": "thermo_scores_cv_groups",
                },
                outputs="thermo_scores_table_cv",
            ),
            # GC scores
            node(
                func=get_pcsaft_thermo_scores,
                name="get_pcsaft_thermo_scores_with_gc",
                inputs={
                    "wandb_runs": "params:thermo_table_wandb_runs",
                    "parameters_dfs": "regressed_only",
                    "experimental_data": "dortmund_base_pure_component_filtered",
                    "dipole_moment_data": "dortmund_dipole_moment_predictions",
                    "pure_gc_data": "pure_gc_data",
                    "segments_gc_data": "sauer2014_homo_segments_gc_data",
                    "wandb_entity": "params:wandb_entity",
                    "wandb_project": "params:wandb_project",
                    "split": "params:split",
                    "parameter_dfs_smiles_column": "params:parameter_dfs_smiles_column",
                    "dipole_moment_data_smiles_column": "params:dipole_moment_data_smiles_column",
                    # "plot_figures": "params:plot_figures",
                },
                outputs=["_tmp_2", "thermo_scores_gc", "all_data_gc", "_tmp_groups_gc"],
            ),
            node(
                func=partial(make_thermo_results_table, only_tabular=True),
                name="make_thermo_results_table_gc",
                inputs={"thermo_scores": "thermo_scores_gc"},
                outputs="thermo_scores_table_gc",
            ),
            # All scores
            node(
                func=get_pcsaft_thermo_scores,
                name="get_pcsaft_thermo_scores",
                inputs={
                    "wandb_runs": "params:thermo_table_wandb_runs",
                    "parameters_dfs": "regressed_only",
                    "dipole_moment_data": "dortmund_dipole_moment_predictions",
                    "experimental_data": "dortmund_base_pure_component_filtered",
                    "wandb_entity": "params:wandb_entity",
                    "wandb_project": "params:wandb_project",
                    "split": "params:split",
                    "parameter_dfs_smiles_column": "params:parameter_dfs_smiles_column",
                    "dipole_moment_data_smiles_column": "params:dipole_moment_data_smiles_column",
                    "plot_figures": "params:plot_figures",
                    "return_all_data": "params:return_all_data",
                },
                outputs=[
                    "feos_predictions_figures",
                    "thermo_scores",
                    "all_data",
                    "_tmp_grops_feos",
                ],
            ),
            node(
                func=partial(make_thermo_results_table, only_tabular=True),
                inputs={"thermo_scores": "thermo_scores"},
                outputs="thermo_scores_table",
            ),
        ],
    )


def create_cosmo_pretrain_results_table_pipeline(**kwargs) -> Pipeline:
    """Create a LaTeX table of the results from the models."""
    return pipeline(
        namespace="results_table_cosmo_pretrain",
        inputs={
            "dortmund_base_pure_component_filtered",
            "pcp_saft_cosmo_regressed_pure_parameters",
            "pcp_saft_regressed_pure_parameters",
            "dortmund_dipole_moment_predictions",
        },
        outputs={
            "thermo_scores_cosmo",
            "feos_predictions_figures_cosmo",
            "parity_plot_figures_cosmo",
            "thermo_scores_table_cosmo",
            "parameters_scores_table_cosmo",
            "model_parameter_predictions_cosmo",
        },
        pipe=[
            # PCP-SAFT parameters
            node(
                func=make_pcsaft_parameters_predictions_results_table,
                inputs={
                    "wandb_runs": "params:thermo_table_wandb_runs",
                    "targets": "params:target_columns",
                    "wandb_entity": "params:wandb_entity",
                    "wandb_project": "params:wandb_project",
                    "split": "params:split",
                    "metric": "params:model_table_score",
                    # "rounding": "params:rounding",
                },
                outputs="parameters_scores_table_cosmo",
            ),
            node(
                func=make_parity_plots,
                inputs={
                    "wandb_runs": "params:thermo_table_wandb_runs",
                    "wandb_entity": "params:wandb_entity",
                    "wandb_project": "params:wandb_project",
                    "split": "params:split",
                },
                outputs=[
                    "parity_plot_figures_cosmo",
                    "model_parameter_predictions_cosmo",
                ],
            ),
            node(
                func=lambda df: rename_column(df, [("smiles", "smiles_1")]),
                inputs={
                    "df": "pcp_saft_cosmo_regressed_pure_parameters",
                },
                outputs="regressed_data_new",
            ),
            node(
                func=reformat_regression_parameter_dfs,
                inputs={
                    "regressed": "pcp_saft_regressed_pure_parameters",
                    "cosmo": "regressed_data_new",
                },
                outputs="regressed_only",
            ),
            node(
                func=get_pcsaft_thermo_scores,
                name="get_pcsaft_thermo_scores",
                inputs={
                    "wandb_runs": "params:thermo_table_wandb_runs",
                    "parameters_dfs": "regressed_only",
                    "dipole_moment_data": "dortmund_dipole_moment_predictions",
                    "experimental_data": "dortmund_base_pure_component_filtered",
                    "wandb_entity": "params:wandb_entity",
                    "wandb_project": "params:wandb_project",
                    "split": "params:split",
                    "parameter_dfs_smiles_column": "params:parameter_dfs_smiles_column",
                    "dipole_moment_data_smiles_column": "params:dipole_moment_data_smiles_column",
                    "plot_figures": "params:plot_figures",
                },
                outputs=[
                    "feos_predictions_figures_cosmo",
                    "thermo_scores_cosmo",
                    "_tmp_grops_feos",
                ],
            ),
            node(
                func=make_thermo_results_table,
                inputs={"thermo_scores": "thermo_scores_cosmo"},
                outputs="thermo_scores_table_cosmo",
            ),
        ],
    )
