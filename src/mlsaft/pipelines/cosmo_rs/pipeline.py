"""
This is a boilerplate pipeline 'cosmo_rs'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from mlsaft.extras.kedro_datasets.checkpoint_dataset import create_dataframe_partitions
from mlsaft.extras.utils.cosmo_calculate import get_cosmobase_df

from .nodes import calculate_properties, generate_atoms


def create_cosmo_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="cosmo_rs_calculations",
        outputs={"cosmo_rs_results", "cosmo_rs_xyz_files"},
        pipe=[
            node(
                func=get_cosmobase_df,
                inputs=[],
                outputs="cosmo_base_df",
                name="get_cosmobase",
            ),
            node(
                func=create_dataframe_partitions,
                inputs={
                    "data": "cosmo_base_df",
                    "partition_size": "params:partition_size",
                },
                outputs="cosmo_base_df_partitions",
            ),
            node(
                func=calculate_properties,
                inputs={
                    "df_partitions": "cosmo_base_df_partitions",
                    "n_cores": "params:n_cores",
                    "timeout": "params:timeout",
                    "save_dir": "params:save_dir",
                    "calculate_kwargs": "params:calculate_kwargs",
                },
                outputs="cosmo_rs_results",
            ),
            node(
                func=generate_atoms,
                name="generate_xyz_files",
                inputs={"dfs": "cosmo_rs_results"},
                outputs="cosmo_rs_xyz_files",
            ),
        ],
    )
