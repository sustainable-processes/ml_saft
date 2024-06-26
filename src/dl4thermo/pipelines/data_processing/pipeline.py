"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from dl4thermo.extras.kedro_datasets.checkpoint_dataset import (
    concat_partitioned_dfs,
    create_dataframe_partitions,
)
from dl4thermo.extras.utils.data_transform import prepare_data

from .nodes import (
    classify_molecules,
    clean_resolve_crc_dipole_data,
    combine_and_deduplicate_molecule_lists,
    concat_dataframes,
    density_filtering,
    generate_rdkit_conformers,
    get_cas_numbers,
    get_dortmund_molecules,
    get_thermoml_molecules,
    get_wikipedia_dipole_moments,
    merge_molecule_ids,
    merge_smiles,
    remove_numbers_dortmund_molecules,
    resolve_smiles,
)


def create_thermoml_pipeline(**kwargs) -> Pipeline:
    journals = [
        "journal_thermophysics",
        "fluid_phase_equilibria",
        "journal_chemical_engineering_data",
    ]
    return pipeline(
        namespace="thermoml",
        # Extract data from ThermoML files
        pipe=[
            node(
                func=concat_partitioned_dfs,
                inputs=f"{journal}",
                outputs=f"imported_{journal}",
                name=f"import_{journal}",
            )
            for journal in journals
        ]
        + [
            node(
                func=concat_dataframes,
                inputs=[f"imported_{journal}" for journal in journals],
                outputs="imported",
                name="concat",
            ),
            node(
                func=get_thermoml_molecules,
                inputs="imported",
                outputs="molecules",
                name="get_thermoml_molecules",
            ),
        ],
    )


def create_ddb_resolve_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="dortmund",
        outputs={"dortmund_base_binary", "dortmund_base_pure_component"},
        pipe=[
            node(
                func=concat_partitioned_dfs,
                inputs={
                    "partitions": "binary",
                    "no_na_columns": "params:dortmund_binary_no_na_columns",
                    "keep_columns": "params:dortmund_binary_keep_columns",
                },
                outputs="imported_binary",
                name="import_dortumnd_binary",
            ),
            node(
                func=concat_partitioned_dfs,
                inputs={
                    "partitions": "pure_component",
                    "no_na_columns": "params:dortmund_pure_component_no_na_columns",
                    "float_columns": "params:float_columns",
                    "str_columns": "params:str_columns",
                    "int_columns": "params:int_columns",
                    "keep_columns": "params:dortmund_pure_component_keep_columns",
                },
                outputs="imported_pure_component",
                name="import_dortumnd_pure_component",
            ),
            node(
                func=get_dortmund_molecules,
                inputs={
                    "data": "imported_binary",
                    "columns": "params:dortmund_binary_component_id_columns",
                },
                outputs="binary_molecule_ids",
                name="get_dortmund_binary_ids",
            ),
            node(
                func=get_dortmund_molecules,
                inputs={
                    "data": "imported_pure_component",
                    "columns": "params:dortmund_pure_component_id_column",
                },
                outputs="pure_molecule_ids",
                name="get_dortmund_pure_ids",
            ),
            node(
                func=combine_and_deduplicate_molecule_lists,
                inputs=["binary_molecule_ids", "pure_molecule_ids"],
                outputs="molecule_ids",
                name="combine_dedup_ids",
            ),
            node(
                func=get_cas_numbers,
                inputs={
                    "ids": "molecule_ids",
                    "ids_to_cas": "ids_cas",
                    "id_column": "params:lookup_id_column",
                },
                outputs="molecule_ids_cas",
            ),
            node(
                func=create_dataframe_partitions,
                inputs={
                    "data": "molecule_ids_cas",
                    "partition_size": "params:smiles_lookup_partition_size",
                },
                outputs="molecule_ids_cas_partitioned",
                name="create_partitions",
            ),
            node(
                func=resolve_smiles,
                inputs={
                    "partitions": "molecule_ids_cas_partitioned",
                    "batch_size": "params:smiles_lookup_batch_size",
                    "db_path": "params:pura_db_path",
                },
                outputs="smiles_lookup_partitioned",
            ),
            node(
                func=concat_partitioned_dfs,
                inputs={
                    "partitions": "smiles_lookup_partitioned",
                    "reset_index": "params:reset_index_after_concat",
                },
                outputs="smiles_lookup",
                name="concat_partitioned_dfs",
            ),
            node(
                func=classify_molecules,
                inputs="smiles_lookup",
                outputs="smiles_lookup_classified",
                name="classify_molecules",
            ),
            node(
                func=merge_smiles,
                inputs={
                    "smiles_lookup": "smiles_lookup_classified",
                    "data": "imported_binary",
                    "data_columns": "params:dortmund_binary_component_id_columns",
                },
                outputs="dortmund_base_binary",
                name="merge_binary",
            ),
            node(
                func=merge_smiles,
                inputs={
                    "smiles_lookup": "smiles_lookup_classified",
                    "data": "imported_pure_component",
                    "data_columns": "params:dortmund_pure_component_id_column",
                },
                outputs="dortmund_base_pure_component",
                name="merge_pure_component",
            ),
        ],
    )


def create_dortmund_model_preparation_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="dortmund_model_prep",
        inputs={"dortmund_base_pure_component"},
        outputs={"dortmund_base_pure_component_filtered", "dortmund_rdkit_xyz_files"},
        pipe=[
            node(
                func=prepare_data,
                inputs={
                    "data": "dortmund_base_pure_component",
                    "smiles_columns": "params:smiles_column",
                    "target_columns": "params:target_columns",
                    "drop_duplicates": "params:drop_duplicates",
                    "bounds": "params:bounds",
                },
                outputs=[
                    "dortmund_base_pure_component_intermediate",
                    "dortmund_base_pure_component_removed_intermediate",
                ],
            ),
            node(
                func=density_filtering,
                inputs={
                    "data": "dortmund_base_pure_component_intermediate",
                    "pressure_column": "params:pressure_column",
                    "density_column": "params:density_column",
                    "density_bounds": "params:density_bounds",
                    "max_pressure": "params:max_pressure_for_density",
                },
                outputs="dortmund_base_pure_component_limited",
            ),
            node(
                func=generate_rdkit_conformers,
                inputs={
                    "data": "dortmund_base_pure_component_limited",
                    "smiles_column": "params:smiles_column",
                    "id_column": "params:id_column",
                    "batch_size": "params:conformer_generation_batch_size",
                },
                outputs="dortmund_rdkit_xyz_files",
            ),
            node(
                func=prepare_data,
                inputs={
                    "data": "dortmund_base_pure_component_limited",
                    "available_conformers": "dortmund_rdkit_xyz_files",
                    "conformer_id_lookup_column": "params:id_column",
                    "smiles_columns": "params:smiles_column",
                    "target_columns": "params:target_columns",
                    "drop_duplicates": "params:drop_duplicates",
                },
                outputs=[
                    "dortmund_base_pure_component_filtered",
                    "dortmund_base_pure_component_removed",
                ],
            ),
        ],
    )


def create_crc_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="crc",
        inputs={"crc_dipole_moments"},
        outputs={"crc_xyz_files", "crc_filtered", "crc_removed"},
        pipe=[
            node(
                func=clean_resolve_crc_dipole_data,
                inputs={
                    "df": "crc_dipole_moments",
                    "pura_db_path": "params:pura_db_path",
                    "batch_size": "params:pura_batch_size",
                    "smiles_column": "params:smiles_column",
                    "dipole_moment_column": "params:dipole_moment_column",
                    "id_column": "params:id_column",
                    "filter_low_quality": "params:filter_low_quality",
                },
                outputs="crc_base",
                name="clean_crc_data",
            ),
            node(
                func=generate_rdkit_conformers,
                inputs={
                    "data": "crc_base",
                    "smiles_column": "params:smiles_column",
                    "id_column": "params:id_column",
                    "batch_size": "params:conformer_generation_batch_size",
                },
                outputs="crc_xyz_files",
            ),
            node(
                func=prepare_data,
                inputs={
                    "data": "crc_base",
                    "available_conformers": "crc_xyz_files",
                    "conformer_id_lookup_column": "params:id_column",
                    "smiles_columns": "params:smiles_column",
                    "target_columns": "params:dipole_moment_column",
                },
                outputs=[
                    "crc_filtered",
                    "crc_removed",
                ],
            ),
        ],
    )


def create_wiki_scrape_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        namespace="wiki_scrape",
        outputs={"wikipedia_dipole_moments"},
        pipe=[
            node(
                func=get_wikipedia_dipole_moments,
                inputs={},
                outputs="wikipedia_dipole_moments",
                name="import_wiki_scrape",
            ),
        ],
    )
