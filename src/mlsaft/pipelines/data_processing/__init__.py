"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from .pipeline import (
    create_crc_pipeline,
    create_ddb_resolve_pipeline,
    create_dortmund_model_preparation_pipeline,
    create_thermoml_pipeline,
    create_wiki_scrape_pipeline,
)

__all__ = [
    "create_thermoml_pipeline",
    "create_ddb_resolve_pipeline",
    "create_wiki_scrape_pipeline",
    "create_crc_pipeline",
    "create_dortmund_model_preparation_pipeline",
]
