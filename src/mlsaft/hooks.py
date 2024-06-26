"""Project hooks."""
import logging
import pdb
import sys
import traceback
import warnings
from typing import Any, Dict, Iterable, Optional

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import AbstractDataSet, DataCatalog
from kedro.utils import load_obj


class TypedParameters:
    def __init__(self, type_indicator: str = "type", inline: bool = False):
        self._type_indicator = type_indicator
        self._type_suffix = f"__{type_indicator}"
        self._inline = inline
        self._logger = logging.getLogger(__name__)

    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog) -> None:
        if self._inline:
            param_types = self._get_param_types_inline(catalog)
        else:
            param_types = self._get_param_types(catalog)

        for param, type_string in param_types.items():
            self._logger.info(f"Loading {type_string}")
            type_obj = load_obj(type_string)
            catalog._data_sets[param]._data = type_obj(  # type: ignore
                **catalog._data_sets[param]._data  # type: ignore
            )

    def _get_param_types(self, catalog: DataCatalog) -> Dict[str, str]:
        param_types = {}

        for name, dataset in catalog._data_sets.items():
            if name.startswith("params:") and name.endswith(self._type_suffix):
                param_name = name[: -len(self._type_suffix)]
                if param_name in catalog._data_sets:
                    param_types[param_name] = dataset._data  # type: ignore
        return param_types

    def _get_param_types_inline(self, catalog: DataCatalog) -> Dict[str, str]:
        param_types = {}

        for name, dataset in catalog._data_sets.items():
            if (
                name.startswith("params:")
                and isinstance(dataset._data, dict)  # type: ignore
                and self._type_indicator in dataset._data  # type: ignore
            ):
                param_types[name] = dataset._data.pop(self._type_indicator)  # type: ignore
        return param_types


# class TapParameters:
#     """Load parameters into a typed-argument-parser"""

#     def __init__(self, type_indicator: str = "type", inline: bool = False):
#         self._type_indicator = type_indicator
#         self._type_suffix = f"__{type_indicator}"
#         self._inline = inline

#     @hook_impl
#     def after_catalog_created(self, catalog: DataCatalog) -> None:
#         if self._inline:
#             param_types = self._get_param_types_inline(catalog)
#         else:
#             param_types = self._get_param_types(catalog)

#         for param, type_string in param_types.items():
#             type_obj = load_obj(type_string)
#             catalog._data_sets[param]._data = type_obj().from_dict(  # type: ignore
#                 catalog._data_sets[param]._data  # type: ignore
#             )

#     def _get_param_types(self, catalog: DataCatalog) -> Dict[str, str]:
#         param_types = {}

#         for name, dataset in catalog._data_sets.items():
#             if name.startswith("params:") and name.endswith(self._type_suffix):
#                 param_name = name[: -len(self._type_suffix) + 1]
#                 if param_name in catalog._data_sets:
#                     param_types[param_name] = dataset._data  # type: ignore
#         return param_types

#     def _get_param_types_inline(self, catalog: DataCatalog) -> Dict[str, str]:
#         param_types = {}

#         for name, dataset in catalog._data_sets.items():
#             if (
#                 name.startswith("params:")
#                 and isinstance(dataset._data, dict)  # type: ignore
#                 and self._type_suffix in dataset._data  # type: ignore
#             ):
#                 param_types[name] = dataset._data.pop(self._type_suffix)  # type: ignore
#         return param_types


class PDBNodeDebugHook:
    """A hook class for creating a post mortem debugging with the PDB debugger
    whenever an error is triggered within a node. The local scope from when the
    exception occured is available within this debugging session.
    """

    @hook_impl
    def on_node_error(self):
        _, _, traceback_object = sys.exc_info()

        #  Print the traceback information for debugging ease
        traceback.print_tb(traceback_object)

        # Drop you into a post mortem debugging session
        pdb.post_mortem(traceback_object)
