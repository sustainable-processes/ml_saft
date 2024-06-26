import logging
import operator
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from cachetools import Cache, cachedmethod
from kedro.io import IncrementalDataSet


class CheckPointDataSet(IncrementalDataSet):
    """DataSet that saves checkpoints without incremental processing

    Useful as a way to confirm processing of an upstream IncrementalDataSet
    """

    @cachedmethod(cache=operator.attrgetter("_partition_cache"))
    def _list_partitions(self) -> List[str]:
        """Don't actually miss previous processed data"""
        return [
            path
            for path in self._filesystem.find(self._normalized_path, **self._load_args)
            if path.endswith(self._filename_suffix)
        ]

    def _save(self, data: Dict[str, Any]) -> None:
        if self._overwrite and self._filesystem.exists(self._normalized_path):
            self._filesystem.rm(self._normalized_path, recursive=True)

        for partition_id, partition_data in sorted(data.items()):
            kwargs = deepcopy(self._dataset_config)
            partition = self._partition_to_path(partition_id)
            # join the protocol back since tools like PySpark may rely on it
            kwargs[self._filepath_arg] = self._join_protocol(partition)
            dataset = self._dataset_type(**kwargs)  # type: ignore
            if callable(partition_data):
                partition_data = partition_data()
            dataset.save(partition_data)
            # Save the checkpoint
            self._checkpoint.save(partition_id)  # checkpoint to last partition
        self._invalidate_caches()


def create_dataframe_partitions(
    data: pd.DataFrame,
    partition_size: int,
) -> Dict[str, pd.DataFrame]:
    """
    Arguments
    ----------
    molecules : list of str
        Molecule names to resolve to SMILES
    partition_size : int, optional
        The size of partitions used for saving intermediate results for lookup.

    """
    n_molecules = len(data)
    n_batches = n_molecules // partition_size
    n_batches += 0 if n_molecules % partition_size == 0 else 1
    return {
        str(i): data.iloc[i * partition_size : (i + 1) * partition_size]
        for i in range(n_batches)
    }


def concat_partitioned_dfs(
    partitions: Dict[str, Callable],
    no_na_columns: Optional[Union[str, List[str]]] = None,
    float_columns: Optional[List[str]] = None,
    str_columns: Optional[List[str]] = None,
    int_columns: Optional[List[str]] = None,
    keep_columns: Optional[List[str]] = None,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Load and concatenate Dortmund batches
    Also remove references at the end of each file
    """
    logger = logging.getLogger(__name__)
    dfs = []
    for _, df_load_func in partitions.items():
        try:
            if callable(df_load_func):
                df = df_load_func()
            else:
                df = df_load_func
            if no_na_columns:
                no_na_columns = (
                    [no_na_columns] if isinstance(no_na_columns, str) else no_na_columns
                )
                # Remove references - component names columns are NA
                for no_na_column in no_na_columns:
                    df = df[~df[no_na_column].isna()]
            if float_columns:
                for col in float_columns:
                    df.loc[:, col] = df.loc[:, col].astype(float)
            if str_columns:
                for col in str_columns:
                    df.loc[:, col] = df.loc[:, col].astype(str)
            if int_columns:
                for col in int_columns:
                    df[col] = df[col].astype(int)
            dfs.append(df)
        except Exception as e:
            logger.error(e)
    final_df = pd.concat(dfs)
    if reset_index:
        final_df = final_df.reset_index(drop=True)
    if keep_columns is not None:
        final_df = final_df[keep_columns]
    return final_df
