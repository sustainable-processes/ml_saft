import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ase import Atoms
from rdkit import Chem
from tqdm import tqdm

from dl4thermo.extras.utils.molecular_fingerprints import _canonicalize_smiles


def prepare_data(
    data: pd.DataFrame,
    smiles_columns: Union[str, List[str]],
    target_columns: Union[str, List[str]],
    fill_na: Optional[Dict[str, float]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    outlier_check_columns: Optional[List[str]] = None,
    outlier_std_devs_cutoff: float = 0.2,
    min_n_atoms: Optional[int] = None,
    drop_duplicates: bool = True,
    dropna: bool = True,
    canonicalize_smiles: bool = True,
    available_conformers: Optional[Dict[str, Callable[[], Atoms]]] = None,
    conformer_id_lookup_column: str = "Name",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare data for downstream tasks

    Arguments
    ----------
    data: pd.DataFrame
        Dataframe containing the data
    smiles_columns: Union[str, List[str]]
        Column(s) containing the SMILES strings
    target_columns: List[str]
        Column(s) containing the target values
    fill_na: Optional[Dict[str, float]]
        Dictionary of columns and values to fill NaNs with
    outlier_check_columns: Optional[List[str]]
        List of columns to remove outliers from
    outlier_std_devs_cutoff: float
        Number of standard deviations to use for outlier cutoff
    drop_duplicates: bool
        Whether to drop duplicate smiles
    dropna: bool
        Whether to drop rows with NaNs in smiles_columns and target_columns
    canonicalize_smiles: bool
        Whether to canonicalize SMILES strings and drop those that fail
    available_conformers: Optional[Dict[str, Callable[[], Atoms]]]
        Dictionary of molecules with conformers calculated.
        Molecules without conformers will be filtered.
    conformer_id_lookup_column: str
        Column to use to lookup as keys in available_conformers

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Dataframe with outliers removed and dataframe with outliers
    """
    logger = logging.getLogger(__name__)

    # Make sure smiles and target columns are lists
    smiles_columns = (
        [smiles_columns] if isinstance(smiles_columns, str) else smiles_columns
    )
    target_columns = (
        [target_columns] if isinstance(target_columns, str) else target_columns
    )

    # Reset index
    data = data.reset_index(drop=True)
    original_data = data.copy()
    original_idx = set(data.index)

    # Make sure targets are floats
    for col in target_columns:
        data[col] = data[col].astype(float)

    # Fill NaNs
    if fill_na is not None:
        for col, val in fill_na.items():
            data[col] = data[col].fillna(val)
    # Drop na, duplicates
    if dropna:
        data = data.dropna(subset=smiles_columns + target_columns)
    if drop_duplicates:
        data = data.drop_duplicates(subset=smiles_columns)

    if bounds is not None:
        for col, b in bounds.items():
            data = data[data[col].between(b[0], b[1], inclusive="neither")]

    # Check if XYZ files are available
    if available_conformers is not None:
        to_drop = []
        for i, row in data.iterrows():
            molecule = str(row[conformer_id_lookup_column])
            if molecule not in available_conformers:
                logger.debug(f"Molecule '{molecule}' not found in available molecules.")
                to_drop.append(i)
        data = data.drop(index=to_drop)
        logger.info(f"Dropped {len(to_drop)} molecules that don't have conformers.")

    # Remove outliers
    if outlier_check_columns:
        n_init = data.shape[0]
        data = remove_outliers(data, outlier_check_columns, outlier_std_devs_cutoff)
        n_dropped = n_init - data.shape[0]
        logger.info(f"Dropped {n_dropped} outliers.")

    # Canonicalize SMILES
    if canonicalize_smiles:
        for smiles_column in smiles_columns:
            n_init = data.shape[0]
            tqdm.pandas(
                total=data.shape[0],
                desc=f"Canonicalizing SMILES for column '{smiles_column}'",
            )
            data[smiles_column] = data[smiles_column].progress_apply(_canonicalize_smiles)  # type: ignore
            data = data.dropna(subset=smiles_column)

            n_dropped = n_init - data.shape[0]
            logger.info(
                f"Dropped {n_dropped} rows with invalid SMILES in column '{smiles_column}'."
            )

    # Remove really small molecules
    def count_n_atoms(smi: str):
        mol = Chem.MolFromSmiles(smi)  # type: ignore
        return mol.GetNumAtoms()

    if min_n_atoms is not None:
        for smiles_column in smiles_columns:
            n_init = data.shape[0]
            tqdm.pandas(
                total=data.shape[0],
                desc=f"Filtering by molecule size in '{smiles_column}'",
            )
            n_atoms: pd.Series = data[smiles_column].progress_apply(
                lambda x: count_n_atoms(x)
            )
            data = data[n_atoms >= min_n_atoms]
            n_dropped = n_init - data.shape[0]
            logger.info(
                f"Dropped {n_dropped} rows with less than {min_n_atoms} atoms in column '{smiles_column}'."
            )

    # Drop again after canonicalization
    if drop_duplicates:
        data = data.drop_duplicates(subset=smiles_columns)

    all_removed_idx = original_idx - set(data.index)
    logger.info(f"Removed {len(all_removed_idx)} rows from original data.")
    removed_data: pd.DataFrame = original_data.iloc[pd.Series(list(all_removed_idx))]  # type: ignore
    return data, removed_data


def remove_outliers(df: pd.DataFrame, columns: List[str], std_devs: float = 1):
    """Remove outliers from a dataframe based on the number of standard deviations.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to remove outliers from.
    columns : List[str]
        List of columns to remove outliers from.
    std_devs : int, optional
        Number of standard deviations to use as a threshold, by default 1


    Returns
    -------
    pd.DataFrame
        Dataframe with outliers removed.

    """
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        tmp_lower_bound = mean - std_devs * std
        tmp_upper_bound = mean + std_devs * std
        df = df[(df[col] >= tmp_lower_bound) & (df[col] <= tmp_upper_bound)]

    return df


def rename_column(df, name_chages: List[Tuple[str, str]]):
    for old, new in name_chages:
        df = df.rename(columns={old: new})
    return df


def jsonify_dict(d, copy=True):
    """Make dictionary JSON serializable"""
    if copy:
        d = deepcopy(d)
    for k, v in d.items():
        if type(v) == np.ndarray:
            d[k] = v.tolist()
        elif type(v) == list:
            d[k] = jsonify_list(v)
        elif type(v) == dict:
            d[k] = jsonify_dict(v)
        elif type(v) in (np.int64, np.int32, np.int8):
            d[k] = int(v)
        elif type(v) in (np.float16, np.float32, np.float64):
            d[k] = float(v)
        elif type(v) in [str, int, float, bool, tuple] or v is None:
            pass
        else:
            raise TypeError(f"Cannot jsonify type for {v}: {type(v)}.")
    return d


def jsonify_list(a, copy=True):
    if copy:
        a = deepcopy(a)
    for i, l in enumerate(a):
        if type(l) == list:
            a[i] = jsonify_list(l)
        elif type(l) == dict:
            a[i] = jsonify_dict(l)
        elif type(l) == np.ndarray:
            a[i] = l.tolist()
        elif type(l) in [str, int, float, bool, tuple] or l is None:
            pass
        else:
            raise TypeError(f"Cannot jsonify type for {l}: {type(l)}.")
    return a
