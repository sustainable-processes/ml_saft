"""
This is a boilerplate pipeline 'cosmo_rs'
generated using Kedro 0.18.4
"""

import logging
import re
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import pandas as pd
from ase import Atoms
from tqdm import tqdm

from dl4thermo.extras.kedro_datasets.checkpoint_dataset import (
    create_dataframe_partitions,
)
from dl4thermo.extras.utils.cosmo_calculate import CosmoCalculate, guess_cosmobase_path


def calculate_properties(
    df_partitions: Dict[str, pd.DataFrame],
    save_dir: str,
    n_cores: Union[int, str] = "max",
    timeout: float = 10.0,
    calculate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, pd.DataFrame]:
    """Calculate boiling points for a set of molecules.

    Arguments
    ---------
    df_partitions: Dict[str, Callable[[], pd.DataFrame]]
        A dictionary of functions that return a dataframe. The keys are the names of the
        batches. The functions are used to lazily load the dataframes.
    n_cores: Optional[Union[int, str]]
        The number of cores to use for the calculation. If "max", use all available
        cores. If None, use the default number of cores.
    calculate_kwargs: Optional[Dict[str, float]]
        Keyword arguments to pass to the CosmoCalculate.calculate method.


    """
    logger = logging.getLogger(__name__)

    # Setup COSMO-RS
    calc_func = CosmoCalculate(
        ["Vapor pressure", "Density"],
        lookup_name="uniqueCode12",
        lookup_type=CosmoCalculate.UNICODE,
        background=True,
        n_cores=n_cores,
    )
    # Connect to database with example
    calc_func.ct.searchName("ethanol")

    bar = tqdm(df_partitions.items())
    calculate_kwargs = calculate_kwargs or {}
    new_dfs = {}
    working_dir = Path(".")
    save_dir_ = Path(save_dir)
    for batch_name, df in bar:
        # Actually run calculations
        bar.set_description(f"Calculating using {calc_func.ct.getNCores()} cores")
        try:
            df_batch = _calculate_batch(
                df,
                calc_func=calc_func,
                timeout=timeout,
                calculate_kwargs=calculate_kwargs,
            )
            if df_batch is not None:
                new_dfs[batch_name] = df_batch
            else:
                new_dfs[batch_name] = pd.DataFrame()

            # Move all files to data directory
            save_dir_.mkdir(parents=True, exist_ok=True)
            for file_type in [".inp", ".out", ".tab"]:
                for file in working_dir.glob(f"*{file_type}"):
                    file.rename(save_dir_ / file.name)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected. Killing queue.")
            calc_func.ct.killQueue()
            break
        except ValueError as e:
            logger.error(e)
    bar.close()
    return new_dfs


def _calculate_batch(
    df: pd.DataFrame,
    calc_func: CosmoCalculate,
    timeout: float,
    calculate_kwargs: Dict,
) -> Union[pd.DataFrame, None]:
    # Setup calculations
    paths = []
    for _, row in df.iterrows():
        try:
            path = calc_func(row, **calculate_kwargs)
            paths.append(path)
            calc_func.ct.waitQueue()  # Make sure there is space in the queue
        except ValueError:
            warnings.warn(f"Cannot find molecule {row[calc_func.lookup_name]}")
            paths.append(None)

    # Finish queue with timeout
    start = datetime.now()
    global _queue_list
    _queue_list = list()
    waiting_jobs = calc_func.ct.checkQueue(submit_jobs=False, verbose=False)
    while len(waiting_jobs) > 0 and (datetime.now() - start).total_seconds() < timeout:
        time.sleep(timeout // 2)
        waiting_jobs = calc_func.ct.checkQueue(submit_jobs=False, verbose=False)
        _queue_list = list()  # No subsequent jobs allowed
    calc_func.ct.killQueue()

    # Read results
    mol_dfs = []
    for i, path in enumerate(paths):
        if path is None:
            continue
        try:
            mols = calc_func.read(path=path + "_ct")
        except IOError:
            continue
        if mols is not None:
            # Get dipole moment
            dipole = mols[0].getProperty("Dipole")
            dipole_x = mols[0].getProperty("DipoleX")
            dipole_y = mols[0].getProperty("DipoleY")
            dipole_z = mols[0].getProperty("DipoleZ")

            # Get sigma moments
            sigma_moments = {
                f"sigma_moment_{i}": mols[0].getProperty(f"sig{i}") for i in range(1, 7)
            }

            # Get density and vapor pressure
            mol_df = pd.DataFrame(
                [
                    {
                        "compoundID": df.iloc[i]["compoundID"],
                        "compoundName": df.iloc[i]["compoundName"],
                        "casNumber": df.iloc[i]["casNumber"],
                        "smiles": df.iloc[i]["smiles"],
                        "uniqueCode12": df.iloc[i]["uniqueCode12"],
                        "Temperature[K]": Ti,
                        "Density[g/ml]": mols[1].getProperty(
                            "Density", T=Ti, verbose=False
                        ),
                        "ExpDensity[g/ml]": mols[1].getProperty(
                            "Exp_Density", T=Ti, verbose=False
                        ),
                        "PVtot[bar]": mols[1].getProperty("PVtot", T=Ti) / 1e3,
                        "Dipole[Debye]": dipole,
                        "DipoleX[Debye]": dipole_x,
                        "DipoleY[Debye]": dipole_y,
                        "DipoleZ[Debye]": dipole_z,
                        **sigma_moments,
                    }
                    for Ti in mols[1].getPropertyTemperatures("PVtot")
                ]
            )
            mol_dfs.append(mol_df)
    if len(mol_dfs) > 0:
        return pd.concat(mol_dfs, axis=0)


def generate_atoms(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Atoms]:
    logger = logging.getLogger(__name__)
    base_path = guess_cosmobase_path().parent / "BP-TZVPD-FINE/"
    atoms_dict = {}
    stop = False
    for _, df in tqdm(dfs.items(), total=len(dfs)):
        names = df["compoundName"]
        names = pd.unique(names)
        if stop:
            break
        for name in names:
            cosmo_filepaths = base_path.glob(f"**/{name}_c0.cosmo")
            try:
                cosmo_filepath = next(cosmo_filepaths)
                xyz = cosmo_to_atoms(cosmo_filepath=cosmo_filepath)
                atoms_dict[name] = xyz
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected. Stopping at end of batch")
                stop = True
            except StopIteration:
                logger.warning(f"Failed to find COSMO file for {name}")
            except Exception as e:
                logger.error(f"Failed to convert COSMO file for {name}.")
    return atoms_dict


def cosmo_to_atoms(cosmo_filepath: Path) -> Atoms:
    with open(cosmo_filepath, "r") as f:
        cosmo_file_lines = f.readlines()
    xyz = False
    xyz_lines = []
    elements = []

    # Read in coordinates
    for line in cosmo_file_lines:
        if "$coord_car" in line:
            xyz = False
        if xyz:
            xyz_line = re.split(r"\s+", line)
            xyz_line = [s.rstrip("\n") for s in xyz_line]
            x = float(xyz_line[2].rstrip("0"))
            y = float(xyz_line[3].rstrip("0"))
            z = float(xyz_line[4].rstrip("0"))
            elements.append(xyz_line[5].title())
            xyz_lines.append([x, y, z])
        if "#atom" in line:
            xyz = True

    # Create Atoms object
    atoms = Atoms(elements, xyz_lines)
    return atoms
