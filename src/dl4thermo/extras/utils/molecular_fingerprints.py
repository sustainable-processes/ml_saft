"""Functions to compute fingerprints for molecules.

From: https://github.com/swansonk14/chem_utils
"""
import logging
from multiprocessing import Pool
from typing import Callable, List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray  # type: ignore
from tqdm import tqdm

MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048
# MORGAN_NUM_BITS = 16384
logger = logging.getLogger(__name__)


def _canonicalize_smiles(smi) -> Union[str, None]:
    try:
        return Chem.CanonSmiles(smi)
    except:
        return None


def compute_morgan_fingerprint(
    mol: Union[str, Chem.Mol],  # type: ignore
    radius: int = MORGAN_RADIUS,
    num_bits: int = MORGAN_NUM_BITS,
) -> Union[np.ndarray, None]:
    """Generates a binary Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e., either a SMILES string or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D boolean numpy array (num_bits,) containing the binary Morgan fingerprint.
    """
    try:
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol  # type: ignore
        morgan_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)  # type: ignore
        morgan_fp = np.zeros((1,))
        ConvertToNumpyArray(morgan_vec, morgan_fp)
        morgan_fp = morgan_fp.astype(bool)
    except Exception as e:
        logger.warning(f"Could not compute Morgan fingerprint for molecule: {mol}.")
        morgan_fp = np.zeros((num_bits,), dtype=bool)

    return morgan_fp


def compute_morgan_fingerprints(
    mols: List[str],
    radius: int = MORGAN_RADIUS,
    num_bits: int = MORGAN_NUM_BITS,
) -> np.ndarray:
    """Generates molecular fingerprints for each molecule in a list of molecules (in parallel).

    :param mols: A list of molecules (i.e., either a SMILES string or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 2D numpy array (num_molecules, num_features) containing the fingerprints for each molecule.
    """

    return np.array(
        [
            compute_morgan_fingerprint(mol, radius=radius, num_bits=num_bits)
            for mol in tqdm(
                mols,
                total=len(mols),
                desc=f"Generating morgan fingerprints",
            )
        ],
        dtype=float,
    )
