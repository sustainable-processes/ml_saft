from typing import Any, Dict, Tuple, Union

import ase
import numpy as np
import pandas as pd
import ray
from rdkit import Chem
from rdkit.Chem import AllChem

from .parallel import ParallelMoleculeTask
import logging

logger = logging.getLogger(__name__)

class RDKitConformerRunner(ParallelMoleculeTask):
    @staticmethod
    def filter_func(
        data: pd.DataFrame, smiles: str, smiles_column: str, id_column: str
    ) -> Tuple[Any, Dict[str, Any], bool]:
        return (
            str(int(data[data[smiles_column] == smiles].iloc[0][id_column])),
            {},
            True,
        )

    @staticmethod  # type: ignore
    @ray.remote  # type: ignore
    def run_func(
        molecule_data: str, smiles: str, **kwargs
    ) -> Tuple[Union[Tuple[str, ase.Atoms], None], bool]:
        """
        molecule_data is unique identifier of the molecule that's saved as the name of the file
        Can pass nconf as a keyword argument to set number of conformers generated
        Default is 10

        Notes:
        Random coordinates initialization improves conformer generation for a slight speed penalty
        https://greglandrum.github.io/rdkit-blog/posts/2021-01-31-looking-at-random-coordinate-embedding.html

        """
        # Create RDKit mol
        mol = Chem.MolFromSmiles(smiles)  # type: ignore
        mol = Chem.AddHs(mol)  # type: ignore

        # Use RDKit to generate conformers
        ps = AllChem.ETKDGv3()  # type: ignore
        ps.useRandomCoords = True
        nconf = kwargs.get("nconf", 10)
        try:
            AllChem.EmbedMultipleConfs(mol, nconf, ps)  # type: ignore
            AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")  # type: ignore
            results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant="MMFF94s")  # type: ignore
        except ValueError as e:
            logger.error(e)
            return None, False

        # Find minimum energy conformer
        energies = [r[1] for r in results if r[0]==0]
        if len(energies) > 0:
            min_energy_idx = int(np.argmin(energies))
            conf = mol.GetConformer(min_energy_idx)
        else:
            return None, False

        # Create Atoms object
        elements = []
        xyz_lines = []
        for ai, a in enumerate(mol.GetAtoms()):
            positions = conf.GetAtomPosition(ai)
            elements.append(a.GetSymbol())
            xyz_lines.append(
                [
                    positions.x,
                    positions.y,
                    positions.z,
                ]
            )
        return (molecule_data, ase.Atoms(elements, xyz_lines)), True

    @staticmethod
    def process_func(
        results: Tuple[str, ase.Atoms], **kwargs
    ) -> Tuple[Tuple[str, ase.Atoms], bool]:
        return results, True
