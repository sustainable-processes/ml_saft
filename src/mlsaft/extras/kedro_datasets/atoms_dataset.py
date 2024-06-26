from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from ase import Atoms
from ase.io.extxyz import read_xyz
from kedro.io import AbstractDataSet


class AtomsDataSet(AbstractDataSet):
    def __init__(self, filepath: str):
        """Creates a new Schnetpack Atoms object from an xyz file.

        Args:
            filepath: The location of the image file to load / save data.
        """
        self._filepath = filepath

    def _load(self) -> Atoms:
        """

        Returns:
            An xyz file parsed into an ASE Atoms object.
        """
        with open(self._filepath, "r") as f:
            return list(read_xyz(f, 0))[0]

    def _save(self, data: Union[Atoms, List[Tuple[str, Atoms]]]):
        p = Path(self._filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, list):
            p = p.parent
            for name, d in data:
                d.write(p / f"{name}.xyz")
        else:
            data.write(p)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath)
