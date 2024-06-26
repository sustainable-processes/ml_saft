import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib import request as request

import ase
import numpy as np
import pandas as pd
import schnetpack as spk
import schnetpack.properties as structure
import schnetpack.transform as trn
import torch
from ase.db import connect
from rdkit import Chem
from rdkit.Chem import Lipinski
from schnetpack.data import ASEAtomsData, AtomsDataFormat, AtomsDataModule, AtomsLoader
from schnetpack.data.atoms import AtomsDataError
from schnetpack.datasets import QM9
from schnetpack.task import AtomisticTask, ModelOutput
from schnetpack.utils import as_dtype
from torchmetrics import Metric
from tqdm import tqdm

from mlsaft.extras.utils.metrics import get_torchmetrics

from .base import LightningValidator, TrainArgs

logger = logging.getLogger(__name__)


@dataclass
class SPKTrainArgs(TrainArgs):
    # Data
    targets_atomistic_type: List[str] = field(default_factory=list)
    molecule_lookup_column: str = "Name"
    database_name: str = "pcp.db"

    # Model architecture
    cutoff: float = 5.0
    n_atom_basis: int = 128
    n_rbf: int = 20
    n_interactions: int = 3
    model_type: str = "PaiNN"
    use_dipole_vector_representation: bool = False
    wandb_artifact_name: str = "spk"
    freeze_encoder: bool = False

    # Training parameters
    targets_transformation: Optional[List[str]] = None
    esp: int = 25
    optimizer: str = "AdamW"
    scheduler: str = "ReduceLROnPlateau"

    def __post_init__(self):
        super().__post_init__()
        if (
            self.use_dipole_vector_representation
            and self.wandb_tags is not None
            and "vector_dipole" not in self.wandb_tags
        ):
            self.wandb_tags.append("vector_dipole")


class NewASEAtomsData(ASEAtomsData):
    def __init__(
        self,
        datapath: str,
        load_properties: Optional[List[str]] = None,
        load_structure: bool = True,
        metadata_fields: Optional[List[str]] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        subset_idx: Optional[List[int]] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
    ):
        super().__init__(
            datapath,
            load_properties,
            load_structure,
            transforms,
            subset_idx,
            property_units,
            distance_unit,
        )
        self.metadata_fields = metadata_fields

    def _get_properties(
        self,
        conn,
        idx: int,
        load_properties: List[str],
        load_structure: bool,
    ):
        row = conn.get(idx + 1)

        # extract properties
        # TODO: can the copies be avoided?
        properties = {}
        properties[structure.idx] = torch.tensor([idx])
        for pname in load_properties:
            properties[pname] = (
                torch.tensor(row.data[pname].copy()) * self.conversions[pname]
            )

        # extract metadata <-- added this
        if self.metadata_fields is not None:
            for mname in self.metadata_fields:
                properties[mname] = row.data[mname]

        Z = row["numbers"].copy()
        properties[structure.n_atoms] = torch.tensor([Z.shape[0]])

        if load_structure:
            properties[structure.Z] = torch.tensor(Z, dtype=torch.long)
            properties[structure.position] = (
                torch.tensor(row["positions"].copy()) * self.distance_conversion
            )
            properties[structure.cell] = (
                torch.tensor(row["cell"][None].copy()) * self.distance_conversion
            )
            properties[structure.pbc] = torch.tensor(row["pbc"])

        return properties

    @staticmethod
    def create(
        datapath: str,
        distance_unit: str,
        property_unit_dict: Dict[str, str],
        atomrefs: Optional[Dict[str, List[float]]] = None,
        **kwargs,
    ) -> "ASEAtomsData":
        """

        Args:
            datapath: Path to ASE DB.
            distance_unit: unit of atom positions and cell
            property_unit_dict: Defines the available properties of the datasetseta and
                provides units for ALL properties of the dataset. If a property is
                unit-less, you can pass "arb. unit" or `None`.
            atomrefs: dictionary mapping properies (the keys) to lists of single-atom
                reference values of the property. This is especially useful for
                extensive properties such as the energy, where the single atom energies
                contribute a major part to the overall value.
            kwargs: Pass arguments to init.

        Returns:
            newly created ASEAtomsData

        """
        if not datapath.endswith(".db"):
            raise AtomsDataError(
                "Invalid datapath! Please make sure to add the file extension '.db' to "
                "your dbpath."
            )

        if os.path.exists(datapath):
            raise AtomsDataError(f"Dataset already exists: {datapath}")

        atomrefs = atomrefs or {}

        with connect(datapath) as conn:  # type: ignore
            conn.metadata = {
                "_property_unit_dict": property_unit_dict,
                "_distance_unit": distance_unit,
                "atomrefs": atomrefs,
            }

        return NewASEAtomsData(datapath, **kwargs)


def load_dataset(datapath: str, format: AtomsDataFormat, **kwargs) -> ASEAtomsData:
    """
    Load dataset.

    Args:
        datapath: file path
        format: atoms data format
        **kwargs: arguments for passed to AtomsData init

    """
    if format is AtomsDataFormat.ASE:
        dataset = NewASEAtomsData(datapath=datapath, **kwargs)
    else:
        raise AtomsDataError(f"Unknown format: {format}")
    return dataset


def _atoms_collate_fn(batch):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    elem = batch[0]
    idx_keys = {structure.idx_i, structure.idx_j, structure.idx_i_triples}
    # Atom triple indices must be treated separately
    idx_triple_keys = {structure.idx_j_triples, structure.idx_k_triples}

    coll_batch = {}
    for key in elem:
        if key == "SMILES":
            coll_batch[key] = [d[key] for d in batch]
        elif (key not in idx_keys) and (key not in idx_triple_keys):
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)
        elif key in idx_keys:
            coll_batch[key + "_local"] = torch.cat([d[key] for d in batch], 0)

    seg_m = torch.cumsum(coll_batch[structure.n_atoms], dim=0)
    seg_m = torch.cat([torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0)
    idx_m = torch.repeat_interleave(
        torch.arange(len(batch)), repeats=coll_batch[structure.n_atoms], dim=0
    )
    coll_batch[structure.idx_m] = idx_m

    for key in idx_keys:
        if key in elem.keys():
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], 0
            )

    # Shift the indices for the atom triples
    for key in idx_triple_keys:
        if key in elem.keys():
            indices = []
            offset = 0
            for idx, d in enumerate(batch):
                indices.append(d[key] + offset)
                offset += d[structure.idx_j].shape[0]
            coll_batch[key] = torch.cat(indices, 0)

    return coll_batch


class SPKDataModule(AtomsDataModule):
    """A general Schnet data module.

    The molecular representation is based on **atom** coordinates and thus stores 3D information.
    """

    # properties
    mu = "mu"
    sigma = "sigma"
    epsilon_k = "epsilon_k"
    m = "m"
    KAB = "KAB"
    epsilonAB = "epsilonAB"

    def __init__(
        self,
        molecules_list: Dict[str, Callable[[], ase.Atoms]],
        smiles_column: str,
        target_property_data: pd.DataFrame,
        database_path: str,
        batch_size: int,
        split_idx: Tuple[np.ndarray, np.ndarray, np.ndarray],
        molecule_lookup_column: str = "Name",
        target_properties: Optional[List[str]] = None,
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        remove_uncharacterized: bool = False,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 2,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        pin_memory: bool = False,
        **kwargs,
    ):
        """

        Args:
            molecules: dictionary of molecules
            target_property_data: dataframe with target properties
            datapath: path to dataset
            batch_size: (train) batch size
            format: dataset format
            load_properties: subset of properties to load
            remove_uncharacterized: do not include uncharacterized molecules.
            val_batch_size: validation batch size. If None, use test_batch_size, then batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then batch_size.
            transforms: Transform applied to each system separately before batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers (overrides num_workers).
            num_test_workers: Number of test data loader workers (overrides num_workers).
            property_units: Dictionary from property to corresponding unit as a string (eV, kcal/mol, ...).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. cluster scratch for faster performance.
        """
        super().__init__(
            datapath=database_path,
            batch_size=batch_size,
            format=format,
            load_properties=load_properties,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            num_test_workers=num_test_workers,
            property_units=property_units,
            distance_unit=distance_unit,
            data_workdir=data_workdir,
            pin_memory=pin_memory,
            **kwargs,
        )
        self.smiles_column = smiles_column
        self.molecule_lookup_column = molecule_lookup_column
        self.target_properties = target_properties
        self.remove_uncharacterized = remove_uncharacterized
        self._predict_dataloader = None
        self._molecules = molecules_list
        self._target_property_data = target_property_data
        # Ensure prepare data is only called once
        self.prepare_data_per_node = False
        self._split_idx = split_idx
        self.train_idx: Optional[List] = None
        self.val_idx: Optional[List] = None
        self.test_idx: Optional[List] = None

    def prepare_data(self):
        """Prepare data for dataloader."""
        datapath = Path(self.datapath)
        if datapath.exists():
            datapath.unlink()

        if self.target_properties is None:
            property_unit_dict = {
                SPKDataModule.mu: "Debye",
                SPKDataModule.sigma: "None",
                SPKDataModule.epsilon_k: "None",
                SPKDataModule.m: "None",
                SPKDataModule.KAB: "None",
                SPKDataModule.epsilonAB: "None",
                "SMILES": "None",
            }
        else:
            property_unit_dict = {p: "None" for p in self.target_properties}
            property_unit_dict.update({"SMILES": "None"})

        """
        From atoms.py:
            atomrefs: dictionary mapping properies (the keys) to lists of single-atom
            reference values of the property. This is especially useful for
            extensive properties such as the energy, where the single atom energies
            contribute a major part to the overall value.
        -> Probably not needed in our case.
        """

        dataset = NewASEAtomsData.create(
            datapath=self.datapath,
            distance_unit="Ang",
            property_unit_dict=property_unit_dict,
            metadata_fields=["SMILES"],
            # atomrefs=atomrefs,
        )

        # Load data
        property_list, atom_list = self._load_data(
            dataset=dataset,
            molecules=self._molecules,
            target_property_data=self._target_property_data,
            molecule_lookup_column=self.molecule_lookup_column,
            smiles_column=self.smiles_column,
        )

        # Save to database
        self._log_with_rank("Writing atoms to db...")
        dataset.add_systems(property_list=property_list, atoms_list=atom_list)

        # Calculate stats
        means, stds = self.get_basic_stats(
            self._target_property_data, self.load_properties
        )
        data_dir = os.path.dirname(self.datapath)
        torch.save(means, os.path.join(data_dir, "means.pt"))
        torch.save(stds, os.path.join(data_dir, "stds.pt"))

        # Clear memory
        self._molecules = {}
        self._target_property_data = pd.DataFrame()

    def setup(self, stage: Optional[str] = None):
        data_dir = os.path.dirname(self.datapath)
        self.means = torch.load(os.path.join(data_dir, "means.pt"))
        self.stds = torch.load(os.path.join(data_dir, "stds.pt"))

        # check whether data needs to be copied
        if self.data_workdir is None:
            datapath = self.datapath
        else:
            datapath = self._copy_to_workdir()

        # (re)load datasets
        if self.dataset is None:
            self.dataset = load_dataset(
                datapath,
                self.format,
                property_units=self.property_units,
                distance_unit=self.distance_unit,
                load_properties=self.load_properties,
                metadata_fields=["SMILES"],
            )

            # load and generate partitions if needed
            if self.train_idx is None or self.val_idx is None or self.test_idx is None:
                self.train_idx = self._split_idx[0].tolist()
                self.val_idx = self._split_idx[1].tolist()
                self.test_idx = self._split_idx[2].tolist()
            self._train_dataset = self.dataset.subset(self.train_idx)
            self._val_dataset = self.dataset.subset(self.val_idx)
            self._test_dataset = self.dataset.subset(self.test_idx)
            self._setup_transforms()

    @staticmethod
    def _load_data(
        dataset: ASEAtomsData,
        molecules: Dict[str, Callable[[], ase.Atoms]],
        target_property_data: pd.DataFrame,
        smiles_column: str = "SMILES",
        molecule_lookup_column: str = "Name",
        smiles_property: bool = True,
    ):
        """Load data from kedro"""
        property_list = []
        atom_list = []
        cols = target_property_data.columns
        for _, row in tqdm(
            target_property_data.iterrows(), total=len(target_property_data)
        ):
            # Atoms
            mol_name = str(row[molecule_lookup_column])
            if mol_name in molecules:
                atoms = molecules[mol_name]()
            else:
                logger.info(f"No xyz file found for {mol_name}. Molecule disregarded.")
                continue

            # Properties
            properties = {}

            for pn in dataset.available_properties:
                if pn != "SMILES" and pn in cols:
                    properties[pn] = np.array([row[pn]])
                if pn != "SMILES" and pn not in cols:
                    properties[pn] = np.array([np.nan])
            if smiles_property:
                properties["SMILES"] = row[smiles_column]

            property_list.append(properties)
            atom_list.append(atoms)

        return property_list, atom_list

    @staticmethod
    def get_basic_stats(
        df: pd.DataFrame, properties: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cols = properties if properties is not None else df.columns
        for col in cols:
            df[col] = df[col].astype(float)
        means = torch.tensor(df[cols].mean().to_numpy())
        stds = torch.tensor(df[cols].std().to_numpy())
        # means = torch.zeros((n_props,))
        # stds = torch.zeros((n_props,))
        # for i, p in enumerate(properties):
        #     x = torch.cat([d[p] for d in self._train_dataset])  # type: ignore
        #     means[i] = x.mean()
        #     stds[i] = x.std()
        return means, stds

    def train_dataloader(self) -> AtomsLoader:
        if self._train_dataloader is None:
            self._train_dataloader = AtomsLoader(
                self.train_dataset,  # type: ignore
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                collate_fn=_atoms_collate_fn,
                pin_memory=self._pin_memory,  # type: ignore
            )
        return self._train_dataloader

    def val_dataloader(self) -> AtomsLoader:
        if self._val_dataloader is None:
            self._val_dataloader = AtomsLoader(
                self.val_dataset,  # type: ignore
                batch_size=self.val_batch_size,
                num_workers=self.num_val_workers,
                collate_fn=_atoms_collate_fn,
                pin_memory=self._pin_memory,  # type: ignore
            )
        return self._val_dataloader

    def test_dataloader(self) -> AtomsLoader:
        if self._test_dataloader is None:
            self._test_dataloader = AtomsLoader(
                self.test_dataset,  # type: ignore
                batch_size=self.test_batch_size,
                num_workers=self.num_test_workers,
                collate_fn=_atoms_collate_fn,
                pin_memory=self._pin_memory,  # type: ignore
            )
        return self._test_dataloader

    def predict_dataloader(self) -> List[AtomsLoader]:
        """Dataloaders for prediction - the training and validation data"""
        if self._predict_dataloader is None:
            self._predict_dataloader = AtomsLoader(
                self.train_dataset,  # type: ignore
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=_atoms_collate_fn,
                pin_memory=self._pin_memory,  # type: ignore
            )

        return [self._predict_dataloader, self.val_dataloader(), self.test_dataloader()]


class NormalizeProperties(trn.Transform):
    """
    Scale an entry of the input or results dictionary.
    The `scale` can be automatically obtained from the AtomsDataModule,
    when it is used. Otherwise, it has to be provided in the init manually.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True
    means: torch.Tensor
    stds: torch.Tensor

    def __init__(self, properties: List[str], inverse: bool = False):
        """
        Args:
            property: The property to normalize
            inverse: Whether to normalize or denormalize
        """
        super().__init__()
        self._properties = properties
        self._inverse = inverse

        n_props = len(self._properties)
        means = torch.zeros((n_props,))
        stds = torch.zeros((n_props,))
        self.register_buffer("means", means, persistent=True)
        self.register_buffer("stds", stds, persistent=True)

    def datamodule(self, _datamodule: SPKDataModule):
        for i, p in enumerate(self._properties):
            j = _datamodule.train_dataset.load_properties.index(p)
            self.means[i] = _datamodule.means[j]
            self.stds[i] = _datamodule.stds[j]

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self._inverse:
            for i, p in enumerate(self._properties):
                inputs[p] = inputs[p] * self.stds[i] + self.means[i]
        else:
            for i, p in enumerate(self._properties):
                inputs[p] = (inputs[p] - self.means[i]) / self.stds[i]
        return inputs


class FilterNonAssociating(trn.Transform):
    """
    Filter out non-associating molecules from the dataset.
    """

    def __init__(self, associating_columns: List[str], smiles_key: str = "SMILES"):
        super().__init__()
        self.associating_columns = associating_columns
        self.smiles_key = smiles_key

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        smiles = inputs[self.smiles_key]
        mols = [Chem.MolFromSmiles(s) for s in smiles]  # type: ignore
        for i, mol in mols:
            num_acceptors = Lipinski.NumHAcceptors(mol)
            num_donors = Lipinski.NumHDonors(mol)
            if num_acceptors == 0 or num_donors == 0:
                for col in self.associating_columns:
                    inputs[col][i] = 0.0
        return inputs


class CustomModelOutput(ModelOutput):
    def __init__(
        self,
        name: str,
        loss_fn: Optional[torch.nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
        filter_associating: bool = False,
    ):
        self.filter_associating = filter_associating
        super().__init__(
            name, loss_fn, loss_weight, metrics, constraints, target_property
        )

    def calculate_loss(self, pred, target):
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0

        # Scope loss to nonzero predictions
        if self.filter_associating:
            zeros_idx = pred[self.target_property] == 0.0
            pred = pred[self.name][~zeros_idx]
            target = target[self.target_property][~zeros_idx]
        else:
            pred = pred[self.name]
            target = target[self.target_property]

        loss = self.loss_weight * self.loss_fn(pred, target)
        return loss


class CustomNeuralNetworkPotential(spk.model.NeuralNetworkPotential):
    def __init__(
        self,
        representation: torch.nn.Module,
        input_modules: List[torch.nn.Module],
        output_modules: List[torch.nn.Module],
        postprocessors: Optional[List[trn.Transform]] = None,
        input_dtype_str: str = "float32",
        do_postprocessing: bool = True,
        associating_columns: Optional[List[str]] = None,
        smiles_key: str = "SMILES",
        use_associating_cache: bool = False,
    ):
        self.associating_columns = associating_columns
        self.smiles_key = smiles_key
        self.use_associating_cache = use_associating_cache
        self.associating_cache = {}
        super().__init__(
            representation,
            input_modules,
            output_modules,
            postprocessors,
            input_dtype_str,
            do_postprocessing,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = super().forward(inputs)
        if self.associating_columns:
            smiles = inputs[self.smiles_key]
            for i, smiles in enumerate(smiles):
                filter = False
                if self.use_associating_cache and smiles in self.associating_cache:
                    filter = not self.associating_cache[smiles]
                else:
                    mol = Chem.MolFromSmiles(smiles)  # type: ignore
                    num_acceptors = Lipinski.NumHAcceptors(mol)
                    num_donors = Lipinski.NumHDonors(mol)
                    if num_acceptors == 0 or num_donors == 0:
                        filter = True
                        if self.use_associating_cache:
                            self.associating_cache[smiles] = False
                if filter:
                    for col in self.associating_columns:
                        outputs[col][i] = 0.0
        return outputs


def setup_spk_tasks(
    args: SPKTrainArgs,
) -> Tuple[
    List[spk.atomistic.Atomwise],
    List[ModelOutput],
    List[trn.Transform],
    List[trn.Transform],
]:
    # Modules for each task
    task_heads = []
    task_outputs = []
    pre_normalization_properties = []
    post_normalization_properties = []
    for target_idx, pred_target in enumerate(args.target_columns):
        # Output modules
        target_atomistic_type = args.targets_atomistic_type[target_idx]
        if target_atomistic_type is not None:
            if target_atomistic_type == "Atomwise":
                pred_task = spk.atomistic.Atomwise(
                    n_in=args.n_atom_basis, output_key=pred_target
                )
            elif target_atomistic_type == "DipoleMoment":
                pred_task = spk.atomistic.DipoleMoment(
                    n_in=args.n_atom_basis,
                    dipole_key=pred_target,
                    predict_magnitude=True,
                    use_vector_representation=args.use_dipole_vector_representation,
                )
            else:
                raise ValueError(
                    f"{target_atomistic_type} is not a valid atomistic type in schnetpack."
                )
            task_heads.append(pred_task)
            weight = args.target_weights[target_idx] if args.target_weights else 1.0
            # filter_associating = False
            # if args.associating_columns is not None:
            #     filter_associating = (
            #         True if pred_target in args.associating_columns else False
            #     )
            task_outputs.append(
                ModelOutput(
                    name=pred_target,
                    target_property=pred_target,
                    loss_fn=torch.nn.MSELoss(),
                    loss_weight=weight,
                    metrics={
                        metric: get_torchmetrics(metric) for metric in args.metrics
                    },
                    # filter_associating=filter_associating,
                )
            )
        else:
            raise ValueError(f"Please specify atomistic type for {pred_target}.")

        # Normalization modules
        if args.targets_transformation is not None:
            ti_transf = args.targets_transformation[target_idx]
            if ti_transf != "None":
                if ti_transf in ["norm", "Norm", "Normalization", "normalization"]:
                    pre_normalization_properties.append(pred_target)
                    post_normalization_properties.append(pred_target)
                else:
                    raise NotImplementedError(
                        f"{ti_transf} target value transfomration is not implemented for schnetpack models."
                    )
    pre_normalization_modules: List[trn.Transform] = [
        NormalizeProperties(properties=pre_normalization_properties)
    ]
    post_normalization_modules: List[trn.Transform] = [
        NormalizeProperties(properties=post_normalization_properties, inverse=True)
    ]
    return (
        task_heads,
        task_outputs,
        pre_normalization_modules,
        post_normalization_modules,
    )


class CastMap(trn.Transform):
    """
    Cast all inputs according to type map.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = True

    def __init__(self, type_map: Dict[str, str]):
        """
        Args:
            type_map: dict with source_type: target_type (as strings)
        """
        super().__init__()
        self.type_map = type_map

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        for k, v in inputs.items():
            try:
                vdtype = str(v.dtype).split(".")[-1]
                if vdtype in self.type_map:
                    inputs[k] = v.to(dtype=as_dtype(self.type_map[vdtype]))
            except AttributeError:
                pass
        return inputs


def setup_spk_data(
    args: SPKTrainArgs,
    molecules: Dict[str, Callable[[], ase.Atoms]],
    target_property_data: pd.DataFrame,
    split_idx: Tuple[np.ndarray, np.ndarray, np.ndarray],
    normalization_modules: Optional[List[trn.Transform]] = None,
) -> SPKDataModule:
    """Setup the lightning datamodule"""
    if normalization_modules is None:
        normalization_modules = []
    transforms = [
        trn.SubtractCenterOfMass(),
        trn.MatScipyNeighborList(cutoff=args.cutoff),
        CastMap(type_map={"float64": "float32"}),
    ]

    train_val_transforms = transforms + normalization_modules
    spk_datamodule = SPKDataModule(
        molecules_list=molecules,
        smiles_column=args.smiles_columns[0],
        target_property_data=target_property_data,
        target_properties=args.target_columns,
        split_idx=split_idx,
        molecule_lookup_column=args.molecule_lookup_column,
        database_path=os.path.join(args.save_dir, args.database_name),
        load_properties=args.target_columns,
        batch_size=args.batch_size,
        train_transforms=train_val_transforms,
        val_transforms=train_val_transforms,
        test_transforms=transforms,
        num_workers=args.num_workers,
        split_file=os.path.join(args.save_dir, "split.npz"),
        pin_memory=True if torch.cuda.is_available() else False,
    )
    return spk_datamodule


class CustomQM9(QM9):
    def __init__(
        self,
        extra_data: Optional[
            List[Tuple[pd.DataFrame, Dict[str, Callable[[], ase.Atoms]]]]
        ] = None,
        smiles_column: str = "smiles",
        molecule_lookup_column: str = "Name",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.extra_data = extra_data
        self.smiles_column = smiles_column
        self.molecule_lookup_column = molecule_lookup_column
        self.extra_data_added = False

    def prepare_data(self):
        super().prepare_data()
        if not self.extra_data_added and self.extra_data is not None:
            for target_property_data, molecules in self.extra_data:
                dataset = load_dataset(
                    self.datapath,
                    self.format,
                    property_units=self.property_units,
                    distance_unit=self.distance_unit,
                    load_properties=self.load_properties,
                )
                property_list, atom_list = SPKDataModule._load_data(
                    dataset,
                    molecules=molecules,
                    target_property_data=target_property_data,
                    smiles_column=self.smiles_column,
                    molecule_lookup_column=self.molecule_lookup_column,
                    smiles_property=False,
                )

                # Save to database
                logger.info("Writing atoms to db...")
                dataset.add_systems(property_list=property_list, atoms_list=atom_list)
            self.extra_data_added = True

    def predict_dataloader(self) -> List[AtomsLoader]:
        """Dataloaders for prediction - the training and validation data"""
        if self._predict_dataloader is None:
            self._predict_dataloader = AtomsLoader(
                self.train_dataset,  # type: ignore
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=_atoms_collate_fn,
                pin_memory=self._pin_memory,  # type: ignore
            )

        return [self._predict_dataloader, self.val_dataloader(), self.test_dataloader()]


def setup_qm9_data(
    args: SPKTrainArgs,
    extra_data: Optional[
        List[Tuple[pd.DataFrame, Dict[str, Callable[[], ase.Atoms]]]]
    ] = None,
):
    # Start with loading QM9 data
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    db_path = save_dir / args.database_name
    return CustomQM9(
        datapath=str(db_path),
        batch_size=args.batch_size,
        extra_data=extra_data,
        smiles_column=args.smiles_columns[0],
        molecule_lookup_column=args.molecule_lookup_column,
        num_train=130000,
        num_val=1000,
        transforms=[
            trn.ASENeighborList(cutoff=5.0),
            trn.CastTo32(),
        ],
        property_units={QM9.mu: "Debye"},
        num_workers=args.num_workers,
        split_file=os.path.join(args.save_dir, "split.npz"),
        pin_memory=True
        if torch.cuda.is_available()
        else False,  # set to false, when not using a GPU
        load_properties=[QM9.mu],  # only load dipole moment
    )


def setup_spk_model(
    args: SPKTrainArgs,
    task_heads: List[spk.atomistic.Atomwise],
    task_outputs: List[ModelOutput],
    post_normalization_modules: List[trn.Transform],
) -> AtomisticTask:
    """Setup the LightningModule for training."""
    # Representation
    pairwise_distance = (
        spk.atomistic.PairwiseDistances()
    )  # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=args.n_rbf, cutoff=args.cutoff)
    if args.model_type == "PaiNN":
        spk_model = spk.representation.PaiNN(
            n_atom_basis=args.n_atom_basis,
            n_interactions=args.n_interactions,
            radial_basis=radial_basis,
            cutoff_fn=spk.nn.CosineCutoff(args.cutoff),
        )
    else:
        raise NotImplementedError(
            "Other model types from schnetpack to be implemented."
        )

    # Model architecture
    postprocessors = [
        CastMap(type_map={"float32": "float64"}),
    ] + post_normalization_modules
    spk_nn = spk.model.NeuralNetworkPotential(
        representation=spk_model,
        input_modules=[pairwise_distance],
        output_modules=task_heads,  # type: ignore
        postprocessors=postprocessors,
        do_postprocessing=True,
        # associating_columns=args.associating_columns
        # if args.associating_columns
        # else None,
        # use_associating_cache=True,
    )

    # Training details
    if args.optimizer == "AdamW":
        optimizer_cls = torch.optim.AdamW
    else:
        raise NotImplementedError(
            f"Optimizer class {args.optimizer} not implemented for schnetpack yet."
        )

    if args.scheduler == "ReduceLROnPlateau":
        scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    else:
        raise NotImplementedError(
            f"Scheduler class {args.scheduler} not implemented for schnetpack yet."
        )

    # Create lightning module
    lit_model = AtomisticTask(
        model=spk_nn,
        outputs=task_outputs,
        optimizer_cls=optimizer_cls,
        optimizer_args={"lr": args.lr},
        scheduler_cls=scheduler_cls,
        scheduler_args={"mode": "min", "patience": 3.0},
        scheduler_monitor="val_loss",
    )
    # lit_model.save_hyperparameters(asdict(args))

    # Freeze encoder
    if args.freeze_encoder:
        for param in lit_model.model.representation.parameters():  # type: ignore
            param.requires_grad = False

    return lit_model


class SPKValidator(LightningValidator):
    lit_model: AtomisticTask
    datamodule: Union[SPKDataModule, CustomQM9]

    def __init__(
        self,
        args: TrainArgs,
        lit_model: AtomisticTask,
        datamodule: Union[SPKDataModule, CustomQM9],
    ):
        super().__init__(args, lit_model, datamodule)

    def get_ground_truth_and_smiles(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
        pp = self.lit_model.model.postprocess
        dls = self.datamodule.predict_dataloader()
        train_ground_truth = torch.column_stack(
            [
                torch.cat([pp(d)[target] for d in dls[0].dataset])
                for target in self.args.target_columns
            ]
        ).numpy()
        val_ground_truth = torch.column_stack(
            [
                torch.cat([pp(d)[target] for d in dls[1].dataset])
                for target in self.args.target_columns
            ]
        ).numpy()
        test_ground_truth = torch.column_stack(
            [
                torch.cat([pp(d)[target] for d in dls[2].dataset])
                for target in self.args.target_columns
            ]
        ).numpy()

        train_smiles = [d["SMILES"] for d in dls[0].dataset]
        val_smiles = [d["SMILES"] for d in dls[1].dataset]
        test_smiles = [d["SMILES"] for d in dls[2].dataset]

        return (
            train_ground_truth,
            val_ground_truth,
            test_ground_truth,
            train_smiles,
            val_smiles,
            test_smiles,
        )

    def _collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            target: torch.cat([t[target] for t in batch])
            for target in self.args.target_columns
        }

    def get_target(
        self, t: Dict[str, torch.Tensor], target: str, idx: int
    ) -> np.ndarray:
        return t[target].cpu().detach().numpy()
