"""
This is a boilerplate pipeline 'train_models'
generated using Kedro 0.18.0
"""
import logging
import random
import string
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from ase import Atoms
from feos import pcsaft, si  # type: ignore
from feos.eos import EquationOfState, State  # type: ignore
from joblib import dump
from matplotlib.figure import Figure
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    callbacks,
    loggers,
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # type: ignore
from wandb.sdk.wandb_run import Run

# Extras
from mlsaft.extras.utils import (
    calculate_metrics,
    compute_morgan_fingerprints,
    functional_group_distribution_plot,
    parity_plot,
    train_test_cluster_split,
    visualize_clusters_umap,
)
from mlsaft.extras.utils.doe import generate_search
from mlsaft.extras.utils.lightning import load_lightningmodule_from_checkpoint
from mlsaft.extras.utils.pcsaft import is_associating, predict_phase_diagram
from mlsaft.extras.utils.wandb_utils import download_best_model  # type: ignore

# Models
from .models import LightningValidator, TrainArgs
from .models.chemprop import (
    ChempropDataModule,
    ChempropLightningModule,
    ChempropTrainArgs,
    ChempropValidator,
    RegressionMoleculeModel,
)
from .models.ffn import (
    FFN,
    FFNDataModule,
    FFNLightningModule,
    FFNTrainArgs,
    FFNValidator,
)
from .models.pcsaft_emulator import (
    PcSaftEmulator,
    PcSaftEmulatorDataModule,
    PcSaftEmulatorLightningModule,
    PcSaftEmulatorTrainArgs,
    PcSaftEmulatorValidator,
)
from .models.pyg import (
    MPNN,
    E2ELightningModule,
    PyGDataModule,
    PyGLightningModule,
    PyGTrainArgs,
    PyGValidator,
)
from .models.sklearn import Estimator, SklearnTrainArgs
from .models.spk import (
    CustomQM9,
    SPKDataModule,
    SPKTrainArgs,
    SPKValidator,
    setup_qm9_data,
    setup_spk_data,
    setup_spk_model,
    setup_spk_tasks,
)


def fingerprints(
    data: pd.DataFrame,
    smiles_columns: Union[str, List[str]],
    radius: int = 2,
    num_bits: int = 2048,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    fps = {}
    if isinstance(smiles_columns, str):
        smiles_columns = [smiles_columns]
    for smiles_column in smiles_columns:
        smiles: List[str] = data[smiles_column].tolist()
        fps[smiles_column] = compute_morgan_fingerprints(
            smiles, radius=radius, num_bits=num_bits
        )
    if len(smiles_columns) == 1:
        return fps[smiles_columns[0]]
    else:
        return fps


def random_split(
    data: pd.DataFrame,
    valid_size: float,
    test_size: float,
):
    """Split data into train, valid, and test sets randomly

    Arguments
    ----------
    data: pd.DataFrame
        Dataframe contain SMILES strings
    valid_size: float
        Fraction of data to use for validation
    test_size: float
        Fraction of data to use for testing

    Returns
    -------
    A dictionary with keys "train_idx", "valid_idx", and "test_idx"

    """
    logger = logging.getLogger(__name__)
    random_state = np.random.RandomState(100)
    if not valid_size + test_size < 1:
        raise ValueError("valid_size + test_size must be less than 1")
    n_test = int(test_size * len(data))
    splitter = ShuffleSplit(n_splits=1, test_size=n_test, random_state=random_state)
    train_valid_idx, test_idx = next(splitter.split(data))
    n_valid = int(valid_size * len(data))
    splitter = ShuffleSplit(n_splits=1, test_size=n_valid, random_state=random_state)
    train_idx, valid_idx = next(splitter.split(data.iloc[train_valid_idx]))
    logging.info(
        f"Train size: {len(train_idx)}, valid size: {len(valid_idx)}, test size: {len(test_idx)}"
    )
    return {"train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx}


def cluster_split(
    fps: np.ndarray,
    test_size: Optional[float] = None,
    valid_size: Optional[float] = None,
    kmeans_kwargs: Optional[Dict[str, Any]] = None,
    umap_before_cluster: bool = True,
    umap_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Dict[str, np.ndarray],
    np.ndarray,
]:
    """Split data into train, valid, and test sets using k-means clustering

    Arguments
    ----------
    fps: np.ndarray
        Morgan fingerprints
    data: pd.DataFrame
        Dataframe
    target_columns: List[str]
        List of target columns
    valid_size: float
        Fraction of data to use for validation
    test_size: float
        Fraction of data to use for testing
    kmeans_kwargs: Dict[str, Any]
        Keyword arguments for sklearn k-means clustering

    Returns
    -------
    clusters: np.ndarray
        The cluster number of each molecule
    train_idx: np.ndarray
        Indices of molecules in the training set
    valid_idx: np.ndarray
        Indices of molecules in the validation set
    test_idx: np.ndarray
        Indices of molecules in the test set

    """

    # Create splits
    train_idx, valid_idx, test_idx, cluster_labels = train_test_cluster_split(
        fps=fps,
        valid_size=valid_size,
        test_size=test_size,
        kmeans_args=kmeans_kwargs,
        umap_before_cluster=umap_before_cluster,
        umap_kwargs=umap_kwargs,
    )

    return (
        {
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": test_idx,
            # "cluster_labels": cluster_labels,
        },
        cluster_labels,
    )


def cv_split(data: pd.DataFrame, n_folds: int, valid_size: float):
    # Create splits using cross validation
    splitter = ShuffleSplit(n_splits=n_folds, test_size=valid_size, random_state=100)
    splits = list(iter(splitter.split(data)))
    splits_dict = {}
    for i, (train_idx, valid_idx) in enumerate(splits):
        splits_dict[f"train_idx_{i}"] = train_idx
        splits_dict[f"valid_idx_{i}"] = valid_idx
    return splits_dict


def setup_cv(
    cv_split_idx: Dict[str, Callable[[], np.ndarray]],
):
    # Reformat into dict of dicts
    splits = {}
    n_folds = 0
    for split_name, split_idx in cv_split_idx.items():
        if split_name not in ["train_idx", "test_idx"]:
            fold_number = int(split_name.split("_")[-1])
            split = split_name.split("_")[0]
            if isinstance(splits.get(fold_number), dict):
                splits[fold_number][f"{split}_idx"] = split_idx
            else:
                splits[fold_number] = {f"{split}_idx": split_idx}
            if fold_number + 1 > n_folds:
                n_folds = fold_number + 1
    test_idx = cv_split_idx["test_idx"]
    for i in range(n_folds):
        splits[i]["test_idx"] = test_idx

    return splits


def id_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def add_group_id(args: TrainArgs):
    # Add wandb group
    group_id = id_generator()
    if args.wandb_kwargs is None:
        args.wandb_kwargs = {"group": f"cv_{group_id}"}
    elif "group" not in args.wandb_kwargs:
        args.wandb_kwargs["group"] = f"cv_{group_id}"
    return f"cv_{group_id}"


def visualize_cluster_split(
    data: pd.DataFrame,
    fps: np.ndarray,
    cluster_labels: np.ndarray,
    split_idx: Dict[str, Callable[[], np.ndarray]],
    umap_kwargs: Dict[str, Any],
    smiles_column: str,
    target_columns: List[str],
    fragments: pd.DataFrame,
    plot_top_k_functional_groups: int,
    umap_kwargs_app: Dict[str, Any],
    run_app: bool,
    label_columns: List[str],
) -> Tuple[Optional[Figure], Figure, Dict[str, Figure], Figure]:
    """Visualize cluster splitting

    Arguments
    ----------
    data: pd.DataFrame
        Dataframe contain SMILES strings
    fps: np.ndarray
        Morgan fingerprints
    cluster_labels: np.ndarray
        Cluster labels
    umap_kwargs: Dict[str, Any]
        Keyword arguments for UMAP

    Returns
    -------
    clusters: np.ndarray
        The cluster number of each molecule
    train_idx: np.ndarray
        Indices of molecules in the training set
    valid_idx: np.ndarray
        Indices of molecules in the validation set
    test_idx: np.ndarray
        Indices of molecules in the test set

    """
    logger = logging.getLogger(__name__)
    train_idx, valid_idx, test_idx = (
        split_idx["train_idx"](),
        split_idx["valid_idx"](),
        split_idx["test_idx"](),
    )

    # UMAP visualization matplotlib
    logger.info("Visualizing clusters with UMAP")
    fig, train_fig, valid_fig, test_fig = visualize_clusters_umap(
        fps=fps,
        cluster_labels=cluster_labels,
        plot_type="matplotlib",
        split_idx=(train_idx, valid_idx, test_idx),
        umap_kwargs=umap_kwargs,
    )  # type: ignore

    # Visualize parameter distributions in pairplot
    logger.info("Visualizing parameter distributions in pairplot")
    data["cluster"] = cluster_labels
    if len(target_columns) > 0:
        pp = sns.pairplot(data, vars=target_columns, corner=True, hue="cluster")
        pp_fig = pp.figure
    else:
        pp_fig = plt.figure()

    # Plot distribution of functional groups betweeen train, test and split
    logger.info("Plotting distribution of functional groups")
    train_df = data.iloc[train_idx]
    train_df["Split"] = "1. Train"
    valid_df = data.iloc[valid_idx]
    valid_df["Split"] = "2. Validation"
    test_df = data.iloc[test_idx]
    test_df["Split"] = "3. Test"
    data_ordered = pd.concat(
        [train_df, valid_df, test_df],
        axis=0,
    )
    fig_functional_groups = functional_group_distribution_plot(
        data_ordered,
        smiles_column=smiles_column,
        fragments_data=fragments,
        group_column="Split",
        frequency_plot=True,
        plot_top_k=plot_top_k_functional_groups,
    )
    fig_functional_groups.tight_layout()

    if run_app:
        visualize_clusters_umap(
            fps=fps,
            cluster_labels=cluster_labels,
            plot_type="plotly",
            label_data=data,
            label_columns=label_columns,
            smiles_column=smiles_column,
            umap_kwargs=umap_kwargs_app,
        )

    return (
        pp_fig,
        fig,
        {"train": train_fig, "valid": valid_fig, "test": test_fig},
        fig_functional_groups,
    )


def predefined_split_by_molecule(
    data: pd.DataFrame,
    holdout_molecules: List[str],
    molecule_column: str,
    holdout_split: str = "test",
) -> Dict[str, np.ndarray]:
    data = data.reset_index(drop=True)
    test_idx = data[data[molecule_column].isin(holdout_molecules)].index.to_numpy()
    train_idx = data[~data[molecule_column].isin(holdout_molecules)].index.to_numpy()
    return {"train_idx": train_idx, holdout_split + "_idx": test_idx}


def combine_splits(
    base_split: Dict[str, np.ndarray], other_split: Dict[str, np.ndarray]
):
    for split_name, split_idx in other_split.items():
        if split_idx.shape[0] > 0:
            base_split[split_name] = split_idx
    return base_split


def pcsaft_parameter_random_design(n_samples: int):
    # Generate quasi random deisgn
    space = {
        "m": {"min": 0.5, "max": 5.0, "scaling": "linear", "type": float},
        "sigma": {"min": 0.5, "max": 5.0, "scaling": "linear", "type": float},
        "epsilon_k": {"min": 100.0, "max": 1000.0, "scaling": "linear", "type": float},
        "mu": {"min": 0, "max": 3.0, "scaling": "linear", "type": float},
        "epsilonAB": {"min": 100.0, "max": 4000.0, "scaling": "linear", "type": float},
        "KAB": {"min": 0.0001, "max": 4.0, "scaling": "log", "type": float},
    }
    # Need to make space balancd betwween associating and non-associating
    search = generate_search(space, num_trials=n_samples // 2)
    search_df = pd.DataFrame(search)
    search_df_non_associating = search_df.copy()
    search_df_non_associating["epsilonAB"] = 0.0
    search_df_non_associating["KAB"] = 0.0
    search_df = pd.concat([search_df, search_df_non_associating], axis=0).reset_index(
        drop=True
    )

    return search_df


def calculate_pcsaft_predictions(parameter_df: pd.DataFrame):
    logger = logging.getLogger(__name__)
    # Create FeOs PCP-SAFT parameters
    feos_parameters = []
    parameter_df.reset_index(drop=True, inplace=True)
    mw = (
        100
    )  # Molecular weight only used for mass density calculation, so can be constant
    for _, row in parameter_df.iterrows():
        identifier = pcsaft.Identifier()
        psr = pcsaft.PcSaftRecord(
            m=row["m"],
            sigma=row["sigma"],
            epsilon_k=row["epsilon_k"],
            mu=row["mu"],
            kappa_ab=row["KAB"],
            epsilon_k_ab=row["epsilonAB"],
        )

        record = pcsaft.PureRecord(identifier, molarweight=mw, model_record=psr)
        parameters = pcsaft.PcSaftParameters.new_pure(record)
        feos_parameters.append(parameters)

    # Predict critical points using PCP-SAFT
    critical_points = []
    phase_diagram_dfs = []
    for i, parameters in tqdm(
        enumerate(feos_parameters),
        total=len(feos_parameters),
        desc="Making PC-SAFT predictions",
    ):
        eos = EquationOfState.pcsaft(parameters)
        try:
            critical_point = State.critical_point(eos)
            critical_temperature = critical_point.temperature / si.KELVIN
            critical_pressure = critical_point.pressure() / si.PASCAL / 1e3  # kPa
            if critical_pressure < 0:
                raise RuntimeError("Negative critical pressure")
            min_temperature = 0.5 * critical_temperature
            phase_diagram_df = predict_phase_diagram(
                parameters,
                min_temperature=min_temperature,
                critical_temperature=critical_temperature,
                n_points=100,
            )
            if (phase_diagram_df["pressure"] < 0).any():
                raise RuntimeError("Negative pressure")
            # Convert densities to mol/cm^3
            phase_diagram_df["density liquid"] = (
                phase_diagram_df["density liquid"] / 1e3
            )
            phase_diagram_df["density vapor"] = phase_diagram_df["density vapor"] / 1e3
            phase_diagram_df["pressure"] /= 1e3  # kPa
            phase_diagram_df = phase_diagram_df.rename(
                columns={
                    "pressure": "P",
                    "temperature": "T",
                    "density liquid": "rho_l",
                    "density vapor": "rho_v",
                }
            )
            cols = ["T", "P", "rho_l", "rho_v"]
            phase_diagram_df = phase_diagram_df[cols]
            for col in parameter_df.columns:
                phase_diagram_df[col] = parameter_df.iloc[i][col]
            phase_diagram_df["inverse_T"] = 1 / phase_diagram_df["T"]
            phase_diagram_df["Tr"] = phase_diagram_df["T"] / critical_temperature
            phase_diagram_df["inverse_Tr"] = 1 / phase_diagram_df["Tr"]
            phase_diagram_df["log_Tr"] = np.log(phase_diagram_df["Tr"])
            phase_diagram_df["Pr"] = phase_diagram_df["P"] / critical_pressure
            phase_diagram_df["log_Pr"] = np.log(phase_diagram_df["Pr"])
            phase_diagram_df["rho_l_s"] = (
                phase_diagram_df["rho_l"] * phase_diagram_df["m"]
            )
            phase_diagram_df["rho_v_s"] = (
                phase_diagram_df["rho_v"] * phase_diagram_df["m"]
            )
            phase_diagram_dfs.append(phase_diagram_df)
        except RuntimeError:
            critical_temperature = None
            critical_pressure = None
        critical_points.append([critical_temperature, critical_pressure])
    crit_df = pd.DataFrame(critical_points, columns=["T_crit", "P_crit"])
    crit_df = pd.concat([parameter_df, crit_df], axis=1).reset_index(drop=True)
    phase_diagram_df_big = pd.concat(phase_diagram_dfs, axis=0).reset_index(drop=True)

    # Clip values
    crit_df["T_crit"] = crit_df["T_crit"].clip(0, 1e4)
    crit_df["P_crit"] = crit_df["P_crit"].clip(-1e5, 1e5)

    # # Set negative pressures as non-converged
    # neg_pressure = crit_df["P_crit"] < 0
    # crit_df.loc[neg_pressure, "P_crit"] = None
    # crit_df.loc[neg_pressure, "T_crit"] = None

    # Identify failed calculations
    crit_df["failed"] = False
    failed = crit_df["P_crit"].isna()
    crit_df.loc[failed, "failed"] = True
    # Log the number of failed calculations
    logger.info(
        f"Number of failed calculations: {failed.sum()}/{len(failed)} ({failed.sum()/len(failed)*100:.2f}%))"
    )
    phase_diagram_df_big["failed"] = False

    # Create log targets
    crit_df["log_T_crit"] = np.log(crit_df["T_crit"])
    crit_df["log_P_crit"] = np.log(crit_df["P_crit"])

    # Make pairplot
    logger.info("Making pairplot")
    g = sns.pairplot(crit_df, hue="failed", corner=True, palette="flare")

    return crit_df, phase_diagram_df_big, g.figure


def create_pcsaft_emulator_modules(
    args: PcSaftEmulatorTrainArgs,
    data: pd.DataFrame,
    split_idx: Dict[str, Callable[[], np.ndarray]],
):
    # train_df = critical_point_data.fillna(0.0)
    model = PcSaftEmulator(args)
    lit_model = PcSaftEmulatorLightningModule(model=model, args=args)
    emul_datamodule = PcSaftEmulatorDataModule(
        args=args,
        data=data,
        split_idx=(
            split_idx["train_idx"](),
            split_idx["valid_idx"](),
            split_idx["test_idx"](),
        ),
    )
    validator = PcSaftEmulatorValidator(
        args=args,
        lit_model=lit_model,
        datamodule=emul_datamodule,
    )
    return lit_model, emul_datamodule, validator


def create_spk_modules(
    args: SPKTrainArgs,
    molecules: Dict[str, Callable[[], Atoms]],
    target_property_data: pd.DataFrame,
    split_idx: Dict[str, Callable[[], np.ndarray]],
) -> Tuple[LightningModule, SPKDataModule, LightningValidator]:
    """Create model and datamodule for SchnetPack"""
    # Setup tasks
    (
        task_heads,
        task_outputs,
        pre_normalization_modules,
        post_normalization_modules,
    ) = setup_spk_tasks(args)

    # Create lightning datamodule
    lit_model = setup_spk_model(
        args, task_heads, task_outputs, post_normalization_modules
    )

    # Setup data
    spk_datamodule = setup_spk_data(
        args,
        molecules=molecules,
        target_property_data=target_property_data,
        split_idx=(
            split_idx["train_idx"](),
            split_idx["valid_idx"](),
            split_idx["test_idx"](),
        ),
        normalization_modules=pre_normalization_modules,
    )

    # Validator
    validator = SPKValidator(args, lit_model, spk_datamodule)

    return lit_model, spk_datamodule, validator


def create_spk_modules_qm9(
    args: SPKTrainArgs,
    extra_data: Optional[
        List[Tuple[pd.DataFrame, Dict[str, Callable[[], ase.Atoms]]]]
    ] = None,
) -> Tuple[LightningModule, Union[SPKDataModule, CustomQM9], LightningValidator]:
    """Create model and datamodule for SchnetPack"""
    # Setup tasks
    (
        task_heads,
        task_outputs,
        pre_normalization_modules,
        post_normalization_modules,
    ) = setup_spk_tasks(args)

    # Create lightning datamodule
    lit_model = setup_spk_model(
        args, task_heads, task_outputs, post_normalization_modules
    )

    # Setup data
    spk_datamodule = setup_qm9_data(args, extra_data=extra_data)

    # Validator
    validator = SPKValidator(args, lit_model, spk_datamodule)

    return lit_model, spk_datamodule, validator


def format_extra_data(
    sepp_data: pd.DataFrame,
    sepp_molecules: Dict[str, Callable[[], ase.Atoms]],
    experimental_data: pd.DataFrame,
    experimental_molecules: Dict[str, Callable[[], ase.Atoms]],
):
    return [
        (sepp_data, sepp_molecules),
        (experimental_data, experimental_molecules),
    ]


def create_chemprop_modules(
    args: ChempropTrainArgs,
    data: pd.DataFrame,
    split_idx: Dict[str, Callable[[], np.ndarray]],
) -> Tuple[LightningModule, LightningDataModule, LightningValidator]:
    datamodule = ChempropDataModule(
        args=args,
        data=data,
        split_idx=(
            split_idx["train_idx"](),
            split_idx["valid_idx"](),
            split_idx["test_idx"](),
        ),
    )
    model = RegressionMoleculeModel(args)
    lit_model = ChempropLightningModule(args=args, model=model)
    validator = ChempropValidator(args=args, lit_model=lit_model, datamodule=datamodule)
    return lit_model, datamodule, validator


def create_chemprop_cv_modules(
    args: ChempropTrainArgs,
    data: pd.DataFrame,
    cv_split_idx: Dict[str, Callable[[], np.ndarray]],
) -> List[Tuple[LightningModule, LightningDataModule, LightningValidator]]:
    logger = logging.getLogger(__name__)

    # Get CV splits
    splits = setup_cv(cv_split_idx)

    # Setup cross validation
    split_items = []
    for cv_splits in splits.values():
        datamodule = ChempropDataModule(
            args=args,
            data=data,
            split_idx=(
                cv_splits["train_idx"](),
                cv_splits["valid_idx"](),
                cv_splits["test_idx"](),
            ),
        )
        model = RegressionMoleculeModel(args)
        lit_model = ChempropLightningModule(args=args, model=model)
        validator = ChempropValidator(
            args=args, lit_model=lit_model, datamodule=datamodule
        )
        split_items.append((lit_model, datamodule, validator))
    return split_items


def create_pyg_modules(
    args: PyGTrainArgs,
    data: pd.DataFrame,
    split_idx: Dict[str, Callable[[], np.ndarray]],
) -> Tuple[LightningModule, LightningDataModule, LightningValidator]:
    """Creates a PyG LightningModule and LightningDataModule."""
    datamodule = PyGDataModule(
        args=args,
        data=data,
        split_idx=(
            split_idx["train_idx"](),
            split_idx["valid_idx"](),
            split_idx["test_idx"](),
        ),
    )

    model = MPNN(args)
    lit_model = PyGLightningModule(args=args, model=model)
    validator = PyGValidator(args=args, lit_model=lit_model, datamodule=datamodule)

    return lit_model, datamodule, validator


def create_pyg_cv_modules(
    args: PyGTrainArgs,
    data: pd.DataFrame,
    cv_split_idx: Dict[str, Callable[[], np.ndarray]],
) -> List[Tuple[LightningModule, LightningDataModule, LightningValidator]]:
    logger = logging.getLogger(__name__)

    # Get CV splits
    splits = setup_cv(cv_split_idx)

    # Setup cross validation
    split_items = []
    for i, cv_splits in splits.items():
        logger.info(f"Running split {i+1}/{len(splits)}")
        datamodule = PyGDataModule(
            args=args,
            data=data,
            split_idx=(
                cv_splits["train_idx"](),
                cv_splits["valid_idx"](),
                cv_splits["test_idx"](),
            ),
        )

        model = MPNN(args)
        lit_model = PyGLightningModule(args=args, model=model)
        validator = PyGValidator(args=args, lit_model=lit_model, datamodule=datamodule)
        split_items.append((lit_model, datamodule, validator))
    return split_items


def create_pyg_e2e_modules(
    args: PyGTrainArgs,
    data: pd.DataFrame,
    phase_equilibria_emulator_wandb_run_id: str,
    critical_point_emulator_wandb_run_id: str,
    wandb_project: str,
    wandb_entity: str,
    split_idx: Dict[str, Callable[[], np.ndarray]],
) -> Tuple[LightningModule, LightningDataModule, LightningValidator]:
    """Creates a PyG LightningModule and LightningDataModule."""
    datamodule = PyGDataModule(
        args=args,
        data=data,
        split_idx=(
            split_idx["train_idx"](),
            split_idx["valid_idx"](),
            split_idx["test_idx"](),
        ),
    )

    # Download pretrained emulators
    api = wandb.Api()
    emulators: List[PcSaftEmulatorLightningModule] = []
    for wandb_run_id in [
        phase_equilibria_emulator_wandb_run_id,
        critical_point_emulator_wandb_run_id,
    ]:
        run = api.run(f"{wandb_entity}/{wandb_project}/{wandb_run_id}")
        emulator_args = PcSaftEmulatorTrainArgs(**(dict(run.config)))
        path, _ = download_best_model(run=run)
        lit_emulator = PcSaftEmulatorLightningModule(
            args=emulator_args, model=PcSaftEmulator(emulator_args)
        )
        load_lightningmodule_from_checkpoint(str(path), lit_emulator)
        emulators.append(lit_emulator)

    # Setup e2e lightning module
    lit_model = E2ELightningModule(
        args=args,
        phase_equilibria_emulator=emulators[0],
        critical_point_emulator=emulators[1],
        pcsaft_emulator_input_columns=emulator_args.input_columns,  # type: ignore
        pcsaft_emulator_output_columns=emulator_args.target_columns,  # type: ignore
    )

    # Validator
    validator = PyGValidator(args=args, lit_model=lit_model, datamodule=datamodule)

    return lit_model, datamodule, validator


def create_ffn_modules(
    args: FFNTrainArgs,
    fps: np.ndarray,
    data: pd.DataFrame,
    split_idx: Dict[str, Callable[[], np.ndarray]],
) -> Tuple[LightningModule, LightningDataModule, LightningValidator]:
    """Creates a FFN LightningModule and LightningDataModule."""
    datamodule = FFNDataModule(
        args=args,
        fps=fps,
        data=data,
        split_idx=(
            split_idx["train_idx"](),
            split_idx["valid_idx"](),
            split_idx["test_idx"](),
        ),
    )

    lit_model = FFNLightningModule(args=args, model=FFN(args))
    validator = FFNValidator(args=args, lit_model=lit_model, datamodule=datamodule)
    return lit_model, datamodule, validator


def create_ffn_modules_cv(
    args: FFNTrainArgs,
    fps: np.ndarray,
    data: pd.DataFrame,
    cv_split_idx: Dict[str, Callable[[], np.ndarray]],
) -> List[Tuple[LightningModule, LightningDataModule, LightningValidator]]:
    """Creates a FFN LightningModule and LightningDataModule."""
    logger = logging.getLogger(__name__)

    # Get CV splits
    splits = setup_cv(cv_split_idx)

    # Setup cross validation
    split_items = []
    for i, cv_splits in splits.items():
        logger.info(f"Running split {i+1}/{len(splits)}")
        datamodule = FFNDataModule(
            args=args,
            fps=fps,
            data=data,
            split_idx=(
                cv_splits["train_idx"](),
                cv_splits["valid_idx"](),
                cv_splits["test_idx"](),
            ),
        )
        lit_model = FFNLightningModule(args=args, model=FFN(args))
        validator = FFNValidator(args=args, lit_model=lit_model, datamodule=datamodule)
        split_items.append((lit_model, datamodule, validator))
    return split_items


def train_validate_pytorch_lighting_cv(
    args: TrainArgs,
    split_items: List[Tuple[LightningModule, LightningDataModule, LightningValidator]],
) -> List[str]:
    """Train and validate a pytorch lightning model"""
    logger = logging.getLogger(__name__)
    run_ids = []
    # Group ID for cross validation
    group_id = add_group_id(args)
    for i, (lit_model, datamodule, validator) in enumerate(split_items):
        logger.info(f"Running split {i+1}/{len(split_items)}")
        wandb_run_id = train_pytorch_lightning(args, lit_model, datamodule)
        run_ids.append(wandb_run_id)
        validate_pytorch_lightning(validator, wandb_run_id)
    return run_ids


def train_pytorch_lightning(
    args: TrainArgs,
    lit_model: LightningModule,
    datamodule: LightningDataModule,
) -> str:
    """Train and validate a pytorch lightning model

    Arguments
    ----------
    args: CustomTrainArgs
        Arguments for training
    lit_model: pl.LightningModule
        Pytorch lightning module
    data: pl.LightningDataModule
        Pytorch lightning data module
    wandb_checkpoint_artifact_id : Optional[str]
        Wandb artifact id for checkpoint if using a pretrained model

    Returns
    -------
    Wandb run id

    """
    # Loggers and Callbacks
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    wandb_kwargs = args.wandb_kwargs if args.wandb_kwargs else {}
    wandb_logger = loggers.WandbLogger(  # type: ignore
        entity=args.wandb_entity,
        project=args.wandb_project,
        tags=args.wandb_tags,
        log_model="all" if args.log_all_models else True,
        **wandb_kwargs,
    )
    wandb_logger.watch(lit_model.model)  # type: ignore
    if args.wandb_checkpoint_artifact_id:
        logger.info("Loading checkpoint from wandb artifact")
        artifact = wandb_logger.experiment.use_artifact(
            args.wandb_checkpoint_artifact_id
        )
        ckpt_path = str(Path(artifact.download()) / args.checkpoint_name)
        loaded_state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
        loaded_model_state_dict = loaded_state_dict["state_dict"]
        model_state_dict = lit_model.state_dict()
        for param_name, param_value in loaded_model_state_dict.items():
            check_1 = param_name in model_state_dict
            check_2 = (
                check_1 and model_state_dict[param_name].shape == param_value.shape
            )
            if check_1 and check_2:
                model_state_dict[param_name] = param_value
            elif check_1 and not check_2:
                logger.info(
                    "Skipping looading parameter %s because shapes do not match",
                    param_name,
                )
        lit_model.load_state_dict(model_state_dict)
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=args.save_dir,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=False,
        every_n_epochs=args.epochs // 10,
    )
    lr_monitor_cb = callbacks.LearningRateMonitor(logging_interval="step")
    callbacks_list = [checkpoint_callback, lr_monitor_cb]
    if args.early_stopping:
        early_stop_callback = callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min"
        )
        callbacks_list.append(early_stop_callback)

    # Train
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=callbacks_list,
        accelerator=accelerator,
        log_every_n_steps=5,
        auto_lr_find=args.auto_lr_find,
    )

    if args.auto_lr_find:
        trainer.tune(lit_model, datamodule)

    try:
        trainer.fit(
            model=lit_model,
            datamodule=datamodule,
        )
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt: Stopping training and uploading checkpoints")
        wandb_logger._scan_and_log_checkpoints(wandb_logger._checkpoint_callback)  # type: ignore

    wandb.finish(0)
    return wandb_logger.experiment.id


def validate_pytorch_lightning(
    validator: LightningValidator,
    wandb_run_id: str,
    update_wandb: bool = True,
    new_wandb_run: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Validate a pytorch lightning model"""
    return validator(
        wandb_run_id=wandb_run_id,
        update_wandb=update_wandb,
        new_wandb_run=new_wandb_run,
    )


def cross_validate_sklearn(
    args: SklearnTrainArgs,
    fps: np.ndarray,
    data: pd.DataFrame,
    cv_split_idx: Dict[str, Callable[[], np.ndarray]],
    target_display_names: Union[None, dict] = None,
    estimator: Optional[Estimator] = None,
) -> str:
    logger = logging.getLogger(__name__)

    # Reformat into dict of dicts
    splits = {}
    n_folds = 0
    for split_name, split_idx in cv_split_idx.items():
        if split_name not in ["train_idx", "test_idx"]:
            fold_number = int(split_name.split("_")[-1])
            split = split_name.split("_")[0]
            if isinstance(splits.get(fold_number), dict):
                splits[fold_number][f"{split}_idx"] = split_idx
            else:
                splits[fold_number] = {f"{split}_idx": split_idx}
            if fold_number + 1 > n_folds:
                n_folds = fold_number + 1
    test_idx = cv_split_idx["test_idx"]
    for i in range(n_folds):
        splits[i]["test_idx"] = test_idx

    # Add wandb group
    group_id = id_generator()
    if args.wandb_kwargs is None:
        args.wandb_kwargs = {"group": f"cv_{group_id}"}
    elif "group" not in args.wandb_kwargs:
        args.wandb_kwargs["group"] = f"cv_{group_id}"
    logger.info(f"Wandb group id: {group_id}")

    # Run cross validation
    for i, cv_splits in splits.items():
        logger.info(f"Running split {i+1}/{len(splits)}")
        train_validate_sklearn(
            args,
            fps,
            data,
            cv_splits,
            target_display_names=target_display_names,
            estimator=estimator,
        )
    return group_id


def train_validate_sklearn(
    args: SklearnTrainArgs,
    fps: np.ndarray,
    data: pd.DataFrame,
    split_idx: Dict[str, Callable[[], np.ndarray]],
    target_display_names: Union[None, dict] = None,
    estimator: Optional[Estimator] = None,
) -> str:
    logger = logging.getLogger(__name__)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    predict_dir = save_dir / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Wandb
    run: Run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        tags=args.wandb_tags,
        config=asdict(args),
        **(args.wandb_kwargs if args.wandb_kwargs else {}),
    )  # type: ignore

    # Indices
    train_idx = split_idx["train_idx"]()
    val_idx = split_idx["valid_idx"]()
    test_idx = split_idx["test_idx"]()

    # Split fps
    train_fps = fps[train_idx]
    val_fps = fps[val_idx]
    test_fps = fps[test_idx]

    # Split targets
    train_targets = data[args.target_columns].iloc[train_idx].to_numpy()
    val_targets = data[args.target_columns].iloc[val_idx].to_numpy()
    test_targets = data[args.target_columns].iloc[test_idx].to_numpy()

    # Create and fit model
    if estimator is not None:
        model = estimator
    else:
        model = TransformedTargetRegressor(
            regressor=RandomForestRegressor(n_estimators=args.num_trees),
            transformer=StandardScaler(),
        )
    logger.info("Fitting model")
    model.fit(train_fps, train_targets)

    # Get predictions
    train_pred: np.ndarray = model.predict(train_fps)
    val_pred: np.ndarray = model.predict(val_fps)
    test_pred: np.ndarray = model.predict(test_fps)

    def filter_non_associating(preds, smiles, associating_col_idx):
        association = np.array([1 if is_associating(smi) else 0 for smi in smiles])[
            :, None
        ]
        idx = associating_col_idx
        preds[:, idx] = association * preds[:, idx]
        return preds

    # SMILES
    smiles_col = args.smiles_columns[0]
    train_smiles = data[smiles_col].iloc[train_idx].to_numpy()
    val_smiles = data[smiles_col].iloc[val_idx].to_numpy()
    test_smiles = data[smiles_col].iloc[test_idx].to_numpy()

    # Filter non-associating
    if args.filter_non_associating and args.associating_columns:
        idx = [args.target_columns.index(col) for col in args.associating_columns]
        train_pred = filter_non_associating(train_pred, train_smiles, idx)
        val_pred = filter_non_associating(val_pred, val_smiles, idx)
        test_pred = filter_non_associating(test_pred, test_smiles, idx)

    # Save predictions
    for split, ground_truth, preds, smiles in zip(
        ["train", "val", "test"],
        [
            train_targets,
            val_targets,
            test_targets,
        ],
        [train_pred, val_pred, test_pred],
        [train_smiles, val_smiles, test_smiles],
    ):
        # Create ground truth dataframe
        ground_truth_target = [
            ground_truth[:, i] for i, target in enumerate(args.target_columns)
        ]
        ground_truth_target = np.column_stack(ground_truth_target)
        df = pd.DataFrame(ground_truth_target, columns=args.target_columns)

        # Create predictions dataframe
        preds_target = [preds[:, i] for i, target in enumerate(args.target_columns)]
        preds_target = np.column_stack(preds_target)
        df_preds = pd.DataFrame(
            preds_target,
            columns=[f"{col}_pred" for col in args.target_columns],
        )

        # Concatenate and save
        df = pd.concat([df, df_preds], axis=1)
        df[args.smiles_columns[0]] = smiles
        df.to_csv(predict_dir / f"{split}_predictions.csv", index=False)
    # Log predictions
    artifact = wandb.Artifact(args.wandb_artifact_name, type="dataset")
    artifact.add_dir(str(predict_dir))  # type: ignore
    run.log_artifact(artifact)

    # Make parity plot and upload scores to wandb
    logger.info("Plotting parity plots and calculating scores")
    n_rows = len(args.target_columns) // 2 + len(args.target_columns) % 2
    fig = plt.figure(figsize=(10, 5 * n_rows))
    fig.subplots_adjust(wspace=0.3)
    fig_test = plt.figure(figsize=(10, 5 * n_rows))
    fig_test.subplots_adjust(wspace=0.3)
    target_display_names = (
        target_display_names if target_display_names is not None else {}
    )
    scores = {"train": {}, "val": {}, "test": {}}
    for i, target in enumerate(args.target_columns):
        ax = fig.add_subplot(n_rows, 2, i + 1)  # type: ignore
        ax_test = fig_test.add_subplot(n_rows, 2, i + 1)  # type: ignore
        for split, ground_truth, preds in zip(
            ["train", "val", "test"],
            [train_targets, val_targets, test_targets],
            [train_pred, val_pred, test_pred],
        ):
            ground_truth_target = ground_truth[:, i]
            preds_target = preds[:, i]
            # Calculate scores
            current_scores = calculate_metrics(
                ground_truth_target, preds_target, scores=["mae", "r2"]
            )
            if split in ["train", "val"]:
                parity_plot(
                    ground_truth_target,
                    preds_target,
                    ax=ax,
                    label=split,
                    scores=current_scores,
                )
            else:
                parity_plot(
                    ground_truth_target,
                    preds_target,
                    ax=ax_test,
                    label=split,
                    scores=current_scores,
                )
            rmse = calculate_metrics(
                ground_truth_target, preds_target, scores=["rmse", "mse", "mape"]
            )
            current_scores.update(rmse)
            scores[split][target] = current_scores
            for score_name, score in current_scores.items():
                run.summary[f"{split}_{target}_{score_name}"] = score
            ax.set_title(target_display_names.get(target, target))

    # Log parity plot
    plot_path = save_dir / "parity.png"
    fig.savefig(str(plot_path), dpi=300)
    run.log({"parity_plot": wandb.Image(str(plot_path))})
    plot_test_path = save_dir / "parity_test.png"
    fig_test.savefig(str(plot_test_path), dpi=300)
    run.log({"test_parity_plot": wandb.Image(str(plot_test_path))})

    # Save model to wandb
    dump(model, save_dir / "model.pkl")
    artifact = wandb.Artifact("rf", type="model")
    artifact.add_file(str(save_dir / "model.pkl"))
    run.log_artifact(artifact)

    # Finish wandb run
    wandb.finish(0)  # type: ignore
    return run.id


def train_validate_lolopy(
    args: SklearnTrainArgs,
    fps: np.ndarray,
    data: pd.DataFrame,
    split_idx: Dict[str, Callable[[], np.ndarray]],
    target_display_names: Union[None, dict] = None,
):
    """
    Train a LOLOpy model that also can predict uncertainties
    """
    from lolopy.learners import RandomForestRegressor as LoloRandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor

    # Setup model
    # Use multioutput regressor because LOLOpy does not support multioutput
    regressor = MultiOutputRegressor(
        LoloRandomForestRegressor(
            num_trees=args.num_trees, uncertainty_calibration=True
        )
    )
    model = TransformedTargetRegressor(
        regressor=regressor, transformer=StandardScaler()
    )

    # Train model
    return train_validate_sklearn(
        args,
        fps,
        data,
        split_idx,
        target_display_names,
        estimator=model,
        predict_std=True,  # type: ignore
    )
