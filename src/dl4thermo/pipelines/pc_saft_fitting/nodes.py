"""
This is a boilerplate pipeline 'pc_saft_fitting'
generated using Kedro 0.17.7
"""
import json
import logging
from datetime import datetime as dt
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import ase
import numpy as np
import pandas as pd
import ray
import schnetpack.transform as trn
import torch
from lmfit import Parameters
from lmfit.minimizer import MinimizerResult
from tqdm import tqdm, trange

import wandb
from dl4thermo.extras.utils.data_transform import jsonify_dict
from dl4thermo.extras.utils.pcsaft import is_associating, pcsaft_lmfit_regression
from dl4thermo.extras.utils.sensitivity import pcsaft_sensitivity_analysis
from dl4thermo.pipelines.train_models.models.spk import (
    ASEAtomsData,
    AtomsLoader,
    SPKTrainArgs,
    _atoms_collate_fn,
    setup_spk_model,
    setup_spk_tasks,
)

logger = logging.getLogger(__name__)


def sensitivity_analysis(
    pcsaft_data: pd.DataFrame,
    pcsaft_data_smiles_column: str,
    experimental_data,
    experimental_data_smiles_column: str,
    experimental_data_density_column: str = "DEN",
    n_samples: int = 100,
    batch_size: int = 100,
):
    """Perform sensitivity analysis on the PC-SAFT model using ray for parallelization

    Parameters
    ----------
    pcsaft_data : pd.DataFrame
        Dataframe containing the PC-SAFT parameters for each molecule
    pcsaft_data_smiles_column : str
        Column name for the SMILES string in the PC-SAFT dataframe
    experimental_data : pd.DataFrame
        Dataframe containing the experimental data for each molecule
    experimental_data_smiles_column : str
        Column name for the SMILES string in the experimental dataframe
    experimental_data_density_column : str, optional
        Column name for the density in the experimental dataframe, by default "DEN"
    n_samples : int, optional
        Number of samples to use for the sensitivity analysis, by default 100
    batch_size : int, optional
        Number of molecules to process in each batch, by default 100

    Returns
    -------

    """
    logger = logging.getLogger(__name__)

    if not ray.is_initialized():  # type: ignore
        ray.init()  # type: ignore

    total_start = dt.now()

    @ray.remote  # type: ignore
    def _get_sensitivity_indices(
        smiles: str,
        molecule_data: pd.DataFrame,
        associating: bool,
        initial_parameters: Dict[str, float],
    ) -> Tuple[Union[dict, None], float]:
        start = dt.now()
        try:
            indices = pcsaft_sensitivity_analysis(
                smiles,
                molecule_data,
                associating=associating,
                initial_parameters=initial_parameters,
                n_samples=n_samples,
                experimental_data_density_column=experimental_data_density_column,
            )
        except TypeError:
            indices = None
        end = dt.now()
        return indices, (end - start).total_seconds()

    # Submit jobs to cluster
    results_ray = {}
    smiles_list = pcsaft_data[pcsaft_data_smiles_column].unique()
    n_tasks = 0
    skipped_list = []
    for smiles in tqdm(smiles_list):
        # Get data
        initial_params = (
            pcsaft_data[pcsaft_data[pcsaft_data_smiles_column] == smiles]
            .iloc[0]
            .to_dict()
        )
        molecule_data = experimental_data[
            experimental_data[experimental_data_smiles_column] == smiles
        ]

        # Only include vapor pressure and reasonable density data
        molecule_data = molecule_data[
            (molecule_data["DEN"].isna()) | (molecule_data["DEN"] < 2.5e3)
        ]

        # Check there is enough data
        check_1 = molecule_data[molecule_data["DEN"].isna()].shape[0] < 10
        check_2 = molecule_data[molecule_data["DEN"].notnull()].shape[0] < 10
        if check_1 or check_2:
            skipped_list.append(smiles)
            continue

        # Only liquid density (no vapor)
        rho_data = molecule_data[
            molecule_data[experimental_data_density_column].notnull()
        ]
        phase_transition_rho = rho_data.iloc[rho_data["T"].argmax()][
            experimental_data_density_column
        ]
        rho_data_liquid = rho_data[
            rho_data[experimental_data_density_column] > phase_transition_rho
        ]
        check_rho_liquid = len(rho_data_liquid) < 2
        if check_rho_liquid:
            skipped_list.append(smiles)
            continue

        # Check if association is predicted
        associating = initial_params["KAB"] > 0

        # Submit job
        out_ref = _get_sensitivity_indices.remote(
            smiles=smiles,  # type: ignore
            molecule_data=molecule_data,
            associating=associating,
            initial_parameters=initial_params,
        )
        result = {"smiles": smiles}
        results_ray.update({out_ref: result})
        n_tasks += 1

    # Get results in batches
    n_batches = n_tasks // batch_size
    n_batches += 1 if n_tasks % batch_size != 0 else 0
    object_refs = list(results_ray.keys())
    # output_dir = Path(save_dir)
    # output_dir.mkdir(exist_ok=True)
    batches = {"pvap": [], "rho": []}
    for batch in trange(n_batches):
        # Select correct refs
        refs = object_refs[batch * batch_size : (batch + 1) * batch_size]

        # Wait for results
        ready_refs, _ = ray.wait(refs, num_returns=len(refs))  # type: ignore

        # Get results
        results_ray_list = []
        for ready_ref in ready_refs:
            # Retrieve existing results
            result = results_ray[ready_ref]

            # Get sensitivity result
            results_dict, elapsed = ray.get(ready_ref, timeout=4)  # type: ignore
            if results_dict is None:
                skipped_list.append(result["smiles"])
            else:
                for name, df in results_dict.items():
                    df = df[["S1"]].T
                    df["smiles"] = result["smiles"]
                    result.update({"time": elapsed, f"{name}_S1": df})

                # Append to results
                results_ray_list.append(result)

        for name in ["pvap", "rho"]:
            first_order_sensitivity_df = pd.concat(
                [r[f"{name}_S1"] for r in results_ray_list], axis=0
            )
            batches[name].append(first_order_sensitivity_df)

    total_end = dt.now()
    elapsed = total_end - total_start
    logger.info(f"Fitting took {elapsed.total_seconds()/3600} hours in total.")
    return {name: pd.concat(b) for name, b in batches.items()}


def make_dipole_moment_predictions(
    wandb_run_id: str,
    wandb_entity: str,
    wandb_project: str,
    data: pd.DataFrame,
    smiles_column: str,
    molecules: Dict[str, Callable[[], ase.Atoms]],
    molecule_lookup_column_name: str,
    dipole_moment_column: str = "dipole_moment",
    datapath: str = "data/07_model_output/",
    db_name: str = "dortmund.db",
    batch_size: int = 100,
):
    """Make predictions of dipole moment using a trained PaiNN model from SchnetPack."""
    # Download best model from run
    logger.info("Downloading model from wandb")
    api = wandb.Api()  # type: ignore
    run = api.run(f"{wandb_entity}/{wandb_project}/{wandb_run_id}")
    artifacts = run.logged_artifacts()  # type: ignore
    ckpt_path = None
    for artifact in artifacts:
        if artifact.type == "model" and "best_k" in artifact.aliases:
            ckpt_path = artifact.download()
    if ckpt_path is None:
        raise ValueError("No best checkpoint found")
    ckpt_path = Path(ckpt_path) / "model.ckpt"

    # Load arguments
    logger.info("Loading arguments")
    args = SPKTrainArgs(**run.config["train_args"])
    args.batch_size = batch_size
    args.num_workers = 8

    # Setup model
    logger.info("Preparing SPK model")
    (
        task_heads,
        task_outputs,
        _,
        post_normalization_modules,
    ) = setup_spk_tasks(args)
    lit_model = setup_spk_model(
        args, task_heads, task_outputs, post_normalization_modules
    )

    # Setup data
    logger.info("Preparing SPK dataset")
    property_unit_dict = {p: "None" for p in args.target_columns}
    db_path = Path(datapath) / db_name
    if db_path.exists():
        db_path.unlink()
    dataset = ASEAtomsData.create(
        datapath=str(db_path),
        distance_unit="Ang",
        property_unit_dict=property_unit_dict,
    )
    unique_data = data.drop_duplicates(subset=[smiles_column])
    property_list = [{p: np.array([1.0]) for p in args.target_columns}] * len(
        unique_data
    )
    atoms_list = [
        molecules[str(id)]()
        for id in tqdm(
            unique_data[molecule_lookup_column_name], desc="Loading conformers"
        )
    ]
    dataset.add_systems(property_list=property_list, atoms_list=atoms_list)
    transforms: List[trn.Transform] = [trn.ASENeighborList(cutoff=5.0), trn.CastTo32()]
    dataset.transforms = transforms
    dl = AtomsLoader(
        dataset,  # type: ignore
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=_atoms_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Load weights
    logger.info("Loading weights")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_state_dict = torch.load(ckpt_path, map_location=device)
    model_state_dict = lit_model.state_dict()
    for param_name, param_value in loaded_state_dict.items():
        check_1 = param_name in model_state_dict
        check_2 = check_1 and model_state_dict[param_name].shape == param_value.shape
        if check_1 and check_2:
            model_state_dict[param_name] = param_value
        elif check_1 and not check_2:
            logger.info(
                f"Skipping loading parameter {param_name} because shapes do not match",
            )

    # Make predictions
    logger.info("Making dipole moment predictions")
    lit_model.eval()
    mu_preds = []
    for batch in tqdm(dl, total=len(dl), desc="Batches"):
        with torch.no_grad():
            batch = lit_model(batch)
        mu_pred = batch[dipole_moment_column]
        mu_preds.append(mu_pred)
    mu_pred_all = torch.cat(mu_preds)
    if not len(mu_pred_all) == len(unique_data):
        raise ValueError(
            f"Number of predictions ({len(mu_pred_all)}) does not match number of "
            f"unique data points ({len(unique_data)})"
        )
    mu_pred_df = pd.DataFrame(
        {
            dipole_moment_column: mu_pred_all.detach().cpu().numpy(),
            smiles_column: unique_data[smiles_column],
        }
    )
    return mu_pred_df


def create_cosmo_dfs(
    cosmo_df: pd.DataFrame,
    density_column: str,
    smiles_column: str,
    pressure_column: str,
    temperature_column: str,
    dipole_moment_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cosmo_df[pressure_column] = cosmo_df["PVtot[bar]"] * 100  # Convert to kPa

    # Combine experimental and predicted density, preferring experimental
    # This should be g/ml
    density_data = cosmo_df["ExpDensity[g/mol]"].combine_first(
        cosmo_df["Density[g/mol]"]
    )
    density_data *= 1000  # Convert to kg/m3
    density_df = pd.DataFrame(
        {
            density_column: density_data,
            smiles_column: cosmo_df["smiles"],
            pressure_column: cosmo_df[pressure_column],
            temperature_column: cosmo_df["Temperature[K]"],
            "compoundName": cosmo_df["compoundName"],
        }
    ).dropna()

    pvap_data = cosmo_df[
        ["Temperature[K]", pressure_column, "smiles", "compoundName"]
    ].dropna()
    pvap_data = pvap_data.rename(columns={"Temperature[K]": temperature_column})
    df = pd.concat([density_df, pvap_data], axis=0, ignore_index=True)
    for col in [density_column, pressure_column, temperature_column]:
        df[col] = df[col].astype(float)

    # Get dipole moment data
    dipole_moment_data = cosmo_df[["Dipole[Debye]", "smiles"]].copy()
    dipole_moment_data = dipole_moment_data.rename(
        columns={"Dipole[Debye]": dipole_moment_column, "smiles": smiles_column}
    ).drop_duplicates()
    dipole_moment_data[dipole_moment_column] = dipole_moment_data[
        dipole_moment_column
    ].astype(float)

    # unique_smiles = df[smiles_column].unique()
    # df = df[df[smiles_column].isin(unique_smiles[:4])]
    return df, dipole_moment_data


def regress_pc_saft_pure(
    experimental_data: pd.DataFrame,
    dipole_moment_data: pd.DataFrame,
    smiles_column: str,
    density_column: str,
    pressure_column: str,
    temperature_column: str,
    dipole_moment_data_smiles_column: str,
    dipole_moment_data_column: str,
    min_num_density_data: int = 10,
    min_num_pvap_data: int = 10,
    fix_kab: bool = True,
    fix_dipole_moment: bool = True,
    fit_log_pressure: bool = True,
    id_column: Optional[str] = None,
    density_weight: float = 1.0,
    batch_size: int = 4,
    start_batch: int = 0,
):
    """Regress PC-SAFT parameters for pure components using experimental data.

    Uses ray for parallelization..

    Parameters
    ----------
    experimental_data : pd.DataFrame
        Experimental data for pure components.
    dipole_moment_data : pd.DataFrame
        Dipole moment data for pure components.
    smiles_column : str
        Name of the column containing SMILES strings.
    density_column : str
        Name of the column containing density data.
    pressure_column : str
        Name of the column containing pressure data.
    temperature_column : str
        Name of the column containing temperature data.
    dipole_moment_data_smiles_column : str
        Name of the column containing SMILES strings in the dipole moment data.
    dipole_moment_data_column : str
        Name of the column containing dipole moment data.
    min_num_density_data : int, optional
        Minimum number of density data points required to include a molecule in the
        regression, by default 10
    min_num_pvap_data : int, optional
        Minimum number of vapor pressure data points required to include a molecule in
        the regression, by default 10
    fix_kab : bool, optional
        Whether to fix the KAB association, by default True
    fix_dipole_moment : bool, optional
        Whether to fix the dipole moment parameter, by default True
    batch_size : int, optional
        Batch size for the number of regressions to run in parallel., by default 4
    start_batch : int, optional
        Batch number to start at, by default 0

    Returns
    -------
    batches: Dict[str, Dict]
        Dictionary of dictionaries containing the results of the regression.

    """
    logger = logging.getLogger(__name__)
    total_start = dt.now()
    if not ray.is_initialized():  # type: ignore
        ray.init()  # type: ignore

    @ray.remote  # type: ignore
    def _lmfit_regression(
        molecule_data: pd.DataFrame, smiles: str, lmfit_params: Parameters
    ) -> Union[MinimizerResult, None]:
        try:
            result_lm, _, _ = pcsaft_lmfit_regression(
                experimental_data=molecule_data,
                smiles=smiles,
                lmfit_params=lmfit_params,
                experimental_data_density_column=density_column,
                experimental_pressure_column=pressure_column,
                experimental_temperature_column=temperature_column,
                minimize_kwargs=dict(method="leastsq", nan_policy="omit"),
                density_weight=density_weight,
                fit_log_pressure=fit_log_pressure,
            )
            return result_lm
        except RuntimeError as e:
            logger.error(e)
        except ValueError as e:
            logger.error(e)

    # Submit jobs to cluster
    jobs_ray = {}
    smiles_list = experimental_data[smiles_column].unique()
    n_tasks = 0
    skipped_list = []
    for smiles in tqdm(smiles_list):
        # Get data
        molecule_data = experimental_data[experimental_data[smiles_column] == smiles]

        # Check there is enough data (including enough liquid density data)
        if not data_check(
            molecule_data,
            temperature_column=temperature_column,
            density_column=density_column,
            min_pvap_data=min_num_pvap_data,
            min_rho_data=min_num_density_data,
        ):
            skipped_list.append(smiles)
            continue

        # Determine if associating
        associating = is_associating(smiles)

        # Get dipole moment
        mu_data = dipole_moment_data[
            dipole_moment_data[dipole_moment_data_smiles_column] == smiles
        ][dipole_moment_data_column]
        if len(mu_data) > 0:
            mu = mu_data.iloc[0]
        else:
            skipped_list.append(smiles)
            continue

        # Create LMFit parameters
        lmfit_params = create_lmfit_parameters(
            mu,
            associating=associating,
            fix_kab=fix_kab,
            fix_dipole_moment=fix_dipole_moment,
        )

        # Submit job
        out_ref = _lmfit_regression.remote(molecule_data, smiles, lmfit_params)

        # Save metadata
        initalization = {
            "smiles": smiles,
            "associating": associating,
            "initial_parameters": lmfit_params.valuesdict(),
        }
        if id_column:
            initalization[id_column] = molecule_data[id_column].iloc[0]
        jobs_ray.update({out_ref: initalization})
        n_tasks += 1

    # Get results in batches
    n_batches = n_tasks // batch_size
    n_batches += 1 if n_tasks % batch_size != 0 else 0
    object_refs = list(jobs_ray.keys())
    batches = {}
    try:
        for batch in tqdm(range(start_batch, n_batches)):
            # Select correct refs
            refs = object_refs[batch * batch_size : (batch + 1) * batch_size]

            # Wait for results
            ready_refs, _ = ray.wait(refs, num_returns=len(refs))  # type: ignore

            # Get results
            results_ray_list = []
            for ready_ref in ready_refs:
                # Retrieve job definition
                initialization = jobs_ray[ready_ref]

                # Get regression results
                result = ray.get(ready_ref, timeout=4)  # type: ignore
                if result is None:
                    continue
                parameters_lm = result.params.valuesdict()  # type: ignore
                parameter_errors = {
                    f"{name}_error": p.stderr for name, p in result.params.items()
                }
                total_residual = np.abs(result.residual).sum()
                mean_residual = np.abs(result.residual).mean()
                max_residual = np.abs(result.residual).max()
                results_ray_list.append(
                    jsonify_dict(
                        {
                            **initialization,
                            "levenberg_marquandt_parameters": parameters_lm,
                            "total_residual": total_residual,
                            "mean_residual": mean_residual,
                            "max_residual": max_residual,
                            "parameter_errors": parameter_errors,
                        }
                    )
                )
            batches[f"batch_{batch}"] = results_ray_list
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down Ray cluster.")
    finally:
        ray.shutdown()  # type: ignore

    total_end = dt.now()
    elapsed = total_end - total_start
    logger.info(f"Fitting took {elapsed.total_seconds()/3600} hours in total.")
    return batches


def create_lmfit_parameters(
    mu: float, associating: bool, fix_dipole_moment: bool, fix_kab: bool
):
    """Create PC-SAFT parameters for LMFit"""
    params = Parameters()
    params.add("m", value=3.26, min=1.0, max=10.0)
    params.add("sigma", value=3.69, min=2.5, max=5.0)
    params.add("epsilon_k", value=284.13, min=100.0, max=1000.0)
    params.add("mu", value=mu, min=0.0, max=10.0)
    params.add("epsilonAB", value=2400, min=0.0, max=4000.0)
    params.add("KAB", value=0.0, min=0.0, max=0.05)
    if fix_dipole_moment:
        params["mu"].vary = False
    if not associating:
        params["epsilonAB"].value = 0.0
        params["epsilonAB"].vary = False
        params["KAB"].value = 0.0
        params["KAB"].vary = False
    elif associating and fix_kab:
        params["KAB"].value = 0.01
        params["KAB"].vary = False
    return params


def data_check(
    molecule_data: pd.DataFrame,
    density_column: str,
    temperature_column: str,
    min_pvap_data: int = 10,
    min_rho_data: int = 10,
):
    """Returns True if the data is good, False is not"""
    check_1 = (
        molecule_data[molecule_data[density_column].isna()].shape[0] < min_pvap_data
    )
    check_2 = (
        molecule_data[molecule_data[density_column].notnull()].shape[0] < min_rho_data
    )
    rho_data = molecule_data[molecule_data[density_column].notnull()]
    if check_1 or check_2:
        return False
    phase_transition_rho = rho_data.iloc[rho_data[temperature_column].argmax()][
        density_column
    ]
    rho_data_liquid = rho_data[rho_data[density_column] > phase_transition_rho]
    check_rho_liquid = len(rho_data_liquid) > 2
    return check_rho_liquid


def parameter_dicts_to_dataframes(
    batches: Dict[str, Callable[[], List[dict]]],
    smiles_column: str = "smiles",
    id_column: Optional[str] = None,
    max_total_residual: Optional[float] = None,
    max_residual: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert batched results to DataFrames"""
    initial_params = []
    final_params = []
    skipped_params = []
    for batch_fn in batches.values():
        for result in batch_fn():
            smiles = result["smiles"]
            mean_residual = result["mean_residual"]
            total_residual = result["total_residual"]
            max_residual_batch = result["max_residual"]
            params = {
                smiles_column: smiles,
                "associating": result["associating"],
                **result["levenberg_marquandt_parameters"],
                "total_residual": result["total_residual"],
                "mean_residual": mean_residual,
                "max_residual": max_residual,
                **result["parameter_errors"],
            }
            initial_params.append(
                {
                    smiles_column: smiles,
                    "associating": result["associating"],
                    **result["initial_parameters"],
                }
            )

            if id_column:
                initial_params[-1][id_column] = result[id_column]
                params[id_column] = result[id_column]

            check_max_total_residual = (
                max_total_residual and total_residual < max_total_residual
            ) or not (max_total_residual)
            check_max_residual = (max_residual and max_residual_batch < max_residual) or not (
                max_residual
            )

            if check_max_total_residual and check_max_residual:
                final_params.append(params)
            else:
                skipped_params.append(params)

    return (
        pd.DataFrame(initial_params),
        pd.DataFrame(final_params),
        pd.DataFrame(skipped_params),
    )


if __name__ == "__main__":
    pass
