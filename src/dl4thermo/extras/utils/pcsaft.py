import json
import logging
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime as dt
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from feos import eos, pcsaft, si  # type: ignore
from feos.eos.estimator import DataSet  # type: ignore
from lmfit import Minimizer, Parameters, conf_interval, minimize
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from scipy.interpolate import griddata
from tqdm import tqdm

from dl4thermo.extras.utils.metrics import calculate_metrics  # type: ignore
from dl4thermo.extras.utils.plotting import plot_pressure_density_phase_diagram

logger = logging.getLogger(__name__)


def create_pcsaft_parameters_from_model_predictions(
    data: pd.DataFrame, smiles_column: str, suffix: str = ""
) -> Dict[str, pcsaft.PcSaftParameters]:
    """Create a dictionary of PC-SAFT parameters from a dataframe."""
    return {
        row[smiles_column]: create_parameters(row.to_dict(), row[smiles_column])
        for _, row in data.iterrows()
    }


def create_parameters(row: Dict[str, float], smiles: str, clamp_small: bool = True):
    """Create PC-SAFT parameters"""
    identifier = pcsaft.Identifier(smiles=smiles)

    # Clamp small parameters to zero
    cols = ["m", "sigma", "epsilon_k", "mu", "KAB", "epsilonAB"]
    if clamp_small:
        for name, param in row.items():
            if param not in cols:
                continue
            row[name] = param if param > 1e-3 else 0.0

    # Create segment records
    psr = pcsaft.PcSaftRecord(
        m=row["m"],
        sigma=row["sigma"],
        epsilon_k=row["epsilon_k"],
        mu=row["mu"],
        kappa_ab=row["KAB"],
        epsilon_k_ab=row["epsilonAB"],
    )
    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    mw = ExactMolWt(mol)
    record = pcsaft.PureRecord(identifier, molarweight=mw, model_record=psr)
    return pcsaft.PcSaftParameters.new_pure(record)


def create_pcsaft_parameters_from_gc(
    smiles_list: List[str],
    segment_records: List[pcsaft.SegmentRecord],
    pure_chemical_records: List[pcsaft.ChemicalRecord],
    skip_failures: bool = True,
) -> Dict[str, Union[pcsaft.PcSaftParameters, None]]:
    return {
        smiles: smiles_to_gc_pcsaft_parameters(
            smiles,
            segment_records=segment_records,
            pure_chemical_records=pure_chemical_records,
            skip_failures=skip_failures,
        )
        for smiles in smiles_list
    }


def smiles_to_gc_pcsaft_parameters(
    smiles: str,
    segment_records: List[pcsaft.SegmentRecord],
    pure_chemical_records: List[pcsaft.ChemicalRecord],
    skip_failures: bool = True,
) -> Union[pcsaft.PcSaftParameters, None]:
    """Create PC-SAFT parameters from a SMILES string using group contribution."""
    # Look up in existing database
    cr = None
    for record in pure_chemical_records:
        if record.identifier.smiles == smiles:
            cr = record
            break

    # If not in database, use fragment algorithm to guess segmenets
    if cr is None:
        # Get segments
        counts, success, status = smarts_fragment(
            catalog=J_BIGGS_JOBACK_SMARTS_id_dict, smi=smiles
        )
        if not success:
            warnings.warn(f"Could not fully fragment {smiles} (status: {status})")
            if skip_failures:
                return
        segment_counts = {J_BIGGS_JOBACK_SMARTS[k - 1][1]: v for k, v in counts.items()}
        segments = []
        for k, v in segment_counts.items():
            segments.extend([k] * v)

        # Create parameters
        cr = pcsaft.ChemicalRecord(
            identifier=pcsaft.Identifier(smiles=smiles),
            segments=segments,
        )

    # Return parameters
    try:
        return pcsaft.PcSaftParameters.from_segments(
            chemical_records=[cr], segment_records=segment_records
        )
    except RuntimeError as e:
        warnings.warn(str(e))
        return None


def predict_phase_diagram(
    parameters: pcsaft.PcSaftParameters,
    min_temperature: float,
    n_points: int = 250,
    critical_temperature: Optional[float] = None,
) -> pd.DataFrame:
    """Predict phase diagram for a given set of PC-SAFT parameters.

    Parameters
    ----------
    min_temperature : float
        Minimum temperature for the phase diagram.
    parameters : pcsaft.PcSaftParameters
        Parameters for the PC-SAFT equation of state.
    n_points : int, optional
        Number of points to use for the phase diagram, by default 250

    """
    my_eos = eos.EquationOfState.pcsaft(parameters)
    phase_diagram = eos.PhaseDiagram.pure(
        my_eos,
        min_temperature=min_temperature * si.KELVIN,
        npoints=n_points,
        critical_temperature=critical_temperature * si.KELVIN
        if critical_temperature
        else None,
    )
    df = pd.DataFrame(phase_diagram.to_dict())
    return df


def pscsaft_pure_prediction(
    experimental_data: pd.DataFrame,
    parameters: pcsaft.PcSaftParameters,
    experimental_data_density_column: str = "DEN",
    experimental_temperature_column: str = "T",
    experimental_pressure_column: str = "P",
):
    """Calculate the predictions and error of a PCSAFT model for a set of experimental data given a set of PC-SAFT parameters

    Parameters
    ----------
    experimental_data : pd.DataFrame
        DataFrame containing the experimental data
    parameters : pcsaft.PcSaftParameters
        PC-SAFT parameters
    experimental_data_density_column : str, optional
        Name of the column containing the density data in kg/m^3, by default "DEN"
    experimental_temperature_column : str, optional
        Name of the column containing the temperature data in Kelvin, by default "T"
    experimental_pressure_column : str, optional
        Name of the column containing the pressure data in kPa, by default "P"

    Returns
    -------
    pvap_pred_df : pd.DataFrame
        DataFrame containing the predictions and error for the vapor pressure
    rho_pred_df : pd.DataFrame
        DataFrame containing the predictions and error for the liquid density

    Both dataframes have a "relative_error" column containing the relative error

    """
    # Equation of state from parameters
    my_eos = eos.EquationOfState.pcsaft(parameters)

    den = experimental_data_density_column
    T = experimental_temperature_column
    P = experimental_pressure_column

    # Vapor pressure data
    pvap_data = experimental_data[experimental_data[den].isna()]
    pvap_data = pvap_data[pvap_data[P] > 0.0]
    if len(pvap_data) > 1:
        temperatures = pvap_data[T].to_numpy() * si.KELVIN
        pressures = pvap_data[P].to_numpy() * 1e3 * si.PASCAL
        p_ds = DataSet.vapor_pressure(
            target=pressures,
            temperature=temperatures,
            extrapolate=True,
        )
        pvap_error = p_ds.relative_difference(my_eos)
        pvap_pred = p_ds.predict(my_eos) / (1e3 * si.PASCAL)
        pvap_pred_df = pd.DataFrame(
            {
                T: temperatures / si.KELVIN,
                P: pvap_pred,
                P + "_true": pressures / (1e3 * si.PASCAL),
                "relative_error": pvap_error,
                den: [None] * len(pvap_error),
                den + "_true": [None] * len(pvap_error),
            }
        )
    else:
        pvap_pred_df = pd.DataFrame(
            {T: [], P: [], P + "_pred": [], "relative_error": []}
        )

    # Liquid Density  (kg/m^3)
    rho_data = experimental_data[experimental_data[den].notnull()]
    rho_liquid_pred_df = pd.DataFrame({T: [], P: [], den: [], "relative_error": []})
    if len(rho_data) > 1:
        phase_transition_rho = rho_data.iloc[rho_data[T].argmax()][den]
        rho_liquid_data = rho_data[rho_data[den] > phase_transition_rho]
        temperatures = rho_liquid_data[T].to_numpy() * si.KELVIN
        pressures = rho_liquid_data[P].to_numpy() * 1e3 * si.PASCAL
        if len(rho_liquid_data) > 1:
            rhos = (
                rho_liquid_data[experimental_data_density_column].to_numpy()
                * 1e3
                * si.GRAM
                / si.METER**3
            )
            rho_ds = DataSet.liquid_density(
                target=rhos, temperature=temperatures, pressure=pressures
            )
            rho_error = rho_ds.relative_difference(my_eos)
            rho_pred = rho_ds.predict(my_eos) / (1e3 * si.GRAM / si.METER**3)
            rho_liquid_pred_df = pd.DataFrame(
                {
                    T: temperatures / si.KELVIN,
                    P: pressures / (1e3 * si.PASCAL),
                    P + "_true": [None] * len(temperatures),
                    den: rho_pred,
                    den + "_true": rhos / (1e3 * si.GRAM / si.METER**3),
                    "relative_error": rho_error,
                }
            )

    return pvap_pred_df, rho_liquid_pred_df


def make_pcsaft_predictions(
    model_predictions: Dict[str, pd.DataFrame],
    experimental_data: pd.DataFrame,
    pure_gc_data: Optional[List[Dict]] = None,
    segments_gc_data: Optional[List[Dict]] = None,
    model_predictions_smiles_column: str = "smiles_1",
    experimental_data_smiles_column: str = "smiles_1",
    experimental_data_name_column: str = "name_1",
    experimental_data_density_column: str = "DEN",
    experimental_temperature_column: str = "T",
    experimental_pressure_column: str = "P",
    plot_figures: bool = True,
    intersection_only: bool = True,
    skip_gc_failures: bool = True,
    return_all_data: bool = False,
):
    """Make predictions for conditions from a set of experimental data

    Parameters
    ----------
    model_predictions : Dict[str, pd.DataFrame]
        Dictionary containing the predictions from the models
    experimental_data : pd.DataFrame
        DataFrame containing the experimental data
    pure_gc_data : List[Dict]
        List of dictionaries containing the pure component group contribution data
    segments_gc_data : List[Dict]
        List of dictionaries containing the group contribution segment data
    model_predictions_smiles_column : str, optional
        Name of the column containing the SMILES in the model predictions, by default "smiles_1"
    experimental_data_smiles_column : str, optional
        Name of the column containing the SMILES in the experimental data, by default "smiles_1"
    experimental_data_name_column : str, optional
        Name of the column containing the name in the experimental data, by default "name_1"
    experimental_data_density_column : str, optional
        Name of the column containing the density in the experimental data, by default "DEN"
    experimental_temperature_column : str, optional
        Name of the column containing the temperature in the experimental data, by default "T"
    experimental_pressure_column : str, optional
        Name of the column containing the pressure in the experimental data, by default "P"
    plot_figures : bool, optional
        Whether to plot the figures, by default True
    intersection_only : bool, optional
        Whether to only make predictions for the intersection of SMILES in  the experimental and model data, by default True
    skip_gc_failures : bool, optional
        Whether to skip molecules for which group contribution parameters could not be found, by default True
    return_all_data : bool, optional
        Whether to save all the dataframes, by default False

    Returns
    -------
    figs : Dict[str, plt.Figure]
        Dictionary containing the figures
    score_dfs : Dict[str, pd.DataFrame]
        Dictionary containing the scores for each model
    predicted_dfs_large : Dict[str, pd.DataFrame]
        Dictionary containing the predictions for each model



    """
    # Get intersection of SMILES from
    experimental_smiles_list = experimental_data[
        experimental_data_smiles_column
    ].tolist()
    # experiment_smiles_list = [Chem.CanonSmiles(s) for s in experimental_smiles_list]
    if intersection_only:
        model_smiles_list = []
        for prediction_df in model_predictions.values():
            model_smiles_list += prediction_df[
                model_predictions_smiles_column
            ].to_list()
        model_smiles = set(model_smiles_list)
        smiles_to_predict = list(
            set(experimental_smiles_list).intersection(model_smiles)
        )
    else:
        smiles_to_predict = list(set(experimental_smiles_list))

    # GC data
    segment_records: List[pcsaft.SegmentRecord]
    chemical_records: List[pcsaft.ChemicalRecord]
    if segments_gc_data is not None and pure_gc_data is not None:
        segment_records = [
            pcsaft.SegmentRecord.from_json_str(json.dumps(d)) for d in segments_gc_data
        ]
        chemical_records = [
            pcsaft.ChemicalRecord.from_json_str(json.dumps(d)) for d in pure_gc_data
        ]
    elif (segments_gc_data is not None and pure_gc_data is None) or (
        segments_gc_data is None and pure_gc_data is not None
    ):
        raise ValueError(
            "Both segments_gc_data and pure_gc_data must be provided for group contribution predictions."
        )

    # Loop through smiles and calculate scores + make figures for each
    figs = {}
    model_scores = {}
    predicted_dfs_large = {}
    model_scores.update({model: [] for model in model_predictions.keys()})
    for smiles in tqdm(smiles_to_predict, desc="Making predictions"):
        try:
            Chem.CanonSmiles(smiles)
        except:
            continue

        # Get experimental data for this SMILES
        molecule_data = experimental_data[
            experimental_data[experimental_data_smiles_column] == smiles
        ]
        if len(molecule_data) < 2:
            continue
        name = molecule_data[experimental_data_name_column].iloc[0]
        mw = ExactMolWt(Chem.MolFromSmiles(smiles))  # type: ignore

        # Get PC-SAFT parameters for this SMILES from prediction dataframes
        parameters_dict = {}
        phase_diagram_dfs = {}
        for model, prediction_df in model_predictions.items():
            # Create parameters
            if smiles in prediction_df[model_predictions_smiles_column].tolist():
                row = prediction_df[
                    prediction_df[model_predictions_smiles_column] == smiles
                ].iloc[0]
            else:
                continue
            parameters_dict[model] = create_parameters(row.to_dict(), smiles)
            # parameters_dict[model] = parameters.pure_records[0].model_record

        if len(parameters_dict) != len(model_predictions):
            continue

        # Add group contribution if available
        skip = False
        if segments_gc_data is not None and pure_gc_data is not None:
            gc_parameters = smiles_to_gc_pcsaft_parameters(
                smiles,
                segment_records=segment_records,  # type: ignore
                pure_chemical_records=chemical_records,  # type: ignore
                skip_failures=skip_gc_failures,
            )
            if gc_parameters is not None:
                parameters_dict["GC"] = gc_parameters
            else:
                skip = True
            if "GC" not in model_scores:
                model_scores["GC"] = []
        if skip:
            continue

        # Make predictions and calculate errors
        for model, parameters in parameters_dict.items():
            try:
                (
                    phase_diagram_df,
                    residuals_pvap,
                    residuals_rho,
                ) = Residual._predictions_residuals(
                    parameters,
                    molecule_data,
                    mw=mw,
                    den=experimental_data_density_column,
                    T=experimental_temperature_column,
                    P=experimental_pressure_column,
                )
            except RuntimeError as e:
                logger.error(e)
                continue
            # Uncomment to save predictions for debugging
            # if model.lower() == "rf":
            #     residuals_pvap.to_csv(
            #         f"data/08_reporting/feos_predictions/data/{name}_{model}_pvap.csv",
            #         index=False,
            #     )
            #     residuals_rho.to_csv(
            #         f"data/08_reporting/feos_predictions/data/{name}_{model}_rho.csv",
            #         index=False,
            #     )
            phase_diagram_dfs[model] = phase_diagram_df
            predicted_df = pd.merge(
                residuals_pvap,
                residuals_rho,
                on=[experimental_temperature_column, experimental_pressure_column],
                how="outer",
            )
            if return_all_data and model not in predicted_dfs_large:
                predicted_dfs_large[model] = predicted_df
            elif return_all_data:
                predicted_dfs_large[model] = pd.concat(
                    (predicted_dfs_large[model], predicted_df)
                )
            scores = {
                "pvap_mape": residuals_pvap["residual"].abs().mean(),  # type: ignore
                "rho_mape": residuals_rho["residual"].abs().mean(),
                "pvap_rmse": (residuals_pvap["residual"] ** 2).mean() ** 0.5,
                "rho_rmse": (residuals_rho["residual"] ** 2).mean() ** 0.5,
                "mean_residual": pd.concat((residuals_pvap["residual"], residuals_rho["residual"]))  # type: ignore
                .abs()
                .mean(),
                "name": name,
                "smiles": smiles,
                **json.loads(parameters.pure_records[0].model_record.to_json_str()),
            }
            model_scores[model].append(scores)

        # Plot
        if plot_figures:
            phase_diagram_dfs = {
                k: df.reset_index(drop=True) for k, df in phase_diagram_dfs.items()
            }
            try:
                fig = plot_pressure_density_phase_diagram(
                    predicted_dfs=phase_diagram_dfs,
                    experimental_data=molecule_data.reset_index(drop=True),
                    name=name,
                    # params_dict=parameters_dict,
                    experimental_liquid_density_column=experimental_data_density_column,
                    temperature_column=experimental_temperature_column,
                    pressure_column=experimental_pressure_column,
                )
                figs[name] = fig
            except ValueError as e:
                logger.error(e)
                continue

    # Turn scores into dataframes
    score_dfs = {}
    for model, scores_list in model_scores.items():
        score_df = pd.DataFrame(scores_list)
        score_dfs[model] = score_df

    return figs, score_dfs, predicted_dfs_large


@dataclass
class Residual:
    """
    Residual for LMfit regression

    Parameters
    ----------
    experimental_data : pd.DataFrame
        Experimental data with temperature, pressure and density columns
    molecular_weight : float
        Molecular weight (used for density conversion)
    experimental_data_density_column : str, optional
        Name of the density column in the experimental data, by default "DEN"
    experimental_temperature_column : str, optional
        Name of the temperature column in the experimental data, by default "T"
    experimental_pressure_column : str, optional
        Name of the pressure column in the experimental data, by default "P"
    density_weight : float, optional
        Weight for the density error, by default 1.0
    vapor_pressure_weight : float, optional
        Weight for the vapor pressure error, by default 1.0

    """

    initial_parameters: Parameters
    experimental_data: pd.DataFrame
    smiles: str
    molecular_weight: Optional[float] = None
    experimental_data_density_column: str = "DEN"
    experimental_temperature_column: str = "T"
    experimental_pressure_column: str = "P"
    fit_log_pressure: bool = False

    minimize_kwargs: Optional[Dict] = None
    density_weight: float = 1.0
    vapor_pressure_weight: float = 1.0
    critical_temperature: Optional[float] = None
    n_failed: int = 0
    n_total: int = 0
    n_data: int = 0

    def __post_init__(self):
        if self.molecular_weight is None:
            self.molecular_weight = ExactMolWt(Chem.MolFromSmiles(self.smiles))  # type: ignore

    def __call__(
        self,
        params: Parameters,
        **kwargs,
    ):
        den = self.experimental_data_density_column
        T = self.experimental_temperature_column
        P = self.experimental_pressure_column
        n = self.experimental_data.shape[0]
        experimental_data = self.experimental_data

        # Create PC-SAFT FeOs parameters
        params_dict: Dict[str, float] = params.valuesdict()  # type: ignore
        feos_parameters = create_parameters(params_dict, self.smiles)

        try:
            _, residuals_pressure, residuals_density = self._predictions_residuals(
                feos_parameters=feos_parameters,
                experimental_data=experimental_data,
                mw=self.molecular_weight,  # type: ignore
                den=den,
                T=T,
                P=P,
                Tc=self.critical_temperature,
                fit_log_pressure=self.fit_log_pressure,
            )
            residuals_pressure = (
                self.vapor_pressure_weight * residuals_pressure["residual"]
            )
            residuals_density = self.density_weight * residuals_density["residual"]
            residuals_density = np.nan_to_num(residuals_density, nan=0)
            residuals = np.concatenate([residuals_pressure, residuals_density])
        except RuntimeError as e:
            logger.error(e)
            self.n_failed += 1
            residuals = np.array([1.5e4] * n)
        finally:
            self.n_total += 1
        return residuals

    @staticmethod
    def _predictions_residuals(
        feos_parameters,
        experimental_data: pd.DataFrame,
        mw: float,
        den="DEN",
        T="T",
        P="P",
        Tc=None,
        fit_log_pressure: bool = False,
    ):
        # Make predictions
        my_eos = eos.EquationOfState.pcsaft(feos_parameters)
        phase_diagram = eos.PhaseDiagram.pure(
            my_eos,
            min_temperature=experimental_data[T].min() * si.KELVIN,
            npoints=250,
            critical_temperature=None,
        )
        phase_diagram_df = pd.DataFrame(phase_diagram.to_dict())

        phase_diagram_df["density liquid"] = (
            phase_diagram_df["density liquid"] / 1e3 * mw
        )  # Convert to kg/m3
        phase_diagram_df["density vapor"] = (
            phase_diagram_df["density vapor"] / 1e3 * mw
        )  # Convert to kg/m3
        phase_diagram_df["pressure"] /= 1e3  # Convert to kPa
        phase_diagram_df = phase_diagram_df.rename(
            columns={"temperature": T, "pressure": P}
        )
        predicted_data = phase_diagram_df.sort_values(T, ascending=True)
        predicted_data[den] = predicted_data["density liquid"]

        # Pressure residual
        exp_psat = experimental_data[experimental_data[den].isna()]
        predicted_pressures = np.interp(
            exp_psat[T].to_numpy(),
            predicted_data[T].to_numpy(),
            predicted_data[P].to_numpy(),
        )
        if fit_log_pressure:
            log_exp_psat = np.log(exp_psat[P].to_numpy())
            residuals_pressure = (
                np.log(predicted_pressures) - log_exp_psat  # type: ignore
            ) / log_exp_psat  # type: ignore

        else:
            residuals_pressure = (predicted_pressures - exp_psat[P]) / exp_psat[P]
        residuals_pressure = pd.DataFrame(
            {
                "residual": residuals_pressure,
                "T": exp_psat[T],
                "P": exp_psat[P],
                "P_pred": predicted_pressures,
            }
        )

        # Density residual
        exp_with_density = experimental_data[
            experimental_data[den].notnull() & experimental_data[P].notnull()
        ]
        rhos = exp_with_density[den].to_numpy() * 1e3 * si.GRAM / si.METER**3
        temperatures = exp_with_density[T].to_numpy() * si.KELVIN
        pressures = exp_with_density[P].to_numpy() * 1e3 * si.PASCAL
        rho_ds = DataSet.liquid_density(
            target=rhos, temperature=temperatures, pressure=pressures
        )
        residuals_density = pd.DataFrame(
            {
                "residual": rho_ds.relative_difference(my_eos),
                "T": exp_with_density[T],
                "P": exp_with_density[P],
                den: exp_with_density[den],
                den + "_pred": rho_ds.predict(my_eos) / (1e3 * si.GRAM / si.METER**3),
            }
        )

        return phase_diagram_df, residuals_pressure, residuals_density

    def reset(self):
        self.n_failed = 0
        self.n_total = 0


def pcsaft_lmfit_regression(
    experimental_data: pd.DataFrame,
    lmfit_params: Parameters,
    smiles: str,
    experimental_data_density_column: str = "DEN",
    experimental_temperature_column: str = "T",
    experimental_pressure_column: str = "P",
    minimize_kwargs: Optional[Dict] = None,
    density_weight: float = 1.0,
    vapor_pressure_weight: float = 1.0,
    critical_temperature: Optional[float] = None,
    fit_log_pressure: bool = False,
):
    """Regression of PC-SAFT parameters

    Parameters
    ----------
    experimental_data : pd.DataFrame
        Experimental data with temperature, pressure and density columns
    smiles : str
        SMILES string of the molecule
    lmfit_params : Parameters
        Parameters to fit
    experimental_data_density_column : str, optional
        Name of the density column in the experimental data, by default "DEN"
    experimental_temperature_column : str, optional
        Name of the temperature column in the experimental data, by default "T"
    experimental_pressure_column : str, optional
        Name of the pressure column in the experimental data, by default "P"
    minimize_kwargs : Optional[Dict], optional
        Keyword arguments for the minimization, by default None
    density_weight : float, optional
        Weight for the density error, by default 1.0
    vapor_pressure_weight : float, optional
        Weight for the vapor pressure error, by default 1.0
    critical_temperature : Optional[float], optional
        Critical temperature for the phase diagram, by default None
    fit_log_pressure : bool, optional
        Whether to fit the log of the pressure, by default False

    Returns
    -------
    result: MinimizerResult
        Result of the minimization from LMfit
    n_failed: int
        Number of failed iterations due to PC-SAFT errors
    n_total: int
        Total number of iterations

    """
    # Get residual
    residual = Residual(
        initial_parameters=lmfit_params,
        experimental_data=experimental_data,
        smiles=smiles,
        critical_temperature=critical_temperature,
        experimental_data_density_column=experimental_data_density_column,
        experimental_temperature_column=experimental_temperature_column,
        experimental_pressure_column=experimental_pressure_column,
        vapor_pressure_weight=vapor_pressure_weight,
        density_weight=density_weight,
        fit_log_pressure=fit_log_pressure,
    )

    # Run minimization
    minimize_kwargs = minimize_kwargs or {}
    result = minimize(
        residual,
        lmfit_params,
        **minimize_kwargs,
    )
    return result, residual.n_failed, residual.n_total


"""The code below is directly copied from the thermo group_contribution code (MIT license) 
https://github.com/CalebBell/thermo/blob/516dee4ceda8e100918c7645e393a42fdfdc4bef/thermo/group_contribution/

I made changes to the SMARTS strings to match the ones used in FeOs
"""
J_BIGGS_JOBACK_SMARTS = [
    ["Methyl", "CH3", "[CX4H3]"],
    ["Secondary acyclic", "CH2", "[!R;CX4H2]"],
    ["Tertiary acyclic", ">CH", "[!R;CX4H]"],
    ["Quaternary acyclic", ">C<", "[!R;CX4H0]"],
    ["Primary alkene", "=CH2", "[CX3H2]"],
    ["Secondary alkene acyclic", "=CH", "[!R;CX3H1;!$([CX3H1](=O))]"],
    ["Tertiary alkene acyclic", "=C<", "[$([!R;CX3H0]);!$([!R;CX3H0]=[#8])]"],
    ["Alkyne", "C≡CH", "[CX2H1]#[CX2]"],
    ["Aliphatic cyclohexane CH2", "CH2_hex", "[r6;CX4H2]"],
    ["Aliphatic cyclohexane", "CH_hex", "[r6;CX4H1]"],
    ["Aliphatic cyclopentane CH2", "CH2_pent", "[r5;CX4H2]"],
    ["Aliphatic cyclopentane CH", "CH_pent", "[r5;CX4H1]"],
    ["Alcohol", "OH", "[#6X3;!$([#6X3H0]=O)][OX2H]"],
    ["Aromatic carbon", "C_arom", "[cX3H0]"],
    ["Aromatic carbon-hydrogen", "CH_arom", "[cX3H1]"],
    [
        "Carbonyl acyclic",
        ">C=O",
        "[$([CX3H0]-[!OX2])]=O",
    ],
    ["Aliphatic Ether", "OCH3", "[CX4H3][OX2H0]"],
    ["Aliphatic Ether", "OCH2", "[CX4H2][OX2H0]"],
    ["Aldehyde", "CH=O", "[CX3H1](=O)"],
    ["Ester", "COO", "[#6X3H0;!$([#6X3H0](~O)(~O)(~O))](=[#8X1])[#8X2H0]"],
    ["Terminal Ester", "HCOO", "[#6X3H1;!$([#6X3H0](~O)(~O)(~O))](=[#8X1])[#8X2H0]"],
    ["Primary amino", "NH2", "[NX3H2]"],
]
"""Metadata for the Joback groups. The first element is the group name; the
second is the group symbol; and the third is the SMARTS matching string.
"""

J_BIGGS_JOBACK_SMARTS_id_dict = {
    i + 1: j[2] for i, j in enumerate(J_BIGGS_JOBACK_SMARTS)
}
J_BIGGS_JOBACK_SMARTS_str_dict = {i[1]: i[2] for i in J_BIGGS_JOBACK_SMARTS}


def smarts_fragment(catalog, mol=None, smi=None, deduplicate=True):
    r"""Fragments a molecule into a set of unique groups and counts as
    specified by the `catalog`. The molecule can either be an rdkit
    molecule object, or a smiles string which will be parsed by rdkit.
    Returns a dictionary of groups and their counts according to the
    indexes of the catalog provided.

    Parameters
    ----------
    catalog : dict
        Dictionary indexed by keys pointing to smarts strings, [-]
    mol : mol, optional
        RDKit Mol object, [-]
    smi : str, optional
        Smiles string representing a chemical, [-]

    Returns
    -------
    counts : dict
        Dictionaty of integer counts of the found groups only, indexed by
        the same keys used by the catalog [-]
    success : bool
        Whether or not molecule was fully and uniquely fragmented, [-]
    status : str
        A string holding an explanation of why the molecule failed to be
        fragmented, if it fails; 'OK' if it suceeds.

    Notes
    -----
    Raises an exception if rdkit is not installed, or `smi` or `rdkitmol` is
    not defined.
    Examples
    --------
    Acetone:
    >>> smarts_fragment(catalog=J_BIGGS_JOBACK_SMARTS_id_dict, smi='CC(=O)C') # doctest:+SKIP
    ({1: 2, 24: 1}, True, 'OK')
    Sodium sulfate, (Na2O4S):
    >>> smarts_fragment(catalog=J_BIGGS_JOBACK_SMARTS_id_dict, smi='[O-]S(=O)(=O)[O-].[Na+].[Na+]') # doctest:+SKIP
    ({29: 4}, False, 'Did not match all atoms present')
    Propionic anhydride (C6H10O3):
    >>> smarts_fragment(catalog=J_BIGGS_JOBACK_SMARTS_id_dict, smi='CCC(=O)OC(=O)CC') # doctest:+SKIP
    ({1: 2, 2: 2, 28: 2}, False, 'Matched some atoms repeatedly: [4]')
    """

    if mol is None and smi is None:
        raise Exception("Either an rdkit mol or a smiles string is required")
    if smi is not None:
        mol = Chem.MolFromSmiles(smi)  # type: ignore
        if mol is None:
            status = "Failed to construct mol"
            success = False
            return {}, success, status

    atom_count = len(mol.GetAtoms())  # type: ignore
    status = "OK"
    success = True

    counts = {}
    all_matches = {}
    for key, smart in catalog.items():
        if isinstance(smart, str):
            patt = Chem.MolFromSmarts(smart)  # type: ignore
        else:
            patt = smart
        hits = list(mol.GetSubstructMatches(patt))  # type: ignore
        if hits:
            all_matches[key] = hits
            counts[key] = len(hits)

    # Duplicate group cleanup
    matched_atoms = []
    for i in all_matches.values():
        for j in i:
            matched_atoms.extend(j)

    if deduplicate:
        dups = [i for i, c in Counter(matched_atoms).items() if c > 1]
        iteration = 0
        while dups and iteration < 100:
            dup = dups[0]

            dup_smart_matches = []
            for group, group_match_list in all_matches.items():
                for i, group_match_i in enumerate(group_match_list):
                    if dup in group_match_i:
                        dup_smart_matches.append(
                            (group, i, group_match_i, len(group_match_i))
                        )

            sizes = [i[3] for i in dup_smart_matches]
            max_size = max(sizes)
            #            #print(sizes, 'sizes', 'dup', dup, 'working_data', dup_smart_matches)
            if sizes.count(max_size) > 1:
                iteration += 1
                #                #print('BAD')
                # Two same size groups, continue, can't do anything
                continue
            else:
                # Remove matches that are not the largest
                for group, idx, positions, size in dup_smart_matches:
                    if size != max_size:
                        # Not handling the case of multiple duplicate matches right, indexes changing!!!
                        del all_matches[group][idx]
                        continue

            matched_atoms = []
            for i in all_matches.values():
                for j in i:
                    matched_atoms.extend(j)

            dups = [i for i, c in Counter(matched_atoms).items() if c > 1]
            iteration += 1

    matched_atoms = set()
    for i in all_matches.values():
        for j in i:
            matched_atoms.update(j)
    if len(matched_atoms) != atom_count:
        status = "Did not match all atoms present"
        success = False

    # Check the atom aount again, this time looking for duplicate matches (only if have yet to fail)
    if success:
        matched_atoms = []
        for i in all_matches.values():
            for j in i:
                matched_atoms.extend(j)
        if len(matched_atoms) < atom_count:
            status = "Matched %d of %d atoms only" % (len(matched_atoms), atom_count)
            success = False
        elif len(matched_atoms) > atom_count:
            status = "Matched some atoms repeatedly: %s" % (
                [i for i, c in Counter(matched_atoms).items() if c > 1]
            )
            success = False

    return counts, success, status


def is_associating(smiles: str) -> bool:
    """Check if molecule is associating.

    Uses the number of hydrogen bond donors and acceptors
    """
    from rdkit.Chem import Lipinski

    mol = Chem.MolFromSmiles(smiles)  # type: ignore
    num_acceptors = Lipinski.NumHAcceptors(mol)
    num_donors = Lipinski.NumHDonors(mol)
    if num_acceptors == 0 or num_donors == 0:
        return False
    else:
        return True


def test(smiles: str, show_fig: bool = False):
    experimental_data = pd.read_parquet(
        "data/02_intermediate/dortmund_base_pure_component.pq"
    )
    dipole_moment_data = pd.read_parquet(
        "data/02_intermediate/dortmund_dipole_moment_predictions.pq"
    )
    # with open("data/01_raw/gc_substances.json", "r") as f:
    #     pure_gc_data = json.load(f)
    # with open("data/01_raw/sauer2014_homo.json", "r") as f:
    #     segments_gc_data = json.load(f)
    # segment_records = [
    #     pcsaft.SegmentRecord.from_json_str(json.dumps(d)) for d in segments_gc_data
    # ]
    # chemical_records = [
    #     pcsaft.ChemicalRecord.from_json_str(json.dumps(d)) for d in pure_gc_data
    # ]

    associating = is_associating(smiles)
    molecule_data = experimental_data[experimental_data["smiles_1"] == smiles]
    name = molecule_data["name_1"].iloc[0]
    # print("Running regression for %s" % name)
    # sepp_params = pd.read_csv("data/03_primary/pcp_saft_sepp_pure_component_params.csv")
    # sepp_params = sepp_params[sepp_params["smiles"] == smiles].iloc[0].to_dict()
    mu = dipole_moment_data[dipole_moment_data["smiles_1"] == smiles][
        "dipole_moment"
    ].iloc[0]
    if smiles == "O":
        associating = True
        mu = 1.8
    # print(f"Dipole moment prediction: {mu:.03f}")
    # critical_temperature = 545 * si.KELVIN
    critical_temperature = None
    Tbounds = [200, 1000]
    Pbounds = [10, 1e3]
    density_bounds = [20, 2e3]
    inc = "neither"
    molecule_data = molecule_data[
        molecule_data["T"].between(Tbounds[0], Tbounds[1], inclusive=inc)
        & molecule_data["P"].between(Pbounds[0], Pbounds[1], inclusive=inc)
        & (
            (molecule_data["DEN"].isna())
            | molecule_data["DEN"].between(
                density_bounds[0], density_bounds[1], inclusive=inc
            )
        )
    ]
    molecule_data = molecule_data[
        ((molecule_data["DEN"].notnull()) & (molecule_data["P"] < 150))
        | molecule_data["DEN"].isna()
    ]
    n_pvap_data = molecule_data[molecule_data["DEN"].isna()].shape[0]
    n_rho_data = molecule_data[molecule_data["DEN"].notnull()].shape[0]
    weight = 10 * n_pvap_data / n_rho_data
    # print("Density weighting factor: %f" % weight)

    # Create parameters
    params = Parameters()
    params.add("m", value=3.26, min=1.5, max=4.5)
    params.add("sigma", value=3.69, min=2.5, max=4.0)
    params.add("epsilon_k", value=284.13, min=100.0, max=1000.0)
    params.add("epsilonAB", value=2400, min=0.0, max=3600.0)
    params.add("KAB", value=0.01, min=0.0, max=0.01)
    params.add("mu", value=mu, min=0.0, max=10.0)
    params["mu"].vary = False
    if not associating:
        params["epsilonAB"].value = 0.0
        params["epsilonAB"].vary = False
        params["KAB"].value = 0.0
        params["KAB"].vary = False

    # gc_feos_parameters = smiles_to_gc_pcsaft_parameters(
    #     smiles, segment_records=segment_records, pure_chemical_records=chemical_records
    # )

    # Parameter estimation
    start = dt.now()
    initial_parameters = params.valuesdict()
    mw = ExactMolWt(Chem.MolFromSmiles(smiles))  # type: ignore
    residual = Residual(
        initial_parameters=params,
        experimental_data=molecule_data,
        smiles=smiles,
        molecular_weight=mw,
        critical_temperature=critical_temperature,
        fit_log_pressure=True
        # regularization_weights={
        #     "m": 2.5,
        #     "sigma": 2.5,
        #     "epsilon_k": 2.5,
        #     "epsilonAB": 10.0,
        #     "KAB": 10.0,
        #     "mu": 0.0,
        # },
    )
    # #print("Running differential evolution")
    # result_de = minimize(
    #     residual,
    #     params,
    #     method="differential_evolution",
    #     nan_policy="omit",
    #     max_nfev=5000,
    # )
    # n_failed_de, n_total_de = residual.n_failed, residual.n_total
    # for param_name, val in result_de.params.valuesdict().items():  # type: ignore
    #     params[param_name].value = val
    # Fine tune with Levenberg-Marquardt
    # print("Running Levenberg-Marquandt")
    result_lm = minimize(residual, params, method="least_sq", nan_policy="omit")
    n_failed_lm, n_total_lm = residual.n_failed, residual.n_total
    end = dt.now()

    # Get initial and final parameters
    loss_fn = lambda r: np.sum(r**2)
    initial_feos_parameters = create_parameters(initial_parameters, smiles)  # type: ignore
    initial_loss = loss_fn(residual(params))
    # de_parameters = result_de.params.valuesdict()  # type: ignore
    # de_feos_parameters = create_parameters(de_parameters, smiles)  # type: ignore
    # de_loss = loss_fn(residual(result_de.params))  # type: ignore
    final_parameters = result_lm.params.valuesdict()  # type: ignore
    lm_feos_parameters = create_parameters(final_parameters, smiles)  # type: ignore
    lm_loss = loss_fn(residual(result_lm.params))  # type: ignore

    # print("\n\n\n")
    # print("Results")
    # print("Associating: ", associating)
    # print("Density weighting factor: %f" % weight)
    # print(f"Regression time: {(end - start).total_seconds()/60:.2f} min")
    # print("-" * 10)
    # print()

    # print(f"Initial parameters (Loss: {initial_loss:.02f}):")
    # print(initial_parameters)
    # print()

    # #print(f"Differential evolution parameters (Loss: {de_loss:.02f}):")
    # #print(de_parameters)
    # #print(
    #     "{} failed out of {} ({:.2f})% during differential evolution".format(
    #         n_failed_de, n_total_de, n_failed_de / n_total_de * 100
    #     )
    # )
    # #print()

    # print(f"LM parameters (Loss: {lm_loss:.02f}):")
    # print(final_parameters)  # type: ignore
    # print(
    #     "{} failed out of {} ({:.2f})% during LM".format(
    #         n_failed_lm, n_total_lm, n_failed_lm / n_total_lm * 100
    #     )
    # )
    # print()

    # #print("GC parameters:")
    # #print(gc_feos_parameters.pure_records[0].model_record)  # type: ignore

    # Make predictions
    mw = ExactMolWt(Chem.MolFromSmiles(smiles))  # type: ignore
    predicted_dfs = {}
    for algo, params in zip(
        [
            "Initial",
            # "Differential Evolution",
            "Levenberg-Marquandt",
            # "GC",
        ],
        [
            initial_feos_parameters,
            # de_feos_parameters,
            lm_feos_parameters,
            # gc_feos_parameters,
        ],
    ):
        pred = predict_phase_diagram(params, min_temperature=molecule_data["T"].min())
        pred = pred.rename(columns={"temperature": "T", "pressure": "P"})
        pred["density liquid"] = pred["density liquid"] / 1e3 * mw
        pred["density vapor"] = pred["density vapor"] / 1e3 * mw
        pred["P"] = pred["P"] / 1e3
        predicted_dfs[algo] = pred

    fig = plot_pressure_density_phase_diagram(
        predicted_dfs=predicted_dfs,
        experimental_data=molecule_data,
        experimental_liquid_density_column="DEN",
        name=name,
    )

    import matplotlib.pyplot as plt

    if show_fig:
        plt.show()
    fig.savefig(f"regression_{name}.png", dpi=300)
    return lm_loss, final_parameters


if __name__ == "__main__":
    to_fit = pd.read_csv("smiles.csv")
    all_losses = []
    all_params = []
    for smiles in tqdm(to_fit["smiles"]):
        loss, params = test(smiles, show_fig=False)
        all_losses.append(loss)
        all_params.append(params)
    print("Mean loss:", np.mean(all_losses))
    # np.savetxt("parameters/losses.txt")
    with open("regressed_params.json", "w") as f:
        json.dump(all_params, f)
