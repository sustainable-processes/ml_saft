from typing import Dict

import numpy as np
import pandas as pd
from mlsaft.extras.utils.pcsaft import create_parameters, pscsaft_pure_prediction
from SALib.analyze import rbd_fast
from SALib.sample import latin


def pcsaft_sensitivity_analysis(
    smiles: str,
    experimental_data: pd.DataFrame,
    associating: bool,
    initial_parameters: Dict[str, float],
    n_samples: int = 100,
    experimental_data_density_column: str = "DEN",
    experimental_temperature_column: str = "T",
    experimental_pressure_column: str = "P",
):
    """Perform Sobol sensitivity analysis on the PC-SAFT model

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule
    molecule_data : pd.DataFrame
        DataFrame containing the density and vapor pressure data for the molecule
    associating : bool
        Whether the molecule is associating or not
    initial_parameters : Dict[str,float]
        Dictionary containing the initial parameters for the molecule
    n_samples : int, optional
        Number of samples to use for the sensitivity analysis, by default 100

    Notes
    -----
    molecule_data should have a column "T" for temperature in Kelvin and "P" for pressure in kPa.

    """
    # Define the problem
    vars = ["sigma", "m", "epsilon_k", "mu", "epsilonAB", "KAB"]
    if associating:
        problem = {
            "num_vars": 6,
            "names": vars,
            "bounds": [
                [
                    initial_parameters[param] * 0.9,
                    initial_parameters[param] * 1.1,
                    0.5,
                ]
                for param in vars
            ],
            "dists": ["triang"] * len(vars),
        }
    else:
        problem = {
            "num_vars": 6,
            "names": vars,
            "bounds": [
                [
                    initial_parameters[param] * 0.9,
                    initial_parameters[param] * 1.1,
                    0.5,
                ]
                for param in vars[:4]
            ]
            + [
                [0, 1.4, 0.5],
                [0, 2400, 0.5],
                [0, 388, 0.5],
            ],
            "dists": ["triang"] * len(vars),
        }

    # Generate samples
    param_values = latin.sample(problem, n_samples)
    parameters_df = pd.DataFrame(param_values, columns=vars)

    # Evaluate samples
    pvap_errors = []
    rho_errors = []
    for _, parameters_row in parameters_df.iterrows():
        # Generate parameters
        pcsaft_parameters = create_parameters(parameters_row.to_dict(), smiles)

        # Calculate errors
        pvap_pred_df, rho_pred_df = pscsaft_pure_prediction(
            experimental_data,
            pcsaft_parameters,
            experimental_data_density_column=experimental_data_density_column,
            experimental_temperature_column=experimental_temperature_column,
            experimental_pressure_column=experimental_pressure_column,
        )
        pvap_mape = pvap_pred_df["relative_error"].abs().mean()
        rho_mape = rho_pred_df["relative_error"].abs().mean()
        pvap_errors.append(pvap_mape)
        rho_errors.append(rho_mape)

    # Analyze results
    indices = {
        target: rbd_fast.analyze(problem, X=param_values, Y=np.array(errors)).to_df()
        for target, errors in zip(["pvap", "rho"], [pvap_errors, rho_errors])
    }
    return indices
