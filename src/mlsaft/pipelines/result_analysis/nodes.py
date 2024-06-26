"""
This is a boilerplate pipeline 'result_analysis'
generated using Kedro 0.17.7
"""

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import dl4thermo.extras.utils.counterfactual as counterfactual
import latextable
import numpy as np
import pandas as pd
import seaborn as sns
import svgutils.transform as sg
import wandb
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from texttable import Texttable
from tqdm import tqdm

from mlsaft.extras.utils.metrics import calculate_metrics
from mlsaft.extras.utils.pcsaft import is_associating, make_pcsaft_predictions
from mlsaft.extras.utils.plotting import (
    conterfactual_plot,
    parity_plot_grid,
    score_display,
)
from mlsaft.extras.utils.wandb_utils import download_runs_wandb

target_display_names: Dict[str, str] = {
    "m": r"$m$",
    "sigma": r"$\sigma$",
    "epsilon_k": r"$\epsilon/k$",
    "mu": r"$\mu$",
    "KAB": r"$\kappa_{AB}$",
    "epsilonAB": r"$\epsilon_{AB}$",
    "rho": r"$\rho^l$",
    "pvap": r"$p^{sat}$",
}


def make_parity_plots(
    wandb_runs: List[Dict[str, str]],
    wandb_entity: str,
    wandb_project: str,
    split: str = "test",
):
    """Make parity plots for a list of wandb runs

    Parameters
    ----------
    wandb_runs: List[Dict[str, str]]
        List of dictionaries with keys "name" and "wandb_run_id"
    wandb_entity: str
        wandb entity name
    wandb_project: str
        wandb project name
    split: str
        Which split to use for the parity plots
    """
    logger = logging.getLogger(__name__)

    # Download from wandb
    logger.info("Downloading predictions from wandb")
    api = wandb.Api()
    dfs = {}
    targets = {}
    for d in tqdm(wandb_runs):
        model_name = d["name"]
        wandb_run_id = d["wandb_run_id"]
        run = api.run(f"{wandb_entity}/{wandb_project}/{wandb_run_id}")
        artifacts = run.logged_artifacts()  # type: ignore
        targets[model_name] = run.config["target_columns"]  # type: ignore
        data_path = None
        for artifact in artifacts:
            if artifact.type == "dataset":
                data_path = artifact.download()
        if data_path is None:
            raise ValueError("No {split} predictions found")
        data_path = Path(data_path)
        df = pd.read_csv(data_path / f"{split}_predictions.csv")
        dfs[model_name] = df

    # Make parity plots
    mpl_figs = parity_plot_grid(
        dfs,
        targets,
        target_display_names=target_display_names,
        x_axis_label_prefix="Regressed",
    )

    # Save figures to svg bytestream
    subfigs = {fig_name: sg.from_mpl(fig) for fig_name, fig in mpl_figs.items()}
    param_names = ["m", "sigma", "epsilon_k", "epsilonAB"]
    model_names = ["FFN", "MPNN", "D-MPNN", "RF"]
    N_COLS = 2
    COL_WIDTH = 450
    ROW_HEIGHT = 450
    svg_figs = {}
    for model_name in model_names:
        # Create figure
        svg_fig = sg.SVGFigure()
        svg_fig.set_size(("680pt", "680pt"))

        # Load parity plots
        i, j = 0, 0
        for param_name in param_names:
            subfig = subfigs[f"{model_name}_{param_name}"]
            r = subfig.getroot()
            r.moveto(x=COL_WIDTH * j, y=ROW_HEIGHT * i)
            svg_fig.append([r])
            j += 1
            if j >= N_COLS:
                i += 1
                j = 0

        svg_figs[model_name] = svg_fig

    return svg_figs, dfs


def make_parity_plots_with_uncertainty(
    wandb_runs: List[Dict[str, str]],
    wandb_entity: str,
    wandb_project: str,
    split: str = "test",
):
    """Make parity plots for a list of wandb runs

    Parameters
    ----------
    wandb_runs: List[Dict[str, str]]
        List of dictionaries with keys "name" and "wandb_run_id"
    wandb_entity: str
        wandb entity name
    wandb_project: str
        wandb project name
    split: str
        Which split to use for the parity plots
    """
    logger = logging.getLogger(__name__)

    # Download from wandb
    logger.info("Downloading predictions from wandb")
    api = wandb.Api()
    dfs = {}
    targets = {}
    for d in tqdm(wandb_runs):
        model_name = d["name"]
        wandb_run_id = d["wandb_run_id"]
        run = api.run(f"{wandb_entity}/{wandb_project}/{wandb_run_id}")
        artifacts = run.logged_artifacts()  # type: ignore
        targets[model_name] = run.config["target_columns"]  # type: ignore
        data_path = None
        for artifact in artifacts:
            if artifact.type == "dataset":
                data_path = artifact.download()
        if data_path is None:
            raise ValueError("No {split} predictions found")
        data_path = Path(data_path)
        df = pd.read_csv(data_path / f"{split}_predictions.csv")
        dfs[model_name] = df

    # Make parity plotss
    figs = {}
    for model_name, df in dfs.items():
        for target in targets[model_name]:
            df[target] = df[target].astype(float).where(df[target] >= 1e-10, 0.0)
            df[target + "_pred"] = (
                df[target + "_pred"]
                .astype(float)
                .where(df[target + "_pred"] >= 1e-10, 0.0)
            )
            scores = calculate_metrics(
                df[target], df[target + "_pred"], scores=["mae", "r2"]
            )
            display_name = target_display_names.get(target, target)
            min_y = df[[target, target + "_pred"]].min().min()
            max_y = df[[target, target + "_pred"]].max().max()
            score_label = ", ".join(
                [
                    f"{score_display.get(score_name, score_name)}={score:.02f}"
                    for score_name, score in scores.items()
                ]
            )
            full_label = f"{display_name} ({score_label})"
            g = sns.jointplot(
                data=df,
                x=target,
                y=target + "_pred",
                color="#896273",
                label=full_label,
                s=100,
            )
            g.set_axis_labels(
                f"Regressed {display_name}", f"Predicted {display_name}", fontsize=15
            )
            g.ax_joint.plot([min_y, max_y], [min_y, max_y], "--k")
            g.figure.tight_layout()
            figs[f"{model_name}_{target}"] = g.figure
    return figs, dfs


def make_pcsaft_parameters_predictions_results_table(
    targets: List[str],
    wandb_entity: str,
    wandb_project: str,
    wandb_runs: Optional[List[Dict[str, str]]] = None,
    wandb_groups: Optional[List[Dict[str, Union[str, dict]]]] = None,
    split: str = "test",
    metric: str = "mae",
    mark_bold: Optional[Literal["max", "min"]] = "min",
    show_std: bool = True,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    only_tabular: bool = False,
):
    """Make a LaTex table of results from wandb runs

    Arguments
    ---------
    target_columns: List[str]
        List of targets to include in the table
    wandb_runs: Optional[List[Dict[str, str]]]
        List of dictionaries with keys "name" and "wandb_run_id"
    wandb_groups: Optional[List[Dict[str, str]]]
        List of dictionaries with keys "wandb_group_id" and "name"
    wandb_entity: str
        wandb entity name
    wandb_project: str
        wandb project name
    split: Optional[str] = "test"
        Which split to use for the table
    score: Optional[str] = "mae"
        Which score to use for the table
    rounding: Optional[int] = 2
        How many decimal places to round the scores to
    mark_bold: Optional[Literal["max", "min"]] = "min"
        Whether to mark the max or min value for each row in the table with bold text
        if None, no values will be marked
    show_std: bool = True
        Whether to show the standard deviation of the scores
    caption: str, optional
        A string that adds a caption to the Latex formatting
    label: str, optional
        A string that adds a referencing label to the Latex formatting.
    only_tabular: bool, optional
        Whether to only return the tabular part of the Latex table. Default is False.

    """
    logger = logging.getLogger(__name__)

    api = wandb.Api()

    # Download runs based on group from wandb

    groups = {}
    if (
        wandb_entity is not None
        and wandb_project is not None
        and wandb_groups is not None
    ):
        wandb_runs = [] if wandb_runs is None else wandb_runs
        for group in wandb_groups:
            filters = {"group": group["wandb_group_id"]}
            if group.get("extra_filters") and isinstance(group["extra_filters"], dict):
                filters.update(group["extra_filters"])
            group_runs = download_runs_wandb(
                api=api,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                extra_filters=filters,
            )
            groups[group["name"]] = [run.id for run in group_runs]
            wandb_runs += [
                {"name": run.id, "wandb_run_id": run.id} for run in group_runs
            ]

    # Get scores
    all_model_scores: Dict[str, Dict[str, float]] = {}
    if (
        wandb_runs is not None
        and wandb_entity is not None
        and wandb_project is not None
    ):
        for wandb_run in wandb_runs:
            name = wandb_run["name"]
            wandb_run_id = wandb_run["wandb_run_id"]
            run = api.run(f"{wandb_entity}/{wandb_project}/{wandb_run_id}")
            summary = run.summary
            scores: Dict[str, float] = {}
            for key, score_val in summary.items():
                if split not in key:
                    continue
                info = key.split("_")
                this_split = info[0]
                target = "_".join(list(info[1:-1]))
                score_name = info[-1].lower()
                if this_split == split and score_name == metric:
                    # scores[target] = round(score_val, rounding)
                    scores[target] = score_val
            all_model_scores[name] = scores  # type: ignore
    elif wandb_runs is not None and (wandb_entity is None or wandb_project is None):
        raise ValueError(
            "wandb_entity and wandb_project must be specified if wandb_runs is not None"
        )

    # Create summary dataframe
    df_dict: Dict[str, List] = {
        "model": [],
    }
    for target in targets:
        df_dict[target] = []
        df_dict[f"{target}_std"] = []
    if len(groups) > 1:
        for model, ids in groups.items():
            df_dict["model"].append(model)

            # Average scores for each model in the group
            for target in targets:
                arr = np.array([all_model_scores[id_][target] for id_ in ids])
                # Average scores for the group
                avg = np.mean(arr)
                std = np.std(arr)
                df_dict[target].append(avg)
                df_dict[f"{target}_std"].append(std)

    else:
        for model, model_scores in all_model_scores.items():
            df_dict["model"].append(model)
            for target in targets:
                df_dict[target].append(model_scores[target])
                df_dict[f"{target}_std"].append(np.nan)
    display_df = pd.DataFrame(df_dict)

    # Make rows
    def format_row(
        target: str,
        target_label: str,
        display_df: pd.DataFrame,
        show_std: bool,
    ):
        r = [target_label]
        for _, row in display_df.iterrows():
            score = row[target]
            std = row[f"{target}_std"] if f"{target}_std" in row.index else np.nan
            if show_std and not np.isnan(std):
                r.append(f"{score:.02f}±{std:.02f}")
            elif type(score) == float:
                r.append(f"{score:.02f}")
            else:
                r.append(str(score))
        return r

    rows: List[List] = []
    rows.append(format_row("model", "", display_df, False))
    for target in targets:
        rows.append(
            format_row(
                target, target_display_names.get(target, target), display_df, True
            )
        )

    # Make best score bold
    if mark_bold is not None and not (groups and show_std):
        for i, target in enumerate(targets):
            col_offset: int = 1
            if mark_bold == "min":
                bold_score_idx = display_df[target].argmin()
            elif mark_bold == "max":
                bold_score_idx = display_df[target].argmax()
            else:
                raise ValueError(f"mark_bold must be 'min' or 'max', not {mark_bold}")
            rows[i + 1][bold_score_idx + col_offset] = (  # type: ignore
                "\\textbf{" + str(rows[i + 1][bold_score_idx + col_offset]) + "}"  # type: ignore
            )  # type: ignore

    # Draw table
    n_cols = len(all_model_scores) + 1 if len(groups) == 0 else len(groups) + 1
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(["t"] * n_cols)
    table.set_cols_align(["l"] + ["r"] * (n_cols - 1))
    table.add_rows(rows, header=True)
    logger.info("\nASCII representation of table\n" + table.draw())  # type: ignore
    if only_tabular:
        latex_code = latextable.draw_latex(table)
        return "\n".join(latex_code.splitlines()[2:-2])
    else:
        return latextable.draw_latex(
            table, caption=caption, caption_above=True, label=label
        )


def reformat_regression_parameter_dfs(**kwargs):
    dfs = dict(**kwargs)
    return {" ".join(name.split("_")).title(): df for name, df in dfs.items()}


def make_regresion_histograms(
    regression_dfs: Dict[str, Callable[[], pd.DataFrame]],
    cutoffs: Optional[List[Optional[float]]] = None,
):
    # Make histograms of mape for pvap and rho using seaborn
    figs = {}
    colors = ["#263B5C", "#97A85E", "#c50027", "#007ab2"]
    if cutoffs is None:
        cutoffs = [None]

    for i, (name, df_func) in enumerate(regression_dfs.items()):
        for target in ["pvap", "rho"]:
            df = df_func()
            df[f"{target}_aard"] = df[f"{target}_mape"] * 100
            for cutoff in cutoffs:
                fig, ax = plt.subplots(figsize=(8, 6))
                if cutoff is None:
                    cutoff = df[f"{target}_aard"].max()
                sns.histplot(
                    data=df,
                    x=f"{target}_aard",
                    stat="proportion",
                    color=colors[i],
                    ax=ax,
                    bins=20,  # type: ignore
                    log_scale=True if cutoff > 1e3 else False,
                )
                ax.set_xlim(0, cutoff)
                target_name = target_display_names.get(target)
                ax.set_xlabel(f"{target_name} %AAD", fontsize=24)
                ax.set_ylabel("Proportion", fontsize=24)
                ax.tick_params(axis="both", labelsize=16)
                fig.tight_layout()
                figs[f"{name}_{target}_max_{cutoff}_histogram"] = fig
    return figs


def merge_dipole_moment_predictions(
    parameters_df: pd.DataFrame,
    dipole_moment_data: pd.DataFrame,
    dipole_moment_data_smiles_column: str = "smiles_1",
    parameter_dfs_smiles_column: str = "smiles_1",
    dipole_moment_data_column: str = "dipole_moment",
):
    if "mu" in parameters_df.columns:
        parameters_df = parameters_df.drop(columns="mu")
    if dipole_moment_data_column != "mu":
        dipole_moment_data = dipole_moment_data.rename(
            columns={dipole_moment_data_column: "mu"}
        )
    merged_df = pd.merge(
        parameters_df,
        dipole_moment_data[[dipole_moment_data_smiles_column, "mu"]],
        left_on=parameter_dfs_smiles_column,
        right_on=dipole_moment_data_smiles_column,
    )
    if dipole_moment_data_smiles_column != parameter_dfs_smiles_column:
        merged_df = merged_df.drop(columns=[dipole_moment_data_smiles_column])
    return merged_df


def get_pcsaft_thermo_scores(
    experimental_data: pd.DataFrame,
    dipole_moment_data: Optional[pd.DataFrame] = None,
    parameters_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    wandb_runs: Optional[List[Dict[str, str]]] = None,
    wandb_groups: Optional[List[Dict[str, Union[str, dict]]]] = None,
    wandb_entity: Optional[str] = None,
    wandb_project: Optional[str] = None,
    split: str = "test",
    pure_gc_data: Optional[List[Dict]] = None,
    segments_gc_data: Optional[List[Dict]] = None,
    dipole_moment_data_smiles_column: str = "smiles_1",
    dipole_moment_data_column: str = "dipole_moment",
    parameter_dfs_smiles_column: str = "smiles_1",
    experimental_data_smiles_column: str = "smiles_1",
    experimental_data_name_column: str = "name_1",
    experimental_data_density_column: str = "DEN",
    experimental_temperature_column: str = "T",
    experimental_pressure_column: str = "P",
    plot_figures: bool = False,
    return_all_data: bool = False,
) -> Tuple[
    Dict[str, Figure],
    Dict[str, Dict[str, float]],
    Dict[str, pd.DataFrame],
    Dict[str, List[str]],
]:
    """PC-SAFT thermo predictions and scores

    Parameters
    ----------
    experimental_data: pd.DataFrame
        Experimental data with thermodynamic values for density and vapor pressure
    parameter_dfs: Optional[Dict[str, pd.DataFrame]] = None
        DataFrames with PC-SAFT parameters for different molecules
    wandb_runs: Optional[List[Dict[str, str]]] = None
        Wandb runs to get predictions from. Should be list of dicts with keys
        "name" and "wandb_run_id"
    wandb_groups: Optional[List[Dict[str, str]]] = None
        Wandb groups to get predictions from. Should be list of dicts with keys
        "wandb_group_id" and "name"
    wandb_entity: Optional[str] = None
        wandb entity to use if wandb_runs is not None
    wandb_project: Optional[str] = None
        wandb project to use if wandb_runs is not None
    split: str = "test"
        Split to download from wandb
    pure_gc_data: Optional[List[Dict]] = None
        Pure component group contribution data for FeOS
    segments_gc_data: Optional[List[Dict]] = None
        Segment group contribution data for FeOS
    parameter_dfs_smiles_column : str, optional
        Name of the column containing the SMILES in the parameters DataFrames, by default "smiles_1"
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
    plot_figures: bool = False
        Whether to plot figures
    return_all_data: bool = False
        Whether to return all data

    Returns
    -------
    figs: Dict[str, Figure]
        Figures of thermodynamic predictions for each molecule
    scores: Dict[str, Dict[str, float]]
        Scores for each molecule

    """
    logger = logging.getLogger(__name__)

    api = wandb.Api()

    # Download runs based on group from wandb
    groups = {}
    if (
        wandb_entity is not None
        and wandb_project is not None
        and wandb_groups is not None
    ):
        wandb_runs = [] if wandb_runs is None else wandb_runs
        for group in wandb_groups:
            filters = {"group": group["wandb_group_id"]}
            if group.get("extra_filters") and isinstance(group["extra_filters"], dict):
                filters.update(group["extra_filters"])
            group_runs = download_runs_wandb(
                api=api,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                extra_filters=filters,
            )
            groups[group["name"]] = [run.id for run in group_runs]
            wandb_runs += [
                {"name": run.id, "wandb_run_id": run.id} for run in group_runs
            ]

    # Download predictions from wandb artifacts
    dfs = {}
    if (
        wandb_runs is not None
        and wandb_entity is not None
        and wandb_project is not None
    ):
        logger.info("Downloading predictions from wandb")
        for d in wandb_runs:
            model_name = d["name"]
            wandb_run_id = d["wandb_run_id"]
            run = api.run(f"{wandb_entity}/{wandb_project}/{wandb_run_id}")
            artifacts = run.logged_artifacts()  # type: ignore
            data_path = None
            for artifact in artifacts:
                if artifact.type == "dataset":
                    data_path = artifact.download()
            if data_path is None:
                raise ValueError("No {split} predictions found")
            data_path = Path(data_path)
            df = pd.read_csv(data_path / f"{split}_predictions.csv")
            for col in df.columns:
                if col.endswith("_pred"):
                    new_col = col.rstrip("_pred")
                    df = df.drop(columns=[new_col])
                    df = df.rename(columns={col: new_col})
            df = df.rename(columns={"smiles": parameter_dfs_smiles_column})
            dfs[model_name] = df
    elif wandb_runs is not None and (wandb_entity is None or wandb_project is None):
        raise ValueError(
            "wandb_entity and wandb_project must be specified if wandb_runs is not None"
        )

    # Instead of wandb, we can also pass in a dictionary of parameter dfs
    if parameters_dfs is not None:
        parameters_dfs = {
            name: df
            for name, df in parameters_dfs.items()
            if isinstance(df, pd.DataFrame)
        }
        dfs.update(parameters_dfs)

    if len(dfs) == 0:
        raise ValueError("Either wandb_runs or parameters_dfs must be specified")

    # Handle association
    for name, df in dfs.items():
        if "KAB" not in df.columns:
            associating = [
                is_associating(smiles) for smiles in df[parameter_dfs_smiles_column]
            ]
            df["KAB"] = [0.01 if a else 0.0 for a in associating]

        # Make sure values are not too small
        for target in ["epsilonAB", "KAB"]:
            if target in df.columns:
                df[target] = df[target].astype(float).where(df[target] >= 1e-5, 0.0)

    # Get dipole moment data
    if dipole_moment_data is not None:
        for name, df in tqdm(dfs.items(), desc="Merging dipole moment predictions"):
            dfs[name] = merge_dipole_moment_predictions(
                df,
                dipole_moment_data=dipole_moment_data,
                dipole_moment_data_column=dipole_moment_data_column,
                dipole_moment_data_smiles_column=dipole_moment_data_smiles_column,
                parameter_dfs_smiles_column=parameter_dfs_smiles_column,
            )

    # Make predictions of density and vapor pressure and compare to exp
    figs, scores, all_predictions = make_pcsaft_predictions(
        dfs,
        experimental_data.dropna(subset=[experimental_data_smiles_column]).reset_index(
            drop=True
        ),
        pure_gc_data=pure_gc_data,
        segments_gc_data=segments_gc_data,
        plot_figures=plot_figures,
        skip_gc_failures=True,
        intersection_only=True,
        model_predictions_smiles_column=parameter_dfs_smiles_column,
        experimental_data_smiles_column=experimental_data_smiles_column,
        experimental_data_name_column=experimental_data_name_column,
        experimental_data_density_column=experimental_data_density_column,
        experimental_temperature_column=experimental_temperature_column,
        experimental_pressure_column=experimental_pressure_column,
        return_all_data=return_all_data,
    )
    return figs, scores, all_predictions, groups


def make_thermo_results_table(
    thermo_scores: Dict[str, Union[Callable[[], pd.DataFrame], pd.DataFrame]],
    groups: Optional[Dict[str, List[str]]] = None,
    mark_bold: Optional[Literal["max", "min"]] = "min",
    intersection_only: bool = True,
    show_std: bool = True,
    metric: str = "mape",
    caption: Optional[str] = None,
    label: Optional[str] = None,
    only_tabular: bool = False,
):
    """
    Make a LaTex table of results

    Arguments
    ---------
    thermo_scores: dict
        Dictionary of thermo scores. Keys are the names of the models and values are
        either functions that return a DataFrame or DataFrames
    groups: dict, optional
        Dictionary of groups. Keys are the names of the groups and values are lists of
        the names of the models in the group
    mark_bold: "min" or "max", optional
        Whether to mark the max or min value for each row in the table with bold text
        if None, no values will be marked. Default is "min".
    intersection_only: bool
        Whether to only include data points that are in all datasets. Default is True.
    show_std: bool
        Whether to show the standard deviation of the scores. Default is True.
    metric: str
        Which metric to use for the table. Default is "mape".
    caption: str, optional
        A string that adds a caption to the Latex formatting.
    label: str, optional
        A string that adds a referencing label to the Latex formatting.
    only_tabular: bool, optional
        Whether to only return the tabular part of the Latex table. Default is False.

    """
    logger = logging.getLogger(__name__)
    groups = {} if groups is None else groups

    # Each group will have as key what should be the index in the table
    # and as value a list of the names of the models in the group

    # Score dfs
    scores_dfs = {
        name: score_func() if callable(score_func) else score_func
        for name, score_func in thermo_scores.items()
    }

    # Only include data points that are in all datasets
    if intersection_only:
        score_dfs_list = list(scores_dfs.values())
        smiles = set(score_dfs_list[0]["smiles"])
        for scores_df in score_dfs_list[1:]:
            smiles = smiles.intersection(set(scores_df["smiles"]))
        for k, scores_df in scores_dfs.items():
            scores_dfs[k] = scores_df[scores_df["smiles"].isin(smiles)]

    # Create summary dataframe
    df_dict: Dict[str, List] = {
        "n": [],
        "model": [],
    }
    targets = ["pvap", "rho"]
    for target in targets:
        df_dict[target] = []
        df_dict[f"{target}_std"] = []
    if len(groups) > 1:
        for i, (model, ids) in enumerate(groups.items()):
            # Number of data points per model
            n = scores_dfs[ids[0]].shape[0]
            df_dict["n"].append(n)
            df_dict["model"].append(model)

            # Average scores for each model in the group
            for target in targets:
                scores_row = np.array(
                    [scores_dfs[id_][f"{target}_{metric}"].mean() for id_ in ids]
                )
                # Average scores for the group
                avg = np.mean(scores_row)
                std = np.std(scores_row)
                df_dict[target].append(avg)
                df_dict[f"{target}_std"].append(std)
    else:
        for i, (model, scores_df) in enumerate(scores_dfs.items()):
            n = scores_df.shape[0]
            df_dict["n"].append(n)
            df_dict["model"].append(model)

            for target in targets:
                avg = scores_df[f"{target}_{metric}"].mean()
                df_dict[target].append(avg)
                df_dict[f"{target}_std"].append(np.nan)
    display_df = pd.DataFrame(df_dict)
    display_df = display_df.sort_values(by="pvap", ascending=False)
    display_df = display_df.rename(columns={"Sepp": "SEPP", "gc": "GC"})

    # Make rows
    def format_row(
        target: str,
        target_label: str,
        display_df: pd.DataFrame,
        show_std: bool,
    ):
        r = [target_label]
        for _, row in display_df.iterrows():
            score = row[target]
            std = row[f"{target}_std"] if f"{target}_std" in row.index else np.nan
            if show_std and not np.isnan(std):
                r.append(f"{score*100:.02f}±{std*100:.02f}")
            elif type(score) == float:
                r.append(f"{score*100:.02f}")
            else:
                r.append(str(score))
        return r

    rows: List[List] = []
    rows.append(format_row("model", "", display_df, False))
    rows.append(format_row("n", "n", display_df, False))
    rows.append(format_row("pvap", r"\%AAD $p_{sat}$", display_df, show_std))
    rows.append(format_row("rho", r"\%AAD $\rho^{L}$", display_df, show_std))

    # Make best score bold
    if mark_bold is not None and not (groups and show_std):
        display_df_filtered = display_df[
            ~(display_df["model"].str.lower() == "regressed")
        ]
        for i, target in enumerate(targets):
            col_offset: int = 1
            if mark_bold == "min":
                bold_score_idx = display_df_filtered[target].argmin()
            elif mark_bold == "max":
                bold_score_idx = display_df_filtered[target].argmax()
            else:
                raise ValueError(f"mark_bold must be 'min' or 'max', not {mark_bold}")
            rows[i + 2][bold_score_idx + col_offset] = (  # type: ignore
                "\\textbf{" + str(rows[i + 2][bold_score_idx + col_offset]) + "}"  # type: ignore
            )  # type: ignore

    # Draw table
    n_cols = len(groups) + 1 if len(groups) > 1 else len(thermo_scores) + 1
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(["t"] * n_cols)
    table.set_cols_align(["l"] + ["r"] * (n_cols - 1))
    table.add_rows(rows, header=True)
    logger.info("\nASCII representation of table\n" + table.draw())  # type: ignore

    if only_tabular:
        latex_code = latextable.draw_latex(table)
        return "\n".join(latex_code.splitlines()[2:-2])
    else:
        return latextable.draw_latex(
            table, caption=caption, caption_above=True, label=label
        )


def plot_counterfactuals(args, df_counterfactuals, target):
    os.makedirs(f"{args['save_dir']}", exist_ok=True)
    for cf_i in range(args["num_counterfactuals_to_plot"]):
        cf_row = df_counterfactuals.iloc[[cf_i]]
        smiles_pair = cf_row["SMILES-Pair"].item().rsplit("_")
        value_pair = cf_row["Value-Pair"].item().rsplit("_")
        conterfactual_plot(
            smiles_1=smiles_pair[0],
            value_1=value_pair[0],
            smiles_2=smiles_pair[1],
            value_2=value_pair[1],
            save_loc=args["save_dir"],
            target=target,
        )


def get_counterfactuals(args, df_results, df_compare_cfs=None):
    os.makedirs(f"{args['save_dir']}", exist_ok=True)
    for target in args["target_columns"]:
        y_pred = df_results[f"Pred_{target}"].to_numpy()
        smiles = df_results[args["smiles_column"]].to_list()
        # provide external search list for counterfactual candidates
        if df_compare_cfs is not None:
            cf_values = df_compare_cfs[f"Pred_{target}"].to_numpy()
            cf_smiles = df_compare_cfs[args["smiles_column"]].to_list()
        # search for counterfactuals within prediction list
        else:
            cf_values = y_pred
            cf_smiles = smiles

        (
            df_all_TYPE_simMol_diffValue,
            df_all_TYPE_diffMol_simValue,
            df_all_TYPE_max_simMol_diffValue,
            df_all_TYPE_min_simMol_diffValue,
        ) = counterfactual.counterfactual_analysis(
            smiles=smiles,
            predictions=y_pred,
            compare_smiles=cf_smiles,
            compare_values=cf_values,
            type_simMol_diffValue_lb_value_diff=args[
                "type_simMol_diffValue_lb_value_diff"
            ],
            type_diffMol_simValue_ub_value_diff=args[
                "type_diffMol_simValue_ub_value_diff"
            ],
            type_max_simMol_diffValue_lambda=args["type_max_simMol_diffValue_lambda"],
            type_min_simMol_diffValue_lambda=args["type_min_simMol_diffValue_lambda"],
        )

        if df_all_TYPE_simMol_diffValue is not None:
            plot_counterfactuals(args, df_all_TYPE_simMol_diffValue, target)
            df_all_TYPE_simMol_diffValue.to_csv(
                f"{args['save_dir']}/cf_simMol_diffValue_{target}.csv"
            )
        if df_all_TYPE_diffMol_simValue is not None:
            plot_counterfactuals(args, df_all_TYPE_diffMol_simValue, target)
            df_all_TYPE_diffMol_simValue.to_csv(
                f"{args['save_dir']}/cf_diffMol_simValue_{target}.csv"
            )
        if df_all_TYPE_max_simMol_diffValue is not None:
            plot_counterfactuals(args, df_all_TYPE_max_simMol_diffValue, target)
            df_all_TYPE_max_simMol_diffValue.to_csv(
                f"{args['save_dir']}/cf_max_simMol_diffValue_{target}.csv"
            )
        if df_all_TYPE_min_simMol_diffValue is not None:
            plot_counterfactuals(args, df_all_TYPE_min_simMol_diffValue, target)
            df_all_TYPE_min_simMol_diffValue.to_csv(
                f"{args['save_dir']}/cf_min_simMol_diffValue_{target}.csv"
            )
            plot_counterfactuals(args, df_all_TYPE_min_simMol_diffValue, target)
            df_all_TYPE_min_simMol_diffValue.to_csv(
                f"{args['save_dir']}/cf_min_simMol_diffValue_{target}.csv"
            )
