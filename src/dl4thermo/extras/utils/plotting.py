from typing import Callable, Dict, List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm, ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS  # type: ignore
from scipy.interpolate import make_interp_spline

from dl4thermo.extras.utils.metrics import calculate_metrics

score_display = {"r2": "$R^2$", "mae": "MAE", "rmse": "RMSE", "mse": "MSE"}


def parity_plot(
    y,
    yhat,
    yerr: Optional[Union[List[float], np.ndarray]] = None,
    ax: Optional[Axes] = None,
    include_parity: bool = True,
    scores: Optional[Dict[str, float]] = None,
    label: Optional[str] = None,
    quantity_name: Optional[str] = None,
    **plot_kwargs,
) -> Axes:
    """Make a parity plot
    Parameters
    ----------
    y : array-like
        Measured values
    yhat : array-like
        Predicted values
    ax : Axes, optional
        Matplotlib axes object, by default None
    include_parity : bool, optional
        Whether to include a parity line, by default True
    scores : Dict[str, float], optional
        A dictionary with scores to display in the legend, by default None
    label : str, optional
        Label for the scatter plot, by default None
    quantity_name : str, optional
        Name of the quantity being plotted. Used for axis labels.
    Returns
    -------
    A matplotlib axes object
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if scores is not None:
        score_label = ", ".join(
            [
                f"{score_display.get(score_name, score_name)}={score:.02f}"
                for score_name, score in scores.items()
            ]
        )
        full_label = f"{label} ({score_label})"
    else:
        full_label = label

    # Scatter plot
    # ax.scatter(y, yhat, label=full_label, alpha=alpha)
    alpha = plot_kwargs.pop("alpha", 0.5)
    if yerr is None:
        sns.scatterplot(
            x=y, y=yhat, ax=ax, label=full_label, alpha=alpha, **plot_kwargs
        )
    else:
        ax.errorbar(
            x=y,
            y=yhat,
            yerr=yerr,
            fmt="o",
            label=full_label,
            alpha=alpha,
            **plot_kwargs,
        )

    # Parity line
    if include_parity:
        combined = np.vstack((y, yhat))
        min_y = np.min(combined)
        max_y = np.max(combined)
        ax.plot([min_y, max_y], [min_y, max_y], "k--")

    # Formatting
    if label and not quantity_name:
        quantity_name = label
    quantity_name = quantity_name or ""
    ax.set_ylabel(f"Predicted {quantity_name}")
    ax.set_xlabel(f"Measured {quantity_name}")
    ax.tick_params(direction="in")
    if scores is not None:
        ax.legend()
    return ax


def parity_plot_grid(
    dfs: Dict[str, pd.DataFrame],
    targets: Dict[str, List[str]],
    target_display_names: Optional[Dict[str, str]] = None,
    x_axis_label_prefix: Optional[str] = "Experimental",
    y_axis_label_prefix: Optional[str] = "Predicted",
) -> Dict[str, Figure]:
    target_display_names = target_display_names or {}
    # Make parity plots
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
                f"{x_axis_label_prefix} {display_name}",
                f"{y_axis_label_prefix} {display_name}",
                fontsize=15,
            )
            g.ax_joint.plot([min_y, max_y], [min_y, max_y], "--k")
            g.figure.tight_layout()
            figs[f"{model_name}_{target}"] = g
    return figs


def get_functional_group_counts(
    smiles: List[str], fragment_lookup: Callable, clip: bool = True
) -> pd.Series:
    # Create mols and get functional groups
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]  # type: ignore
    functional_groups = [fragment_lookup(mol) for mol in mols]
    functional_group_df = pd.DataFrame(functional_groups)
    if clip:
        functional_group_df = functional_group_df.clip(upper=1)
    sums = functional_group_df[functional_group_df > 0].count()
    sums = sums.sort_values(ascending=False)
    return sums


def functional_group_distribution_plot(
    data: pd.DataFrame,
    smiles_column: str,
    fragments_data: pd.DataFrame,
    plot_top_k: Optional[int] = 10,
    group_column: Optional[str] = None,
    frequency_plot: bool = True,
    clip: bool = True,
) -> Figure:
    """Plot the distribution of functional groups in a dataset

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the SMILES strings
    smiles_column : str
        Column name of the SMILES strings
    fragments_data : pd.DataFrame
        Dataframe containing the SMARTS strings for the functional groups
    plot_top_k : Optional[int], optional
        Plot the top k functional groups, by default 10
    group_column : Optional[str], optional
        Column name of the group column, by default None
    frequency_plot : bool, optional
        Plot the frequency of the functional groups, by default True
    clip : bool, optional
        Clip to the number of instances of a functional group in a molecule to 1
        By default True

    Returns
    -------
    A matplotlib figure


    """
    # Set up fragments
    fragments = {
        row["name"]: Chem.MolFromSmarts(row["smarts"])  # type: ignore
        for _, row in fragments_data.iterrows()
    }
    _get_functional_groups = lambda mol: {
        label: len(mol.GetSubstructMatches(fragment))
        for label, fragment in fragments.items()
    }

    # Get counts
    if group_column:
        counts_dfs = []
        groups = data[group_column].unique()
        for group in groups:
            group_data = data[data[group_column] == group]
            counts_group = get_functional_group_counts(
                group_data[smiles_column].astype(str).tolist(),
                _get_functional_groups,
                clip=clip,
            )
            if frequency_plot:
                counts_group = counts_group / len(group_data)
            counts_group = counts_group.to_frame().rename(columns={0: "counts"})
            counts_group[group_column] = group
            counts_dfs.append(counts_group)
        counts = pd.concat(counts_dfs)
    else:
        counts = get_functional_group_counts(
            data[smiles_column].astype(str).tolist(), _get_functional_groups
        )
        if frequency_plot:
            counts = counts / len(data)
        counts = counts.to_frame().rename(columns={0: "counts"})
    counts = counts.reset_index().rename(columns={"index": "Functional group"})

    # Filter
    if plot_top_k:
        groups = counts.groupby("Functional group").mean()
        top_groups = (
            groups.sort_values(by="counts", ascending=False)
            .iloc[:plot_top_k]
            .index.tolist()
        )
        counts = counts[counts["Functional group"].isin(top_groups)]

    # Make figure
    g = sns.catplot(
        data=counts,
        kind="bar",
        y="Functional group",
        x="counts",
        palette="dark",
        hue=group_column if group_column else None,
        orient="h",
        alpha=0.6,
        height=6,
        legend=False,
    )
    # g.ax.set_xticklabels(
    #     g.ax.get_xticklabels(), rotation=45, horizontalalignment="right"
    # )
    g.ax.tick_params(axis="both", labelsize=15)
    g.set_axis_labels(
        "Frequency" if frequency_plot else "Counts",
        "Functional group",
        fontsize=15,
    )
    g.ax.legend(loc="lower right", fontsize=15, title_fontsize=20)
    g.fig.tight_layout()
    return g.figure


def conterfactual_plot(smiles_1, value_1, smiles_2, value_2, save_loc, target):
    mol1 = Chem.MolFromSmiles(smiles_1)  # type: ignore
    mol2 = Chem.MolFromSmiles(smiles_2)  # type: ignore

    Draw.MolsToGridImage([mol1, mol2])

    # view difference from: https://www.rdkit.org/docs/Cookbook.html and https://gist.github.com/iwatobipen/6d8708d8c77c615cfffbb89409be730d
    def view_difference(mol1, mol2):
        mcs = rdFMCS.FindMCS([mol1, mol2])
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)  # type: ignore
        match1 = mol1.GetSubstructMatch(mcs_mol)
        target_atm1 = []
        for atom in mol1.GetAtoms():
            if atom.GetIdx() not in match1:
                target_atm1.append(atom.GetIdx())
        match2 = mol2.GetSubstructMatch(mcs_mol)
        target_atm2 = []
        for atom in mol2.GetAtoms():
            if atom.GetIdx() not in match2:
                target_atm2.append(atom.GetIdx())
        vals = [
            f"SMILES: {smiles_1} \nPrediction: {value_1}",
            f"SMILES: {smiles_2} \nPrediction: {value_2}",
        ]
        img = Draw.MolsToGridImage(
            [mol1, mol2],
            highlightAtomLists=[target_atm1, target_atm2],
            legends=vals,
            subImgSize=(300, 300),
            molsPerRow=2,
        )
        with open(
            f"{save_loc}/cf_{target}_{smiles_1.replace('/', '').replace(chr(92), '')}_{smiles_2.replace('/', '').replace(chr(92), '')}",
            "w",
        ) as f_handle:
            f_handle.write(img.data)  # type: ignore

    view_difference(mol1, mol2)


def parallel_plot(
    df: pd.DataFrame,
    cols: List[str],
    color_col: str,
    log_cols: Optional[List[str]] = None,
    cmap="Spectral",
    spread=None,
    curved: bool = False,
    curvedextend: float = 0.1,
    alpha: float = 0.4,
    categorical_cutoff: int = 10,
    highlight_idx: Optional[List[int]] = None,
    pretty_print_labels: bool = True,
):
    """Produce a parallel coordinates plot from pandas dataframe with line colour with respect to a column.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to plot
    cols: List[str]
        Columns to use for axes
    color_col: str
        Column to use for colorbar
    cmap: str
        Colour palette to use for ranking of lines
    spread:
        Spread to use to separate lines at categorical values
    curved: bool, optional
        Spline interpolation along lines. Default is False
    curvedextend: float, optional
        Fraction extension in y axis, adjust to contain curvature. Default is 0.1

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure object
    axes: matplotlib.axes.Axes
        Axes object

    Notes
    -----
    Copied directly from: https://github.com/jraine/parallel-coordinates-plot-dataframe

    """
    colmap = cm.get_cmap(cmap)
    cols = cols + [color_col]
    log_cols = log_cols or []
    cols = [col for col in cols if df[col].nunique() > 1]
    skipped = [col for col in cols if df[col].nunique() == 1]

    fig, axes = plt.subplots(
        1, len(cols) - 1, sharey=False, figsize=(1.5 * len(cols) + 3, 5)
    )
    valmat = np.ndarray(shape=(len(cols), len(df)))
    x = np.arange(0, len(cols), 1)
    ax_info = {}
    for i, col in enumerate(cols):
        vals = df[col]
        n_unique = len(np.unique(vals))
        if ((vals.dtype == float) or (vals.dtype == int)) & (
            n_unique > categorical_cutoff
        ):
            dtype = vals.dtype
            if col in log_cols:
                vals = np.log(vals)
            minval = np.min(vals)
            maxval = np.max(vals)
            vals = np.true_divide(vals - minval, maxval - minval)
            rangeval = maxval - minval
            nticks = 5
            if rangeval < 1:
                rounding = 2
            elif rangeval > 1 and rangeval < 10:
                rounding = 0
            elif rangeval > 10 and rangeval < 100:
                rounding = -1
            elif rangeval > 100 and rangeval < 1000:
                rounding = -2
            else:
                rounding = 4
            if dtype == float and col not in log_cols:
                tick_labels = [
                    round(minval + i * (rangeval / nticks), rounding)
                    for i in range(nticks + 1)
                ]
            elif dtype == int and col not in log_cols:
                tick_labels = [
                    str(int(minval + i * (rangeval // nticks)))
                    for i in range(nticks + 1)
                ]
            else:
                tick_labels = [
                    "{:.0e}".format(np.exp(minval + i * (rangeval / nticks)))
                    for i in range(nticks + 1)
                ]

            # tick_labels = clean_axis_labels(tick_labels)
            ticks = [0 + i * (1.0 / nticks) for i in range(nticks + 1)]
            valmat[i] = vals
            ax_info[col] = [tick_labels, ticks]
        else:
            vals = vals.astype("category")
            cats = vals.cat.categories
            c_vals = vals.cat.codes
            minval = -0.5
            maxval = len(cats) - 0.5
            if maxval == 0:
                c_vals = 0.5
            else:
                c_vals = np.true_divide(c_vals - minval, maxval - minval)
            tick_labels = cats
            ticks = np.unique(c_vals)
            ax_info[col] = [tick_labels, ticks]
            if spread is not None:
                offset = np.arange(-1, 1, 2.0 / (len(c_vals))) * 2e-2  # type: ignore
                np.random.shuffle(offset)
                c_vals = c_vals + offset
            valmat[i] = c_vals

    extendfrac = curvedextend if curved else 0.05
    grey = "#454545"
    for i, ax in enumerate(axes):
        # remove_frame(ax, sides=["top", "bottom"])
        set_axis_color(ax, color=grey)
        ax.tick_params(colors=grey, which="both")
        for idx in range(valmat.shape[-1]):
            if curved:
                x_new = np.linspace(0, len(x), len(x) * 20)
                a_BSpline = make_interp_spline(
                    x, valmat[:, idx], k=3, bc_type="natural"
                )
                y_new = a_BSpline(x_new)
                if highlight_idx is None:
                    color = colmap(valmat[-1, idx])
                else:
                    if idx in highlight_idx:
                        color = colmap(valmat[-1, idx])
                    else:
                        color = "#DBDBDB"
                ax.plot(x_new, y_new, color=color, alpha=alpha)
            else:
                ax.plot(x, valmat[:, idx], color=colmap(valmat[-1, idx]), alpha=alpha)
        ax.set_ylim(0 - extendfrac, 1 + extendfrac)
        ax.set_xlim(i, i + 1)

    for dim, (ax, col) in enumerate(zip(axes, cols)):
        if col in skipped:
            continue
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        ax.yaxis.set_major_locator(ticker.FixedLocator(ax_info[col][1]))
        ax.set_yticklabels(ax_info[col][0])
        txt = cols[dim]
        if pretty_print_labels:
            txt = txt.replace("_", " ").title()
        ax.set_xticklabels([txt])

    plt.subplots_adjust(wspace=0)
    norm = mpl.colors.Normalize(0, 1)  # *axes[-1].get_ylim())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(
        sm,
        pad=0,
        ticks=ax_info[color_col][1],
        extend="both",
        extendrect=True,
        extendfrac=extendfrac,
    )
    cbar.ax.set_yticklabels(ax_info[color_col][0])
    txt = color_col
    if pretty_print_labels:
        txt = txt.replace("_", " ").title()
    cbar.ax.set_xlabel(txt, labelpad=30.0, color=grey)

    return fig, axes


def plot_pressure_density_phase_diagram(
    predicted_dfs: Dict[str, pd.DataFrame],
    experimental_data: Optional[pd.DataFrame] = None,
    name: Optional[str] = None,
    params_dict: Optional[dict] = None,
    experimental_liquid_density_column: str = "DEN",
    experimental_vapor_density_column: Optional[str] = None,
    temperature_column: str = "T",
    pressure_column: str = "P",
):
    l_den = experimental_liquid_density_column
    v_den = experimental_vapor_density_column
    T = temperature_column
    P = pressure_column
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    if params_dict is not None:
        params_str = ""
        for model, params_record in params_dict.items():
            params = {
                r"$m$": params_record.m,
                r"$\sigma$": params_record.sigma,
                r"$\epsilon/k$": params_record.epsilon_k,
                r"$\mu$": params_record.mu,
                r"$\epsilon_{AB}$": params_record.epsilon_k_ab,
                r"$\kappa_{AB}$": params_record.kappa_ab,
            }
            params = {
                k: f"{v:.02f}" if v is not None else "N/A" for k, v in params.items()
            }
            params_str += f"{model}:\t"
            params_str += r"  ".join(
                [f"{param}={val}" for param, val in params.items()]
            )
            params_str += "\n"
        params_str = params_str.rstrip("\n")
        props = dict(boxstyle="round", facecolor="white", alpha=0.5)
        fig.text(
            0.5,
            1.25,
            params_str,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="center",
            bbox=props,
        )

    colors = ["#97A85E", "#263B5C", "#A8574D", "#9E8557", "#896273"]
    lw = 6
    name = name or ""
    if name is not None:
        ax[0].set_title(f"Saturation pressure of {name}")
    for i, (predictor, predicted_df) in enumerate(predicted_dfs.items()):
        predicted_df = predicted_df.sort_values(by=T, ascending=True)
        sns.lineplot(
            y=predicted_df[P],
            x=1000.0 / predicted_df[T],
            ax=ax[0],
            label=predictor,
            color=colors[i],
            linewidth=lw,
        )
    if experimental_data is not None:
        pvap_data = experimental_data[experimental_data[l_den].isna()]
        sns.scatterplot(
            y=pvap_data[P].astype(float),
            x=1000.0 / pvap_data[T].astype(float),
            ax=ax[0],
            color="k",
            label="Experiments",
            s=200,
        )

    # axis and styling
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\frac{1000}{T}$ / K$^{-1}$", fontsize=20)
    ax[0].set_ylabel(r"$p$ / kPa", fontsize=20)
    ax[0].xaxis.set_tick_params(labelsize=15)
    ax[0].yaxis.set_tick_params(labelsize=15)
    ax[0].legend(fontsize=20)

    # Density predictions
    if name is not None:
        ax[1].set_title(r"$T$-$\rho$-diagram of {}".format(name))
    for i, (predictor, predicted_df) in enumerate(predicted_dfs.items()):
        # sns.lineplot(
        #     data=predicted_df,
        #     y=T,
        #     x="density vapor",
        #     ax=ax[1],
        #     label=predictor + " (vapor)",
        #     color=colors[i],
        #     linewidth=lw,
        # )
        sns.lineplot(
            data=predicted_df,
            y=T,
            x="density liquid",
            ax=ax[1],
            label=predictor + " (liquid)",
            linestyle="--",
            color=colors[i],
            linewidth=lw,
        )
    if experimental_data is not None:
        sns.scatterplot(
            y=experimental_data[T].astype(float),
            x=experimental_data[l_den].astype(float),
            ax=ax[1],
            color="k",
            label="Experiments",
            s=200,
        )
    if experimental_data is not None and v_den is not None:
        sns.scatterplot(
            y=experimental_data[T].astype(float),
            x=experimental_data[v_den].astype(float),
            ax=ax[1],
            color="k",
            label="Experiments",
            s=200,
        )

    # axis and styling
    ax[1].set_ylabel(r"$T$ / K", fontsize=20)
    ax[1].set_xlabel(r"$\rho$ / kg/m³", fontsize=20)
    ax[1].xaxis.set_tick_params(labelsize=15)
    ax[1].yaxis.set_tick_params(labelsize=15)
    # ax[1].legend
    ax[1].get_legend().remove()  # type: ignore
    # sns.despine(offset=10)
    fig.tight_layout()
    return fig


def sensitivity_boxplot(pvap_df: pd.DataFrame, rho_df: pd.DataFrame) -> Figure:
    """Make a boxplot of the sensitivity indices"""
    fig, axes = plt.subplots(2, 1, figsize=(4, 10))
    # fig.subplots_adjust(wspace=0.25)
    target_display = {
        "m": r"$m$",
        "sigma": r"$\sigma$",
        "epsilon_k": r"$\epsilon/k$",
        "mu": r"$\mu$",
        "KAB": r"$\kappa_{AB}$",
        "epsilonAB": r"$\epsilon_{AB}$",
    }
    for i, (label, df) in enumerate(zip([r"$p_{sat}$", r"$\rho$"], [pvap_df, rho_df])):
        ax = axes[i]
        df_melt = pd.melt(df.drop("smiles", axis=1))
        df_melt = df_melt.replace(target_display)
        sns.boxplot(data=df_melt, x="variable", y="value", ax=ax)
        ax.set_xlabel("PCP-SAFT parameter")
        ax.set_ylabel(f"Sensitivity index of {label} MAPE")
    fig.tight_layout()
    return fig


def remove_frame(ax, sides=["top", "left", "right"]):
    for side in sides:
        ax_side = ax.spines[side]
        ax_side.set_visible(False)


def remove_spines(ax, sides=["top", "left", "right"]):
    for side in sides:
        ax_side = ax.spines[side]
        ax_side.set_visible(False)


def set_axis_color(ax, sides=["bottom", "left", "top", "right"], color="#D1D1D1"):
    for side in sides:
        ax_side = ax.spines[side]
        ax_side.set_color(color)
