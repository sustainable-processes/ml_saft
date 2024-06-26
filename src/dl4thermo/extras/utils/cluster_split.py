import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import molplotly
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from umap import UMAP

from dl4thermo.extras.utils.plotting import remove_frame, remove_spines

logger = logging.getLogger(__name__)


def train_test_cluster_split(
    fps: np.ndarray,
    valid_size: Optional[float] = None,
    test_size: Optional[float] = None,
    kmeans_args: Optional[Dict[str, Any]] = None,
    umap_before_cluster: bool = True,
    umap_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train, validation, and test split based on k-means clustering of molecular fingerprints.

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        Array-like of morgan fingerprint vectors
    test_size : float
        Fraction of data to use for testing.
    valid_size : float, optional
        Fraction of data to use for validation. If none, returns empty
        array for validation indices.
    kmeans_args : dict, optional
        Keyword arguments to pass to KMeans.

    Returns
    -------
    train_idx : array-like of shape (n_train_samples,)
        Indices of training data.
    valid_idx : array-like of shape (n_valid_samples,)
        Indices of validation data.
    test_idx : array-like of shape (n_test_samples,)
        Indices of test data.
    cluster_labels : array-like of shape (n_samples,)
        Cluster labels for each data point.

    """
    # K-means clustering
    logger.info("Clustering data...")
    kmeans_kwargs = kmeans_args if kmeans_args is not None else {}
    if umap_before_cluster:
        if umap_kwargs is None:
            umap_kwargs = {}
        if "n_components" not in umap_kwargs:
            umap_kwargs["n_components"] = 5
        if "n_neighbors" not in umap_kwargs:
            umap_kwargs["n_neighbors"] = 15
        if "min_dist" not in umap_kwargs:
            umap_kwargs["min_dist"] = 0.1
        if "metric" not in umap_kwargs:
            umap_kwargs["metric"] = "jaccard"
        if "random_state" not in umap_kwargs:
            umap_kwargs["random_state"] = 0
        reducer = UMAP(**umap_kwargs)
        X: np.ndarray = reducer.fit_transform(fps)  # type: ignore
        num_nan = np.isnan(X).any(axis=1).sum()
        if num_nan > 0:
            raise ValueError("UMAP returned NaN values.")
    else:
        X = fps
    kmeans = KMeans(**kmeans_kwargs)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_ + 1  # type: ignore

    # Split clusters
    if test_size is None and valid_size is not None:
        n_test = int(valid_size * fps.shape[0])
    elif test_size is not None:
        n_test = int(test_size * fps.shape[0])
    else:
        raise ValueError("Must specify either test_size or valid_size")
    splitter_test = StratifiedShuffleSplit(
        n_splits=1, test_size=n_test, random_state=10
    )
    train_valid_indx, test_idx = next(
        splitter_test.split(fps, y=cluster_labels),
    )
    if valid_size and test_size:
        n_valid = int(valid_size * fps.shape[0])
        splitter_valid = StratifiedShuffleSplit(
            n_splits=1, test_size=n_valid, random_state=10
        )
        train_idx, valid_idx = next(
            splitter_valid.split(
                fps[train_valid_indx], y=cluster_labels[train_valid_indx]
            )
        )
    elif valid_size and not test_size:
        train_idx = train_valid_indx
        valid_idx = test_idx
        test_idx = np.array([])
    else:
        train_idx = train_valid_indx
        valid_idx = np.array([])
    return train_idx, valid_idx, test_idx, cluster_labels


def visualize_clusters_umap(
    fps: np.ndarray,
    cluster_labels: np.ndarray,
    label_data: Optional[pd.DataFrame] = None,
    label_columns: Optional[List[str]] = None,
    umap_kwargs: Optional[Dict[str, Any]] = None,
    plot_type: Literal["matplotlib", "plotly"] = "matplotlib",
    smiles_column: Optional[str] = None,
    split_idx: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Optional[Union[Figure, Tuple[Figure, Figure, Figure, Figure]]]:
    """Visualize clusters using UMAP dimensionality reduction.

    Parameters
    ----------
    fps : array-like of shape (n_samples, n_features)
        Array-like of morgan fingerprint vectors
    data : pd.DataFrame
        Dataframe containing cluster labels.

    """
    # UMAP
    if umap_kwargs is None:
        umap_kwargs = {}
    if "n_components" not in umap_kwargs:
        umap_kwargs["n_components"] = 2
    if "n_neighbors" not in umap_kwargs:
        umap_kwargs["n_neighbors"] = 15
    if "min_dist" not in umap_kwargs:
        umap_kwargs["min_dist"] = 0.1
    if "metric" not in umap_kwargs:
        umap_kwargs["metric"] = "jaccard"
    if "random_state" not in umap_kwargs:
        umap_kwargs["random_state"] = 0
    logger.debug("UMAP dimensionality reduction")
    reducer = UMAP(**umap_kwargs)
    X: np.ndarray = reducer.fit_transform(fps)  # type: ignore

    # Plot
    if plot_type == "matplotlib":
        fig = make_clusters_plot_matplotlib(X=X, cluster_labels=cluster_labels)
        if split_idx is not None:
            split_figs = (
                make_clusters_plot_matplotlib(
                    X=X[split],
                    cluster_labels=cluster_labels[split],
                )
                for split in split_idx
            )

            return fig, *split_figs
        else:
            return fig
    elif plot_type == "plotly":
        if label_data is None:
            raise ValueError("label_data must be provided if plot_type is plotly")

        make_clusters_plot_plotly(
            X=X,
            cluster_labels=cluster_labels,
            label_data=label_data,
            label_columns=label_columns,
            smiles_column=smiles_column,
        )


def make_clusters_plot_matplotlib(
    X: np.ndarray,
    cluster_labels: np.ndarray,
) -> Figure:
    if X.shape[1] == 2:
        fig, ax = plt.subplots(1)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        data = pd.DataFrame(X, columns=["UMAP 1", "UMAP 2"])
        data["Cluster"] = cluster_labels
        sns.scatterplot(
            data=data,
            x="UMAP 1",
            y="UMAP 2",
            hue="Cluster",
            ax=ax,
        )
        ax.legend(title="Cluster", fontsize=15, title_fontsize=20)
    elif X.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_labels)
        ax.legend()
    else:
        raise ValueError("X must be 2 or 3 dimensional")
    ax.set_xlabel("UMAP 1", fontsize=20)
    ax.set_ylabel("UMAP 2", fontsize=20)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    fig.tight_layout()
    remove_frame(ax, sides=["top", "right"])
    remove_spines(ax, sides=["top", "right"])
    return fig


def make_clusters_plot_plotly(
    X: np.ndarray,
    cluster_labels: Union[np.ndarray, pd.Series],
    label_data: pd.DataFrame,
    label_columns: Optional[List[str]] = None,
    smiles_column: Optional[str] = None,
):
    if X.shape[1] == 2:
        fig_scatter = px.scatter(
            label_data,
            x=X[:, 0],
            y=X[:, 1],
            color=cluster_labels,
            title="UMAP",
            width=800,
            height=600,
        )
    elif X.shape[1] == 3:
        fig_scatter = px.scatter_3d(
            label_data,
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            color=cluster_labels,
            title="UMAP",
            width=800,
            height=600,
        )
    else:
        raise ValueError("X must be 2 or 3 dimensional.")
    if smiles_column is None or label_columns is None:
        raise ValueError("smiles_column and label_columns must be provided")
    app_scatter = molplotly.add_molecules(
        fig=fig_scatter,
        df=label_data,
        smiles_col=smiles_column,
        caption_cols=label_columns if label_columns is not None else [],
        caption_transform={
            col: lambda x: f"{x:.02f}"
            for col in label_columns
            if label_data[col].dtype == float
        },
        show_coords=False,
    )
    app_scatter.run_server(mode="external", port=8001, height=1000)
