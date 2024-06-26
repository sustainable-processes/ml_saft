from typing import Dict, List, Literal

import matplotlib.pyplot as plt
import torchmetrics
from scipy.stats import linregress, spearmanr
from sklearn import metrics

SCORE_NAMES = ["mae", "mse", "rmse", "mape", "r2", "maxe", "expl_var"]


def rsquared(x, y):
    """Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2


def calculate_metrics(
    y_true, y_pred, scores: List[str] = SCORE_NAMES
) -> Dict[str, float]:
    """Calculate metrics on a given dataset."""

    def _get_score(score):
        # calculate metric values
        if score == "mae":
            return metrics.mean_absolute_error(y_true, y_pred)
        if score == "mse":
            return metrics.mean_squared_error(y_true, y_pred)
        if score == "rmse":
            return metrics.mean_squared_error(y_true, y_pred) ** (1 / 2)
        if score == "mape":
            return metrics.mean_absolute_percentage_error(y_true, y_pred)
        if score == "r2":
            try:
                return rsquared(y_true, y_pred)
            except ValueError:
                return metrics.r2_score(y_true, y_pred)
        if score == "maxe":
            return [
                metrics.max_error(y_true[:, i], y_pred[:, i])
                for i in range(y_true.shape[1])
            ]
        if score == "expl_var":
            return metrics.explained_variance_score(y_true, y_pred)
        if score == "spearman":
            return spearmanr(y_true, y_pred).correlation  # type: ignore

    result_metric = {}
    for s in scores:
        result_metric[s] = _get_score(s)

    return result_metric


def get_torchmetrics(metric_name: str):
    metrics = {
        "mae": torchmetrics.MeanAbsoluteError(),
        "mse": torchmetrics.MeanSquaredError(),
        "r2": torchmetrics.R2Score(),
        "mape": torchmetrics.MeanAbsolutePercentageError(),
        "binary_accuracy": torchmetrics.Accuracy(task="binary"),
        "binary_aucroc": torchmetrics.AUROC(task="binary"),
        "binary_f1": torchmetrics.F1Score(task="binary"),
    }
    if metric_name not in metrics:
        raise ValueError(f"Metric {metric_name} not supported.")
    return metrics[metric_name]
