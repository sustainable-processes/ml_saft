import logging
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from tap import Tap
from wandb.sdk.wandb_run import Run

import wandb
from dl4thermo.extras.utils.metrics import calculate_metrics
from dl4thermo.extras.utils.plotting import parity_plot

logger = logging.getLogger(__name__)

from .base import TrainArgs


@dataclass
class SklearnTrainArgs(TrainArgs):
    model_type: str = "RF"
    wandb_artifact_name: str = "sklearn"
    num_trees: int = 100


class Estimator(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        pass

    def predict(self, X) -> Any:
        pass
