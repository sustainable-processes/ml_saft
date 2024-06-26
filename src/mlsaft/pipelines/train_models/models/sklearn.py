import logging
from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator, RegressorMixin


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
