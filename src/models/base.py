from abc import ABC, abstractmethod

import polars as pl
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    r2_score,
)


class BaseRegressionModel(ABC):
    model_name: str
    metrics = {
        "RMSE": root_mean_squared_error,
        "MAE": mean_absolute_error,
        "MAPE": mean_absolute_percentage_error,
        "R2": r2_score,
    }

    def __init__(self):
        self.model = None

    @abstractmethod
    def fit(self, X_train: pl.DataFrame, y_train: pl.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pl.DataFrame):
        raise NotImplementedError

    @classmethod
    def evaluate(cls, y_true: pl.DataFrame, y_pred: pl.DataFrame):
        return {
            metric: func(y_true.to_pandas(), y_pred.to_pandas())
            for metric, func in cls.metrics.items()
        }
