import polars as pl
from autogluon.tabular import TabularPredictor

from src.models.base import BaseRegressionModel


class AutoGluonModel(BaseRegressionModel):
    model_name: str = "AutoGluon"

    def __init__(self, **parameters):
        self.model = TabularPredictor(**parameters)

    def fit(self, X_train: pl.DataFrame, y_train: pl.DataFrame):
        data = pl.concat([X_train, y_train], how="horizontal").to_pandas()
        self.model.fit(data)

    def predict(self, X, **parameters):
        return self.model.predict(X, **parameters)
