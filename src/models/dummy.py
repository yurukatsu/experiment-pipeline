import random
import polars as pl


from src.models.base import BaseRegressionModel


class DummyRegressor(BaseRegressionModel):
    model_name: str = "DummyRegressor"

    def __init__(self, **parameters):
        self.random = random.randint(0, 100)

    def fit(self, X_train: pl.DataFrame, y_train: pl.DataFrame):
        return

    def predict(self, X: pl.DataFrame, **parameters):
        size = X.shape[0]
        return pl.DataFrame({"prediction": [self.random] * size}).to_series()
