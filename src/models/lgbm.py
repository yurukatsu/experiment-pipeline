from lightgbm import LGBMRegressor

from src.models.base import BaseRegressionModel


class LGBMModel(BaseRegressionModel):
    model_name: str = "LGBMRegressor"

    def __init__(self, **parameters):
        self.model = LGBMRegressor(**parameters)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X, **parameters):
        return self.model.predict(X, **parameters)
