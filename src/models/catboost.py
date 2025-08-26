from catboost import CatBoostRegressor

from src.models.base import BaseRegressionModel


class CatBoostModel(BaseRegressionModel):
    model_name: str = "CatBoostRegressor"

    def __init__(self, **parameters):
        self.model = CatBoostRegressor(**parameters)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X, **parameters):
        return self.model.predict(X, **parameters)
