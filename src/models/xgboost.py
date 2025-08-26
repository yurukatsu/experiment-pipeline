from xgboost import XGBRegressor

from src.models.base import BaseRegressionModel


class XGBoostModel(BaseRegressionModel):
    model_name: str = "XGBRegressor"

    def __init__(self, **parameters):
        self.model = XGBRegressor(**parameters)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X, **parameters):
        return self.model.predict(X, **parameters)
