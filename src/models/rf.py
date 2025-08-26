from sklearn.ensemble import RandomForestRegressor

from src.models.base import BaseRegressionModel


class RandomForestModel(BaseRegressionModel):
    model_name: str = "RandomForestRegressor"

    def __init__(self, **parameters):
        self.model = RandomForestRegressor(**parameters)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
