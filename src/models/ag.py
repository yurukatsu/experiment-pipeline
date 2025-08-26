import polars as pl
from autogluon.tabular import TabularPredictor

from src.models.base import BaseRegressionModel


class AutoGluonModel(BaseRegressionModel):
    model_name: str = "AutoGluon"

    def __init__(self, label, **parameters):
        # TabularPredictorの初期化時に必要なパラメータのみ抽出
        init_params = {
            k: v
            for k, v in parameters.items()
            if k in ["learner_type", "quantile_levels", "learner_kwargs"]
        }
        self.model = TabularPredictor(label=label, **init_params)
        # fit時に使用するパラメータを保存
        self.fit_params = {
            k: v
            for k, v in parameters.items()
            if k not in ["learner_type", "quantile_levels", "learner_kwargs"]
        }

    def fit(self, X_train: pl.DataFrame, y_train: pl.DataFrame):
        data = pl.concat([X_train, y_train], how="horizontal").to_pandas()
        self.model.fit(data, **self.fit_params)

    def predict(self, X, **parameters):
        predictions = self.model.predict(X.to_pandas(), **parameters)
        # polars Seriesに変換して返す
        return pl.Series("prediction", predictions)
