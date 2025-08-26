import logging

import luigi
import polars as pl

from src.tasks.base import MlflowTask
from src.utils.preprocess import preprocess
from src.tasks.preprocess import IngestDataTask

logger = logging.getLogger(__name__)


class FeatureCreateTask(MlflowTask):
    file_path = luigi.Parameter()
    target = luigi.Parameter()
    derive_date_features = luigi.BoolParameter(default=False)
    onehot_max_card = luigi.IntParameter(default=200)

    def requires(self):
        return IngestDataTask(file_path=self.file_path)

    def _run(
        self, target: str, derive_date_features: bool, onehot_max_card: int
    ) -> pl.DataFrame:
        df = self.load()
        X, y, _ = preprocess(df, target, derive_date_features, onehot_max_card)
        preprocessed = pl.concat([X, y], how="horizontal")
        return preprocessed

    @property
    def parameters(self) -> dict:
        return {
            "target": self.target,
            "derive_date_features": self.derive_date_features,
            "onehot_max_card": self.onehot_max_card,
        }
