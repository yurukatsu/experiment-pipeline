import logging
from pathlib import Path
from typing import Generator

import luigi

from src.tasks.base import MlflowTask
from src.utils.splitter import TimeSeriesDataSplitter
from src.settings import CVConfig
from src.tasks.preprocess.features import FeatureCreateTask

logger = logging.getLogger(__name__)


class LoadCVConfigTask(MlflowTask):
    """
    Task for loading cross-validation configuration.
    """

    file_path = luigi.Parameter()

    @classmethod
    def _run(cls, file_path: str) -> CVConfig:
        config = CVConfig.from_yaml(Path(file_path))
        logger.info(f"Loaded CV config from {file_path}: {config}")
        return config

    @property
    def parameters(self) -> dict:
        """
        Parameters for the data ingestion task.
        """
        return {"file_path": self.file_path}


class SplitDataTask(MlflowTask):
    """
    Task for splitting data into train and test sets.
    """

    data_path = luigi.Parameter()
    target = luigi.Parameter()
    derive_date_features = luigi.BoolParameter(default=False)
    onehot_max_card = luigi.IntParameter(default=200)
    config_path = luigi.Parameter()

    def requires(self):
        return {
            "data": FeatureCreateTask(
                file_path=self.data_path,
                target=self.target,
                derive_date_features=self.derive_date_features,
                onehot_max_card=self.onehot_max_card,
            ),
            "cv_config": LoadCVConfigTask(file_path=self.config_path),
        }

    def _run(self) -> Generator:
        df = self.load("data")
        cv_config = self.load("cv_config")
        train_val_pairs = []
        for train, val in TimeSeriesDataSplitter.from_config(cv_config).split(df):
            train_val_pairs.append((train, val))
        logger.info(f"Generated {len(train_val_pairs)} train/validation splits.")
        return train_val_pairs

    @property
    def parameters(self) -> dict:
        """
        Parameters for the data splitting task.
        """
        return {}
