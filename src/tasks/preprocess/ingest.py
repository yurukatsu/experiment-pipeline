import logging

import luigi
import polars as pl

from src.tasks.base import MlflowTask

logger = logging.getLogger(__name__)


class IngestDataTask(MlflowTask):
    """
    Task for ingesting data.
    """

    file_path = luigi.Parameter()

    def _run(self, file_path: str) -> pl.DataFrame:
        df = pl.read_csv(file_path, try_parse_dates=True)
        logger.info(f"Loaded data from {file_path}: {df.shape}")
        return df

    @property
    def parameters(self) -> dict:
        """
        Parameters for the data ingestion task.
        """
        return {"file_path": self.file_path}
