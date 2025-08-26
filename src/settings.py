from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file from the specified path and return its contents as a dictionary.

    :param path: The path to the YAML file.
    :type path: Path
    :raises FileNotFoundError: If the file does not exist at the specified path.
    :return: The contents of the YAML file.
    :rtype: Dict[str, Any]
    """

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CVConfig(BaseModel):
    """
    Cross-validation configuration schema (configs/cv/*.yml).

    :ivar strategy: The splitting strategy (currently 'sliding_window' and 'expanding_window' are supported).
    :ivar n_splits: The number of folds.
    :ivar validation_duration: The validation duration. Polars duration string (e.g., "3M", "4w", "14d").
    :ivar gap_duration: The gap duration between training and validation data.
    :ivar train_duration: The training duration (optional). If None, it will be inferred from the entire dataset.
    :ivar test_start_date: The start date of the first validation period (YYYY-MM-DD).
    """

    strategy: Literal["sliding_window", "expanding_window"]
    date_column_name: str
    n_splits: int
    validation_duration: str
    step_duration: str
    gap_duration: str
    train_duration: str | None = None
    test_end_date: str

    @classmethod
    def from_yaml(cls, path: Path) -> "CVConfig":
        data = load_yaml(path)
        return cls(**data)


class DataConfig(BaseModel):
    """
    Data configuration schema (configs/data.yml).
    """

    train_path: str
    test_path: str
    target_column: str
    date_column: str
    derive_date_features: bool
    onehot_max_card: int

    @classmethod
    def from_yaml(cls, path: Path) -> "DataConfig":
        data = load_yaml(path)
        return cls(**data)


class ModelConfig(BaseModel):
    """
    Model configuration schema (configs/models/*.yml).
    """

    name: str
    model_class: str  # e.g., "src.models.rf.RandomForestModel"
    fit_params: dict[str, Any] = {}
    prediction_params: dict[str, Any] = {}

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelConfig":
        data = load_yaml(path)
        return cls(**data)


class Settings(BaseModel):
    data_config_path: str
    model_config_path: str
    cv_config_path: str

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        return cls(**load_yaml(path))

    @property
    def data_cfg(self) -> DataConfig:
        return DataConfig.from_yaml(Path(self.data_config_path))

    @property
    def model_cfg(self) -> ModelConfig:
        return ModelConfig.from_yaml(Path(self.model_config_path))

    @property
    def cv_cfg(self) -> CVConfig:
        return CVConfig.from_yaml(Path(self.cv_config_path))
