import logging
import datetime
from typing import Any, Dict, Type
from pathlib import Path

import luigi
import mlflow
import numpy as np
import polars as pl


from src.models.base import BaseRegressionModel
from src.tasks.base import MlflowTask
from src.tasks.preprocess.split import SplitDataTask
from src.settings import Settings

logger = logging.getLogger(__name__)


class TrainTask(MlflowTask):
    """
    Generic task for training regression models and logging metrics to MLflow.
    """

    settings_file_path = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = Settings.from_yaml(Path(self.settings_file_path))

    def requires(self):
        return SplitDataTask(
            data_path=self.settings.data_cfg.train_path,
            target=self.settings.data_cfg.target_column,
            derive_date_features=self.settings.data_cfg.derive_date_features,
            onehot_max_card=self.settings.data_cfg.onehot_max_card,
            config_path=self.settings.cv_config_path,
        )

    def _get_model_class(self) -> Type[BaseRegressionModel]:
        """
        Dynamically import and return the model class.
        """
        module_path, class_name = self.settings.model_cfg.model_class.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def _run(
        self,
        model_class: str,
        model_params: dict,
        target_column: str,
    ) -> Dict[str, Any]:
        """
        Train regression model and evaluate on validation sets.
        """
        # Load train/validation splits
        train_val_pairs = self.load()

        # Get model class
        ModelClass = self._get_model_class()

        all_metrics = []
        models = []

        for fold_idx, (train_df, val_df) in enumerate(train_val_pairs):
            logger.info(f"Training fold {fold_idx + 1}/{len(train_val_pairs)}")

            # Convert to polars if not already
            if not isinstance(train_df, pl.DataFrame):
                train_df = pl.DataFrame(train_df)
            if not isinstance(val_df, pl.DataFrame):
                val_df = pl.DataFrame(val_df)

            # Prepare training data
            X_train = train_df.drop(target_column)
            y_train = train_df.select(target_column)

            # Prepare validation data
            X_val = val_df.drop(target_column)
            y_val = val_df.select(target_column)

            # Initialize and train model
            model = ModelClass(**model_params)

            # Train model
            model.fit(X_train, y_train)

            # Evaluate on validation set
            y_pred = model.predict(X_val)

            # Convert predictions to DataFrame if necessary
            if not isinstance(y_pred, pl.DataFrame):
                if isinstance(y_pred, pl.Series):
                    y_pred = y_pred.to_frame("prediction")
                else:
                    y_pred = pl.DataFrame({"prediction": y_pred})

            val_metrics = model.evaluate(y_val, y_pred)

            # Add fold information to metrics
            fold_metrics = {
                f"fold_{fold_idx}_val_{key}": value
                for key, value in val_metrics.items()
            }
            all_metrics.append(fold_metrics)

            # Store model
            models.append(model)

            logger.info(f"Fold {fold_idx + 1} validation metrics: {val_metrics}")

        # Calculate average metrics across all folds
        avg_metrics = {}
        if all_metrics:
            metric_names = list(all_metrics[0].keys())
            metric_names = [m.split("_val_")[-1] for m in metric_names]
            metric_names = list(set(metric_names))

            for metric_name in metric_names:
                fold_values = [
                    metrics.get(f"fold_{i}_val_{metric_name}", 0)
                    for i, metrics in enumerate(all_metrics)
                ]
                avg_metrics[f"avg_val_{metric_name}"] = np.mean(fold_values)
                avg_metrics[f"std_val_{metric_name}"] = np.std(fold_values)

        # Store metrics in the instance for automatic logging
        # Combine all fold metrics and average metrics
        combined_metrics = {}
        for fold_metric in all_metrics:
            combined_metrics.update(fold_metric)
        combined_metrics.update(avg_metrics)
        self.metrics = combined_metrics

        # Log model and settings information
        mlflow.log_params(
            {
                "model_class": model_class,
                "model_params": str(model_params),
                "target_column": target_column,
                "n_folds": len(train_val_pairs),
                "model_name": ModelClass.model_name
                if hasattr(ModelClass, "model_name")
                else model_class,
            }
        )
        
        # Log settings information
        mlflow.log_params({
            "settings": {
                "data_config": self.settings.data_cfg.model_dump(),
                "model_config": self.settings.model_cfg.model_dump(),
                "cv_config": self.settings.cv_cfg.model_dump(),
            }
        })

        # Log feature importance if available (for tree-based models)
        if (
            models
            and hasattr(models[0], "model")
            and hasattr(models[0].model, "feature_importances_")
        ):
            feature_names = X_train.columns
            feature_importance = models[0].model.feature_importances_

            # Create feature importance dict
            feature_importance_dict = {
                f"feature_importance_{name}": importance
                for name, importance in zip(feature_names, feature_importance)
            }

            # Log top 10 most important features
            sorted_features = sorted(
                feature_importance_dict.items(), key=lambda x: x[1], reverse=True
            )[:10]

            for feature_name, importance in sorted_features:
                mlflow.log_metric(feature_name, importance)

        logger.info(f"Average validation metrics: {avg_metrics}")

        return {
            "models": models,
            "all_metrics": all_metrics,
            "avg_metrics": avg_metrics,
        }

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Parameters for the Train task.
        """
        return {
            "model_class": self.settings.model_cfg.model_class,
            "model_params": dict(self.settings.model_cfg.fit_params),
            "target_column": self.settings.data_cfg.target_column,
        }

    @property
    def mlflow_run_name(self) -> str:
        """
        Run name for MLflow.
        """
        now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.settings.model_cfg.name}-{now}"
