"""
Simplified MLFlow experiment tracking for voice biomarker experiments.
Replaces custom experiment tracking with MLFlow's built-in functionality.
"""

import os
import mlflow
from typing import Any
from contextlib import contextmanager
from datetime import datetime

from src.models.config_models import ExperimentConfig
import logging

logger = logging.getLogger(__name__)


class MLFlowTracker:
    """Simplified experiment tracking using MLFlow."""

    def __init__(self, experiment_name: str, run_name: str):
        """
        Initialize MLFlow tracker.

        Args:
            experiment_name: Name of the MLFlow experiment
            run_name: Name of this specific run
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = None

        # Set MLFlow tracking URI from environment
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLFlow tracking URI: {tracking_uri}")
        else:
            raise ValueError("MLFLOW_TRACKING_URI environment variable not set")

        # Set artifact URI from environment
        artifact_uri = os.getenv("MLFLOW_ARTIFACT_URI")
        if artifact_uri:
            logger.info(f"MLFlow artifact URI: {artifact_uri}")
        else:
            raise ValueError("MLFLOW_ARTIFACT_URI environment variable not set")

    @contextmanager
    def start_run(self):
        """Context manager for MLFlow run."""
        try:
            # Set or create experiment
            mlflow.set_experiment(self.experiment_name)

            # Start run
            self.run = mlflow.start_run(run_name=self.run_name)
            logger.info(f"Started MLFlow run: {self.run_name}")

            # Add basic system tags
            mlflow.set_tags(
                {
                    "slurm_job_id": os.getenv("SLURM_JOB_ID", "local"),
                    "node": os.uname().nodename,
                }
            )

            yield self

        finally:
            if self.run:
                mlflow.end_run()
                logger.info(f"Ended MLFlow run: {self.run_name}")

    def get_tags_from_config(self, config: ExperimentConfig) -> dict[str, str]:
        """Extract tags from experiment configuration.

        Tag order from generate_experiment_configs: [exp, scope, feature, hyperparam, task, splitting?]
        """
        tags: dict[str, str] = {
            "exp": config.tags[0],
            "scope": config.tags[1],
            "feature": config.tags[2],
            "hyperparam": config.tags[3],
            "task": config.tags[4],
        }
        if len(config.tags) > 5:
            tags["split"] = config.tags[5]
        return tags

    def log_experiment_config(self, config: ExperimentConfig):
        """
        Log experiment configuration as parameters and tags.

        Args:
            config: ExperimentConfig object
        """
        # Log wildcards as tags for easy filtering
        tags = self.get_tags_from_config(config)
        mlflow.set_tags(tags)
        logger.info(f"Logged wildcards as tags: {tags}")
        # Log configuration parameters, skipping top-level config section names
        config_dict = config.dict() if hasattr(config, "dict") else config

        # Flatten each top-level section independently to avoid long prefixes
        flattened_params = {}
        for section_name, section_config in config_dict.items():
            if isinstance(section_config, dict):
                # Flatten this section without the section name prefix
                section_params = self._flatten_dict(section_config)
                flattened_params.update(section_params)
            else:
                # Keep non-dict values as-is
                flattened_params[section_name] = section_config

        mlflow.log_params(flattened_params)
        logger.info("Logged experiment configuration")

    def log_log_file_path(self, log_file_path: str):
        """
        Log the training log file path as a parameter for easy access from MLFlow GUI.

        Args:
            log_file_path: Path to the training log file
        """
        mlflow.log_param("log_file_path", log_file_path)
        logger.info(f"Logged log file path: {log_file_path}")

    def log_log_file_artifact(self, log_file_path: str):
        """
        Log the training log file directly as an MLFlow artifact for easy download.

        Args:
            log_file_path: Path to the training log file
        """
        if os.path.exists(log_file_path):
            mlflow.log_artifact(log_file_path, "logs")
            logger.info(f"Logged log file as artifact: {log_file_path}")
        else:
            logger.warning(f"Log file not found for artifact logging: {log_file_path}")

    def log_fold_metrics(
        self,
        fold: int,
        metrics: dict[str, float],
    ):
        """
        Log metrics for unsplit evaluation in a single fold.

        Args:
            fold: Fold number
            metrics: Metrics from unsplit evaluation
        """
        mlflow.log_metrics(metrics, step=fold)
        logger.info(f"Logged evaluation metrics for fold {fold}")

    def log_final_metrics(self, metrics: dict[str, Any]):
        """Log final aggregated metrics."""
        float_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, str):
                if value == "empty_dataset":
                    float_metrics[f"{key}_code"] = -1.0
                else:
                    logger.warning(f"Skipping non-numeric metric {key}={value}")
            else:
                float_metrics[key] = float(value)

        mlflow.log_metrics(float_metrics)
        logger.info(f"Logged final metrics: {list(float_metrics.keys())}")

    def log_file_artifact(self, file_path: str, artifact_path: str | None = None):
        """Log file as artifact."""
        mlflow.log_artifact(file_path, artifact_path)
        logger.info(f"Logged artifact: {file_path}")

    def _flatten_dict(self, d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dictionary for MLFlow parameter logging.

        Uses dot-separated prefixes to avoid key collisions between sections.
        """
        items = []
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, prefix=f"{key}.").items())
            elif isinstance(v, list | tuple):
                # Convert lists to strings for MLFlow
                items.append((key, str(v)))
            else:
                items.append((key, v))
        return dict(items)


def create_experiment_name(
    task_name: str,
    date: str | None = None,
    experiment_version: str | None = None,
) -> str:
    """
    Create MLFlow experiment name following convention.

    Args:
        task_name: Name of the task (e.g., "copd_classification")
        date: Date string (defaults to current date)
        experiment_version: Optional version suffix (e.g. "v2") appended as
            ``{date}-{version}``.  Data paths still use *date* alone.

    Returns:
        Formatted experiment name
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    if experiment_version:
        date = f"{date}-{experiment_version}"
    return f"{task_name}_{date}"


def get_wildcards_from_env() -> dict[str, str]:
    """Extract Snakemake wildcards from environment variables."""
    return {
        "exp": os.getenv("SNAKEMAKE_EXP", "unknown"),
        "feature": os.getenv("SNAKEMAKE_FEATURE", "unknown"),
        "hyperparam": os.getenv("SNAKEMAKE_HYPERPARAM", "unknown"),
        "task": os.getenv("SNAKEMAKE_TASK", "unknown"),
    }
