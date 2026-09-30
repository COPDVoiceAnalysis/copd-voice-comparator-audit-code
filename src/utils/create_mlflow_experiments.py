"""
Pre-create MLFlow experiments to prevent race conditions.
Scans generated config files and creates all unique experiments before parallel training starts.
"""

import os
import glob
import yaml
import argparse
import mlflow
from src.utils.mlflow_tracker import create_experiment_name
import logging

logger = logging.getLogger(__name__)


def extract_experiment_names_from_configs(config_dir: str) -> set[str]:
    """
    Extract unique experiment names from all generated config files.

    Args:
        config_dir: Directory containing generated config files

    Returns:
        Set of unique experiment names
    """
    experiment_names = set()

    # Find all config files
    config_pattern = os.path.join(config_dir, "*__config.yml")
    config_files = glob.glob(config_pattern)

    logger.info(f"Found {len(config_files)} config files in {config_dir}")

    for config_file in config_files:
        try:
            with open(config_file) as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                logger.warning(f"Empty config file: {config_file}")
                continue

            # Extract task name and date from config — prefer classification_task
            # (e.g. "copd_classification_charite_only"), fall back to target_column
            task_name = config_data.get("classification_task") or config_data.get(
                "task_config", {}
            ).get("target_column")
            if not task_name:
                logger.warning(f"No classification_task or target_column found in {config_file}")
                continue
            date = config_data.get("date")
            experiment_version = config_data.get("experiment_version")

            # Create experiment name using the same logic as training script
            experiment_name = create_experiment_name(
                task_name, date=date, experiment_version=experiment_version,
            )
            experiment_names.add(experiment_name)

        except Exception as e:
            logger.error(f"Error processing config file {config_file}: {e}")
            continue

    logger.info(f"Extracted {len(experiment_names)} unique experiment names")
    return experiment_names


def setup_mlflow_connection():
    """Setup MLFlow connection using environment variables."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable not set")

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLFlow tracking URI: {tracking_uri}")

    artifact_uri = os.getenv("MLFLOW_ARTIFACT_URI")
    if artifact_uri:
        logger.info(f"MLFlow artifact URI: {artifact_uri}")
    else:
        logger.warning("MLFLOW_ARTIFACT_URI environment variable not set")


def create_experiment_safe(experiment_name: str) -> str:
    """
    Safely create an MLFlow experiment, handling the case where it already exists.

    Args:
        experiment_name: Name of the experiment to create

    Returns:
        Experiment ID
    """
    try:
        # Try to get existing experiment first
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            logger.info(
                f"Experiment '{experiment_name}' already exists (ID: {experiment.experiment_id})"
            )
            return experiment.experiment_id

        # Create new experiment
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new experiment '{experiment_name}' (ID: {experiment_id})")
        return experiment_id

    except Exception as e:
        # Handle race condition where experiment was created between our check and creation
        logger.warning(f"Error creating experiment '{experiment_name}': {e}")

        # Try to get it again in case it was created by another process
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is not None:
                logger.info(
                    f"Experiment '{experiment_name}' was created by another process (ID: {experiment.experiment_id})"
                )
                return experiment.experiment_id
        except Exception as e2:
            logger.error(f"Failed to retrieve experiment after creation error: {e2}")

        raise e


def create_all_experiments(experiment_names: set[str]) -> dict[str, str]:
    """
    Create all experiments and return mapping of names to IDs.

    Args:
        experiment_names: Set of experiment names to create

    Returns:
        Dictionary mapping experiment names to IDs
    """
    experiment_ids = {}

    for experiment_name in sorted(experiment_names):
        try:
            experiment_id = create_experiment_safe(experiment_name)
            experiment_ids[experiment_name] = experiment_id
        except Exception as e:
            logger.error(f"Failed to create experiment '{experiment_name}': {e}")
            # Continue with other experiments rather than failing completely
            continue

    return experiment_ids


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Pre-create MLFlow experiments to prevent race conditions"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Directory containing generated config files",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Setup MLFlow connection
        setup_mlflow_connection()

        # Extract experiment names from config files
        experiment_names = extract_experiment_names_from_configs(args.config_dir)

        if not experiment_names:
            logger.warning("No experiment names found in config files")
            return

        # Create all experiments
        logger.info(f"Creating {len(experiment_names)} experiments...")
        experiment_ids = create_all_experiments(experiment_names)

        # Summary
        logger.info(f"Successfully processed {len(experiment_ids)} experiments:")
        for name, exp_id in experiment_ids.items():
            logger.info(f"  - {name} (ID: {exp_id})")

        if len(experiment_ids) < len(experiment_names):
            failed_count = len(experiment_names) - len(experiment_ids)
            logger.warning(f"{failed_count} experiments failed to create")

        logger.info("MLFlow experiment pre-creation completed")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise


if __name__ == "__main__":
    main()
