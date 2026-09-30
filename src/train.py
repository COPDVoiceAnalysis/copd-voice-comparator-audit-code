"""
Simplified training script with inlined nested CV loops.
Reduces complexity from 9 functions to 3 functions while maintaining all functionality.
"""

from pathlib import Path
import numpy as np
from datetime import datetime
from typing import Any
from dataclasses import dataclass
from sklearn import clone
from sklearn.calibration import cross_val_predict
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from src.utils.person_aware_cv import (
    PersonAwareRepeatedStratifiedKFold,
    PersonAwareStratifiedKFold,
    validate_no_person_leakage,
)
from src.utils.person_aware_gridsearch import PersonAwareGridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import argparse
import yaml
import logging

# Import Pydantic models and utilities
from src.models.config_models import ExperimentConfig, TrainingConfig
from src.models.metadata import Metadata
from src.features.preloader import FeaturePreloader
from src.features.preprocessing import W2VParselPCA
from src.utils.logging_config import setup_application_logging
from src.utils.model_evaluation_mlflow import MLFlowEvaluationManager
from src.utils.mlflow_tracker import (
    MLFlowTracker,
    create_experiment_name,
)
from src.utils.custom_metrics import partial_auroc_weighted
from src.utils.sample_weight_strategies import compute_fold_weights, compute_evaluation_strata_weights

import sklearn

sklearn.set_config(enable_metadata_routing=True)

# Module-level logger - used throughout the file
logger = logging.getLogger(__name__)


def _tau_at_fpr(y, s, alpha=0.2, sample_weight=None):
    """Compute the threshold at which FPR <= alpha with maximum TPR.

    When sample_weight is provided, the ROC curve is computed with weighted
    counts, so FPR is controlled in the reweighted (target) population.
    """
    fpr, tpr, thr = roc_curve(y, s, sample_weight=sample_weight)
    idx = np.where(fpr <= alpha)[0]
    j = np.argmax(tpr[idx])
    return thr[idx][j]


@dataclass
class TrainingContext:
    """Container for all training-related data and configurations."""

    # Metadata
    training_metadata: Metadata

    # Features
    concatenated_features: np.ndarray
    parsel_feature_count: int
    demographic_feature_count: int

    # Training components
    classifier_name: str
    classifier: BaseEstimator
    param_grid: dict[str, Any]
    outer_cv_strategy: PersonAwareRepeatedStratifiedKFold
    inner_cv_strategy: PersonAwareRepeatedStratifiedKFold
    scorer: Any

    # Prepared data
    y_training: np.ndarray
    label_encoder: LabelEncoder
    audio_ids: np.ndarray

    # Evaluation
    eval_manager: MLFlowEvaluationManager


def get_classifier_map(training_config: TrainingConfig) -> BaseEstimator:
    """Map classifier names to sklearn estimator instances."""
    classifier_name = training_config.classifier_name
    if classifier_name == "RandomForestClassifier":
        return RandomForestClassifier(
            random_state=training_config.random_seed,
            n_jobs=training_config.n_jobs,
        )
    elif classifier_name == "XGBClassifier":
        return XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=training_config.random_seed,
            n_jobs=training_config.n_jobs,
        )
    elif classifier_name == "SVC":
        return SVC(
            probability=True,
            random_state=training_config.random_seed,
        )
    elif classifier_name == "LogisticRegression":
        return LogisticRegression(
            random_state=training_config.random_seed,
            max_iter=1000,
            n_jobs=training_config.n_jobs,
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")


def create_training_pipeline(
    training_config: TrainingConfig,
    parsel_feature_count: int,
    demographic_feature_count: int,
    classifier: BaseEstimator,
    use_sample_weights: bool = False,
) -> Pipeline:
    """Create sklearn pipeline with preprocessing steps."""
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("vt", VarianceThreshold(0.01)),
            (
                "scaler",
                StandardScaler().set_fit_request(sample_weight=use_sample_weights),
            ),
            (
                "PCA",
                W2VParselPCA(
                    random_state=training_config.random_seed,
                    parsel_feature_count=parsel_feature_count,
                    demographic_feature_count=demographic_feature_count,
                ),
            ),
            ("clf", classifier.set_fit_request(sample_weight=use_sample_weights)),
        ]
    )


def load_data_and_setup(
    config: ExperimentConfig,
    mlflow_tracker: MLFlowTracker,
) -> TrainingContext | None:
    """Load data, setup classifiers, and create complete training context."""
    # Load and prepare training data
    logger.info("=== LOADING DATA ===")
    training_metadata = Metadata(
        csv_path=config.metadata_file,
        ps_covariates_file=config.ps_covariates_file,
    )
    training_metadata.filter_for_experiment(
        task_config=config.task_config,
        feature_config=config.feature_config,
    )

    # Generate recording identifier sets for training
    recording_identifier_sets_list = training_metadata.get_recording_identifier_sets(
        feature_config=config.feature_config
    )

    # Load features and demographic data (optionally baseline-only when include_voice_features=False)
    training_config = config.training_config
    demographic_feature_columns = training_config.demographic_feature_columns
    include_voice_features = training_config.include_voice_features
    demographic_features = training_metadata.get_demographic_features(
        recording_identifier_sets_list, demographic_feature_columns
    )
    preloader = FeaturePreloader(config.feature_config)
    concatenated_features, parsel_feature_count, demographic_feature_count = (
        preloader.load_features(
            recording_identifier_sets_list,
            demographic_features,
            include_voice_features=include_voice_features,
        )
    )

    feature_info = preloader.get_feature_info()
    logger.info(f"Feature loading complete: {feature_info}")
    logger.info(f"Concatenated features shape: {concatenated_features.shape}")

    # Setup classifiers
    classifier = get_classifier_map(training_config)

    # Build parameter grid with proper prefixes for sklearn pipeline
    param_grid = {
        **training_config.classifier_hyperparams,
        **training_config.pca_hyperparams,
    }

    # Always use PersonAwareRepeatedStratifiedKFold for robust person-aware cross-validation
    logger.info("Using PersonAwareRepeatedStratifiedKFold for all data types")
    cv_strategy = PersonAwareRepeatedStratifiedKFold(
        n_splits=training_config.n_splits_outer,
        n_repeats=training_config.n_repeats_outer,
        random_state=training_config.random_seed,
    )
    inner_cv_strategy = PersonAwareRepeatedStratifiedKFold(
        n_splits=training_config.n_splits_inner,
        n_repeats=training_config.n_repeats_inner,
        random_state=training_config.random_seed,
    )

    # Use configurable scoring metric (default: MCC)
    if training_config.scoring_metric == "matthews_corrcoef":
        scorer = make_scorer(
            matthews_corrcoef, greater_is_better=True
        ).set_score_request(sample_weight=True)
    elif training_config.scoring_metric == "balanced_accuracy":
        scorer = make_scorer(
            balanced_accuracy_score, greater_is_better=True
        ).set_score_request(sample_weight=True)
    elif training_config.scoring_metric == "auroc":
        scorer = make_scorer(
            roc_auc_score, greater_is_better=True, needs_proba=True
        ).set_score_request(sample_weight=True)
    elif training_config.scoring_metric == "partial_auroc_weighted":
        pi = training_config.pi_deploy if training_config.pi_deploy is not None else 0.2
        scorer = make_scorer(
            partial_auroc_weighted, needs_proba=True, max_fpr=0.25, pi_deploy=pi
        ).set_score_request(sample_weight=True)
    elif training_config.scoring_metric == "average_prec":
        scorer = make_scorer(
            average_precision_score, greater_is_better=True, needs_proba=True
        ).set_score_request(sample_weight=True)
    else:
        raise ValueError(
            f"Unsupported scoring metric: {training_config.scoring_metric}"
        )

    # Prepare training data (labels and audio_ids)
    y_training, label_encoder = training_metadata.get_class_labels_for_sets(
        recording_identifier_sets_list
    )
    audio_ids = training_metadata.extract_audio_ids(recording_identifier_sets_list)

    eval_manager = MLFlowEvaluationManager(
        mlflow_tracker=mlflow_tracker,
        metadata=training_metadata,
        feature_config=config.feature_config,
        preloader=preloader,
        demographic_feature_columns=demographic_feature_columns,
        include_voice_features=include_voice_features,
    )

    logger.info(f"Configured classifier: {training_config.classifier_name}")

    return TrainingContext(
        training_metadata=training_metadata,
        concatenated_features=concatenated_features,
        parsel_feature_count=parsel_feature_count,
        demographic_feature_count=demographic_feature_count,
        classifier_name=training_config.classifier_name,
        classifier=classifier,
        param_grid=param_grid,
        outer_cv_strategy=cv_strategy,
        inner_cv_strategy=inner_cv_strategy,
        scorer=scorer,
        y_training=y_training,
        label_encoder=label_encoder,
        audio_ids=audio_ids,
        eval_manager=eval_manager,
    )


def run_training(
    context: TrainingContext,
    config: ExperimentConfig,
) -> None:
    """Run nested cross-validation using TrainingContext."""
    training_config = config.training_config

    classifier_name = context.classifier_name
    logger.info(f"=== Processing classifier: {classifier_name} ===")

    # Create pipeline
    pipeline = create_training_pipeline(
        training_config=training_config,
        parsel_feature_count=context.parsel_feature_count,
        demographic_feature_count=context.demographic_feature_count,
        classifier=context.classifier,
        use_sample_weights=True,
    )

    # Always use person-aware CV split iterator with audio_ids
    outer_splits = context.outer_cv_strategy.split(
        context.concatenated_features,
        context.y_training,
        audio_ids=context.audio_ids,
    )

    for fold, (train_idx, test_idx) in enumerate(outer_splits, 1):
        logger.info(f"=== Processing {classifier_name} - Fold {fold} ===")

        # Split training data
        X_train = context.concatenated_features[train_idx]
        y_train = context.y_training[train_idx]

        # Get test audio IDs for this fold
        test_audio_ids_fold = context.audio_ids[test_idx]

        # Create test sets
        X_test_training = context.concatenated_features[test_idx]
        y_test_training = context.y_training[test_idx]

        # Always validate no person leakage (works for both chunked and non-chunked data)
        train_audio_ids = context.audio_ids[train_idx]
        if not validate_no_person_leakage(train_audio_ids, test_audio_ids_fold):
            raise ValueError(f"Person leakage detected in fold {fold}!")

        weights_train = compute_fold_weights(
            training_config.sample_weight_strategy,
            context.training_metadata,
            context.audio_ids[train_idx],
            y_train,
            training_config,
        )
        assert weights_train.shape[0] == y_train.shape[0]

        grid_search = PersonAwareGridSearchCV(
            pipeline,
            param_grid=context.param_grid,
            cv=context.inner_cv_strategy,
            scoring=context.scorer,
            n_jobs=training_config.n_jobs,
            return_train_score=False,
            verbose=1,
            audio_ids=train_audio_ids,
        )

        logger.info(
            f"Starting hyperparameter optimization for {classifier_name} - fold {fold}..."
        )
        grid_search.fit(X_train, y_train, sample_weight=weights_train)
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")

        # Use non-repeated CV for cross_val_predict (which requires partitions, not overlapping splits)
        cv_for_predict = PersonAwareStratifiedKFold(
            n_splits=training_config.n_splits_inner,
            shuffle=True,
            random_state=training_config.random_seed,
        )

        est = clone(grid_search.best_estimator_)
        oof_scores = cross_val_predict(
            est, X_train, y_train,
            cv=cv_for_predict.split(X_train, y_train, audio_ids=train_audio_ids),
            method="predict_proba",
            fit_params={"sample_weight": weights_train},
        )[:, 1]

        # Compute strata weights for threshold finding (target population FPR control)
        tau_strata_weights = compute_evaluation_strata_weights(
            context.training_metadata, train_audio_ids, y_train
        )
        tau = _tau_at_fpr(y_train, oof_scores, alpha=0.2, sample_weight=tau_strata_weights)

        # Evaluate this fold (manager lazy-loads original features for unsplit evaluation)
        chunked_test_data = (X_test_training, y_test_training, test_audio_ids_fold)

        context.eval_manager.evaluate_fold(
            model=best_model,
            chunked_test_data=chunked_test_data,
            fold=fold,
            classifier_name=classifier_name,
            label_encoder=context.label_encoder,
            threshold=tau,
        )

    # Finalize experiment - this logs summary statistics to MLFlow
    context.eval_manager.finalize_experiment(context.label_encoder)


def main():
    """Main function with argument parsing and MLFlow integration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config_data = yaml.safe_load(f)

    # Exit if config is empty (happens e.g. when using invalid feature+task combination)
    if not config_data:
        return
    config = ExperimentConfig(**config_data)

    # Extract config_id from config file path
    config_filename = Path(args.config).name
    if not config_filename.endswith("__config.yml"):
        raise ValueError("Config filename must end with '__config.yml'")
    config_id = config_filename.replace("__config.yml", "")

    # Add datetime stamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config.output_dir = Path(args.output_dir) / timestamp

    # Create structured log directory and file
    log_dir = Path(config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{config_id}_training_{timestamp}.log"

    # Setup application-wide logging (this configures logging for the entire application)
    setup_application_logging(log_file_path=str(log_file_path))

    logger.info("=== TRAINING SCRIPT STARTED ===")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Description: {config.description}")
    logger.info(f"Log file: {log_file_path}")

    # Setup MLFlow tracking
    task_name = config.classification_task or config.task_config.target_column
    experiment_name = create_experiment_name(
        task_name, date=config.date, experiment_version=config.experiment_version,
    )

    mlflow_tracker = MLFlowTracker(experiment_name, config_id)

    with mlflow_tracker.start_run():
        # Log experiment configuration and wildcards
        mlflow_tracker.log_experiment_config(config)

        # Log the log file path for easy access from MLFlow GUI
        mlflow_tracker.log_log_file_path(str(log_file_path))

        context = load_data_and_setup(config, mlflow_tracker)

        # Check if training was skipped due to empty dataset
        if context is None:
            logger.info("Training was skipped due to empty dataset after filtering")
            # Log failure to MLFlow
            mlflow_tracker.log_final_metrics({"status": 0, "error": "empty_dataset"})
            return

        # Log dataset info
        mlflow_tracker.log_final_metrics(
            {
                "n_samples": len(context.concatenated_features),
                "n_features": context.concatenated_features.shape[1],
                "n_patients": len(np.unique(context.audio_ids)),
            }
        )

        # Run training
        run_training(context, config)

        logger.info("All results have been logged to MLFlow automatically")
        mlflow_tracker.log_final_metrics({"status": 1})

        # Log the log file as an artifact for easy download from MLFlow
        mlflow_tracker.log_log_file_artifact(str(log_file_path))

    logger.info(f"Training complete! Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()
