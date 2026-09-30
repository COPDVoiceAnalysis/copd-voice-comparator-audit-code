"""
Seed-sweep training entry point for the Phase II robustness analysis.

Purpose
-------
The headline results from the main snakemake_matrix.yml workflow use a single
random_seed (42) per configuration. For the demographics-only baselines this
gives only one realisation of a small, EPV-constrained nested CV — and the
existing data showed that LR/RF/SVC on N~65 PSM cohort + 4-6 covariates can
yield sign-flipped fits (mean fold AUROC ~0.28 = inverted prediction).

This script runs the *same* nested CV but iterates over many random seeds
within a single job, sharing the expensive feature-loading step across seeds.
Each seed produces one fresh MLflow run; the 50 per-fold AUROC values are
the same shape as the main workflow, so downstream aggregation (mean ± CI
across seeds, fold-level CIs within seeds) is straightforward.

Two design choices that differ from src/train.py:

1. **One config -> many MLflow runs.** Each seed gets a separate run with
   suffix ``__seedNN``, so MLflow / parquet aggregation treats them as
   independent realisations.

2. **Features loaded once.** Metadata, demographic features, concatenated
   features, and label encoding are computed before the seed loop and reused.
   Only the per-seed components (classifier `random_state`, CV iterators with
   `random_state=seed`, MLflow tracker, eval manager) are rebuilt each
   iteration.

CLI
---
    uv run python -m src.train_seed_sweep \
        --config <path>/<tuple_id>__seed_sweep_config.yml \
        --seeds 1 2 3 ... 20 \
        --weighting ps_overlap \
        --output_dir <work>/<tuple_id>

The ``--weighting`` flag overrides ``training_config.sample_weight_strategy``
in the loaded config; this is how the seed-sweep matrix runs the same
(scope, cohort, voice_config) tuple under both ``ps_overlap`` and
``class_only`` without duplicating the config file.
"""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    make_scorer,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

import sklearn

from src.features.preloader import FeaturePreloader
from src.models.config_models import ExperimentConfig
from src.models.metadata import Metadata
from src.train import (
    TrainingContext,
    get_classifier_map,
    run_training,
)
from src.utils.custom_metrics import partial_auroc_weighted
from src.utils.logging_config import setup_application_logging
from src.utils.mlflow_tracker import MLFlowTracker, create_experiment_name
from src.utils.model_evaluation_mlflow import MLFlowEvaluationManager
from src.utils.person_aware_cv import PersonAwareRepeatedStratifiedKFold

sklearn.set_config(enable_metadata_routing=True)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Cached data loading: done once per job, shared across all seeds.
# ----------------------------------------------------------------------------


class _CachedDataset:
    """Holds everything that does NOT depend on the random seed.

    All seed-dependent objects (classifier, CV strategies, eval manager,
    MLflow tracker) are rebuilt fresh per iteration in
    :func:`_build_seeded_context`.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        logger.info("=== LOADING DATA (cached, shared across seeds) ===")
        self.training_metadata = Metadata(
            csv_path=config.metadata_file,
            ps_covariates_file=config.ps_covariates_file,
        )
        self.training_metadata.filter_for_experiment(
            task_config=config.task_config,
            feature_config=config.feature_config,
        )

        self.recording_identifier_sets_list = (
            self.training_metadata.get_recording_identifier_sets(
                feature_config=config.feature_config
            )
        )

        tc = config.training_config
        demographic_features = self.training_metadata.get_demographic_features(
            self.recording_identifier_sets_list,
            tc.demographic_feature_columns,
        )
        self.preloader = FeaturePreloader(config.feature_config)
        (
            self.concatenated_features,
            self.parsel_feature_count,
            self.demographic_feature_count,
        ) = self.preloader.load_features(
            self.recording_identifier_sets_list,
            demographic_features,
            include_voice_features=tc.include_voice_features,
        )
        logger.info(
            "Loaded features: shape=%s, parsel=%d, demographic=%d",
            self.concatenated_features.shape,
            self.parsel_feature_count,
            self.demographic_feature_count,
        )

        self.y_training, self.label_encoder = (
            self.training_metadata.get_class_labels_for_sets(
                self.recording_identifier_sets_list
            )
        )
        self.audio_ids = self.training_metadata.extract_audio_ids(
            self.recording_identifier_sets_list
        )

    def n_samples(self) -> int:
        return int(len(self.concatenated_features))

    def n_persons(self) -> int:
        return int(len(np.unique(self.audio_ids)))


def _build_scorer(scoring_metric: str, pi_deploy: float | None) -> Any:
    """Replicates the scorer construction from src.train.load_data_and_setup."""
    if scoring_metric == "matthews_corrcoef":
        return make_scorer(
            matthews_corrcoef, greater_is_better=True
        ).set_score_request(sample_weight=True)
    if scoring_metric == "balanced_accuracy":
        return make_scorer(
            balanced_accuracy_score, greater_is_better=True
        ).set_score_request(sample_weight=True)
    if scoring_metric == "auroc":
        return make_scorer(
            roc_auc_score, greater_is_better=True, needs_proba=True
        ).set_score_request(sample_weight=True)
    if scoring_metric == "partial_auroc_weighted":
        pi = pi_deploy if pi_deploy is not None else 0.2
        return make_scorer(
            partial_auroc_weighted, needs_proba=True, max_fpr=0.25, pi_deploy=pi
        ).set_score_request(sample_weight=True)
    if scoring_metric == "average_prec":
        return make_scorer(
            average_precision_score, greater_is_better=True, needs_proba=True
        ).set_score_request(sample_weight=True)
    raise ValueError(f"Unsupported scoring metric: {scoring_metric}")


def _build_seeded_context(
    cached: _CachedDataset,
    seed: int,
    mlflow_tracker: MLFlowTracker,
    lr_class_weight: str | None = None,
) -> TrainingContext:
    """Construct a fresh TrainingContext for one seed, reusing cached features.

    Mirrors the seed-dependent half of src.train.load_data_and_setup.

    If ``lr_class_weight`` is provided, override the LR's ``class_weight``
    attribute after construction (only applies if the classifier is LR).
    """
    config = cached.config
    tc = config.training_config.model_copy(update={"random_seed": seed})

    classifier = get_classifier_map(tc)
    if lr_class_weight is not None and tc.classifier_name == "LogisticRegression":
        # Pass through sklearn convention: 'balanced' = inverse-frequency
        # class weights; 'none' (our CLI value) -> None (sklearn default).
        cw = None if lr_class_weight == "none" else lr_class_weight
        classifier.set_params(class_weight=cw)
        logger.info("LR class_weight override applied: %s", cw)
    param_grid = {**tc.classifier_hyperparams, **tc.pca_hyperparams}

    outer_cv = PersonAwareRepeatedStratifiedKFold(
        n_splits=tc.n_splits_outer,
        n_repeats=tc.n_repeats_outer,
        random_state=seed,
    )
    inner_cv = PersonAwareRepeatedStratifiedKFold(
        n_splits=tc.n_splits_inner,
        n_repeats=tc.n_repeats_inner,
        random_state=seed,
    )
    scorer = _build_scorer(tc.scoring_metric, tc.pi_deploy)

    eval_manager = MLFlowEvaluationManager(
        mlflow_tracker=mlflow_tracker,
        metadata=cached.training_metadata,
        feature_config=config.feature_config,
        preloader=cached.preloader,
        demographic_feature_columns=tc.demographic_feature_columns,
        include_voice_features=tc.include_voice_features,
    )

    return TrainingContext(
        training_metadata=cached.training_metadata,
        concatenated_features=cached.concatenated_features,
        parsel_feature_count=cached.parsel_feature_count,
        demographic_feature_count=cached.demographic_feature_count,
        classifier_name=tc.classifier_name,
        classifier=classifier,
        param_grid=param_grid,
        outer_cv_strategy=outer_cv,
        inner_cv_strategy=inner_cv,
        scorer=scorer,
        y_training=cached.y_training,
        label_encoder=cached.label_encoder,
        audio_ids=cached.audio_ids,
        eval_manager=eval_manager,
    )


# ----------------------------------------------------------------------------
# Per-seed run wrapper
# ----------------------------------------------------------------------------


def _run_one_seed(
    cached: _CachedDataset,
    seed: int,
    base_config_id: str,
    experiment_name: str,
    lr_class_weight: str | None = None,
) -> dict[str, Any]:
    """Open a fresh MLflow run, build a seed-specific TrainingContext, and
    execute the nested CV. Returns a tiny summary dict for the JSON index.
    """
    config = cached.config
    seed_config_id = f"{base_config_id}__seed{seed:02d}"

    # IMPORTANT: write the seed into the training_config that gets logged.
    # Pydantic v2 BaseModels are immutable in practice — copy with update.
    logged_config = deepcopy(config)
    logged_config.training_config = config.training_config.model_copy(
        update={"random_seed": seed}
    )

    mlflow_tracker = MLFlowTracker(experiment_name, seed_config_id)
    summary: dict[str, Any] = {
        "seed": seed,
        "config_id": seed_config_id,
        "experiment_name": experiment_name,
        "status": "started",
    }

    try:
        with mlflow_tracker.start_run():
            mlflow_tracker.log_experiment_config(logged_config)
            mlflow_tracker.log_final_metrics(
                {
                    "n_samples": cached.n_samples(),
                    "n_features": int(cached.concatenated_features.shape[1]),
                    "n_patients": cached.n_persons(),
                    "seed_sweep_seed": seed,
                }
            )

            context = _build_seeded_context(
                cached, seed, mlflow_tracker,
                lr_class_weight=lr_class_weight,
            )
            run_training(context, logged_config)

            summary["status"] = "finished"
            if mlflow_tracker.run is not None:
                summary["run_id"] = mlflow_tracker.run.info.run_id
    except Exception as exc:  # noqa: BLE001 — surface as JSON, keep loop alive
        logger.exception("seed=%d failed: %s", seed, exc)
        summary["status"] = "failed"
        summary["error"] = repr(exc)
    return summary


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True,
                        help="Path to seed-sweep tuple config YAML")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for logs + per-tuple summary JSON")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="List of random seeds to iterate over")
    parser.add_argument(
        "--weighting",
        type=str,
        choices=["ps_overlap", "class_only", "overlap_ps_and_deploy",
                 "strata_only", "flat"],
        default=None,
        help=(
            "Override training_config.sample_weight_strategy. If omitted, "
            "the value from the config YAML is used."
        ),
    )
    parser.add_argument(
        "--lr-class-weight",
        type=str,
        choices=["balanced", "none"],
        default=None,
        help=(
            "Override LogisticRegression's class_weight argument. "
            "Default (None): sklearn default = unweighted. 'balanced': "
            "inverse class frequency. Only applies when the classifier is "
            "LogisticRegression; ignored otherwise."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    with open(args.config) as f:
        config_data = yaml.safe_load(f)

    if not config_data:
        logger.error("Config %s is empty — nothing to run", args.config)
        return

    config = ExperimentConfig(**config_data)
    config_filename = Path(args.config).name
    if not config_filename.endswith("__seed_sweep_config.yml"):
        raise ValueError(
            "Seed-sweep config filename must end with '__seed_sweep_config.yml'"
        )
    base_config_id = config_filename.replace("__seed_sweep_config.yml", "")

    if args.weighting is not None:
        config.training_config = config.training_config.model_copy(
            update={"sample_weight_strategy": args.weighting}
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.output_dir) / timestamp
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{base_config_id}_seed_sweep_{timestamp}.log"
    setup_application_logging(log_file_path=str(log_file))

    logger.info("=== SEED SWEEP STARTED ===")
    logger.info("Config:           %s", args.config)
    logger.info("Base config id:   %s", base_config_id)
    logger.info("Seeds:            %s", args.seeds)
    logger.info("Weighting:        %s", config.training_config.sample_weight_strategy)
    logger.info("LR class_weight:  %s", args.lr_class_weight or "(default = None)")
    logger.info("Log file:         %s", log_file)

    # Build cached dataset ONCE.
    cached = _CachedDataset(config)
    logger.info("Cached dataset ready: n_samples=%d, n_persons=%d",
                cached.n_samples(), cached.n_persons())

    task_name = config.classification_task or config.task_config.target_column
    experiment_name = create_experiment_name(
        task_name, date=config.date,
        experiment_version=config.experiment_version,
    )

    summaries = []
    for seed in args.seeds:
        logger.info("--- seed %d / %d ---", seed, args.seeds[-1])
        summaries.append(
            _run_one_seed(
                cached, seed, base_config_id, experiment_name,
                lr_class_weight=args.lr_class_weight,
            )
        )

    index_path = out_dir / f"{base_config_id}__seed_sweep_index.json"
    with open(index_path, "w") as f:
        json.dump(
            {
                "base_config_id": base_config_id,
                "experiment_name": experiment_name,
                "weighting": config.training_config.sample_weight_strategy,
                "lr_class_weight": args.lr_class_weight,
                "seeds": args.seeds,
                "summaries": summaries,
            },
            f,
            indent=2,
        )
    logger.info("Wrote summary index: %s", index_path)


if __name__ == "__main__":
    main()
