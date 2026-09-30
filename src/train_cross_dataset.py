"""
Cross-dataset validation: train on one cohort, evaluate on the other.

Trains models on Charité (or UK COVID-19 Sounds) with inner-CV for
hyperparameter selection, then evaluates the final model on the held-out
cohort.  Voice-only scope (no demographics).

Usage:
    uv run python src/train_cross_dataset.py \
        --date 2025-08-13 \
        --preproc norm_on \
        --output_dir outputs/cross_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.calibration import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.features.preprocessing import W2VParselPCA
from src.features.preloader import FeaturePreloader
from src.models.config_models import (
    ExperimentConfig,
    FeatureConfig,
    ParselmouthConfig,
    TaskConfig,
    TrainingConfig,
)
from src.models.metadata import Metadata
from src.utils.person_aware_cv import (
    PersonAwareRepeatedStratifiedKFold,
    PersonAwareStratifiedKFold,
)
from src.utils.person_aware_gridsearch import PersonAwareGridSearchCV
from src.utils.sample_weight_strategies import (
    compute_evaluation_strata_weights,
    compute_fold_weights,
)
from src.utils.logging_config import setup_application_logging

import sklearn

sklearn.set_config(enable_metadata_routing=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRATCH_TPL = "{home}/work/pipeline_features/{date}"
RAW_DIR_TPL = "/data/cephfs-1/work/groups/mittermaier/stimmaufnahmen/{date}"

# Charité / UK task definitions (mirror pipeline_config.yml)
CHARITE_TASK = dict(
    target_column="lung_disease_main",
    target_classes=["copd", "control"],
    exclude_longitudinal=True,
    only_longitudinal=False,
    use_temporal_pairs=False,
    included_data_sources=["charite"],
)
UK_TASK = dict(
    target_column="lung_disease_main",
    target_classes=["copd", "control"],
    exclude_longitudinal=True,
    only_longitudinal=False,
    use_temporal_pairs=False,
    included_data_sources=["uk_covid"],
)

# Extended hyperparameter grids (from pipeline_config.yml)
CLF_HYPERPARAMS = {
    "RandomForestClassifier": {
        "clf__n_estimators": [100, 200, 500],
        "clf__max_depth": [5, 10, 25],
        "clf__max_features": ["sqrt", "log2"],
        "clf__min_samples_leaf": [2, 4],
    },
    "SVC": {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", "auto"],
        "clf__kernel": ["linear", "rbf"],
    },
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 0.5],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["saga"],
    },
}

PCA_HYPERPARAMS = {
    "PCA__wav2vec_n_components": [15, 30, 50],
}

CLASSIFIERS = {
    "RandomForestClassifier": lambda seed, n_jobs: RandomForestClassifier(
        random_state=seed, n_jobs=n_jobs,
    ),
    "SVC": lambda seed, _: SVC(
        probability=True, random_state=seed,
    ),
    "LogisticRegression": lambda seed, n_jobs: LogisticRegression(
        random_state=seed, max_iter=1000, n_jobs=n_jobs,
    ),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class CohortData:
    """All data for a single cohort."""
    name: str
    features: np.ndarray
    labels: np.ndarray
    audio_ids: np.ndarray
    label_encoder: LabelEncoder
    metadata: Metadata
    parsel_feature_count: int
    demographic_feature_count: int


def _build_feature_config(
    preproc: str,
    splitting: str,
    scratch_base: str,
) -> FeatureConfig:
    """Create a wav2vec-poem FeatureConfig."""
    embeddings_dir = Path(scratch_base) / f"wav2vec2__{preproc}"
    # parsel_feature_file is required by FeatureConfig validation but unused
    # for wav2vec-only configs (recordings_parsel is empty). Pick the first
    # path that exists; the file is never actually read.
    candidate_parsel_files = [
        Path(scratch_base) / f"parselmouth__{preproc}" / "acoustic_features.npy",
        Path(scratch_base) / "parselmouth__norm_on" / "acoustic_features.npy",
        Path(scratch_base) / "parselmouth__norm_off" / "acoustic_features.npy",
    ]
    parsel_feature_file = next(
        (p for p in candidate_parsel_files if p.exists()),
        candidate_parsel_files[0],
    )
    return FeatureConfig(
        recordings_parsel=(),
        recordings_wav2vec2=("poem",),
        wav2vec2_layer=4,
        wav2vec2_embedding_statistics="mean",
        parsel_feature_file=parsel_feature_file,
        embeddings_dir=embeddings_dir,
        parselmouth_config=ParselmouthConfig(),
    )


def load_cohort(
    name: str,
    task_dict: dict,
    splitting: str,
    feature_config: FeatureConfig,
    metadata_csv: Path,
    ps_covariates_file: Path | None,
) -> CohortData:
    """Load features and metadata for one cohort."""
    task_config = TaskConfig(
        **task_dict,
        use_split_poems=(splitting == "split"),
    )

    md = Metadata(csv_path=metadata_csv, ps_covariates_file=ps_covariates_file)
    md.filter_for_experiment(task_config=task_config, feature_config=feature_config)

    rec_id_sets = md.get_recording_identifier_sets(feature_config)
    preloader = FeaturePreloader(feature_config)
    features, parsel_count, demo_count = preloader.load_features(
        rec_id_sets, demographic_features=None, include_voice_features=True,
    )

    labels, le = md.get_class_labels_for_sets(rec_id_sets)
    audio_ids = md.extract_audio_ids(rec_id_sets)

    logger.info(
        f"Cohort {name}: {features.shape[0]} samples, "
        f"{features.shape[1]} features, "
        f"classes={dict(zip(le.classes_, np.bincount(labels)))}"
    )
    return CohortData(
        name=name,
        features=features,
        labels=labels,
        audio_ids=audio_ids,
        label_encoder=le,
        metadata=md,
        parsel_feature_count=parsel_count,
        demographic_feature_count=demo_count,
    )


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def _make_pipeline(clf_name: str, seed: int, n_jobs: int) -> Pipeline:
    """Build preprocessing + classifier pipeline (voice-only, no demographics)."""
    clf = CLASSIFIERS[clf_name](seed, n_jobs)
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("vt", VarianceThreshold(0.01)),
        ("scaler", StandardScaler().set_fit_request(sample_weight=True)),
        (
            "PCA",
            W2VParselPCA(
                random_state=seed,
                parsel_feature_count=0,      # wav2vec-only
                demographic_feature_count=0,  # voice-only
            ),
        ),
        ("clf", clf.set_fit_request(sample_weight=True)),
    ])


def _make_param_grid(clf_name: str) -> dict:
    """Combine classifier + PCA hyperparams into a single grid."""
    return {**CLF_HYPERPARAMS[clf_name], **PCA_HYPERPARAMS}


def bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_weight: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute AUROC (optionally strata-weighted) with bootstrapped confidence interval.

    When ``sample_weight`` is provided, the point estimate and every bootstrap
    resample use ``roc_auc_score(..., sample_weight=...)``, yielding a
    strata-weighted (DMP-target) AUROC suitable for direct comparison with the
    within-cohort wAUROC reported elsewhere in the thesis.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    point = roc_auc_score(y_true, y_score, sample_weight=sample_weight)
    rng = np.random.RandomState(seed)
    boot_scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        # Ensure both classes are present in the bootstrap sample
        if len(np.unique(y_true[idx])) < 2:
            continue
        w = sample_weight[idx] if sample_weight is not None else None
        boot_scores.append(roc_auc_score(y_true[idx], y_score[idx], sample_weight=w))
    boot_scores = np.array(boot_scores)
    ci_lo = np.percentile(boot_scores, 100 * alpha / 2)
    ci_hi = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return point, ci_lo, ci_hi


def _find_optimal_wba_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, sample_weight: np.ndarray | None = None
) -> float:
    """Find threshold maximising (weighted) balanced accuracy via ROC curve.

    Skips the degenerate first point (threshold=inf → all-negative) so the
    search only considers reachable operating points.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, sample_weight=sample_weight)
    wba = (tpr + 1.0 - fpr) / 2.0
    if len(thresholds) > 2:
        search_slice = slice(1, len(thresholds))
        best_idx = search_slice.start + np.argmax(wba[search_slice])
        return float(thresholds[best_idx])
    elif len(thresholds) == 2:
        return float(thresholds[1])
    return 0.5


def _balanced_accuracy_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> float:
    """Compute balanced accuracy at a given decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return float(balanced_accuracy_score(y_true, y_pred))


def bootstrap_wba(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute wBA at *threshold* with bootstrapped CI.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    point = _balanced_accuracy_at_threshold(y_true, y_score, threshold)
    rng = np.random.RandomState(seed)
    boot_scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_scores.append(_balanced_accuracy_at_threshold(y_true[idx], y_score[idx], threshold))
    boot_scores = np.array(boot_scores)
    ci_lo = float(np.percentile(boot_scores, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return point, ci_lo, ci_hi


def train_and_evaluate(
    train_data: CohortData,
    test_data: CohortData,
    clf_name: str,
    seed: int = 42,
    n_jobs: int = -1,
    use_ps_weights: bool = True,
) -> dict:
    """Train on one cohort with GridSearchCV, evaluate on the other.

    Returns a dict with AUROC, CI, best params, wBA_opt, etc.
    """
    direction = f"{train_data.name} -> {test_data.name}"
    logger.info(f"=== {direction} | {clf_name} ===")

    # --- Inner CV for hyperparameter selection ---
    inner_cv = PersonAwareRepeatedStratifiedKFold(
        n_splits=5, n_repeats=3, random_state=seed,
    )
    pipeline = _make_pipeline(clf_name, seed, n_jobs)
    param_grid = _make_param_grid(clf_name)

    # Compute sample weights for training
    if use_ps_weights:
        training_config_stub = TrainingConfig(
            classifier_name=clf_name,
            sample_weight_strategy="ps_overlap",
        )
        weights_train = compute_fold_weights(
            "ps_overlap",
            train_data.metadata,
            train_data.audio_ids,
            train_data.labels,
            training_config_stub,
        )
    else:
        weights_train = np.ones(len(train_data.labels))

    scorer = sklearn.metrics.make_scorer(
        roc_auc_score, greater_is_better=True, needs_proba=True,
    ).set_score_request(sample_weight=True)

    grid_search = PersonAwareGridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring=scorer,
        n_jobs=n_jobs,
        return_train_score=False,
        verbose=1,
        audio_ids=train_data.audio_ids,
    )

    logger.info(f"Running GridSearchCV ({len(param_grid)} HP dimensions)...")
    grid_search.fit(
        train_data.features, train_data.labels, sample_weight=weights_train,
    )
    logger.info(f"Best inner-CV score: {grid_search.best_score_:.4f}")
    logger.info(f"Best params: {grid_search.best_params_}")

    # --- OOF predictions for optimal threshold selection ---
    # Use non-repeated CV (partitions, no overlap) for cross_val_predict
    cv_for_predict = PersonAwareStratifiedKFold(
        n_splits=5, shuffle=True, random_state=seed,
    )
    est = clone(grid_search.best_estimator_)
    oof_scores = cross_val_predict(
        est, train_data.features, train_data.labels,
        cv=cv_for_predict.split(train_data.features, train_data.labels,
                                audio_ids=train_data.audio_ids),
        method="predict_proba",
        fit_params={"sample_weight": weights_train},
    )[:, 1]

    tau_opt = _find_optimal_wba_threshold(train_data.labels, oof_scores,
                                          sample_weight=weights_train)
    logger.info(f"Optimal wBA threshold from OOF: {tau_opt:.4f}")

    # --- Evaluate on external test cohort ---
    best_model = grid_search.best_estimator_
    y_score_chunks = best_model.predict_proba(test_data.features)[:, 1]

    # Aggregate chunk-level predictions to person level (no-op if unchunked)
    from src.utils.model_evaluation_mlflow import _aggregate_chunks_to_persons

    y_score, y_true_agg, person_ids = _aggregate_chunks_to_persons(
        y_score_chunks, test_data.labels, test_data.audio_ids,
    )
    auroc, ci_lo, ci_hi = bootstrap_auroc(y_true_agg, y_score)

    # Strata-weighted AUROC against the same DMP target population used for
    # within-cohort wAUROC. Strata are defined by (age-bin, sex) from the
    # test cohort's metadata; weights are independent of the training cohort.
    # This yields a direct, like-for-like comparison with the within-cohort
    # wAUROC headlines (0.79 / 0.61) reported elsewhere in the thesis.
    w_strata = compute_evaluation_strata_weights(
        test_data.metadata, person_ids, y_true_agg,
    )
    if w_strata is not None:
        weighted_auroc, w_ci_lo, w_ci_hi = bootstrap_auroc(
            y_true_agg, y_score, sample_weight=w_strata,
        )
    else:
        logger.warning("Strata weights unavailable for %s; wAUROC will be NaN", direction)
        weighted_auroc, w_ci_lo, w_ci_hi = float("nan"), float("nan"), float("nan")

    wba_opt, wba_ci_lo, wba_ci_hi = bootstrap_wba(y_true_agg, y_score, tau_opt)

    logger.info(f"External AUROC: {auroc:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
    logger.info(f"External wAUROC: {weighted_auroc:.3f} [{w_ci_lo:.3f}, {w_ci_hi:.3f}]  (DMP-target strata)")
    logger.info(f"External wBA_opt: {wba_opt:.3f} [{wba_ci_lo:.3f}, {wba_ci_hi:.3f}]  (τ={tau_opt:.3f})")

    return {
        "direction": direction,
        "train_cohort": train_data.name,
        "test_cohort": test_data.name,
        "classifier": clf_name,
        "n_train": len(train_data.labels),
        "n_test": len(y_true_agg),
        "best_inner_cv_score": round(grid_search.best_score_, 4),
        "best_params": {k: str(v) for k, v in grid_search.best_params_.items()},
        "auroc": round(auroc, 4),
        "auroc_ci_lower": round(ci_lo, 4),
        "auroc_ci_upper": round(ci_hi, 4),
        "weighted_auroc": round(weighted_auroc, 4) if not np.isnan(weighted_auroc) else None,
        "weighted_auroc_ci_lower": round(w_ci_lo, 4) if not np.isnan(w_ci_lo) else None,
        "weighted_auroc_ci_upper": round(w_ci_hi, 4) if not np.isnan(w_ci_hi) else None,
        "wba_opt": round(wba_opt, 4),
        "wba_opt_ci_lower": round(wba_ci_lo, 4),
        "wba_opt_ci_upper": round(wba_ci_hi, 4),
        "threshold_opt": round(tau_opt, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cross-dataset validation")
    parser.add_argument("--date", required=True, help="Pipeline date (e.g. 2025-08-13)")
    parser.add_argument("--preproc", default="norm_on", help="Preprocessing variant")
    parser.add_argument("--output_dir", default="outputs/cross_dataset", help="Output directory")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--splittings", nargs="+", default=["unsplit", "split"],
        help="Splitting variants to test",
    )
    parser.add_argument(
        "--classifiers", nargs="+",
        default=["RandomForestClassifier", "SVC", "LogisticRegression"],
        help="Classifiers to test",
    )
    parser.add_argument("--no_ps_weights", action="store_true", help="Disable PS weights")
    parser.add_argument(
        "--experiment_version", default=None,
        help="Optional version suffix for MLFlow experiment names (e.g. 'v2')",
    )
    parser.add_argument(
        "--charite_metadata_csv", default=None, type=Path,
        help=(
            "Optional override for the Charité-cohort metadata CSV. When set "
            "(e.g. the PSM-matched cohort produced by rule match_charite_psm) "
            "this file is used for the Charité side only; the UK cohort still "
            "loads from the standard preprocessed metadata. Used by the "
            "Snakemake cross_dataset_validation rule to run PSM-matched "
            "cross-cohort validation as the primary analysis."
        ),
    )
    args = parser.parse_args()

    setup_application_logging(logging.INFO)

    home = Path.home()
    scratch_base = SCRATCH_TPL.format(home=home, date=args.date)
    raw_dir = RAW_DIR_TPL.format(date=args.date)

    metadata_csv = Path(scratch_base) / f"preprocessed__{args.preproc}" / "metadata.csv"
    # Charité metadata may be overridden (e.g. PSM-matched cohort). The UK
    # cohort always loads from the standard preprocessed metadata.
    charite_metadata_csv = args.charite_metadata_csv or metadata_csv
    ps_covariates_charite = Path(raw_dir) / "comorbidities.csv"
    ps_covariates_uk = Path(raw_dir) / "uk_matched_metadata_ps_covariates.csv"
    if args.charite_metadata_csv:
        logger.info(
            "Charité metadata overridden: %s (UK still uses %s)",
            charite_metadata_csv, metadata_csv,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for splitting in args.splittings:
        logger.info(f"\n{'='*60}")
        logger.info(f"Splitting: {splitting}")
        logger.info(f"{'='*60}")

        feature_config = _build_feature_config(args.preproc, splitting, scratch_base)

        # Load both cohorts
        charite = load_cohort(
            name="Charité",
            task_dict=CHARITE_TASK,
            splitting=splitting,
            feature_config=feature_config,
            metadata_csv=charite_metadata_csv,
            ps_covariates_file=ps_covariates_charite,
        )
        uk = load_cohort(
            name="UK",
            task_dict=UK_TASK,
            splitting=splitting,
            feature_config=feature_config,
            metadata_csv=metadata_csv,
            ps_covariates_file=ps_covariates_uk,
        )

        for clf_name in args.classifiers:
            # Direction 1: Charité → UK
            result = train_and_evaluate(
                train_data=charite,
                test_data=uk,
                clf_name=clf_name,
                seed=args.seed,
                n_jobs=args.n_jobs,
                use_ps_weights=not args.no_ps_weights,
            )
            result["splitting"] = splitting
            all_results.append(result)

            # Direction 2: UK → Charité
            result = train_and_evaluate(
                train_data=uk,
                test_data=charite,
                clf_name=clf_name,
                seed=args.seed,
                n_jobs=args.n_jobs,
                use_ps_weights=not args.no_ps_weights,
            )
            result["splitting"] = splitting
            all_results.append(result)

    # --- Save results ---
    output = {
        "date": args.date,
        "experiment_version": args.experiment_version,
        "preproc": args.preproc,
        "charite_metadata_csv": str(charite_metadata_csv),
        "charite_cohort_variant": (
            "psm_matched" if args.charite_metadata_csv else "full"
        ),
        "results": all_results,
    }
    results_file = output_dir / "cross_dataset_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    # --- Print summary table ---
    print("\n" + "=" * 100)
    print("CROSS-DATASET VALIDATION RESULTS")
    print("=" * 100)
    print(
        f"{'Direction':<25} {'Clf':<15} {'Split':<10} "
        f"{'AUROC':>8}  {'95% CI':>18}  "
        f"{'wAUROC':>8}  {'95% CI':>18}  "
        f"{'wBA_opt':>8}  {'95% CI':>18}  {'τ':>6}"
    )
    print("-" * 100)
    for r in all_results:
        auroc_ci = f"[{r['auroc_ci_lower']:.3f}, {r['auroc_ci_upper']:.3f}]"
        wauroc = r.get('weighted_auroc')
        wauroc_str = f"{wauroc:.3f}" if wauroc is not None else "  n/a"
        wauroc_ci_str = (
            f"[{r['weighted_auroc_ci_lower']:.3f}, {r['weighted_auroc_ci_upper']:.3f}]"
            if wauroc is not None else "  n/a"
        )
        wba_ci = f"[{r['wba_opt_ci_lower']:.3f}, {r['wba_opt_ci_upper']:.3f}]"
        print(
            f"{r['direction']:<25} {r['classifier']:<15} {r['splitting']:<10} "
            f"{r['auroc']:>8.3f}  {auroc_ci:>18}  "
            f"{wauroc_str:>8}  {wauroc_ci_str:>18}  "
            f"{r['wba_opt']:>8.3f}  {wba_ci:>18}  {r['threshold_opt']:>6.3f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
