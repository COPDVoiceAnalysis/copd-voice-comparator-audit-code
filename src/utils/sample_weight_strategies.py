"""
Sample weight computation and strategy orchestration.

Training weights:
- flat: no reweighting (uniform weights of 1.0)
- class_only: inverse class frequency (balanced)
- ps_overlap: propensity score overlap weights (confounding reduction)
- overlap_ps_and_deploy: PS overlap (alternative parameterization)
- strata_only: age×sex strata weights (DMP target population)

Evaluation weights:
- compute_evaluation_strata_weights: shared helper for threshold calibration
  and target-population-weighted metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from src.utils.stratum_weights import (
    clip_weights,
    compute_strata_weights,
    effective_sample_size,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight building blocks
# ---------------------------------------------------------------------------


def compute_sample_weights(y: np.ndarray, pi_deploy: float | None = None) -> np.ndarray:
    """Compute class-balance sample weights.

    If pi_deploy is None: balanced weights (inverse class frequency).
    If pi_deploy is given: reweight to target prevalence.
    """
    pi = y.mean()
    if pi_deploy is None:
        w_pos = 0.5 / (pi + 1e-12)
        w_neg = 0.5 / (1 - pi + 1e-12)
    else:
        w_pos = pi_deploy / (pi + 1e-12)
        w_neg = (1 - pi_deploy) / (1 - pi + 1e-12)
    return np.where(y == 1, w_pos, w_neg)


def compute_overlap_weights(
    metadata_df: pd.DataFrame,
    y: np.ndarray,
    covariate_cols: list[str],
    random_state: int = 42,
    max_weight: float = 10.0,
    ps_clip_low: float = 0.05,
    ps_clip_high: float = 0.95,
    C: float = 1.0,
) -> np.ndarray:
    """
    Compute overlap (propensity) weights for confounding reduction.

    w_i = 1 - p_i for cases (y=1), w_i = p_i for controls (y=0),
    where p_i = P(y=1|X) from L2 logistic regression on covariates.
    Propensity p is clipped to [ps_clip_low, ps_clip_high] for stability.

    Missing values are imputed with median.
    """
    use_cols = [c for c in covariate_cols if c in metadata_df.columns]
    if not use_cols:
        return np.ones(len(y), dtype=float)
    X = metadata_df[use_cols].copy()
    for c in use_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)
    clf = LogisticRegression(
        penalty="l2",
        max_iter=1000,
        random_state=random_state,
        C=C,
        solver="lbfgs",
    )
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    p = np.clip(p, ps_clip_low, ps_clip_high)
    w = np.where(y == 1, 1.0 - p, p)
    if max_weight is not None and max_weight < np.inf:
        w = np.clip(w, 1e-6, max_weight)
    return w


# ---------------------------------------------------------------------------
# Evaluation / threshold strata weights (shared by train.py and evaluator)
# ---------------------------------------------------------------------------


def compute_evaluation_strata_weights(
    metadata: Any,
    audio_ids: np.ndarray,
    y: np.ndarray,
    min_controls_per_stratum: int = 2,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> np.ndarray | None:
    """Compute strata weights for evaluation or threshold calibration.

    Reweights the sample to match the DMP target population (age×sex).
    Returns clipped strata weights, or None if metadata is unavailable.
    """
    try:
        meta = metadata.get_metadata_for_audio_id_sequence(
            audio_ids, columns=["age", "sex"]
        )
        age = pd.to_numeric(meta["age"], errors="coerce").to_numpy()
        sex = meta["sex"].to_numpy().astype(float)
        w_strata, _ = compute_strata_weights(
            age, sex, y,
            min_controls_per_stratum=min_controls_per_stratum,
        )
        return clip_weights(w_strata, clip_low, clip_high)
    except Exception as e:
        logger.warning("Could not compute strata weights: %s", e)
        return None


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def _log_overlap_weight_stats(
    w: np.ndarray,
    y: np.ndarray,
    covariate_cols: list[str],
    prefix: str = "Overlap (PS) weights",
) -> None:
    """Log ESS, min/max/mean, and mean-by-class for propensity overlap weights."""
    if w is None or len(w) == 0:
        return
    ess = effective_sample_size(w)
    mask1 = y == 1
    mask0 = ~mask1
    mean_c = w[mask0].mean() if mask0.any() else float("nan")
    mean_case = w[mask1].mean() if mask1.any() else float("nan")
    logger.info(
        "%s: ESS=%.2f, min=%.4f, max=%.4f, mean=%.4f",
        prefix,
        ess,
        float(w.min()),
        float(w.max()),
        float(w.mean()),
    )
    logger.info(
        "%s by class: mean(control)=%.4f, mean(case)=%.4f",
        prefix,
        mean_c,
        mean_case,
    )
    logger.info("%s covariates used: %s", prefix, covariate_cols)


# ---------------------------------------------------------------------------
# Strategy orchestration
# ---------------------------------------------------------------------------


def _normalize_weights_by_chunk_count(
    weights: np.ndarray,
    audio_ids: np.ndarray,
) -> np.ndarray:
    """Divide each sample's weight by the number of samples sharing its audio_id.

    When poems are chunked, multiple samples share the same audio_id.  Without
    normalization, participants with more chunks receive proportionally greater
    total weight, undermining propensity-score adjustment.  Dividing by chunk
    count ensures equal per-person total weight.  For unchunked data every
    audio_id appears once, so the operation is a no-op.
    """
    unique_ids, inverse, counts = np.unique(
        audio_ids, return_inverse=True, return_counts=True
    )
    chunk_counts = counts[inverse]  # per-sample count of siblings
    if chunk_counts.max() > 1:
        logger.info(
            "Chunk-count weight normalization: %d unique persons, "
            "chunk counts %d–%d (mean %.1f)",
            len(unique_ids),
            int(chunk_counts.min()),
            int(chunk_counts.max()),
            float(chunk_counts.mean()),
        )
    return weights / chunk_counts


def compute_fold_weights(
    strategy: str,
    training_metadata: Any,
    audio_ids_train: np.ndarray,
    y_train: np.ndarray,
    training_config: Any,
) -> np.ndarray:
    """
    Compute training sample weights for a fold by strategy.

    Returns:
        weights_train: (n_train,) sample weights for training.
    """

    if strategy == "class_only":
        weights_train = compute_sample_weights(y_train)
    elif strategy == "ps_overlap":
        _override = getattr(training_config, "overlap_covariate_columns", None)
        if _override is not None:
            simplified_cols = list(_override)
            logger.info(
                "ps_overlap: using explicit overlap_covariate_columns "
                "override from training_config: %s",
                simplified_cols,
            )
        else:
            simplified_cols = getattr(
                training_metadata,
                "overlap_covariate_columns_simplified",
                ["age", "sex"],
            )
        metadata_train = training_metadata.get_metadata_for_audio_id_sequence(
            audio_ids_train, columns=simplified_cols
        )
        covariate_cols = [c for c in simplified_cols if c in metadata_train.columns]
        weights_train = compute_overlap_weights(
            metadata_train,
            y_train,
            covariate_cols,
            random_state=training_config.random_seed,
            ps_clip_low=0.05,
            ps_clip_high=0.95,
            C=0.1,
        )
        _log_overlap_weight_stats(
            weights_train,
            y_train,
            covariate_cols,
            prefix="Overlap (PS) weights (ps_overlap)",
        )
        ess = effective_sample_size(weights_train)
        if ess < 20:
            logger.warning(
                "Overlap PS ESS=%.1f < 20; using flat weights for this fold.",
                ess,
            )
            weights_train = np.ones(len(y_train), dtype=float)
    elif strategy == "overlap_ps_and_deploy":
        _override = getattr(training_config, "overlap_covariate_columns", None)
        if _override is not None:
            simplified_cols = list(_override)
            logger.info(
                "overlap_ps_and_deploy: using explicit overlap_covariate_columns "
                "override from training_config: %s",
                simplified_cols,
            )
        else:
            simplified_cols = getattr(
                training_metadata,
                "overlap_covariate_columns_simplified",
                ["age", "sex"],
            )
        metadata_train = training_metadata.get_metadata_for_audio_id_sequence(
            audio_ids_train, columns=simplified_cols
        )
        covariate_cols = [c for c in simplified_cols if c in metadata_train.columns]
        w_overlap = compute_overlap_weights(
            metadata_train,
            y_train,
            covariate_cols,
            random_state=training_config.random_seed,
        )
        _log_overlap_weight_stats(
            w_overlap, y_train, covariate_cols, prefix="Overlap (PS) weights"
        )
        weights_train = w_overlap
    elif strategy == "strata_only":
        metadata_train = training_metadata.get_metadata_for_audio_id_sequence(
            audio_ids_train, columns=["age", "sex"]
        )
        age_train = metadata_train["age"].to_numpy(dtype=float)
        sex_train = metadata_train["sex"].to_numpy(dtype=float)
        w_strata_train, info_train = compute_strata_weights(
            age_train,
            sex_train,
            y_train,
            min_controls_per_stratum=training_config.min_controls_per_stratum,
        )
        weights_train = clip_weights(
            w_strata_train,
            low_percentile=training_config.weight_clip_low_percentile,
            high_percentile=training_config.weight_clip_high_percentile,
        )
        logger.info(
            "Strata-only weights ESS=%.1f (after clip)",
            effective_sample_size(weights_train),
        )
    elif strategy == "flat":
        weights_train = np.ones(len(y_train), dtype=float)
        logger.info("Flat weights (no reweighting): all weights = 1.0")
    else:
        raise ValueError(f"Unknown sample_weight_strategy: {strategy}")

    # Normalize weights by per-participant chunk count so that each person
    # contributes equal total weight regardless of how many chunks they have.
    # This is a no-op for unchunked data (each audio_id appears once).
    weights_train = _normalize_weights_by_chunk_count(weights_train, audio_ids_train)

    return weights_train
