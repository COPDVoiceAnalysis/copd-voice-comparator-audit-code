"""
Age/sex stratum weights for target population standardization.

DMP reference (Germany): age bands ≤55: 8.6%, 56-65: 27.5%, 66-75: 34.1%, >75: 29.8%;
sex: female ~46.3%, male ~53.7%.

Coarse 3-band aggregation: ≤65: 36.1%, 66-75: 34.1%, >75: 29.8%.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# DMP reference: fine-grained 4-band age distribution
DMP_AGE_EDGES = (55, 65, 75)  # bins: <=55, 56-65, 66-75, >75
DMP_AGE_LABELS = ("<=55", "56-65", "66-75", ">75")
DMP_AGE_PROPORTIONS = (0.086, 0.275, 0.341, 0.298)

# Coarse 3-band aggregation (recommended for small strata)
COARSE_AGE_EDGES = (65, 75)
COARSE_AGE_LABELS = ("<=65", "66-75", ">75")
COARSE_AGE_PROPORTIONS = (0.361, 0.341, 0.298)  # DMP-based: <=65 = 0.086+0.275

# DMP sex distribution
DMP_SEX_P_FEMALE = 0.463  # P(sex=0/w)
DMP_SEX_P_MALE = 0.537    # P(sex=1/m)


def assign_age_band(
    age: np.ndarray,
    edges: tuple[float, ...] | None = None,
    labels: tuple[str, ...] | None = None,
) -> np.ndarray:
    """
    Assign age band label per sample.

    Args:
        age: (n,) ages in years.
        edges: band boundaries. Default: COARSE_AGE_EDGES (3 bands).
        labels: band labels. Default: COARSE_AGE_LABELS.
    """
    if edges is None:
        edges = COARSE_AGE_EDGES
    if labels is None:
        labels = COARSE_AGE_LABELS
    age = np.asarray(age, dtype=float)
    n_edges = len(edges)
    out = np.full(age.shape, labels[-1], dtype=object)
    out[age <= edges[0]] = labels[0]
    for i in range(1, n_edges):
        out[(age > edges[i - 1]) & (age <= edges[i])] = labels[i]
    return out


def stratum_id_string(age_band: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """Stratum key per sample: 'age_band_sex' with sex 0/1 (w/m)."""
    return np.array([f"{b}_{int(s)}" for b, s in zip(age_band, sex)], dtype=object)


def _count_controls_per_stratum(
    stratum_ids: np.ndarray, y: np.ndarray
) -> dict[str, int]:
    """Count controls (y=0) per stratum."""
    ctrl = y == 0
    ids = stratum_ids[ctrl]
    out: dict[str, int] = {}
    for g in np.unique(ids):
        out[str(g)] = int(np.sum(ids == g))
    return out


def merge_age_bands_if_needed(
    age_bands: np.ndarray,
    sex: np.ndarray,
    y: np.ndarray,
    min_controls_per_stratum: int = 2,
    age_proportions: tuple[float, ...] = COARSE_AGE_PROPORTIONS,
    age_labels: tuple[str, ...] = COARSE_AGE_LABELS,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    If any (age_band, sex) stratum has < min_controls_per_stratum controls,
    merge adjacent age bands until all strata have >= min_controls_per_stratum.

    Returns:
        merged_band_labels: (n,) str after merging.
        P_target: dict stratum_key -> target probability.
    """
    labels = list(age_labels)
    proportions = list(age_proportions)
    current_bands = np.array(age_bands, dtype=object, copy=True)
    n_bands = len(labels)
    merged_labels = labels.copy()
    merged_props = proportions.copy()

    while n_bands >= 2:
        stratum_ids = stratum_id_string(current_bands, sex)
        counts = _count_controls_per_stratum(stratum_ids, y)
        ok = True
        for key, c in counts.items():
            if c < min_controls_per_stratum:
                ok = False
                break
        if ok:
            break
        # Merge first two bands
        b0, b1 = merged_labels[0], merged_labels[1]
        if b0 == "<=65" and b1 == "66-75":
            new_label = "<=75"
        else:
            new_label = f"{b0}+{b1}"
        p_merged = merged_props[0] + merged_props[1]
        merged_labels = [new_label] + merged_labels[2:]
        merged_props = [p_merged] + merged_props[2:]
        n_bands = len(merged_labels)
        np.place(current_bands, current_bands == b1, new_label)
        np.place(current_bands, current_bands == b0, new_label)
        logger.debug(
            "Merged age bands %s and %s -> %s (min_controls=%d).",
            b0, b1, new_label, min_controls_per_stratum,
        )

    # Build P_target for final band labels
    P_target = {}
    for b, p_b in zip(merged_labels, merged_props):
        P_target[f"{b}_0"] = p_b * DMP_SEX_P_FEMALE
        P_target[f"{b}_1"] = p_b * DMP_SEX_P_MALE
    return current_bands, P_target


def compute_stratum_weights(
    stratum_ids: np.ndarray,
    y: np.ndarray,
    P_target: dict[str, float],
) -> np.ndarray:
    """
    w_i^(strata) = P_target(G=g_i) / P_fold(G=g_i | Y=y_i).

    P_fold(G|Y) is estimated from counts in the given (train or test) sample.
    """
    stratum_ids = np.asarray(stratum_ids, dtype=object)
    y = np.asarray(y, dtype=int)
    n = len(y)
    w = np.ones(n, dtype=float)
    for yi in (0, 1):
        mask = y == yi
        if not np.any(mask):
            continue
        ids_y = stratum_ids[mask]
        n_y = mask.sum()
        unique_g, cnt_g = np.unique(ids_y, return_counts=True)
        P_fold_given_y = {str(g): c / n_y for g, c in zip(unique_g, cnt_g)}
        for g in unique_g:
            g_str = str(g)
            pt = P_target.get(g_str, 1.0 / max(len(P_target), 1))
            pf = P_fold_given_y.get(g_str, 1e-12)
            wi = pt / pf
            w[np.where(mask & (stratum_ids == g))[0]] = wi
    return w


def clip_weights(
    w: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Clip weights to [low_percentile, high_percentile] of the weight distribution."""
    low = np.nanpercentile(w, low_percentile)
    high = np.nanpercentile(w, high_percentile)
    return np.clip(w, low, high)


def effective_sample_size(w: np.ndarray) -> float:
    """ESS = (sum w)^2 / sum(w^2)."""
    w = np.asarray(w, dtype=float)
    s1 = np.nansum(w)
    s2 = np.nansum(w * w)
    if s2 <= 0:
        return 0.0
    return float(s1 * s1 / s2)


def compute_strata_weights(
    age: np.ndarray,
    sex: np.ndarray,
    y: np.ndarray,
    min_controls_per_stratum: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Compute strata weights to reweight a sample to the DMP target population.

    Uses coarse 3-band age groups (≤65, 66-75, >75) with DMP-derived proportions
    and DMP sex distribution. Merges adjacent bands if any stratum has fewer than
    min_controls_per_stratum controls.

    Returns:
        w_strata: (n,) per-sample weights.
        info: dict with 'merged_bands', 'P_target', 'ess'.
    """
    age_bands = assign_age_band(age, edges=COARSE_AGE_EDGES, labels=COARSE_AGE_LABELS)
    merged_bands, P_target = merge_age_bands_if_needed(
        age_bands,
        sex,
        y,
        min_controls_per_stratum=min_controls_per_stratum,
        age_proportions=COARSE_AGE_PROPORTIONS,
        age_labels=COARSE_AGE_LABELS,
    )
    stratum_ids = stratum_id_string(merged_bands, sex)
    w_strata = compute_stratum_weights(stratum_ids, y, P_target)

    info = {
        "merged_bands": merged_bands,
        "P_target": P_target,
        "ess": effective_sample_size(w_strata),
    }
    logger.info("Strata weights (before clip): ESS=%.1f", info["ess"])
    return w_strata, info
