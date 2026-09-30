"""
Deployment metrics and scoring functions.

PPV, NPV, expected FP at deployment prevalence, and partial AUROC scorer.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.utils.sample_weight_strategies import compute_sample_weights


def ppv_deploy(pi: float, tpr: float, fpr: float) -> float:
    """PPV at deployment prevalence pi: P(disease | positive) = pi*TPR / (pi*TPR + (1-pi)*FPR)."""
    if pi <= 0 or pi >= 1:
        return float("nan")
    num = pi * tpr
    den = pi * tpr + (1.0 - pi) * fpr
    return num / den if den > 0 else float("nan")


def npv_deploy(pi: float, tpr: float, fpr: float) -> float:
    """NPV at deployment prevalence pi: P(no disease | negative) = (1-pi)*(1-FPR) / ((1-pi)*(1-FPR) + pi*(1-TPR))."""
    if pi <= 0 or pi >= 1:
        return float("nan")
    tnr = 1.0 - fpr
    fnr = 1.0 - tpr
    num = (1.0 - pi) * tnr
    den = (1.0 - pi) * tnr + pi * fnr
    return num / den if den > 0 else float("nan")


def expected_fp_per_1000(pi: float, fpr: float) -> float:
    """Expected number of false positives per 1000 screened: 1000 * (1 - pi) * FPR (in non-disease)."""
    return 1000.0 * (1.0 - pi) * fpr


def partial_auroc_weighted(y_true, y_score, max_fpr=0.2, pi_deploy=None, sample_weight=None):
    """Partial AUC (up to max_fpr), optionally weighted. Accepts sample_weight for sklearn scorer compatibility."""
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
    else:
        w = compute_sample_weights(y_true, pi_deploy=pi_deploy)
    return roc_auc_score(y_true, y_score, sample_weight=w, max_fpr=max_fpr)
