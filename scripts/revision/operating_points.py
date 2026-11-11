"""Sensitivity and specificity at fixed operating points (Reviewer 1, point 7).

Reads the out-of-fold predictions of one or more MLflow runs, averages each
participant's COPD probability over recordings and outer repeats (as
paper_bootstrap_charite.py does), and reports strata-weighted sensitivity at
specificity 0.80 and 0.90 and specificity at sensitivity 0.80 and 0.90, with a
class-stratified participant bootstrap (B = 5000).

Usage: uv run python scripts/revision/operating_points.py <run_dir> [<run_dir> ...]
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20261111)
B = 5000
TARGETS = {"sens_at_spec80": ("spec", 0.80), "sens_at_spec90": ("spec", 0.90),
           "spec_at_sens80": ("sens", 0.80), "spec_at_sens90": ("sens", 0.90)}


def load(run_dir):
    files = glob.glob(f"{run_dir}/artifacts/fold_predictions/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    g = df.groupby("audio_id")
    return (g["true"].first().astype(int).to_numpy(), g["proba_copd"].mean().to_numpy(),
            g["strata_weight"].mean().to_numpy())


def weighted_roc(y, p, w):
    """Weighted (fpr, tpr) over all thresholds, highest threshold first."""
    order = np.argsort(-p, kind="mergesort")
    y, p, w = y[order], p[order], w[order]
    tp = np.cumsum(w * (y == 1)) / np.sum(w * (y == 1))
    fp = np.cumsum(w * (y == 0)) / np.sum(w * (y == 0))
    keep = np.r_[np.diff(p) != 0, True]
    return np.r_[0, fp[keep]], np.r_[0, tp[keep]]


def points(y, p, w):
    fpr, tpr = weighted_roc(y, p, w)
    spec = 1 - fpr
    out = {}
    for name, (kind, level) in TARGETS.items():
        if kind == "spec":
            out[name] = float(tpr[spec >= level].max())
        else:
            out[name] = float(spec[tpr >= level].max())
    out["wauroc"] = float(roc_auc_score(y, p, sample_weight=w))
    return out


def main():
    res = {}
    for run_dir in sys.argv[1:]:
        y, p, w = load(run_dir)
        est = points(y, p, w)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        draws = {k: [] for k in est}
        for _ in range(B):
            idx = np.r_[RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))]
            for k, v in points(y[idx], p[idx], w[idx]).items():
                draws[k].append(v)
        res[run_dir] = {"n_copd": int(len(pos)), "n_control": int(len(neg)),
                        **{k: [round(est[k], 3), round(float(np.percentile(draws[k], 2.5)), 3),
                               round(float(np.percentile(draws[k], 97.5)), 3)] for k in est}}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
