"""Sensitivity and specificity at fixed operating points (Reviewer 1, point 7).

Reads the out-of-fold predictions of one or more MLflow runs, averages each
participant's COPD probability over recordings and outer repeats (as
paper_bootstrap_charite.py does), and reports strata-weighted sensitivity at
specificity 0.80 and 0.90 and specificity at sensitivity 0.80 and 0.90, with a
class-stratified participant bootstrap (B = 5000).

These "apparent" values read the threshold off the pooled out-of-fold ROC, so
the threshold is chosen on the same predictions it is scored on, which is
optimistic. The "cross_fitted" block avoids that: within each outer repeat,
the threshold for test fold k is chosen on the participants of the other folds
(their out-of-fold scores) and applied to fold k; sensitivity and specificity
are then pooled over the folds of the repeat, and the block reports the mean
and the range over repeats. A threshold that targets specificity 0.90 then
reaches roughly that specificity on new participants, not exactly.

Usage: uv run python scripts/revision/operating_points.py <run_dir> [<run_dir> ...]
       uv run python scripts/revision/operating_points.py arms.json
       (arms.json as for paired_bootstrap.py; results are keyed by arm label)
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


K_OUTER = 5  # outer splits per repeat (training_config.n_splits_outer of all revision runs)


def pick_threshold(y, p, w, kind, level):
    """Score threshold (positive if p >= t) meeting the target on these participants."""
    ts = np.unique(p)
    pos, neg = y == 1, y == 0
    sens = np.array([w[pos & (p >= t)].sum() for t in ts]) / w[pos].sum()
    spec = np.array([w[neg & (p < t)].sum() for t in ts]) / w[neg].sum()
    if kind == "spec":  # lowest threshold with spec >= level, i.e. the most sensitive one
        ok = np.where(spec >= level)[0]
        return ts[ok[0]] if len(ok) else np.inf
    ok = np.where(sens >= level)[0]  # highest threshold with sens >= level
    return ts[ok[-1]] if len(ok) else -np.inf


def cross_fitted(run_dir):
    files = glob.glob(f"{run_dir}/artifacts/fold_predictions/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["repeat"] = (df["fold"] - 1) // K_OUTER
    per_repeat = {k: [] for k in TARGETS}
    for _, r in df.groupby("repeat"):
        part = r.groupby(["fold", "audio_id"]).agg(y=("true", "first"), p=("proba_copd", "mean"),
                                                   w=("strata_weight", "mean")).reset_index()
        if part["y"].nunique() < 2:
            continue
        for name, (kind, level) in TARGETS.items():
            pred = np.zeros(len(part), dtype=bool)
            for k in part["fold"].unique():
                te = (part["fold"] == k).to_numpy()
                tr = ~te
                t = pick_threshold(part.y[tr].to_numpy(), part.p[tr].to_numpy(), part.w[tr].to_numpy(), kind, level)
                pred[te] = part.p[te].to_numpy() >= t
            y, w = part.y.to_numpy(), part.w.to_numpy()
            sens = w[(y == 1) & pred].sum() / w[y == 1].sum()
            spec = w[(y == 0) & ~pred].sum() / w[y == 0].sum()
            per_repeat[name].append((sens, spec))
    out = {}
    for name, vals in per_repeat.items():
        if not vals:
            continue
        v = np.array(vals)
        out[name] = {"sensitivity": [round(float(v[:, 0].mean()), 3), round(float(v[:, 0].min()), 3),
                                     round(float(v[:, 0].max()), 3)],
                     "specificity": [round(float(v[:, 1].mean()), 3), round(float(v[:, 1].min()), 3),
                                     round(float(v[:, 1].max()), 3)],
                     "n_repeats": len(v)}
    return out


def main():
    res = {}
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        targets = json.load(open(sys.argv[1]))["arms"]
    else:
        targets = {d: d for d in sys.argv[1:]}
    for label, run_dir in targets.items():
        y, p, w = load(run_dir)
        est = points(y, p, w)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        draws = {k: [] for k in est}
        for _ in range(B):
            idx = np.r_[RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))]
            for k, v in points(y[idx], p[idx], w[idx]).items():
                draws[k].append(v)
        res[label] = {"n_copd": int(len(pos)), "n_control": int(len(neg)),
                      **{k: [round(est[k], 3), round(float(np.percentile(draws[k], 2.5)), 3),
                             round(float(np.percentile(draws[k], 97.5)), 3)] for k in est},
                      "cross_fitted": cross_fitted(run_dir),
                      "note": "apparent values: [estimate, 2.5%, 97.5%] from the pooled ROC; "
                              "cross_fitted: [mean, min, max] over outer repeats, threshold chosen on the other folds"}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
