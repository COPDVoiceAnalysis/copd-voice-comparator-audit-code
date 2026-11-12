"""Small simulation of the comparator collapse under ps_overlap weights (Reviewer 2, point 1).

Synthetic cohorts with the Charite class sizes and class-wise age/sex summary
(77 COPD: age 69.6 +- 8.7, 52% male; 22 controls: age 61.4 +- 17.8, 36% male).
A synthetic "voice" vector has 10 dimensions: one carries age plus noise, one
carries a disease shift `delta` independent of age and sex, the rest are noise.

For each cohort, logistic models are fitted in 5-fold x 5-repeat stratified CV
with training weights either flat or ps_overlap (the pipeline's own
compute_overlap_weights on age and sex, fitted in the training fold). Scores are
averaged per participant over repeats and evaluated with the pipeline's
strata weights (wAUROC) and plain AUROC.

Why it collapses: the propensity model is a logistic regression on age and sex,
and its score equations make the overlap-weighted means of age and sex equal in
both classes. A weighted logistic regression on the same age and sex then has
slope near 0 (exactly 0 without PS clipping and L2), so it predicts a constant
and its AUROC is 0.5 whatever age and sex carry.

Usage: PYTHONPATH=. uv run python scripts/revision/simulate_comparator_collapse.py out_dir [R]
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.sample_weight_strategies import compute_overlap_weights
from src.utils.stratum_weights import clip_weights, compute_strata_weights

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N1, N0 = 77, 22
AGE = {1: (69.6, 8.7), 0: (61.4, 17.8)}
MALE = {1: 0.52, 0: 0.36}
DELTAS = [0.0, 0.5, 1.0]
D, AGE_NOISE = 10, 0.5
K, REPEATS = 5, 5


def cohort(rng, delta):
    y = np.r_[np.ones(N1, int), np.zeros(N0, int)]
    age = np.clip(np.where(y == 1, rng.normal(*AGE[1], len(y)), rng.normal(*AGE[0], len(y))), 25, 90)
    sex = (rng.random(len(y)) < np.where(y == 1, MALE[1], MALE[0])).astype(float)
    x = rng.normal(size=(len(y), D))
    x[:, 0] = (age - 66) / 12 + AGE_NOISE * rng.normal(size=len(y))
    x[:, 1] += delta * y
    return y, age, sex, x


def oof(rng, y, age, sex, feats, weighting):
    p = np.zeros((REPEATS, len(y)))
    meta = pd.DataFrame({"age": age, "sex": sex})
    for r in range(REPEATS):
        for tr, te in StratifiedKFold(K, shuffle=True, random_state=int(rng.integers(1e9))).split(feats, y):
            w = (compute_overlap_weights(meta.iloc[tr], y[tr], ["age", "sex"]) if weighting == "ps_overlap"
                 else np.ones(len(tr)))
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            m.fit(feats[tr], y[tr], logisticregression__sample_weight=w)
            p[r, te] = m.predict_proba(feats[te])[:, 1]
    return p.mean(0)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rng = np.random.default_rng(20261112)
    rows = []
    for delta in DELTAS:
        for rep in range(R):
            y, age, sex, x = cohort(rng, delta)
            ws, _ = compute_strata_weights(age, sex, y)
            ws = clip_weights(ws, 1.0, 99.0)
            for model, feats in [("Age+sex", np.c_[age, sex]), ("Voice", x)]:
                for weighting in ["flat", "ps_overlap"]:
                    p = oof(rng, y, age, sex, feats, weighting)
                    rows.append({"delta": delta, "rep": rep, "model": model, "weighting": weighting,
                                 "wauroc": roc_auc_score(y, p, sample_weight=ws),
                                 "auroc": roc_auc_score(y, p), "sd_p": float(p.std())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "simulation.csv", index=False)
    summ = (df.groupby(["delta", "model", "weighting"])
              .agg(wauroc_mean=("wauroc", "mean"), wauroc_lo=("wauroc", lambda v: np.percentile(v, 2.5)),
                   wauroc_hi=("wauroc", lambda v: np.percentile(v, 97.5)), auroc_mean=("auroc", "mean"),
                   sd_p_median=("sd_p", "median")).round(3).reset_index())
    summ.to_csv(out / "simulation_summary.csv", index=False)
    print(summ.to_string(index=False))

    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    arms = [("Age+sex", "flat"), ("Age+sex", "ps_overlap"), ("Voice", "flat"), ("Voice", "ps_overlap")]
    colors = ["#8c8c8c", "#4d4d4d", "#DD8452", "#C44E52"]
    labels = ["Age+sex, unweighted", "Age+sex, overlap-weighted", "Voice, unweighted", "Voice, overlap-weighted"]
    for j, (m, w) in enumerate(arms):
        for i, delta in enumerate(DELTAS):
            v = df[(df.delta == delta) & (df.model == m) & (df.weighting == w)].wauroc
            pos = i * 5 + j
            ax.boxplot(v, positions=[pos], widths=0.7, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": colors[j], "alpha": 0.8}, medianprops={"color": "black"})
        ax.plot([], [], "s", color=colors[j], label=labels[j])
    ax.axhline(0.5, ls="--", color="grey", lw=0.8)
    ax.set_xticks([i * 5 + 1.5 for i in range(len(DELTAS))],
                  [f"Voice disease signal = {d:g} SD" if d else "Voice carries age only" for d in DELTAS])
    ax.set_ylabel("Weighted AUROC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax.set_ylim(0.2, 1.05)
    fig.tight_layout()
    fig.savefig(out / "simulation.pdf")
    fig.savefig(out / "simulation.png", dpi=200)
    json.dump({"R": R, "deltas": DELTAS, "folds": K, "repeats": REPEATS}, open(out / "simulation_meta.json", "w"))


if __name__ == "__main__":
    main()
