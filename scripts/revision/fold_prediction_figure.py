"""Fold-wise prediction figure (Reviewer 2, point 1).

(a) Out-of-fold COPD probability per participant (mean over recordings and the
10 outer repeats), by class, for the voice model and the age+sex comparator
trained with and without the ps_overlap weights.
(b) Strata-weighted AUROC of each of the 50 outer test folds (10 repeats x 5
splits) for the same three arms. A fold whose controls all fall in one stratum
or whose scores are constant still gets a value; folds where one class is
missing are skipped.

Usage: uv run python scripts/revision/fold_prediction_figure.py arms.json out.pdf
arms.json: {"arms": {"label": "<run_dir>", ...}} in plotting order.
"""
import glob
import json
import sys

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 9})
COLORS = {0: "#4C72B0", 1: "#DD8452"}


def load(run_dir):
    files = glob.glob(f"{run_dir}/artifacts/fold_predictions/*.parquet")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def fold_wauroc(df):
    out = []
    for _, g in df.groupby("fold"):
        if g["true"].nunique() == 2:
            out.append(roc_auc_score(g["true"], g["proba_copd"], sample_weight=g["strata_weight"]))
    return np.array(out)


def main():
    arms = json.load(open(sys.argv[1]))["arms"]
    data = {k: load(v) for k, v in arms.items()}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.4, 1]})
    rng = np.random.default_rng(0)
    for i, (label, df) in enumerate(data.items()):
        part = df.groupby("audio_id").agg(y=("true", "first"), p=("proba_copd", "mean"))
        for cls, off in [(0, -0.18), (1, 0.18)]:
            v = part.loc[part.y == cls, "p"].to_numpy()
            ax1.scatter(i + off + rng.uniform(-0.07, 0.07, len(v)), v, s=7, alpha=0.6,
                        color=COLORS[cls], label=("Control" if cls == 0 else "COPD") if i == 0 else None)
            ax1.hlines(np.median(v), i + off - 0.12, i + off + 0.12, color="black", lw=1.2)
        f = fold_wauroc(df)
        ax2.boxplot(f, positions=[i], widths=0.5, showfliers=False)
        ax2.scatter(i + rng.uniform(-0.15, 0.15, len(f)), f, s=6, alpha=0.5, color="grey")
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(data)), list(data), rotation=0)
        ax.spines[["top", "right"]].set_visible(False)
    ax1.set_ylabel("Out-of-fold P(COPD)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(frameon=False, loc="lower right", fontsize=8)
    ax1.set_title("(a) Participant-level predictions", loc="left", fontsize=9)
    ax2.axhline(0.5, ls="--", color="grey", lw=0.8)
    ax2.set_ylabel("Weighted AUROC per test fold")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("(b) 50 outer test folds", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(sys.argv[2])
    fig.savefig(sys.argv[2].rsplit(".", 1)[0] + ".png", dpi=200)
    for label, df in data.items():
        f = fold_wauroc(df)
        print(f"{label}: folds={len(f)} median={np.median(f):.3f} IQR=[{np.percentile(f, 25):.3f}, "
              f"{np.percentile(f, 75):.3f}] sd_of_P={df.groupby('audio_id').proba_copd.mean().std():.3f}")


if __name__ == "__main__":
    main()
