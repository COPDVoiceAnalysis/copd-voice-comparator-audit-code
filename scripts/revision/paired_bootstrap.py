"""Paired participant bootstrap of AUROC differences between arms (Reviewer 1, point 1).

Each run's out-of-fold predictions are averaged per participant over recordings
and outer repeats (as in operating_points.py). Every bootstrap draw resamples
participants within class once and scores all arms on the same draw, so the
differences are paired. Reports strata-weighted AUROC (wAUROC) and plain AUROC,
each arm with a 95% CI, and each listed contrast with a 95% CI and the share of
draws in which the difference is <= 0.

Usage: uv run python scripts/revision/paired_bootstrap.py arms.json
arms.json: {"arms": {"name": "<run_dir>", ...}, "contrasts": [["a", "b"], ...]}
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20261111)
B = 5000


def load(run_dir):
    files = glob.glob(f"{run_dir}/artifacts/fold_predictions/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    g = df.groupby("audio_id")
    return pd.DataFrame({"y": g["true"].first().astype(int), "p": g["proba_copd"].mean(),
                         "w": g["strata_weight"].mean()})


def scores(y, p, w):
    return roc_auc_score(y, p, sample_weight=w), roc_auc_score(y, p)


def ci(a):
    return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]


def main():
    spec = json.load(open(sys.argv[1]))
    data = {k: load(v) for k, v in spec["arms"].items()}
    ids = sorted(set.intersection(*[set(d.index) for d in data.values()]))
    data = {k: d.loc[ids] for k, d in data.items()}
    ref = next(iter(data.values()))
    y, w = ref["y"].to_numpy(), ref["w"].to_numpy()
    for k, d in data.items():
        assert (d["y"].to_numpy() == y).all(), k
    P = {k: d["p"].to_numpy() for k, d in data.items()}
    est = {k: scores(y, p, w) for k, p in P.items()}
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    draws = {k: [] for k in P}
    for _ in range(B):
        idx = np.r_[RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))]
        for k, p in P.items():
            draws[k].append(scores(y[idx], p[idx], w[idx]))
    draws = {k: np.array(v) for k, v in draws.items()}
    out = {"n": len(ids), "n_copd": int(len(pos)), "n_control": int(len(neg)), "arms": {}, "contrasts": {}}
    for k in P:
        out["arms"][k] = {"wauroc": [round(est[k][0], 3), *ci(draws[k][:, 0])],
                          "auroc": [round(est[k][1], 3), *ci(draws[k][:, 1])]}
    for a, b in spec["contrasts"]:
        d = draws[a] - draws[b]
        out["contrasts"][f"{a} - {b}"] = {
            m: [round(est[a][i] - est[b][i], 3), *ci(d[:, i]), round(float((d[:, i] <= 0).mean()), 4)]
            for i, m in enumerate(["wauroc", "auroc"])}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
