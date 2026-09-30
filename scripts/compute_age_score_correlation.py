"""Among held-out controls, correlate the voice model's predicted COPD
probability with participant age (main text Section 4.3, r-bar values).

Traceability script for the claim that the voice score tracks age even among
controls. For each cohort it pools the voice model's out-of-fold predictions,
restricts to controls, computes Pearson r(proba_copd, age) within each of the
10 outer repeats, and reports the mean (r-bar). Emits a small JSON.

Run: TMPDIR=/tmp/claude UV_CACHE_DIR=/tmp/claude/uv-cache \
        uv run python scripts/compute_age_score_correlation.py
"""
import glob
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

MF = "data/2025-08-13/mlflow_cluster"
# voice OOF runs (same configs the paper bootstraps)
CHARITE_VOICE = f"{MF}/457012274041432773/00a61b66628b45e8906c95097abd7348/artifacts/fold_predictions/*.parquet"
UK_VOICE = f"{MF}/271009020022062271/de720542f5974ee8998d966ca75a536f/artifacts/fold_predictions/*.parquet"


def rbar(pred_glob, age_by_id, control_label="control"):
    df = pd.concat([pd.read_parquet(f) for f in glob.glob(pred_glob)], ignore_index=True)
    df["age"] = df["audio_id"].map(age_by_id)
    df = df[(df["true_label"] == control_label) & df["age"].notna()]
    # one correlation per outer repeat; fold ids run 1..50 and encode
    # repeat*5 + k with k = 1..5, so the repeat is (fold - 1) // 5. Fall back
    # to a single pooled correlation if no repeat structure is recoverable.
    rs = []
    if "fold" in df.columns and df["fold"].nunique() >= 5:
        df["repeat"] = (df["fold"].astype(int) - 1) // 5
        for _, g in df.groupby("repeat"):
            gg = g.groupby("audio_id").agg(p=("proba_copd", "mean"), age=("age", "first"))
            if gg["age"].nunique() > 1:
                rs.append(np.corrcoef(gg["p"], gg["age"])[0, 1])
    if not rs:
        gg = df.groupby("audio_id").agg(p=("proba_copd", "mean"), age=("age", "first"))
        rs = [np.corrcoef(gg["p"], gg["age"])[0, 1]]
    return float(np.mean(rs)), int(df["audio_id"].nunique()), len(rs)


# Charité age = reference_year - birth_year
md_c = pd.read_csv("data/2025-08-13/metadata.csv").groupby("audio_id")["birth_year"].first()
age_c = (2025 - md_c).to_dict()

# UK age = decade midpoints already stored as 'age'
uk = pd.read_csv("data/uk_matched_metadata.csv")
age_col = "age" if "age" in uk.columns else [c for c in uk.columns if "age" in c.lower()][0]
age_uk = uk.groupby("audio_id")[age_col].first().to_dict()

out = {}
for name, glb, ages in [("charite", CHARITE_VOICE, age_c), ("uk", UK_VOICE, age_uk)]:
    r, n, nrep = rbar(glb, ages)
    out[name] = {"rbar_age_score_controls": round(r, 4), "n_controls": n, "n_repeats": nrep}
    print(f"{name}: r-bar(age, voice score | controls) = {r:+.3f}  "
          f"(n_controls={n}, repeats={nrep})")

json.dump(out, open("outputs/age_score_correlation.json", "w"), indent=2)
print("\nwrote outputs/age_score_correlation.json")
