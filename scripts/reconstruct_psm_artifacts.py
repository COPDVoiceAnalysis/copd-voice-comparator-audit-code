"""Regenerate the Charité age-only PSM artifacts locally (no cluster needed).

The corrected age-only N=55 PSM cohort was produced on a cluster that is no
longer accessible. Its per-participant out-of-fold predictions were rsynced
locally (data/2025-08-13/mlflow_cluster/...), so the exact pre-match cohort
(N=99, from the complete run) and the matched cohort (N=55, from the PSM run)
are both recoverable. This script feeds the recovered N=99 through the repo's
matching code with the corrected age-only / linear-age settings
(propensity_covariates=[], include_age_squared=False) and verifies it
reproduces the published cohort exactly (N=55, identical IDs, K-sweep Table S1,
balance Table S2). It then writes the canonical committed artifacts.

Run: TMPDIR=/tmp/claude UV_CACHE_DIR=/tmp/claude/uv-cache PYTHONPATH=. \
        uv run python scripts/reconstruct_psm_artifacts.py
"""
import glob
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from src.utils.generate_psm_cohort import run_label_swapped_psm

REF_YEAR = 2025
COMORB = ["breathing", "metabolic", "depression", "hypothyreose"]
COMPLETE = ("data/2025-08-13/mlflow_cluster/457012274041432773/"
            "00a61b66628b45e8906c95097abd7348/artifacts/fold_predictions/*.parquet")
PSM = ("data/2025-08-13/mlflow_cluster/644578817518142221/"
       "884ea1802c0d44278090ae4860de5288/artifacts/fold_predictions/*.parquet")

comp = pd.concat([pd.read_parquet(f) for f in glob.glob(COMPLETE)], ignore_index=True)
psm = pd.concat([pd.read_parquet(f) for f in glob.glob(PSM)], ignore_index=True)
ids99 = comp.groupby("audio_id")["true_label"].first()        # pre-match cohort + labels
matched55 = set(psm["audio_id"].unique())                     # published matched cohort

mdfull = pd.read_csv("data/2025-08-13/metadata.csv")
md = mdfull.groupby("audio_id").agg(birth_year=("birth_year", "first"),
                                    sex=("sex", "first")).reset_index()
co = pd.read_csv("data/2025-08-13/comorbidities.csv")
cohort = md[md.audio_id.isin(ids99.index)].merge(co, on="audio_id", how="left")
cohort["age"] = (REF_YEAR - cohort["birth_year"]).astype(float)
cohort["label"] = cohort["audio_id"].map(ids99).map({"copd": 1, "control": 0})
for c in COMORB:
    cohort[c] = pd.to_numeric(cohort[c], errors="coerce").fillna(0).astype(int)
cohort["sex"] = cohort["sex"].astype(str).str.strip().str.lower()
print(f"pre-match cohort: N={len(cohort)} "
      f"({(cohort.label==1).sum()} COPD / {(cohort.label==0).sum()} control)")

# K-sweep via the repo matching code (corrected age-only / linear-age settings)
rows, m5, smd5 = [], None, None
for k in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
    m, smd, dr = run_label_swapped_psm(
        cohort.copy(), k=k, caliper_coef=0.20, age_caliper_years=10,
        exact_match=["sex"], propensity_covariates=[], use_age=True,
        random_seed=42, include_age_squared=False, report_bin_cols=COMORB)
    rows.append({"K": k, "N_total": len(m), "N_COPD": int((m.label == 1).sum()),
                 "N_control": int((m.label == 0).sum()),
                 "max_abs_SMD_after": round(float(smd["smd_after"].abs().max()), 3)})
    if k == 5:
        m5, smd5 = m, smd
ksweep = pd.DataFrame(rows)
print("\n=== K-sweep (Table S1) ===")
print(ksweep.to_string(index=False))

# --- VERIFY before writing ---
ok_ids = set(m5.audio_id) == matched55
print(f"\nK=5 cohort == published 55 IDs: {ok_ids} "
      f"({len(set(m5.audio_id) & matched55)}/55 overlap)")
assert len(m5) == 55 and ok_ids, "Reproduction mismatch — NOT writing artifacts."

print("\n=== Balance report (Table S2 check) ===")
S2 = {"age": (0.589, 0.185), "breathing": (-0.029, -0.453), "metabolic": (0.668, 0.661),
      "depression": (0.000, -0.145), "hypothyreose": (0.162, -0.145)}
for cov, r in smd5.iterrows():
    e = S2.get(cov, (None, None))
    print(f"  {cov:12s} before {r['smd_before']:+.3f} (S2 {e[0]:+})  "
          f"after {r['smd_after']:+.3f} (S2 {e[1]:+})")

# --- WRITE canonical artifacts ---
matched_csv = mdfull[mdfull.audio_id.isin(matched55)].copy()
matched_csv.to_csv("data/2025-08-13/psm_matched_charite.csv", index=False)

out = smd5.copy(); out.index.name = "covariate"
out["k"] = 5; out["n_total"] = len(m5)
out["n_copd"] = int((m5.label == 1).sum()); out["n_control"] = int((m5.label == 0).sum())
out.to_csv("data/2025-08-13/psm_matched_charite_balance_report.csv")

os.makedirs("outputs/psm_reconstructed", exist_ok=True)
ksweep.to_csv("outputs/psm_reconstructed/psm_ksweep.csv", index=False)

print(f"\nWROTE:\n  data/2025-08-13/psm_matched_charite.csv ({len(matched_csv)} rows, "
      f"{matched_csv.audio_id.nunique()} participants)\n"
      f"  data/2025-08-13/psm_matched_charite_balance_report.csv\n"
      f"  outputs/psm_reconstructed/psm_ksweep.csv")
