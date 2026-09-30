"""Seed-sweep-FREE patient bootstrap for Charite (complete + PSM).

All numbers from SINGLE reproducible runs:
 - voice / age+sex / age+sex+comorbidity : main-workflow single-seed
   (seed 42) runs from the local mlflow_cluster mirror.
 - age-only : one single seed of the age_only run (no averaging, no seed
   CI, NOT a seed-sweep aggregate) — functionally a single standalone run.
Same scheme as the UK bootstrap (class-stratified person resampling,
B=5000, strata-weighted AUROC, paired diffs).

Writes outputs/_tmp_boot/charite_singleseed_results.json, which Figure 2
reads; run from the repository root. Its inputs are local-only, since data/
and outputs/ are gitignored.

Recovered from the 2026-06-14 session that first wrote that file: it was
outputs/_tmp_boot/run_charite_singleseed_bootstrap.py, never committed and
later lost. The only change is that results are no longer rounded to four
decimals. Unrounded, it reproduces every value of the file it first wrote.

The PSM block mixes cohorts. voice_psm and age_only_psm are the published
N = 55 analysis (43/12); age_sex_psm, age_sexcomorb_psm and their paired rows
are the superseded 50-participant cohort (37/13, paired 24/11).
"""
import glob, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20260516)
B = 5000
MF = "data/2025-08-13/mlflow_cluster"
AGE = "outputs/_tmp_age_only_mlruns"

# (experiment_dir, run_id) for single-seed main-workflow runs; or explicit parquet
RUNS = {
    "voice_complete":        (f"{MF}/457012274041432773", "00a61b66628b45e8906c95097abd7348"),
    "age_sex_complete":      (f"{MF}/457012274041432773", "b9dc00497d77462e8a76f9515b348af2"),
    "age_sexcomorb_complete":(f"{MF}/457012274041432773", "56f48804a8674b4ab6a5ed2098886c19"),
    "age_only_complete":     ("PARQUET", "outputs/_tmp_age_only_mlruns/821226550920393827/a0b5bd9da0384ecdb66d70c5c0206cac/artifacts/fold_predictions/tmp_ll2hj2w.parquet"),
    # PSM rows: CORRECTED rerun v2 (2026-05-20). Both complete-case AND
    # exclude_longitudinal applied BEFORE matching -> zero post-match pairing
    # break. age-only PS, flat weights. Cohort N=55 (43 COPD / 12 control),
    # fully consistent (matched == trained). voice_psm (parselmouth poem
    # spectral SVC, split) is the only voice_only run on the N = 55 cohort; in
    # the superseded N = 50 grid the same configuration ranked 100 of 222.
    # age_only_psm is an in-pipeline MLflow run (LR, extended), not the old
    # _tmp parquet -- but it resolved no scope and trained on 193 Parselmouth
    # /a/ features plus age, sex and BMI, so it is not an age-only model.
    # Figure 2 uses raw age instead (paper_comparator_weighting_analysis.py).
    "voice_psm":             (f"{MF}/644578817518142221", "884ea1802c0d44278090ae4860de5288"),
    "age_sex_psm":           (f"{MF}/644578817518142221", "f9fe235589d14de0a4c69b0fb2428829"),     # OLD (not rerun; not in paper PSM block)
    "age_sexcomorb_psm":     (f"{MF}/644578817518142221", "e387533845b0449498f4d6241ca900ee"),     # OLD (not rerun; not in paper PSM block)
    "age_only_psm":          (f"{MF}/644578817518142221", "8f8cfddc0ae14767b5a72d18e7e5f4db"),
}

def load(spec):
    kind, ref = spec
    if kind == "PARQUET":
        files = [ref]
    else:
        files = glob.glob(f"{kind}/{ref}/artifacts/fold_predictions/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    g = df.groupby("audio_id")
    return pd.DataFrame({"y": g["true"].first().astype(int),
                         "p": g["proba_copd"].mean(),
                         "w": g["strata_weight"].mean()})

def boot(person, paired=None):
    y, p, w = person["y"].to_numpy(), person["p"].to_numpy(), person["w"].to_numpy()
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    pw = roc_auc_score(y, p, sample_weight=w)
    aw, dd = [], ([] if paired is not None else None)
    if paired is not None:
        p2, w2 = paired["p"].to_numpy(), paired["w"].to_numpy()
        pdiff = pw - roc_auc_score(y, p2, sample_weight=w2)
    for _ in range(B):
        idx = np.concatenate([RNG.choice(pos, len(pos), True),
                              RNG.choice(neg, len(neg), True)])
        try:
            a = roc_auc_score(y[idx], p[idx], sample_weight=w[idx])
        except ValueError:
            continue
        aw.append(a)
        if paired is not None:
            try: dd.append(a - roc_auc_score(y[idx], p2[idx], sample_weight=w2[idx]))
            except ValueError: pass
    aw = np.array(aw)
    # Unrounded on purpose: Figure 2 rounds once, for display. Rounding here
    # as well is double rounding -- how a bound of 0.4147 came to print 0.42.
    r = {"point": float(pw),
         "ci": [float(np.quantile(aw, .025)),
                float(np.quantile(aw, .975))],
         "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    if paired is not None:
        d = np.array(dd)
        r["diff_point"] = float(pdiff)
        r["diff_ci"] = [float(np.quantile(d, .025)),
                        float(np.quantile(d, .975))]
        r["frac_diff_gt0"] = float((d > 0).mean())
    return r

P = {k: load(v) for k, v in RUNS.items()}
for k, df in P.items():
    print(f"{k}: {len(df)} persons ({int(df['y'].sum())}/{int((df['y']==0).sum())})")

print()
out = {}
for k in RUNS:
    out[k] = boot(P[k])
    print(k, json.dumps(out[k]))

print()
for coh in ["complete", "psm"]:
    v = P[f"voice_{coh}"]
    for demo in ["age_only", "age_sex", "age_sexcomorb"]:
        d = P[f"{demo}_{coh}"]
        common = v.index.intersection(d.index)
        vv = v.loc[common].reset_index(drop=True)
        dd = d.loc[common].reset_index(drop=True)
        assert (vv["y"].to_numpy() == dd["y"].to_numpy()).all(), f"label mismatch {coh}/{demo}"
        r = boot(vv, paired=dd)
        out[f"paired_{coh}_voice_minus_{demo}"] = r
        print(f"paired {coh} voice-{demo} (n={len(common)}):", json.dumps(r))

json.dump(out, open("outputs/_tmp_boot/charite_singleseed_results.json", "w"), indent=2)
print("\nwrote outputs/_tmp_boot/charite_singleseed_results.json")
