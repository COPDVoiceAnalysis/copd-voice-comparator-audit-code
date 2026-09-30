"""Full UK comparator set, patient bootstrap, all on the voice_uk person set
so paired diffs are clean.

voice_uk / age_sex_uk / age_sex_comorb_uk : main-workflow single-seed runs
(stored OOF predictions; same as before).
age_only_uk : computed as the strata-weighted AUROC of `age` on exactly the
voice_uk participants. For a single monotone feature the pipeline's age-only
logistic regression is rank-equivalent to age within each fold, so
wAUROC(age-only) = wAUROC(age) up to a per-fold monotone refit; this avoids a
cohort-mismatched re-run and keeps the paired comparison exact.

Writes outputs/_tmp_boot/uk_full_results.json, which Figure 2 reads; run from
the repository root. Its inputs are local-only, since data/ is gitignored.

Formerly outputs/_tmp_boot/run_uk_full.py, which rounded every result to three
decimals and is otherwise unchanged here. Unrounded, it reproduces every value
of the three-decimal file it used to write.
"""
import glob, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(20260517)
B = 5000
EXP = "data/2025-08-13/mlflow_cluster/271009020022062271"
UKMETA = "data/uk_matched_metadata.csv"
RUNS = {
    "voice_uk":          ["de720542f5974ee8998d966ca75a536f"],
    "age_sex_comorb_uk": ["2569dbc2b5444f50b6187910b263c8ff",
                          "4906d6f489c24893a804cb14cdd7be68"],
    "age_sex_uk":        ["629aed426f2f4a4b9df891fc8b08b402"],
}

def pool(run_ids):
    dfs = []
    for rid in run_ids:
        for p in glob.glob(f"{EXP}/{rid}/artifacts/fold_predictions/*.parquet"):
            dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)
    g = df.groupby("audio_id")
    return pd.DataFrame({"y": g["true"].first().astype(int),
                         "p": g["proba_copd"].mean(),
                         "w": g["strata_weight"].mean()})

P = {k: pool(v) for k, v in RUNS.items()}

# age-only on the voice_uk person set: p = age (higher age -> higher COPD)
vk = P["voice_uk"]
m = pd.read_csv(UKMETA).drop_duplicates("audio_id").set_index("audio_id")
age = pd.to_numeric(m["age"], errors="coerce")
common = vk.index.intersection(age.dropna().index)
P["age_only_uk"] = pd.DataFrame({
    "y": vk.loc[common, "y"],
    "p": age.loc[common].astype(float),
    "w": vk.loc[common, "w"],
})

def boot(df, paired=None):
    y = df["y"].to_numpy(); p = df["p"].to_numpy(); w = df["w"].to_numpy()
    po, ne = np.where(y == 1)[0], np.where(y == 0)[0]
    pt = roc_auc_score(y, p, sample_weight=w)
    a, dd = [], ([] if paired is not None else None)
    if paired is not None:
        p2 = paired["p"].to_numpy(); w2 = paired["w"].to_numpy()
        pdiff = pt - roc_auc_score(y, p2, sample_weight=w2)
    for _ in range(B):
        ix = np.concatenate([RNG.choice(po, len(po), True),
                             RNG.choice(ne, len(ne), True)])
        try:
            a.append(roc_auc_score(y[ix], p[ix], sample_weight=w[ix]))
            if paired is not None:
                dd.append(a[-1] - roc_auc_score(y[ix], p2[ix], sample_weight=w2[ix]))
        except ValueError:
            pass
    a = np.array(a)
    # Unrounded on purpose: Figure 2 rounds once, for display. Rounding here
    # as well is double rounding -- how a bound of 0.4147 came to print 0.42.
    r = {"point": float(pt),
         "ci": [float(np.quantile(a, .025)),
                float(np.quantile(a, .975))],
         "n_pos": int(len(po)), "n_neg": int(len(ne))}
    if paired is not None:
        d = np.array(dd)
        r["diff"] = float(pdiff)
        r["diff_ci"] = [float(np.quantile(d, .025)),
                        float(np.quantile(d, .975))]
        r["frac_gt0"] = float((d > 0).mean())
    return r

out = {}
for k in ["voice_uk", "age_only_uk", "age_sex_uk", "age_sex_comorb_uk"]:
    out[k] = boot(P[k])
    print(k, json.dumps(out[k]))
print()
v = P["voice_uk"]
for dn in ["age_only_uk", "age_sex_uk", "age_sex_comorb_uk"]:
    d = P[dn]
    c = v.index.intersection(d.index)
    vv = v.loc[c].reset_index(drop=True)
    dd = d.loc[c].reset_index(drop=True)
    assert (vv["y"].to_numpy() == dd["y"].to_numpy()).all(), dn
    r = boot(vv, paired=dd)
    out[f"paired_voice_minus_{dn}"] = r
    print(f"paired voice-{dn} (n={len(c)}):", json.dumps(r))
json.dump(out, open("outputs/_tmp_boot/uk_full_results.json", "w"), indent=2)
