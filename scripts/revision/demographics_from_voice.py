"""How much age and sex the voice embeddings carry, and whether the voice COPD
score is an age proxy (Reviewer 1, point 2; Charite only).

Features: wav2vec2-large-robust layer 4, mean-pooled poem chunks (norm_off), as in
the selected voice model, averaged per participant. Models: StandardScaler ->
PCA(30) -> Ridge (age) or logistic regression (sex), 5-fold x 10-repeat CV, one
row per participant so folds are person-disjoint.

Part A decodes age (R^2, MAE against the training-fold mean) and sex (AUROC) in
the full cohort (N = 99), the PSM-matched cohort (N = 55) and within COPD
(n = 77). Part B takes the voice model's out-of-fold COPD score and reports its
Spearman correlation with age within each class, and the likelihood-ratio test
for the score in a logistic model COPD ~ age + sex + score (full and PSM).

Usage: uv run python scripts/revision/demographics_from_voice.py <voice_run_dir> out.json
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "data/2025-08-13"
REPEATS, K, NPC = 10, 5, 30


def features():
    a = np.load(f"{DATA}/wav2vec2__norm_off/layer_4/poem/mean.npy", allow_pickle=True)[0]
    emb = pd.DataFrame(a["mean_matrix"], index=a["recording_identifiers"])
    m = pd.read_csv(f"{DATA}/preprocessed__norm_off/metadata.csv")
    m = m[m.recording_identifier.isin(emb.index)]
    x = emb.loc[m.recording_identifier].groupby(m.audio_id.to_numpy()).mean()
    meta = m.drop_duplicates("audio_id").set_index("audio_id")[["age", "sex", "control"]]
    meta["male"] = (meta.sex == "m").astype(int)
    return x, meta


def cv_age(x, age, seed=0):
    preds = np.zeros((REPEATS, len(age)))
    base = np.zeros_like(preds)
    for r in range(REPEATS):
        for tr, te in KFold(K, shuffle=True, random_state=seed + r).split(x):
            m = make_pipeline(StandardScaler(), PCA(min(NPC, len(tr) - 1)), Ridge(alpha=10.0))
            m.fit(x[tr], age[tr])
            preds[r, te] = m.predict(x[te])
            base[r, te] = age[tr].mean()
    p, b = preds.mean(0), base.mean(0)
    return {"r2": round(r2_score(age, p), 3), "mae": round(mean_absolute_error(age, p), 2),
            "mae_mean_baseline": round(mean_absolute_error(age, b), 2),
            "pearson_r": round(float(np.corrcoef(age, p)[0, 1]), 3)}


def cv_sex(x, male, seed=0):
    preds = np.zeros((REPEATS, len(male)))
    for r in range(REPEATS):
        for tr, te in StratifiedKFold(K, shuffle=True, random_state=seed + r).split(x, male):
            m = make_pipeline(StandardScaler(), PCA(min(NPC, len(tr) - 1)), LogisticRegression(C=0.1, max_iter=2000))
            m.fit(x[tr], male[tr])
            preds[r, te] = m.predict_proba(x[te])[:, 1]
    return {"auroc": round(roc_auc_score(male, preds.mean(0)), 3), "n_male": int(male.sum()), "n": int(len(male))}


def lr_test(df):
    X0 = sm.add_constant(df[["age", "male"]].astype(float))
    X1 = sm.add_constant(df[["age", "male", "score"]].astype(float))
    f0 = sm.Logit(df.y, X0).fit(disp=0)
    f1 = sm.Logit(df.y, X1).fit(disp=0)
    stat = 2 * (f1.llf - f0.llf)
    return {"lr_chi2": round(float(stat), 2), "p": float(chi2.sf(stat, 1)),
            "coef_score": round(float(f1.params["score"]), 3), "coef_age_without_score": round(float(f0.params["age"]), 4),
            "coef_age_with_score": round(float(f1.params["age"]), 4)}


def main():
    x, meta = features()
    files = glob.glob(f"{sys.argv[1]}/artifacts/fold_predictions/*.parquet")
    fp = pd.concat([pd.read_parquet(f) for f in files])
    score = fp.groupby("audio_id").agg(y=("true", "first"), score=("proba_copd", "mean"))
    ids = score.index.intersection(x.index)
    psm = pd.read_csv(f"{DATA}/psm_matched_charite.csv").audio_id.unique()
    df = meta.loc[ids].join(score)
    out = {"n": int(len(ids)), "A": {}, "B": {}}
    cohorts = {"full": ids, "psm": ids.intersection(psm), "copd_only": df.index[df.y == 1]}
    for name, idx in cohorts.items():
        X = x.loc[idx].to_numpy()
        out["A"][name] = {"n": int(len(idx)), "age": cv_age(X, df.loc[idx, "age"].to_numpy(float)),
                          "sex": cv_sex(X, df.loc[idx, "male"].to_numpy())}
    for name, idx in [("full", cohorts["full"]), ("psm", cohorts["psm"])]:
        d = df.loc[idx]
        corr = {}
        for cls, lab in [(1, "copd"), (0, "control")]:
            s = d[d.y == cls]
            rho, p = spearmanr(s.age, s.score)
            corr[lab] = {"n": int(len(s)), "rho": round(float(rho), 3), "p": round(float(p), 4)}
        out["B"][name] = {"score_vs_age_within_class": corr, "lr_test_score_beyond_age_sex": lr_test(d)}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
