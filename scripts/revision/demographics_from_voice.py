"""How much age and sex the voice embeddings carry, and whether the voice COPD
score is an age proxy (Reviewer 1, point 2).

Features: wav2vec2-large-robust layer 4 (default), mean-pooled poem recordings,
as in the selected voice models, averaged per participant. Split poem chunks
(Charité model) or whole poem recordings (UK model) are chosen with --split /
--no-split. Models: StandardScaler -> PCA(30) -> Ridge (age) or logistic
regression (sex), 5-fold x 10-repeat CV, one row per participant so folds are
person-disjoint.

Part A decodes age (R^2, MAE against the training-fold mean) and sex (AUROC) in
the whole cohort, in the matched subset if one is given (--matched-ids-csv), and
within COPD. Part B takes the voice model's out-of-fold COPD score and reports
its Spearman correlation with age within each class, and the likelihood-ratio
test for the score in a logistic model COPD ~ age + sex + score.

The UK cohort used in the paper is already matched (COVID-19 Sounds, PSM), so
for --cohort uk the whole cohort is the matched one. For the Charité cohort the
PSM-matched subset (N = 55) is given with --matched-ids-csv.

Output is aggregate only (counts, R^2, MAE, AUROC, correlations, p-values).

Usage (from the repository root):
  uv run python scripts/revision/demographics_from_voice.py --cohort charite \
      --preproc norm_on --split --voice-run <run_dir> \
      --matched-ids-csv data/2025-08-13/psm_matched_charite.csv --out out.json
  uv run python scripts/revision/demographics_from_voice.py --cohort uk \
      --preproc norm_on --no-split --voice-run <run_dir> --out out.json
"""
import argparse
import glob
import json

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

REPEATS, K, NPC = 10, 5, 30
SOURCE = {"charite": "charite", "uk": "uk_covid"}


def features(data_dir, preproc, layer, source, split):
    a = np.load(f"{data_dir}/wav2vec2__{preproc}/layer_{layer}/poem/mean.npy", allow_pickle=True)[0]
    emb = pd.DataFrame(a["mean_matrix"], index=a["recording_identifiers"])
    m = pd.read_csv(f"{data_dir}/preprocessed__{preproc}/metadata.csv")
    m = m[(m.data_source == source) & (m.recording_category == "poem")]
    if "is_chunked" in m.columns:
        m = m[m.is_chunked.astype(bool) == split]
    m = m[m.recording_identifier.isin(emb.index)]
    if m.empty:
        raise ValueError(f"No {source} poem embeddings (split={split}) found in {data_dir}")
    x = emb.loc[m.recording_identifier].groupby(m.audio_id.to_numpy()).mean()
    meta = m.drop_duplicates("audio_id").set_index("audio_id")[["age", "sex"]].copy()
    meta["male"] = meta.sex.astype(str).str.lower().isin(["m", "male", "1", "1.0"]).astype(int)
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
            "pearson_r": round(float(np.corrcoef(age, p)[0, 1]), 3),
            "age_sd": round(float(np.std(age, ddof=1)), 2)}


def cv_sex(x, male, seed=0):
    if male.min() == male.max() or min(male.sum(), len(male) - male.sum()) < K:
        return {"auroc": None, "n_male": int(male.sum()), "n": int(len(male)), "note": "too few of one sex"}
    preds = np.zeros((REPEATS, len(male)))
    for r in range(REPEATS):
        for tr, te in StratifiedKFold(K, shuffle=True, random_state=seed + r).split(x, male):
            m = make_pipeline(StandardScaler(), PCA(min(NPC, len(tr) - 1)),
                              LogisticRegression(C=0.1, max_iter=2000))
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
            "coef_score": round(float(f1.params["score"]), 3),
            "coef_age_without_score": round(float(f0.params["age"]), 4),
            "coef_age_with_score": round(float(f1.params["age"]), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", choices=list(SOURCE), required=True)
    ap.add_argument("--voice-run", required=True, help="MLflow run directory of the cohort's voice model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="data/2025-08-13")
    ap.add_argument("--preproc", default="norm_on")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--split", action=argparse.BooleanOptionalAction, required=True,
                    help="--split: poem chunks (Charité model); --no-split: whole poems (UK model)")
    ap.add_argument("--matched-ids-csv", default=None,
                    help="CSV with an audio_id column listing the matched subset (Charité PSM)")
    a = ap.parse_args()

    x, meta = features(a.data_dir, a.preproc, a.layer, SOURCE[a.cohort], a.split)
    files = glob.glob(f"{a.voice_run}/artifacts/fold_predictions/*.parquet")
    if not files:
        raise FileNotFoundError(f"No fold predictions under {a.voice_run}")
    fp = pd.concat([pd.read_parquet(f) for f in files])
    score = fp.groupby("audio_id").agg(y=("true", "first"), score=("proba_copd", "mean"))
    ids = score.index.intersection(x.index)
    missing = len(score) - len(ids)
    df = meta.loc[ids].join(score)
    out = {"cohort": a.cohort, "preproc": a.preproc, "layer": a.layer, "split": a.split,
           "n": int(len(ids)), "n_scored_without_embedding": int(missing), "A": {}, "B": {}}
    cohorts = {"all": ids, "copd_only": df.index[df.y == 1]}
    if a.matched_ids_csv:
        cohorts["matched"] = ids.intersection(pd.read_csv(a.matched_ids_csv).audio_id.unique())
    for name, idx in cohorts.items():
        X = x.loc[idx].to_numpy()
        out["A"][name] = {"n": int(len(idx)), "age": cv_age(X, df.loc[idx, "age"].to_numpy(float)),
                          "sex": cv_sex(X, df.loc[idx, "male"].to_numpy())}
    for name in [n for n in ("all", "matched") if n in cohorts]:
        d = df.loc[cohorts[name]]
        corr = {}
        for cls, lab in [(1, "copd"), (0, "control")]:
            s = d[d.y == cls]
            rho, p = spearmanr(s.age, s.score)
            corr[lab] = {"n": int(len(s)), "rho": round(float(rho), 3), "p": round(float(p), 4)}
        out["B"][name] = {"auroc_of_score": round(roc_auc_score(d.y, d.score), 3),
                          "score_vs_age_within_class": corr,
                          "lr_test_score_beyond_age_sex": lr_test(d)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
