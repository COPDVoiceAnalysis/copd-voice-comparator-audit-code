"""Does the voice score survive the recording sample rate? (Reviewer 2, point 3)

In the Charité cohort the sample rate of the original poem recording differs
between classes (16 / 44.1 / 48 kHz, i.e. different devices or recording
paths), so the voice model could partly detect the device. For one cohort this
script takes each participant's sample rate (metadata column `sampling_rate`,
else the audio file header) and the voice model's out-of-fold COPD score, and
reports:

  rate_only_auroc       AUROC of the sample rate alone as a COPD predictor:
                        cross-validated (P(COPD | rate) from the training folds,
                        add-one smoothing; 5-fold, mean over 20 repeats) and
                        in-sample (an upper bound)
  per_stratum           voice-score AUROC within each sample rate, with
                        class-stratified bootstrap CIs
  stratified_auroc      voice-score AUROC counting only COPD-control pairs
                        recorded at the same sample rate (rate held fixed),
                        with bootstrap CI; next to the overall AUROC
  lr_test               likelihood-ratio test of the score in
                        COPD ~ age + sex + C(rate) [+ duration] + score
  rate_from_embeddings  how well the voice embeddings themselves predict the
                        sample rate (cross-validated AUROC, one-vs-rest)

All outputs are aggregate. AUROCs here are unweighted (no age x sex strata
weights), so the overall value is the plain participant-level AUROC.

Usage (from the repository root):
  uv run python scripts/revision/device_confound.py --cohort charite \
      --voice-run <run_dir> --metadata $OUT/merged_metadata_remapped.csv \
      --audio-dir $RAW_DIR --rate-metadata data/2025-08-13/preprocessed__norm_on/metadata.csv \
      --data-dir data/2025-08-13 --preproc norm_on --split --out device_confound_charite.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demographics_from_voice import SOURCE, features  # noqa: E402
from recording_summary import header, resolve  # noqa: E402

RNG = np.random.default_rng(20261116)
B = 2000


def participant_rates(a, ids):
    meta = pd.read_csv(a.metadata, low_memory=False)
    if "sampling_rate" not in meta.columns:
        meta["sampling_rate"] = np.nan
    for extra in a.rate_metadata:
        e = pd.read_csv(extra, usecols=lambda c: c in {"recording_identifier", "sampling_rate"}, low_memory=False)
        if {"recording_identifier", "sampling_rate"} <= set(e.columns):
            r = e.dropna().drop_duplicates("recording_identifier").set_index("recording_identifier").sampling_rate
            meta["sampling_rate"] = meta.sampling_rate.fillna(meta.recording_identifier.map(r))
    m = meta[(meta.data_source == SOURCE[a.cohort]) & (meta.recording_category == "poem") & meta.audio_id.isin(ids)].copy()
    if "is_chunked" in m.columns:
        m = m[~m.is_chunked.fillna(False).astype(bool)]
    hdr = [header(p) if p else (np.nan,) * 3 for p in (resolve(p, a.audio_dir) for p in m.audio_sample_path)]
    m["hdr_sr"], _, m["duration_s"] = zip(*hdr) if hdr else ([], [], [])
    m["rate"] = m.sampling_rate.fillna(m.hdr_sr)
    g = m.groupby("audio_id")
    return pd.DataFrame({"rate": g.rate.agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan),
                         "duration_s": g.duration_s.mean()})


def pair_auroc(y, s, strata=None):
    """AUROC over COPD-control pairs; with strata, only pairs within the same stratum."""
    if strata is None:
        strata = np.zeros(len(y))
    conc = total = 0.0
    for k in np.unique(strata):
        pos, neg = s[(strata == k) & (y == 1)], s[(strata == k) & (y == 0)]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            conc += (d > 0).sum() + 0.5 * (d == 0).sum()
            total += d.size
    return conc / total if total else np.nan


def boot(y, s, strata, fn):
    cells = [np.where((y == c) & (strata == k))[0] for c in (0, 1) for k in np.unique(strata)]
    cells = [c for c in cells if len(c)]
    vals = []
    for _ in range(B):
        idx = np.concatenate([RNG.choice(c, len(c)) for c in cells])
        vals.append(fn(y[idx], s[idx], strata[idx]))
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)] if len(vals) else None


def cv_rate_auroc(y, rate, repeats=20):
    """Sample rate as the only predictor. Leave-one-out is avoided on purpose:
    with group frequencies it is biased below 0.5 (removing a case lowers its own
    group's estimate)."""
    aucs = []
    for r in range(repeats):
        p = np.empty(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(rate, y):
            freq = {k: (y[tr][rate[tr] == k].sum() + 1) / ((rate[tr] == k).sum() + 2) for k in np.unique(rate)}
            p[te] = [freq[k] for k in rate[te]]
        aucs.append(roc_auc_score(y, p))
    in_sample = pd.Series(y).groupby(rate).transform("mean").to_numpy()
    return round(float(np.mean(aucs)), 3), round(float(roc_auc_score(y, in_sample)), 3)


def lr_test(df, with_duration):
    base = pd.get_dummies(df["rate"].astype(int).astype(str), prefix="sr", drop_first=True, dtype=float)
    X0 = pd.concat([df[["age", "male"]].astype(float), base], axis=1)
    if with_duration:
        X0["duration_s"] = df["duration_s"].astype(float)
    X0 = sm.add_constant(X0)
    X1 = X0.assign(score=df["score"].astype(float))
    try:
        f0 = sm.Logit(df.y, X0).fit(disp=0, maxiter=200)
        f1 = sm.Logit(df.y, X1).fit(disp=0, maxiter=200)
    except Exception as e:  # e.g. perfect separation in a small stratum
        return {"error": str(e)[:200]}
    stat = 2 * (f1.llf - f0.llf)
    return {"covariates": [c for c in X0.columns if c != "const"], "lr_chi2": round(float(stat), 2),
            "p": float(chi2.sf(stat, 1)), "coef_score": round(float(f1.params["score"]), 3)}


def rate_from_embeddings(a, ids, rate):
    x, _ = features(a.data_dir, a.preproc, a.layer, SOURCE[a.cohort], a.split)
    keep = [i for i in ids if i in x.index]
    X, r = x.loc[keep].to_numpy(), rate.loc[keep].astype(int).to_numpy()
    classes, counts = np.unique(r, return_counts=True)
    if len(classes) < 2 or counts.min() < 5:
        return {"note": "fewer than two rates with at least 5 participants", "counts": dict(zip(map(int, classes), map(int, counts)))}
    model = make_pipeline(StandardScaler(), PCA(min(30, len(keep) // 2)), LogisticRegression(C=0.1, max_iter=3000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    if len(classes) == 2:
        p = cross_val_predict(model, X, r, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(r == classes[1], p)
    else:
        p = cross_val_predict(model, X, r, cv=cv, method="predict_proba")
        auc = roc_auc_score(r, p, multi_class="ovr", labels=classes)
    return {"auroc_ovr": round(float(auc), 3), "counts": dict(zip(map(int, classes), map(int, counts)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", choices=list(SOURCE), required=True)
    ap.add_argument("--voice-run", required=True)
    ap.add_argument("--metadata", required=True, help="Raw (remapped) merged metadata with original audio paths")
    ap.add_argument("--audio-dir", action="append", default=[])
    ap.add_argument("--rate-metadata", action="append", default=[])
    ap.add_argument("--data-dir", default="data/2025-08-13")
    ap.add_argument("--preproc", default="norm_on")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--split", action=argparse.BooleanOptionalAction, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = glob.glob(f"{a.voice_run}/artifacts/fold_predictions/*.parquet")
    if not files:
        raise FileNotFoundError(f"No fold predictions under {a.voice_run}")
    fp = pd.concat([pd.read_parquet(f) for f in files])
    sc = fp.groupby("audio_id").agg(y=("true", "first"), score=("proba_copd", "mean"))
    meta = pd.read_csv(a.metadata, low_memory=False).drop_duplicates("audio_id").set_index("audio_id")
    df = sc.join(participant_rates(a, sc.index)).join(meta[["age", "sex"]])
    df["male"] = df.sex.astype(str).str.lower().isin(["m", "male", "1", "1.0"]).astype(int)
    n_all = len(df)
    df = df.dropna(subset=["rate"])
    out = {"cohort": a.cohort, "n_scored": n_all, "n_with_rate": len(df)}
    if df.empty or df.rate.nunique() < 2:
        out["note"] = "fewer than two sample rates among scored participants; nothing to adjust for"
        out["rates"] = {str(int(k)): int(v) for k, v in df.rate.value_counts().items()}
    else:
        y, s, k = df.y.to_numpy(int), df.score.to_numpy(float), df.rate.to_numpy(float)
        out["counts"] = {f"{int(r)}": {"copd": int(((k == r) & (y == 1)).sum()), "control": int(((k == r) & (y == 0)).sum())}
                         for r in np.unique(k)}
        cv, ins = cv_rate_auroc(y, k)
        out["rate_only_auroc"] = {"cross_validated": cv, "in_sample": ins}
        out["overall_auroc"] = [round(pair_auroc(y, s), 3), *(boot(y, s, k, lambda yy, ss, kk: pair_auroc(yy, ss)) or [])]
        out["stratified_auroc"] = [round(pair_auroc(y, s, k), 3), *(boot(y, s, k, pair_auroc) or [])]
        out["per_stratum"] = {}
        for r in np.unique(k):
            m = k == r
            if len(np.unique(y[m])) == 2:
                ci = boot(y[m], s[m], np.zeros(m.sum()), lambda yy, ss, kk: pair_auroc(yy, ss))
                out["per_stratum"][f"{int(r)}"] = {"auroc": round(pair_auroc(y[m], s[m]), 3), "ci": ci,
                                                    "copd": int(y[m].sum()), "control": int((y[m] == 0).sum())}
        out["lr_test"] = {"rate_age_sex": lr_test(df, False)}
        if df.duration_s.notna().all():
            out["lr_test"]["rate_age_sex_duration"] = lr_test(df, True)
        out["rate_from_embeddings"] = rate_from_embeddings(a, df.index, df.rate)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
