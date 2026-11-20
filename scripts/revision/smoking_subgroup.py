"""Voice score by smoking history (Reviewer 1, point 4; Reviewer 3, point 2).

Controls in both cohorts have little or no smoking history, so a voice signal may
reflect COPD, smoking or both. This script uses what the metadata allow:

  COVID-19 Sounds (uk): controls are never-smokers by construction, but COPD
    cases were kept irrespective of smoking. Comparing never-smoking COPD cases
    with the controls holds smoking fixed. Reported: counts by class and smoking
    category; AUROC of the voice score for never-smoking COPD vs controls and for
    ever-smoking COPD vs controls (class-stratified bootstrap CIs), and the AUROC
    of the score for ever- vs never-smoking within COPD (does the score track
    smoking among patients?).
  Charité: no never-smoking COPD patient is expected; reported are counts and the
    Spearman correlation of the voice score with pack-years within COPD (a dose
    check), with a bootstrap CI.

All outputs are aggregate counts and statistics; no identifiers.

Usage (from the repository root):
  uv run python scripts/revision/smoking_subgroup.py --cohort uk --voice-run <run_dir> \
      --metadata $OUT/merged_metadata_remapped.csv \
      --metadata data/2025-08-13/preprocessed__norm_on/metadata.csv --out smoking_uk.json
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demographics_from_voice import SOURCE  # noqa: E402

RNG = np.random.default_rng(20261120)
B = 2000
NEVER = {"never"}
# values that say nothing about smoking: prefer not to say, missing
UNKNOWN = {"pnts", "", "nan", "none", "unknown"}


def scores(run):
    files = glob.glob(f"{run}/artifacts/fold_predictions/*.parquet")
    if not files:
        raise FileNotFoundError(f"No fold predictions under {run}")
    fp = pd.concat([pd.read_parquet(f, columns=["audio_id", "true", "proba_copd"]) for f in files])
    return fp.groupby("audio_id").agg(y=("true", "first"), score=("proba_copd", "mean"))


def metadata(paths, source, cols):
    """First non-missing value per participant across the given metadata files."""
    out = None
    for p in paths:
        m = pd.read_csv(p, low_memory=False, usecols=lambda c: c in {"audio_id", "data_source", *cols})
        if "data_source" in m.columns:
            m = m[m.data_source == source]
        have = [c for c in cols if c in m.columns]
        if not have:
            continue
        g = m.dropna(subset=have, how="all").drop_duplicates("audio_id").set_index("audio_id")[have]
        out = g if out is None else out.combine_first(g)
    return out if out is not None else pd.DataFrame(columns=cols)


ID_COLS = ("audio_id", "uid", "Uid", "UID", "userid", "user_id")


def find_smoking_files(root, max_depth=3):
    """CSV files under root whose header has a smoking column and an id column."""
    hits = []
    root = os.path.abspath(root)
    for d, dirs, files in os.walk(root):
        if d[len(root):].count(os.sep) >= max_depth:
            dirs[:] = []
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            path = os.path.join(d, f)
            try:
                head = pd.read_csv(path, nrows=0, sep=None, engine="python").columns
            except Exception:  # unreadable or not a table
                continue
            smk = [c for c in head if re.match(r"(?i)^smok", str(c))]
            ids = [c for c in head if c in ID_COLS]
            if smk and ids:
                hits.append((path, smk[0], ids))
    return hits


def smoking_from_raw(root, id_map):
    """Smoking per audio_id from raw participant files, joined via audio_id directly or via uid.

    id_map: CSV with audio_id and uid (the matched-cohort metadata). Returns the series
    and a report with file names (no values) and join coverage.
    """
    m = pd.read_csv(id_map, low_memory=False, usecols=lambda c: c in {"audio_id", "uid"})
    report = {"searched": os.path.abspath(root), "files": []}
    out = pd.Series(dtype=object)
    for path, col, ids in find_smoking_files(root):
        t = pd.read_csv(path, sep=None, engine="python", usecols=lambda c: c == col or c in ids)
        t[col] = t[col].astype(str).str.strip().str.lower()
        t = t[~t[col].isin({"", "nan", "none"})]
        if "audio_id" in ids:
            s = t.drop_duplicates("audio_id").set_index("audio_id")[col]
            via = "audio_id"
        else:
            k = ids[0]
            # a participant can submit several times: keep the most frequent answer
            per = t.groupby(k)[col].agg(lambda v: v.value_counts().index[0])
            s = m.dropna(subset=["uid"]).set_index("audio_id").uid.astype(str).map(per.rename(index=str))
            via = k
        s = s.dropna()
        n = int(m.audio_id.isin(s.index).sum())
        report["files"].append({"file": os.path.basename(path), "column": col, "via": via, "matched_audio_ids": n})
        out = out.combine_first(s) if len(out) else s
    report["matched_total"] = int(m.audio_id.isin(out.index).sum())
    report["cohort_size"] = int(m.audio_id.nunique())
    return out, report


def auc_ci(y, s):
    if len(np.unique(y)) < 2:
        return None
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = [roc_auc_score(y[i], s[i]) for i in
            (np.r_[RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))] for _ in range(B))]
    return {"auroc": round(float(roc_auc_score(y, s)), 3),
            "ci": [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)],
            "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", choices=list(SOURCE), required=True)
    ap.add_argument("--voice-run", required=True)
    ap.add_argument("--metadata", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoking-search", help="Directory to search for raw participant CSVs with a smoking column")
    ap.add_argument("--id-map", help="CSV with audio_id and uid, used to join raw files keyed by uid")
    a = ap.parse_args()

    df = scores(a.voice_run).join(metadata(a.metadata, SOURCE[a.cohort], ["smoking", "pack_years"]))
    raw_report = None
    if a.smoking_search:
        raw, raw_report = smoking_from_raw(a.smoking_search, a.id_map)
        have = df["smoking"] if "smoking" in df.columns else pd.Series(index=df.index, dtype=object)
        df["smoking"] = have.combine_first(raw.reindex(df.index))
    out = {"cohort": a.cohort, "n_scored": int(len(df)), "n_copd": int(df.y.sum()), "n_control": int((df.y == 0).sum())}
    y, s = df.y.to_numpy(int), df.score.to_numpy(float)

    if raw_report is not None:
        out["raw_smoking_source"] = raw_report
    if "smoking" in df.columns and df.smoking.notna().any():
        sm = df.smoking.astype(str).str.strip().str.lower()
        out["smoking_by_class"] = {cls: {k: int(v) for k, v in sm[df.y == code].value_counts().items()}
                                   for cls, code in (("copd", 1), ("control", 0))}
        never, known = sm.isin(NEVER).to_numpy(), ~sm.isin(UNKNOWN).to_numpy()
        ever = known & ~never
        ctrl = y == 0
        out["n_control_not_never"] = int((ctrl & ~never).sum())
        out["overall"] = auc_ci(y, s)
        m = ((y == 1) & never) | ctrl
        out["never_copd_vs_controls"] = auc_ci(y[m], s[m])
        m = ((y == 1) & never) | (ctrl & never)
        out["never_copd_vs_never_controls"] = auc_ci(y[m], s[m])
        m = ((y == 1) & ever) | ctrl
        out["ever_copd_vs_controls"] = auc_ci(y[m], s[m])
        m = (y == 1) & known
        out["within_copd_ever_vs_never"] = auc_ci(ever[m].astype(int), s[m])
    if "pack_years" in df.columns:
        py_ = pd.to_numeric(df.pack_years, errors="coerce")
        out["pack_years_recorded"] = {"copd": int(py_[df.y == 1].notna().sum()), "control": int(py_[df.y == 0].notna().sum())}
        c = (df.y == 1).to_numpy() & py_.notna().to_numpy()
        if c.sum() >= 10:
            pv, sv = py_.to_numpy()[c], s[c]
            rho, p = spearmanr(sv, pv)
            idx = np.arange(c.sum())
            bs = [spearmanr(sv[i], pv[i])[0] for i in (RNG.choice(idx, len(idx)) for _ in range(B))]
            out["copd_score_vs_pack_years"] = {"n": int(c.sum()), "rho": round(float(rho), 3), "p": float(p),
                                               "ci": [round(float(np.nanpercentile(bs, 2.5)), 3),
                                                      round(float(np.nanpercentile(bs, 97.5)), 3)],
                                               "n_zero_pack_years": int((pv == 0).sum()),
                                               "min_pack_years": round(float(pv.min()), 1)}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
