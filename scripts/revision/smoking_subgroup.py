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
    a = ap.parse_args()

    df = scores(a.voice_run).join(metadata(a.metadata, SOURCE[a.cohort], ["smoking", "pack_years"]))
    out = {"cohort": a.cohort, "n_scored": int(len(df)), "n_copd": int(df.y.sum()), "n_control": int((df.y == 0).sum())}
    y, s = df.y.to_numpy(int), df.score.to_numpy(float)

    if "smoking" in df.columns:
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
