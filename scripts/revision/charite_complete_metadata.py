"""Charité metadata restricted to the analysed cohort (Reviewer 1, point 5).

The cross-cohort transfer in Table 3 uses the full Charité cohort (N = 99: 77
COPD, 22 controls). Without a filter, src/train_cross_dataset.py would load
every non-longitudinal Charité participant in the preprocessed metadata (more
than 99, because the complete-case restriction of the within-cohort analysis is
not applied there). This script takes the participants that the selected
Charité voice run actually scored (its out-of-fold predictions) and writes the
preprocessed metadata reduced to their rows, for --charite_metadata_csv.

It refuses to write a file whose class counts differ from --expect.

Usage: uv run python scripts/revision/charite_complete_metadata.py \
    --voice-run <run_dir> --metadata data/2025-08-13/preprocessed__norm_on/metadata.csv \
    --out charite_complete_metadata.csv [--expect 77,22]
"""
import argparse
import glob
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--voice-run", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", default="77,22", help="COPD,control participant counts; empty to skip the check")
    a = ap.parse_args()

    files = glob.glob(f"{a.voice_run}/artifacts/fold_predictions/*.parquet")
    if not files:
        sys.exit(f"No fold predictions under {a.voice_run}")
    fp = pd.concat([pd.read_parquet(f, columns=["audio_id", "true"]) for f in files])
    y = fp.groupby("audio_id")["true"].first()
    n_copd, n_ctrl = int((y == 1).sum()), int((y == 0).sum())

    meta = pd.read_csv(a.metadata, low_memory=False)
    keep = (meta["data_source"] == "charite") & meta["audio_id"].isin(y.index)
    sub = meta[keep]
    found = sub.groupby("lung_disease_main")["audio_id"].nunique().to_dict()
    missing = sorted(set(y.index) - set(sub["audio_id"]))
    print(f"scored participants: {n_copd} COPD, {n_ctrl} control; found in metadata: {found}; missing: {len(missing)}")
    if missing:
        sys.exit("some scored participants are not in the metadata; refusing to write")
    if a.expect:
        want = tuple(int(x) for x in a.expect.split(","))
        got = (found.get("copd", 0), found.get("control", 0))
        if got != want:
            sys.exit(f"class counts {got} differ from expected {want}; refusing to write")
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
