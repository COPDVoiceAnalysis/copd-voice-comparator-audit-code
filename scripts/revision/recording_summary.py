"""Recording sample rates and durations per cohort and class (Reviewer 2, point 3).

For the participants of each cohort's voice model (read from its fold
predictions), counts the original (unprocessed) poem recordings by sample rate,
and reports channel count and duration. The sample rate comes from the
metadata column `sampling_rate` when it is filled, otherwise from the audio
file header (soundfile.info), so the raw audio must be readable.

Output is one aggregate CSV (no participant rows):
  cohort, class, sample_rate_hz, n_recordings, n_participants,
  duration_s_median, duration_s_p10, duration_s_p90, channels, source

Usage (from the repository root):
  uv run python scripts/revision/recording_summary.py \
      --metadata <RAW_DIR>/merged_metadata.csv --audio-dir <RAW_DIR> \
      --cohort charite=<charite_voice_run_dir> --cohort uk_covid=<uk_voice_run_dir> \
      --out recording_summary.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import soundfile as sf


def header(path):
    try:
        i = sf.info(path)
        return i.samplerate, i.channels, i.frames / i.samplerate
    except Exception:
        return np.nan, np.nan, np.nan


def resolve(path, audio_dirs):
    if not isinstance(path, str):
        return None
    if os.path.exists(path):
        return path
    for d in audio_dirs:
        cand = os.path.join(d, os.path.basename(path))
        if os.path.exists(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Raw merged metadata with original audio paths")
    ap.add_argument("--cohort", action="append", required=True,
                    help="data_source=voice_run_dir; participants come from the run's fold predictions")
    ap.add_argument("--audio-dir", action="append", default=[],
                    help="Fallback directory searched by file name when a metadata path does not exist "
                         "(repeatable; e.g. the raw data directory)")
    ap.add_argument("--rate-metadata", action="append", default=[],
                    help="Further metadata CSV(s) with recording_identifier and sampling_rate; used to fill "
                         "sample rates missing in --metadata (e.g. the preprocessed metadata, which carries "
                         "the COVID-19 Sounds source rate)")
    ap.add_argument("--category", default="poem")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta = pd.read_csv(a.metadata, low_memory=False)
    if "sampling_rate" not in meta.columns:
        meta["sampling_rate"] = np.nan
    for extra in a.rate_metadata:
        e = pd.read_csv(extra, usecols=lambda c: c in {"recording_identifier", "sampling_rate"}, low_memory=False)
        if {"recording_identifier", "sampling_rate"} <= set(e.columns):
            rate = e.dropna().drop_duplicates("recording_identifier").set_index("recording_identifier").sampling_rate
            meta["sampling_rate"] = meta.sampling_rate.fillna(meta.recording_identifier.map(rate))
    rows = []
    for spec in a.cohort:
        source, run = spec.split("=", 1)
        files = glob.glob(f"{run}/artifacts/fold_predictions/*.parquet")
        if not files:
            raise FileNotFoundError(f"No fold predictions under {run}")
        ids = pd.concat([pd.read_parquet(f) for f in files]).groupby("audio_id")["true"].first()
        m = meta[(meta.data_source == source) & (meta.recording_category == a.category)
                 & meta.audio_id.isin(ids.index)].copy()
        if "is_chunked" in m.columns:
            m = m[~m.is_chunked.fillna(False).astype(bool)]
        info = [header(p) if p else (np.nan,) * 3
                for p in (resolve(p, a.audio_dir) for p in m.audio_sample_path)]
        m["hdr_sr"], m["channels"], m["duration_s"] = zip(*info) if info else ([], [], [])
        meta_sr = m["sampling_rate"]
        m["sample_rate_hz"] = meta_sr.fillna(m["hdr_sr"])
        m["source"] = np.where(meta_sr.notna(), "metadata", np.where(m.hdr_sr.notna(), "header", "unknown"))
        m["class"] = m.audio_id.map(ids).map({1: "copd", 0: "control"})
        print(f"{source}: {m.audio_id.nunique()} of {len(ids)} participants, {len(m)} recordings, "
              f"{int((m.source == 'unknown').sum())} without sample rate")
        for (cls, sr, src), g in m.groupby(["class", m.sample_rate_hz.fillna(-1), "source"]):
            d = g.duration_s.dropna()
            rows.append({"cohort": source, "class": cls, "sample_rate_hz": None if sr == -1 else int(sr),
                         "n_recordings": len(g), "n_participants": g.audio_id.nunique(),
                         "duration_s_median": round(float(d.median()), 1) if len(d) else None,
                         "duration_s_p10": round(float(d.quantile(0.1)), 1) if len(d) else None,
                         "duration_s_p90": round(float(d.quantile(0.9)), 1) if len(d) else None,
                         "channels": ",".join(str(int(c)) for c in sorted(g.channels.dropna().unique())),
                         "source": src})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
