"""Check every input of the revision runs before anything is submitted.

Reads the paths from the environment (set by revision.env), checks that each
input exists and looks right, and prints one line per check (OK / WARN / FAIL)
plus a verdict:
  features   reuse the published feature directory, or build new ones
  published  whether the published MLflow runs (fold predictions) are readable

It writes the report to $OUT/preflight_report.txt and the feature decision to
$OUT/state/features_base. It only reads; it never changes data.
Exit code 1 if any FAIL.

Run through 00_preflight.sh, not directly.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arms import PUBLISHED, published_files  # noqa: E402

E = os.environ
DATE, PREPROC = E["DATE"], E["PREPROC"]
RAW = Path(E["RAW_DIR"])
OLD_FEAT, NEW_FEAT = Path(E["OLD_FEATURES_BASE"]), Path(E["NEW_FEATURES_BASE"])
OLD_MLFLOW, OUT = Path(E["OLD_MLFLOW"]), Path(E["OUT"])
lines, fails = [], 0


def say(level, msg):
    global fails
    fails += level == "FAIL"
    lines.append(f"[{level:4s}] {msg}")
    print(lines[-1], flush=True)


def readable(p):
    try:
        return p.exists() and os.access(p, os.R_OK)
    except PermissionError:
        return False


def check_features(base):
    """Return (ok, notes) for a feature directory with the files the revision reads."""
    notes = []
    meta = base / f"preprocessed__{PREPROC}" / "metadata.csv"
    w2v = base / f"wav2vec2__{PREPROC}" / "layer_4" / "poem" / "mean.npy"
    parsel = base / f"parselmouth__{PREPROC}" / "acoustic_features.npy"
    for p in (meta, w2v, parsel):
        if not readable(p):
            return False, [f"missing or unreadable: {p}"]
    m = pd.read_csv(meta, low_memory=False)
    ids = np.load(w2v, allow_pickle=True)[0]["recording_identifiers"]
    poem = m[m.recording_category == "poem"]
    for src in ("charite", "uk_covid"):
        s = poem[poem.data_source == src]
        cover = s.recording_identifier.isin(ids).mean() if len(s) else 0
        notes.append(f"{src}: {s.audio_id.nunique()} participants with poem rows, "
                     f"{cover:.0%} of poem rows have a layer-4 embedding")
        if not len(s) or cover < 0.95:
            return False, notes
    if "language" not in m.columns:
        notes.append("metadata has no 'language' column (needed for the same-language subset)")
        return False, notes
    layers = sorted((base / f"wav2vec2__{PREPROC}").glob("layer_*"))
    notes.append(f"{len(layers)} wav2vec2 layers present (25 needed for the layer sweep)")
    return True, notes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "state").mkdir(exist_ok=True)
    say("INFO", f"DATE={DATE} PREPROC={PREPROC} OUT={OUT}")

    # --- tools -----------------------------------------------------------
    for tool, level in (("uv", "FAIL"), ("git", "FAIL"), ("sbatch", "WARN")):
        say("OK" if shutil.which(tool) else level, f"command available: {tool}")
    repo = Path(E["REPO"])
    branch = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    say("OK" if branch == "revision-sensors-r1" else "WARN", f"repo branch {branch} at {commit}")
    say("OK" if not dirty else "WARN", "tracked files unchanged" if not dirty else f"tracked files modified:\n{dirty}")

    # --- raw / group directory -------------------------------------------
    need = ["merged_metadata.csv", "comorbidities.csv", "psm_matched_charite.csv",
            "uk_matched_metadata.csv", "uk_matched_metadata_ps_covariates.csv"]
    for f in need:
        say("OK" if readable(RAW / f) else "FAIL", f"raw input {RAW / f}")
    if readable(RAW / "merged_metadata.csv"):
        mm = pd.read_csv(RAW / "merged_metadata.csv", low_memory=False)
        for src in ("charite", "uk_covid"):
            say("INFO", f"merged_metadata: {mm[mm.data_source == src].audio_id.nunique()} {src} participants")
        paths = mm.audio_sample_path.dropna()
        sample = paths.sample(min(20, len(paths)), random_state=0)
        found = sum(os.path.exists(p) or os.path.exists(RAW / os.path.basename(p)) for p in sample)
        say("OK" if found == len(sample) else "WARN",
            f"raw audio: {found} of {len(sample)} sampled recordings found (needed only to build features "
            f"and for the sample-rate table)")
    if readable(RAW / "uk_matched_metadata.csv"):
        uk = pd.read_csv(RAW / "uk_matched_metadata.csv").drop_duplicates("audio_id")
        say("INFO", f"uk_matched_metadata: {len(uk)} participants")
        if "language" in uk.columns:
            tab = pd.crosstab(uk.language, uk.lung_disease_main)
            say("INFO", "UK participants by language and class:\n" + tab.to_string())

    # --- features --------------------------------------------------------
    ok_old, notes = check_features(OLD_FEAT)
    for n in notes:
        say("INFO", f"published features: {n}")
    ok_new, notes_new = check_features(NEW_FEAT)
    for n in notes_new:
        say("INFO", f"own features: {n}")
    if E.get("FEATURES_BASE_OVERRIDE"):
        base, why = Path(E["FEATURES_BASE_OVERRIDE"]), "set by FEATURES_BASE_OVERRIDE"
    elif ok_new:
        base, why = NEW_FEAT, "own features already built"
    elif ok_old:
        base, why = OLD_FEAT, "published features readable"
    else:
        base, why = NEW_FEAT, "BUILD NEEDED (run with features stage)"
    (OUT / "state" / "features_base").write_text(str(base))
    (OUT / "state" / "features_need_build").write_text("1" if base == NEW_FEAT and not ok_new else "0")
    say("INFO", f"VERDICT features: {base} ({why})")

    # --- published runs --------------------------------------------------
    found = 0
    for label in PUBLISHED:
        ok = bool(published_files(OLD_MLFLOW, label))
        found += ok
        say("OK" if ok else "INFO", f"published run {label}: {'readable' if ok else 'not readable'}")
    (OUT / "state" / "published_available").write_text("1" if found == len(PUBLISHED) else "0")
    say("INFO", f"VERDICT published runs: {found} of {len(PUBLISHED)} readable"
        + ("" if found == len(PUBLISHED) else " (export step will be skipped)"))

    # --- space and model cache ------------------------------------------
    free_gb = shutil.disk_usage(OUT).free / 1e9
    say("OK" if free_gb > 50 else "WARN", f"{free_gb:.0f} GB free under {OUT} (feature build needs ~50 GB)")
    hf = Path(E.get("HF_HOME", Path.home() / ".cache/huggingface"))
    cached = any(hf.glob("hub/models--facebook--wav2vec2-large-robust"))
    say("OK" if cached else "INFO", f"wav2vec2-large-robust in {hf}: {'cached' if cached else 'not cached'} "
        "(01_setup.sh downloads it when features must be built)")

    (OUT / "preflight_report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nreport: {OUT / 'preflight_report.txt'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
