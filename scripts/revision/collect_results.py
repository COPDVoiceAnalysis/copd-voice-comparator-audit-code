"""Copy the revision's aggregate outputs into the repository for commit.

Copies only allowed file types from $OUT/analysis (and the preflight report)
into results/revision_r1/ in the repository, and refuses any file that could
carry participant-level data:
  - tables or JSON with a participant/recording identifier field
    (audio_id, study_id, uid, recording_identifier, birth_year, audio_sample_path)
  - CSV files with more than MAX_ROWS rows, any file over MAX_BYTES
Fold predictions, configs, MLflow runs and logs are never copied.
Writes MANIFEST.txt (git commit, date, SMOKE flag, file list) next to the files.

Usage: uv run python scripts/revision/collect_results.py <OUT> <repo>/results/revision_r1
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ALLOWED = {".json", ".csv", ".png", ".pdf", ".txt"}
FORBIDDEN = {"audio_id", "study_id", "uid", "recording_identifier", "recording_identifiers",
             "birth_year", "audio_sample_path"}
MAX_ROWS, MAX_BYTES = 500, 2_000_000


def json_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from json_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from json_keys(v)


def problem(path):
    if path.stat().st_size > MAX_BYTES:
        return f"larger than {MAX_BYTES} bytes"
    if path.suffix == ".json":
        bad = FORBIDDEN & set(json_keys(json.load(open(path))))
        return f"identifier fields {sorted(bad)}" if bad else None
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        bad = FORBIDDEN & {c.lower() for c in df.columns}
        if bad:
            return f"identifier columns {sorted(bad)}"
        if len(df) > MAX_ROWS:
            return f"{len(df)} rows (looks row-per-participant)"
    return None


def main():
    out, dest = Path(sys.argv[1]), Path(sys.argv[2])
    src = [p for p in sorted((out / "analysis").rglob("*")) if p.is_file() and p.suffix in ALLOWED]
    src += [out / "preflight_report.txt"]
    dest.mkdir(parents=True, exist_ok=True)
    copied, refused = [], []
    for p in src:
        if not p.exists():
            continue
        why = problem(p)
        if why:
            refused.append(f"{p}: {why}")
            continue
        rel = p.relative_to(out / "analysis") if (out / "analysis") in p.parents else Path(p.name)
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest / rel)
        copied.append(str(rel))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = [f"collected {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
                f"code commit {commit}", f"SMOKE={os.environ.get('SMOKE', '?')} PREPROC={os.environ.get('PREPROC', '?')}",
                f"features {(out / 'state' / 'features_base').read_text().strip() if (out / 'state' / 'features_base').exists() else '?'}",
                "", "files:", *copied]
    if refused:
        manifest += ["", "REFUSED (not copied):", *refused]
    (dest / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print("\n".join(manifest))
    sys.exit(1 if refused else 0)


if __name__ == "__main__":
    main()
