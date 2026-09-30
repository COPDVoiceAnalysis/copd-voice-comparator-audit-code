"""Export v2 MLFlow experiment results to parquet snapshots.

Reads the FileStore directory structure directly instead of using the
MLFlow API, which is extremely slow on CephFS under heavy I/O load.
"""

from pathlib import Path

import pandas as pd
import yaml

MLFLOW_DIR = Path("~/work/mlflow/").expanduser()  # cluster home
OUTPUT_DIR = Path("data/experiment_runs")

# Known MLflow experiment IDs. `id` is either a single string (one MLflow
# experiment) or a list of strings (read all and merge, deduped by run_name
# downstream in build_canonical). Multiple IDs occur because experiment_version
# in our YAML configs is null, which routes new runs to the "no-v2" experiment,
# while historical runs still live in the original "-v2" experiment. We merge
# both so the parquet contains every successfully-trained config regardless of
# which MLflow generation it belongs to.
EXPERIMENTS = {
    "charite_complete": {
        "id": [
            "446658743422895027",  # -v2 (historical, ~501 unique runs)
            "457012274041432773",  # no-v2 (newer runs, ~1090 unique runs)
        ],
        "output": "copd_classification_charite_complete_2025-08-13-v2.parquet",
    },
    "charite_psm": {
        # PSM-matched Charité (primary). Only one experiment id (no historical
        # -v2 sibling — PSM was introduced after the v2/no-v2 split).
        "id": "644578817518142221",
        "output": "copd_classification_charite_psm_2025-08-13-v2.parquet",
    },
    "uk_only": {
        "id": [
            "686123853244627686",  # -v2 (historical, ~180 runs)
            "271009020022062271",  # no-v2 (newer runs incl May 7 _individual scope additions)
        ],
        "output": "copd_classification_uk_only_2025-08-13-v2.parquet",
    },
    "layer_sweep": {
        "id": None,  # will be discovered
        "name": "copd_classification_charite_only_2025-08-13-v2",
        "output": "copd_classification_charite_only_2025-08-13-v2_layer_split_config_sweep.parquet",
    },
}


def find_experiment_id(name: str) -> str | None:
    """Find experiment ID by scanning top-level meta.yaml files."""
    for d in MLFLOW_DIR.iterdir():
        if not d.is_dir():
            continue
        meta_file = d / "meta.yaml"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = yaml.safe_load(f)
                exp_name = meta.get("name", "")
                if exp_name == name:
                    return d.name
                # Also print v2 experiments for debugging
                if "v2" in exp_name:
                    print(f"  Found v2 experiment: {exp_name} (id: {d.name})", flush=True)
            except Exception:
                continue
    return None


def read_run(run_dir: Path) -> dict | None:
    """Read a single run from its directory. Returns a flat dict."""
    meta_file = run_dir / "meta.yaml"
    if not meta_file.exists():
        return None

    with open(meta_file) as f:
        meta = yaml.safe_load(f)

    # Skip non-finished runs (FileStore uses numeric: 3=FINISHED).
    # We additionally accept status=RUNNING because cephfs I/O errors during
    # mlflow.end_run() leave runs stuck at status=1 even though the training
    # script ran to completion (see "Training complete" in log). The metric-
    # existence guard at the end of read_run() filters out anything that
    # didn't reach the final weighted_auroc_mean log call, so accepting
    # RUNNING here is safe — it only adds genuinely completed runs whose
    # MLflow state-transition failed.
    raw_status = meta.get("status", "UNKNOWN")
    status_map = {1: "RUNNING", 2: "SCHEDULED", 3: "FINISHED", 4: "FAILED"}
    status = status_map.get(raw_status, raw_status) if isinstance(raw_status, int) else raw_status
    if status not in ("FINISHED", "RUNNING"):
        return None

    row = {
        "run_id": meta.get("run_id") or meta.get("run_uuid"),
        "status": status,
        "start_time": meta.get("start_time"),
        "end_time": meta.get("end_time"),
    }

    # Read params
    params_dir = run_dir / "params"
    if params_dir.is_dir():
        for pfile in params_dir.iterdir():
            if pfile.is_file():
                row[f"params.{pfile.name}"] = pfile.read_text().strip()

    # Read metrics (take last value from each metric file)
    metrics_dir = run_dir / "metrics"
    if metrics_dir.is_dir():
        for mfile in metrics_dir.iterdir():
            if mfile.is_file():
                lines = mfile.read_text().strip().split("\n")
                if lines and lines[-1]:
                    # Format: timestamp value step
                    parts = lines[-1].split()
                    if len(parts) >= 2:
                        try:
                            row[f"metrics.{mfile.name}"] = float(parts[1])
                        except ValueError:
                            row[f"metrics.{mfile.name}"] = parts[1]

    # Read tags
    tags_dir = run_dir / "tags"
    if tags_dir.is_dir():
        for tfile in tags_dir.iterdir():
            if tfile.is_file():
                row[f"tags.{tfile.name}"] = tfile.read_text().strip()

    # Skip runs that reported FINISHED but logged no primary metric
    # (e.g. legacy mixed vowel+split-poem configs that failed silently).
    if "metrics.weighted_auroc_mean" not in row and "metrics.auroc_mean" not in row:
        return None

    return row


def export_experiment(label: str, experiment_id, output_filename: str) -> int:
    """Export all finished runs from one or more MLflow experiment directories.

    ``experiment_id`` may be either a single string (legacy single-experiment
    case) or a list of strings. When multiple IDs are given, all runs are
    read into a single rowset; downstream dedup in build_canonical collapses
    duplicates by ``params.experiment_name`` keeping the latest ``end_time``.
    """
    exp_ids = [experiment_id] if isinstance(experiment_id, str) else list(experiment_id)

    rows: list[dict] = []
    for eid in exp_ids:
        exp_dir = MLFLOW_DIR / eid
        if not exp_dir.is_dir():
            print(f"  ERROR: directory {exp_dir} not found", flush=True)
            continue

        run_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name != "meta.yaml"]
        print(f"  Scanning {eid}: {len(run_dirs)} run directories...", flush=True)

        kept = 0
        for run_dir in run_dirs:
            row = read_run(run_dir)
            if row is not None:
                rows.append(row)
                kept += 1
        print(f"    -> {kept} kept", flush=True)

    if not rows:
        print(f"  No FINISHED runs found", flush=True)
        return 0

    df = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / output_filename
    df.to_parquet(output_path, index=False)
    print(f"  Exported {len(df)} FINISHED runs -> {output_path}", flush=True)
    return len(df)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover layer sweep experiment ID
    layer_info = EXPERIMENTS["layer_sweep"]
    if layer_info["id"] is None:
        print(f"Looking up layer sweep experiment ID...", flush=True)
        exp_id = find_experiment_id(layer_info["name"])
        if exp_id:
            layer_info["id"] = exp_id
            print(f"  Found: {exp_id}", flush=True)
        else:
            print(f"  NOT FOUND: {layer_info['name']}", flush=True)

    print("=== Exporting v2 experiments ===", flush=True)
    totals = {}
    for label, info in EXPERIMENTS.items():
        if info["id"] is None:
            totals[label] = 0
            continue
        print(f"\n{label}:", flush=True)
        totals[label] = export_experiment(label, info["id"], info["output"])

    print(f"\n=== Summary ===", flush=True)
    for label, count in totals.items():
        print(f"  {label}: {count} runs", flush=True)
    print(f"  Total: {sum(totals.values())} runs exported", flush=True)


if __name__ == "__main__":
    main()
