"""Build canonical, versioned Parquet files for every experiment in the thesis.

One row per unique experiment_name / per unique cross-dataset configuration,
with all metrics populated. These files are the single source of truth for
the thesis, paper figures, and downstream notebooks.

Sources:
    * current v2 exports in data/experiment_runs/ (committed + uncommitted)
    * committed v2 history (for the composite feature family, which was
      dropped from the most recent charite re-export)
    * outputs/cross_dataset_results.json (cross-dataset transfer runs)

Outputs (written to data/experiment_runs/):
    copd_classification_charite_only_2025-08-13-v2.parquet
    copd_classification_charite_psm_2025-08-13-v2.parquet
    copd_classification_uk_only_2025-08-13-v2.parquet
    copd_classification_charite_only_2025-08-13-v2_layer_split_config_sweep.parquet
    copd_classification_cross_dataset_2025-08-13-v2.parquet
"""

import io
import json
import subprocess
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/experiment_runs")
PRIMARY_METRIC = "metrics.weighted_auroc_mean"
KEY = "params.experiment_name"

# Local MLflow file-backend snapshot (rsync from cluster) used to enrich each
# run with its per-fold metric history. Avoids a running MLflow server and
# makes the canonical parquets self-contained for downstream fold-level work.
MLFLOW_SNAPSHOT = Path("data/2025-08-13/mlflow_cluster")
# MLflow experiment IDs for fold-history enrichment. Each task can have
# multiple experiment generations (-v2 vs no-v2; see scripts/export_v2_results.py
# for context). Lists are searched in order — the first directory containing
# a given run_id wins.
CHARITE_EXP_IDS = [
    "446658743422895027",  # copd_classification_charite_complete_2025-08-13-v2
    "457012274041432773",  # copd_classification_charite_complete_2025-08-13 (no -v2)
    "654388009243800258",  # copd_classification_charite_only_2025-08-13 (legacy; holds runs from the RAW_CHARITE_REF era)
]
CHARITE_PSM_EXP_IDS = ["644578817518142221"]
UK_EXP_IDS = [
    "686123853244627686",  # copd_classification_uk_only_2025-08-13-v2
    "271009020022062271",  # copd_classification_uk_only_2025-08-13 (no -v2)
]
FOLD_METRICS = ("auroc", "weighted_auroc")


def _pick_best(group: pd.DataFrame) -> pd.Series:
    """One run per config: prefer valid primary metric, then newest end_time."""
    valid = group[group[PRIMARY_METRIC].notna()]
    pool = valid if len(valid) else group
    sort_col = "end_time" if "end_time" in pool.columns else pool.columns[0]
    return pool.sort_values(sort_col, ascending=False).iloc[0]


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(KEY, group_keys=False, dropna=False)
        .apply(_pick_best)
        .reset_index(drop=True)
    )


def _drop_basic_hyperparam(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exploratory basic-hyperparam runs (5-repeat, narrower grids).

    Only extended-hyperparam runs (10 outer repeats, full grids) are canonical
    for the thesis. Basic runs were an earlier exploratory sweep.
    """
    if "tags.hyperparam" not in df.columns:
        return df
    return df[df["tags.hyperparam"] != "basic"].reset_index(drop=True)


def _read_fold_history(metric_path: Path) -> list[float]:
    """Parse a single MLflow file-backend metric file into an ordered value list.

    Each line is ``<timestamp_ms> <value> <step>``. We sort by step so the
    resulting list is deterministic regardless of on-disk ordering.
    """
    if not metric_path.exists():
        return []
    rows: list[tuple[int, float]] = []
    for line in metric_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            step = int(parts[2])
            value = float(parts[1])
        except ValueError:
            continue
        rows.append((step, value))
    rows.sort(key=lambda r: r[0])
    return [v for _, v in rows]


def _enrich_with_fold_history(df: pd.DataFrame, experiment_ids) -> pd.DataFrame:
    """Add per-fold metric lists (``fold_<metric>``) by reading the local MLflow snapshot.

    ``experiment_ids`` may be a single string (legacy) or a list of IDs to search
    in order — multiple are needed because experiment-name routing changed
    between -v2 and no-v2 generations. For each run_id the first directory that
    contains it wins; runs whose metric files are missing receive empty lists.
    """
    ids = [experiment_ids] if isinstance(experiment_ids, str) else list(experiment_ids)
    exp_dirs = [MLFLOW_SNAPSHOT / eid for eid in ids]
    existing = [d for d in exp_dirs if d.exists()]
    if not existing:
        print(f"    [warn] No MLflow snapshot dirs found for {ids}; skipping fold enrichment")
        return df

    def _lookup(rid: str, metric: str) -> list[float]:
        for d in existing:
            path = d / rid / "metrics" / metric
            if path.exists():
                return _read_fold_history(path)
        return []

    for metric in FOLD_METRICS:
        col = f"fold_{metric}s"
        df[col] = df["run_id"].map(lambda rid: _lookup(rid, metric))

    fold_col = f"fold_{FOLD_METRICS[0]}s"
    n_full = (df[fold_col].str.len() == 50).sum()
    n_any = (df[fold_col].str.len() > 0).sum()
    print(f"    fold enrichment: {n_any}/{len(df)} runs have fold data, {n_full} with full 50-fold history")
    return df


def _git_show(relpath: Path, ref: str = "HEAD") -> pd.DataFrame:
    out = subprocess.run(
        ["git", "show", f"{ref}:{relpath}"], capture_output=True, check=True
    ).stdout
    return pd.read_parquet(io.BytesIO(out))


# Historical pin: the raw charite_complete snapshot was once removed from the
# working tree (commit 000dc64) before being re-tracked. Kept as a fallback
# source so any runs that lived only in that frozen export still survive
# rebuilds, even after schema-changing matrix expansions.
RAW_CHARITE_REF = "6046c06"


def build_charite() -> None:
    """charite_only: live raw export + frozen historical raw + prior canonical.

    Now mirrors :func:`build_uk` / :func:`build_charite_psm` and reads the
    current local raw export (``copd_classification_charite_complete_*.parquet``)
    as the primary source so matrix expansions like the ``_individual`` scope
    are picked up automatically. The frozen historical raw at
    ``RAW_CHARITE_REF`` and the prior canonical at HEAD are merged in too,
    so any runs only present in those snapshots are preserved across rebuilds.
    """
    src = DATA_DIR / "copd_classification_charite_complete_2025-08-13-v2.parquet"
    out = DATA_DIR / "copd_classification_charite_only_2025-08-13-v2.parquet"

    raw_live = pd.read_parquet(src) if src.exists() else pd.DataFrame()
    raw_frozen = _git_show(src, ref=RAW_CHARITE_REF)
    prior_canonical = _git_show(out)

    composite_mask = raw_frozen[KEY].str.contains(
        "wav2vec_poem_parsel_o_basic_acoustics", na=False
    ) & raw_frozen[KEY].str.contains("__unsplit__", na=False)
    composite = raw_frozen[composite_mask & raw_frozen[PRIMARY_METRIC].notna()]

    merged = pd.concat(
        [raw_live, raw_frozen, composite, prior_canonical], ignore_index=True
    )
    final = _dedup(merged)
    final = final[final[PRIMARY_METRIC].notna()].reset_index(drop=True)

    # Layer-sweep runs were logged into the same MLFlow experiment as the
    # main grid; they belong exclusively in the separate layer-sweep parquet.
    final = final[~final[KEY].str.startswith("layer_sweep__", na=False)].reset_index(drop=True)

    # Drop exploratory basic-hyperparam runs; only extended is canonical.
    final = _drop_basic_hyperparam(final)

    # Enrich with fold-level metric history from local MLflow snapshot.
    final = _enrich_with_fold_history(final, CHARITE_EXP_IDS)

    final.to_parquet(out, index=False)

    composite_count = final[KEY].str.contains(
        "wav2vec_poem_parsel_o_basic_acoustics", na=False
    ).sum()
    print(
        f"  charite_only:  {len(final)} configs "
        f"(incl. {composite_count} composite unsplit) -> {out.name}"
    )


def build_uk() -> None:
    src = DATA_DIR / "copd_classification_uk_only_2025-08-13-v2.parquet"
    out = src  # overwrite in place with deduplicated version

    raw = pd.read_parquet(src)
    final = _dedup(raw)
    final = final[final[PRIMARY_METRIC].notna()].reset_index(drop=True)
    final = _drop_basic_hyperparam(final)
    final = _enrich_with_fold_history(final, UK_EXP_IDS)
    final.to_parquet(out, index=False)
    print(f"  uk_only:       {len(final)} configs -> {out.name}")


def build_charite_psm() -> None:
    """charite_psm: PSM-matched Charité primary analysis (in-place dedup).

    Mirrors :func:`build_uk`'s minimal pipeline (no composite-merge logic
    needed) — read the raw export, dedup to one row per config, drop runs
    without primary metric, drop basic-hyperparam exploratory runs, enrich
    with per-fold metric history, write back in place.
    """
    src = DATA_DIR / "copd_classification_charite_psm_2025-08-13-v2.parquet"
    out = src  # overwrite in place with deduplicated version

    raw = pd.read_parquet(src)
    final = _dedup(raw)
    final = final[final[PRIMARY_METRIC].notna()].reset_index(drop=True)
    final = _drop_basic_hyperparam(final)
    final = _enrich_with_fold_history(final, CHARITE_PSM_EXP_IDS)
    final.to_parquet(out, index=False)
    print(f"  charite_psm:   {len(final)} configs -> {out.name}")


def build_layer_sweep() -> None:
    src = DATA_DIR / "copd_classification_charite_only_2025-08-13-v2_layer_split_config_sweep.parquet"
    raw = pd.read_parquet(src)
    final = _dedup(raw)
    final = final[final[PRIMARY_METRIC].notna()].reset_index(drop=True)
    final.to_parquet(src, index=False)
    print(f"  layer_sweep:   {len(final)} configs -> {src.name}")


def build_cross_dataset() -> None:
    """Cross-dataset transfer: merge norm_on + norm_off + PSM JSONs into Parquet.

    Reads three source JSONs produced by ``src/train_cross_dataset.py``:
      * outputs/cross_dataset_results_v2_norm_on.json  (full Charité, norm_on)
      * outputs/cross_dataset_results_v2_norm_off.json (full Charité, norm_off)
      * outputs/cross_dataset_results_v2_psm.json      (PSM-matched, norm_on)

    Each row corresponds to one (preproc, cohort_variant, direction,
    splitting, classifier) tuple. The ``cohort_variant`` column distinguishes
    PSM-primary ("psm_matched") from full-Charité sensitivity ("full").
    Includes wba_opt + 95% CI alongside auroc + 95% CI. best_params is
    JSON-encoded so the table remains schema-stable.
    """
    sources = [
        (Path("outputs/cross_dataset_results_v2_norm_on.json"), "full"),
        (Path("outputs/cross_dataset_results_v2_norm_off.json"), "full"),
        (Path("outputs/cross_dataset_results_v2_psm.json"), "psm_matched"),
    ]
    out = DATA_DIR / "copd_classification_cross_dataset_2025-08-13-v2.parquet"

    rows = []
    for src, default_variant in sources:
        if not src.exists():
            print(f"    [warn] {src} missing; skipping")
            continue
        with open(src) as f:
            payload = json.load(f)
        preproc = payload["preproc"]
        cohort_variant = payload.get("charite_cohort_variant", default_variant)
        for e in payload["results"]:
            row = dict(e)
            row["preproc"] = preproc
            row["cohort_variant"] = cohort_variant
            row["best_params"] = json.dumps(
                row.get("best_params") or {}, sort_keys=True
            )
            row["config_key"] = (
                f"{preproc}__{cohort_variant}__"
                f"{row['train_cohort']}_to_{row['test_cohort']}__"
                f"{row['splitting']}__{row['classifier']}"
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    # one row per config_key (defensive dedup — JSON already unique)
    df = df.drop_duplicates(subset=["config_key"]).reset_index(drop=True)
    df.to_parquet(out, index=False)
    counts = df.groupby(["cohort_variant", "preproc"]).size().to_dict()
    print(f"  cross_dataset: {len(df)} configs ({counts}) -> {out.name}")


def main() -> None:
    print("Building canonical Parquet files...")
    build_charite()
    build_charite_psm()
    build_uk()
    build_layer_sweep()
    build_cross_dataset()
    print("\nDone. These files are the single source of truth for the thesis/paper.")


if __name__ == "__main__":
    main()
