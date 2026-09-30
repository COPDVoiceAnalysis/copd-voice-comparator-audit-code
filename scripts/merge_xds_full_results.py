"""Merge the 6 full-Charité per-config JSONs into outputs/cross_dataset_results_v2_norm_on.json
and the 6 PSM-matched per-config JSONs into outputs/cross_dataset_results_v2_psm.json.
"""
import json
from pathlib import Path


def merge(src_dir: Path, out_path: Path, cohort_variant: str, version: str):
    merged = {
        "date": "2025-08-13",
        "experiment_version": version,
        "preproc": "norm_on",
        "charite_metadata_csv": None,
        "charite_cohort_variant": cohort_variant,
        "results": [],
    }
    for sub in sorted(src_dir.iterdir()):
        if not sub.is_dir():
            continue
        f = sub / "cross_dataset_results.json"
        if not f.exists():
            continue
        payload = json.loads(f.read_text())
        if merged["charite_metadata_csv"] is None:
            merged["charite_metadata_csv"] = payload["charite_metadata_csv"]
        merged["results"].extend(payload["results"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"Wrote {out_path} ({len(merged['results'])} results)")
    return merged


def summarize(merged: dict, label: str):
    print(f"\n=== {label}: best wAUROC per direction (over all configs) ===")
    by_direction: dict[str, list] = {}
    for r in merged["results"]:
        by_direction.setdefault(r["direction"], []).append(r)
    for direction, rows in by_direction.items():
        # Sort by wauroc descending
        rows_sorted = sorted(rows, key=lambda r: -(r.get("weighted_auroc") or -1))
        print(f"\n  {direction}:")
        for r in rows_sorted:
            wa = r.get("weighted_auroc")
            wa_str = (
                f"{wa:.3f} [{r['weighted_auroc_ci_lower']:.3f}, {r['weighted_auroc_ci_upper']:.3f}]"
                if wa is not None else "n/a"
            )
            au_str = (
                f"{r['auroc']:.3f} [{r['auroc_ci_lower']:.3f}, {r['auroc_ci_upper']:.3f}]"
            )
            print(f"    {r['splitting']:>7} {r['classifier']:>22}: "
                  f"AUROC {au_str}  | wAUROC {wa_str}")


full = merge(
    Path("outputs/cross_dataset_v2_wauroc_full"),
    Path("outputs/cross_dataset_results_v2_norm_on.json"),
    cohort_variant="full",
    version="v2_wauroc_full",
)
psm = merge(
    Path("outputs/cross_dataset_v2_wauroc"),
    Path("outputs/cross_dataset_results_v2_psm.json"),
    cohort_variant="psm_matched",
    version="v2_wauroc",
)

summarize(full, "FULL Charité (thesis primary)")
summarize(psm, "PSM-matched (appendix sensitivity)")
