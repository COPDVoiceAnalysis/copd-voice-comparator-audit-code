"""One table from the cross-cohort runs (Reviewer 1, point 5).

Reads one or more cross_dataset_results.json files written by
src/train_cross_dataset.py (e.g. the run on all UK participants and the run on
the German-language subset) and writes one CSV row per
(UK subset, direction, classifier, splitting) with AUROC, weighted AUROC and
balanced accuracy at the optimal threshold, each with its 95% CI.

Usage: uv run python scripts/revision/summarize_crossdataset.py out.csv <results.json> [...]
"""
import json
import sys

import pandas as pd

COLS = ["direction", "classifier", "splitting", "n_train", "n_test", "auroc", "auroc_ci_lower",
        "auroc_ci_upper", "weighted_auroc", "weighted_auroc_ci_lower", "weighted_auroc_ci_upper",
        "wba_opt", "wba_opt_ci_lower", "wba_opt_ci_upper", "threshold_opt", "best_params"]


def main():
    out, files = sys.argv[1], sys.argv[2:]
    rows = []
    for f in files:
        d = json.load(open(f))
        subset = "+".join(d.get("uk_languages") or []) or "all"
        for r in d["results"]:
            rows.append({"uk_subset": subset, "charite_cohort": d.get("charite_cohort_variant"),
                         "preproc": d.get("preproc"), **{c: r.get(c) for c in COLS}})
    df = pd.DataFrame(rows)
    df["best_params"] = df["best_params"].astype(str)
    df.to_csv(out, index=False)
    print(df.drop(columns=["best_params"]).to_string(index=False))


if __name__ == "__main__":
    main()
