"""Generate a propensity-score-matched Charité cohort.

This module is the reproducible replacement for the interactive exploration
in ``notebooks/charite_psm_matching.py``. Given a metadata CSV, a comorbidities
CSV and a PSM configuration (under ``psm_matching.<cohort>`` in
``pipeline_config.yml``), it produces:

* A filtered metadata CSV containing all recording-level rows for the
  matched Charité participants — a drop-in replacement for the standard
  preprocessed metadata file, consumable by downstream training.
* A balance report CSV with standardised-mean-difference (SMD) values before
  and after matching for every covariate used in the propensity model.

The algorithm mirrors the notebook exactly:

1. Deduplicate metadata to one row per person (``audio_id``).
2. Merge comorbidities, derive ``age = reference_year − birth_year``.
3. Swap labels (controls become the rare "treated" group) so the shared
   ``src.utils.psm_matching.match_within_strata`` infrastructure — which
   matches ``label == 1`` to ``label == 0`` — can be reused as-is.
4. Fit a logistic propensity model on the configured covariates (+ age).
5. Greedy 1:K matching within the configured exact-match strata, enforcing
   the PS caliper and optional absolute-age caliper.
6. Un-swap labels for output semantics.
7. Filter the *full* metadata back to the matched persons (all recording
   categories / chunks preserved).

Invocation (from project root)::

    uv run python -m src.utils.generate_psm_cohort \
        --metadata     data/2025-08-13/preprocessed__norm_on/metadata.csv \
        --comorbidities data/2025-08-13/comorbidities.csv \
        --config       config/pipeline_config.yml \
        --output-csv   data/2025-08-13/psm_matched_charite.csv \
        --balance-report data/2025-08-13/psm_matched_charite_balance_report.csv

The same Snakemake rule (``match_charite_psm``) wires this into the pipeline
using paths derived from ``DATE`` so the output is fully reproducible from a
fresh checkout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.utils.psm_matching import (
    compute_propensity_scores,
    match_within_strata,
    smd_table,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cohort loading
# ---------------------------------------------------------------------------
def load_person_level_cohort(
    metadata_path: Path,
    comorbidities_path: Path,
    cohort: str,
    reference_year: int,
    comorbidity_cols: list[str],
    required_recording_categories: list[str] | None = None,
    exclude_longitudinal: bool = False,
) -> pd.DataFrame:
    """Build a person-level DataFrame for PSM.

    Parameters
    ----------
    metadata_path : Path
        Full metadata CSV (may contain multiple rows per person).
    comorbidities_path : Path
        CSV with ``audio_id`` and one binary column per comorbidity.
    cohort : str
        Value of ``data_source`` to filter on (e.g. ``"charite"``).
    reference_year : int
        Calendar year used to derive ``age = reference_year − birth_year``.
    comorbidity_cols : list[str]
        Binary covariate columns expected in ``comorbidities_path``.
    """
    meta_full = pd.read_csv(metadata_path).query("data_source == @cohort")
    if required_recording_categories:
        if "recording_category" not in meta_full.columns:
            raise KeyError(
                "recording_category column required for the complete-case "
                "restriction but is absent from the metadata"
            )
        _req = set(required_recording_categories)
        _cats = meta_full.groupby("audio_id")["recording_category"].apply(set)
        _complete = {a for a, av in _cats.items() if _req.issubset(av)}
        _n0 = meta_full["audio_id"].nunique()
        meta_full = meta_full[meta_full["audio_id"].isin(_complete)]
        logger.info(
            "Complete-case restriction %s applied BEFORE matching: "
            "%d -> %d persons",
            sorted(_req), _n0, meta_full["audio_id"].nunique(),
        )
    if exclude_longitudinal:
        if "longitudinal" not in meta_full.columns:
            raise KeyError(
                "longitudinal column required for exclude_longitudinal but "
                "is absent from the metadata"
            )
        _n0 = meta_full["audio_id"].nunique()
        meta_full = meta_full[meta_full["longitudinal"] != 1]
        logger.info(
            "exclude_longitudinal=True applied BEFORE matching: "
            "%d -> %d persons",
            _n0, meta_full["audio_id"].nunique(),
        )
    meta = meta_full.drop_duplicates("audio_id")
    ps = pd.read_csv(comorbidities_path)
    df = meta.merge(ps, on="audio_id", how="left")

    for c in comorbidity_cols:
        if c not in df.columns:
            logger.warning("Comorbidity column %r absent; defaulting to 0", c)
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "birth_year" not in df.columns:
        raise KeyError(
            f"Metadata at {metadata_path} is missing required 'birth_year' column"
        )
    df["age"] = reference_year - df["birth_year"]
    df["label"] = df["lung_disease_main"].map({"copd": 1, "control": 0})
    df = df[df["label"].isin([0, 1]) & df["age"].notna()].copy()
    df["age"] = df["age"].astype(float)
    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    return df


# ---------------------------------------------------------------------------
# Matching (label-swap pattern for rare-control cohorts)
# ---------------------------------------------------------------------------
def run_label_swapped_psm(
    cohort_df: pd.DataFrame,
    k: int,
    caliper_coef: float,
    age_caliper_years: int | None,
    exact_match: list[str],
    propensity_covariates: list[str],
    use_age: bool,
    random_seed: int,
    include_age_squared: bool = True,
    report_bin_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run 1:K matching with swapped labels.

    ``match_within_strata`` always iterates over ``label == 1`` as the
    "treated" group. In the Charité cohort controls are rare — the intuitive
    mapping (COPD=1) would therefore match many controls to each COPD case,
    which is not what we want. Swapping labels (``ctrl → 1``, ``copd → 0``)
    causes each control to acquire K COPD matches; after matching we restore
    the original label semantics on the output.
    """
    swapped = cohort_df.copy()
    swapped["orig_label"] = swapped["label"]
    swapped["label"] = 1 - swapped["label"]

    df_ps, _clf = compute_propensity_scores(
        swapped,
        propensity_covariates,
        use_age=use_age,
        include_age_squared=include_age_squared,
        random_state=random_seed,
    )
    matched, drop_report = match_within_strata(
        df_ps,
        exact_keys=list(exact_match),
        k=k,
        caliper_coef=caliper_coef,
        age_caliper_years=age_caliper_years,
    )

    # Restore original label semantics for output and balance reporting
    matched["label"] = matched["orig_label"]
    df_ps["label"] = df_ps["orig_label"]

    smd = smd_table(
        df_ps,
        matched,
        bin_cols=list(
            report_bin_cols if report_bin_cols is not None else propensity_covariates
        ),
        cont_cols=["age"],
    )
    return matched, smd, drop_report


# ---------------------------------------------------------------------------
# Metadata expansion
# ---------------------------------------------------------------------------
def filter_metadata_to_matched(
    metadata_path: Path,
    matched_audio_ids: set[int],
    cohort: str,
) -> pd.DataFrame:
    """Keep all recording-level rows that belong to a matched ``audio_id``.

    We only emit rows from ``cohort``; the resulting file is deliberately
    single-cohort so downstream tasks configured with
    ``included_data_sources: [<cohort>]`` get exactly the matched set.
    """
    full = pd.read_csv(metadata_path)
    mask = (full["data_source"] == cohort) & full["audio_id"].isin(matched_audio_ids)
    return full.loc[mask].copy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True,
                   help="Full (recording-level) metadata CSV.")
    p.add_argument("--comorbidities", type=Path, required=True,
                   help="Comorbidities / PS-covariates CSV (audio_id keyed).")
    p.add_argument("--config", type=Path, required=True,
                   help="Pipeline config YAML (reads psm_matching.<cohort>).")
    p.add_argument("--output-csv", type=Path, required=True,
                   help="Path for filtered (matched) metadata CSV.")
    p.add_argument("--balance-report", type=Path, required=True,
                   help="Path for SMD before/after report CSV.")
    p.add_argument("--cohort", default="charite",
                   help="data_source to match (and key into psm_matching.<cohort>).")
    p.add_argument("--require-recording-categories", nargs="*", default=None,
                   help="Restrict to persons having ALL these recording "
                        "categories BEFORE matching (mirrors the consuming "
                        "task's complete-case restriction; prevents the "
                        "filter-after-match pairing break).")
    p.add_argument("--exclude-longitudinal", action="store_true",
                   help="Drop persons with longitudinal=1 BEFORE matching "
                        "(mirrors the consuming task's exclude_longitudinal "
                        "filter; prevents post-match pairing break).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        pipeline_config = yaml.safe_load(f)

    try:
        psm_cfg = pipeline_config["psm_matching"][args.cohort]
    except KeyError as e:
        raise SystemExit(
            f"Config section psm_matching.{args.cohort} not found in {args.config}"
        ) from e

    logger.info("PSM config for cohort=%s: %s", args.cohort, psm_cfg)

    _prop = list(psm_cfg["propensity_covariates"])
    _report = list(psm_cfg.get("report_covariates", []))
    _load_cols = list(dict.fromkeys(_prop + _report))
    person_df = load_person_level_cohort(
        metadata_path=args.metadata,
        comorbidities_path=args.comorbidities,
        cohort=args.cohort,
        reference_year=int(psm_cfg["reference_year"]),
        comorbidity_cols=_load_cols,
        required_recording_categories=args.require_recording_categories,
        exclude_longitudinal=args.exclude_longitudinal,
    )
    n_copd = int((person_df["label"] == 1).sum())
    n_ctrl = int((person_df["label"] == 0).sum())
    logger.info(
        "Loaded %s cohort: %d persons (%d COPD / %d control)",
        args.cohort, len(person_df), n_copd, n_ctrl,
    )

    matched, smd, drop_report = run_label_swapped_psm(
        person_df,
        k=int(psm_cfg["k"]),
        caliper_coef=float(psm_cfg["caliper_coef"]),
        age_caliper_years=psm_cfg.get("age_caliper_years"),
        exact_match=list(psm_cfg.get("exact_match", [])),
        propensity_covariates=_prop,
        use_age=bool(psm_cfg.get("use_age", True)),
        random_seed=int(psm_cfg["random_seed"]),
        include_age_squared=bool(psm_cfg.get("age_squared", True)),
        report_bin_cols=(_report or _prop),
    )

    n_copd_m = int((matched["label"] == 1).sum())
    n_ctrl_m = int((matched["label"] == 0).sum())
    logger.info(
        "Matched cohort: %d persons (%d COPD / %d control). Drops: %s",
        len(matched), n_copd_m, n_ctrl_m, dict(drop_report),
    )
    logger.info(
        "Max |SMD_after|: %.3f (over %d covariates)",
        smd["smd_after"].abs().max(), len(smd),
    )

    matched_audio_ids = set(matched["audio_id"].astype(int).tolist())
    output_meta = filter_metadata_to_matched(
        metadata_path=args.metadata,
        matched_audio_ids=matched_audio_ids,
        cohort=args.cohort,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_meta.to_csv(args.output_csv, index=False)
    logger.info(
        "Wrote matched metadata: %s (%d rows, %d unique audio_ids)",
        args.output_csv, len(output_meta), output_meta["audio_id"].nunique(),
    )

    smd_out = smd.copy()
    smd_out.index.name = "covariate"
    smd_out["cohort"] = args.cohort
    smd_out["k"] = int(psm_cfg["k"])
    smd_out["n_total"] = len(matched)
    smd_out["n_copd"] = n_copd_m
    smd_out["n_control"] = n_ctrl_m
    for drop_key, drop_val in drop_report.items():
        smd_out[f"drop_{drop_key}"] = int(drop_val)
    args.balance_report.parent.mkdir(parents=True, exist_ok=True)
    smd_out.to_csv(args.balance_report)
    logger.info("Wrote balance report: %s", args.balance_report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
