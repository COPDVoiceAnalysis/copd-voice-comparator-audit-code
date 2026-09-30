"""PSM-based control matching for UK COVID voice biomarker data.

Performs propensity score matching (PSM) as a pre-filter for the main
training pipeline. Reduces the extreme case:control imbalance (~1:73)
to a manageable ratio (default 1:10) while balancing covariates.

The output CSV is pipeline-compatible with src/models/metadata.py.

Usage:
    uv run python scripts/match_uk_controls.py \
        --input notebooks/UK/all_metadata/uk_metadata_cleaned_deduplicated.csv \
        --output data/uk_matched_metadata.csv

Quality gates (configurable):
    --smd-threshold 0.20    Max |SMD| after matching (default: 0.20)
    --min-matched-pct 80    Min % of cases matched (default: 80)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.psm_matching import (
    compute_propensity_scores,
    match_within_strata,
    smd_table,
)
from src.utils.recording_identifier import generate_recording_identifier

logger = logging.getLogger(__name__)

# ── UK comorbidity columns (binary flags present in UK metadata) ────────────
UK_COMORBIDITY_COLS = [
    "angina",
    "asthma",
    "cancer",
    "cystic",
    "diabetes",
    "hbp",
    "heart",
    "hiv",
    "long",
    "longterm",
    "lung",
    "organ",
    "otherheart",
    "pulmonary",
    "stroke",
    "valvular",
]

REQUIRED_INPUT_COLS = ["audio_id", "age", "sex", "language", "date", "audio_sample_path"]


class MatchingQualityError(Exception):
    """Raised when matching quality checks fail, stopping the pipeline."""


# ── Helpers ─────────────────────────────────────────────────────────────────


def _setup_logging() -> None:
    """Configure structured logging to stderr."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def _filter_voice_recordings(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose audio_sample_path is a voice (poem) recording that exists on disk.

    The UK metadata contains cough, breath, and voice recordings.  Only voice
    recordings are used for classification, so PSM must operate on the
    analysis-eligible subset — participants who actually have a voice recording.
    Without this filter, matched controls are silently lost when the pipeline
    later restricts to voice data.
    """
    col = "audio_sample_path"
    if col not in df.columns:
        logger.warning("No '%s' column — skipping voice recording filter.", col)
        return df

    before = len(df)

    # Keep only voice recordings (filename starts with "voice_")
    is_voice = df[col].apply(
        lambda p: pd.notna(p) and Path(str(p)).name.startswith("voice_")
    )
    # Also require the file to exist on disk
    exists = df[col].apply(
        lambda p: Path(str(p)).exists() if pd.notna(p) else False
    )
    mask = is_voice & exists
    df = df[mask].copy()
    dropped = before - len(df)

    n_not_voice = before - int(is_voice.sum())
    n_missing = int(is_voice.sum()) - int(mask.sum())

    if dropped:
        n_cases = (df["label"] == 1).sum() if "label" in df.columns else "?"
        n_ctrls = (df["label"] == 0).sum() if "label" in df.columns else "?"
        logger.info(
            "Voice recording filter: dropped %d / %d rows "
            "(%d not voice, %d file missing; remaining: %d — cases=%s, controls=%s)",
            dropped,
            before,
            n_not_voice,
            n_missing,
            len(df),
            n_cases,
            n_ctrls,
        )
    else:
        logger.info(
            "Voice recording filter: all %d rows are existing voice recordings.", before
        )

    return df


def _load_and_validate(path: Path, require_audio: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Load UK metadata CSV, validate required columns, normalise types.

    Parameters
    ----------
    path : Path to input CSV.
    require_audio : If True, drop rows whose audio file does not exist on disk
        *before* returning.  This guarantees that PSM operates only on
        analysis-eligible participants.

    Returns
    -------
    df : DataFrame with ``label`` column (1=COPD, 0=control).
    present_comorb : list of comorbidity columns actually found in the data.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Resolve label column
    if "copd" in df.columns:
        df["label"] = df["copd"].astype(int)
    elif "lung_disease_main" in df.columns:
        df["label"] = (df["lung_disease_main"] == "copd").astype(int)
    else:
        raise ValueError("Input CSV must have a 'copd' or 'lung_disease_main' column.")

    # Check required columns
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Type coercion and filtering
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[(df["age"] >= 10) & (df["age"] <= 100)].copy()
    df = df.dropna(subset=["label", "sex", "language", "audio_id", "age"])

    # Filter to voice recordings with existing files BEFORE PSM
    if require_audio:
        df = _filter_voice_recordings(df)

    # Detect available comorbidity columns
    present_comorb = [c for c in UK_COMORBIDITY_COLS if c in df.columns]

    logger.info(
        "Loaded %d rows from %s  (COPD=%d, controls=%d)",
        len(df),
        path.name,
        (df.label == 1).sum(),
        (df.label == 0).sum(),
    )
    logger.info("Available comorbidity columns (%d): %s", len(present_comorb), present_comorb)

    return df, present_comorb


def _log_demographics(df: pd.DataFrame, tag: str) -> None:
    """Log age/sex/language distribution for a dataset slice."""
    cases = df[df.label == 1]
    ctrls = df[df.label == 0]

    logger.info("─── Demographics [%s] ───", tag)
    logger.info(
        "  Age — cases: mean=%.1f (sd=%.1f), controls: mean=%.1f (sd=%.1f)",
        cases["age"].mean(),
        cases["age"].std(),
        ctrls["age"].mean(),
        ctrls["age"].std(),
    )

    for grp_name, grp in [("cases", cases), ("controls", ctrls)]:
        sex_dist = grp["sex"].value_counts().to_dict()
        lang_dist = grp["language"].value_counts().head(5).to_dict()
        logger.info("  Sex [%s]: %s", grp_name, sex_dist)
        logger.info("  Language [%s] (top 5): %s", grp_name, lang_dist)


def _log_smd_table(smd_df: pd.DataFrame) -> None:
    """Log SMD table as formatted text."""
    if smd_df.empty:
        logger.info("  (no covariates to report)")
        return
    header = f"  {'Covariate':<30s} {'SMD before':>12s} {'SMD after':>12s}"
    logger.info(header)
    logger.info("  " + "─" * 56)
    for cov, row in smd_df.iterrows():
        flag = " ⚠" if abs(row["smd_after"]) > 0.10 else ""
        logger.info(
            "  %-30s %12.3f %12.3f%s",
            cov,
            row["smd_before"],
            row["smd_after"],
            flag,
        )


def _check_stratum_sparsity(matched_df: pd.DataFrame, exact_keys: list[str]) -> None:
    """Warn about strata with very few samples."""
    if not exact_keys:
        return
    stratum = matched_df[exact_keys].astype(str).agg("|".join, axis=1)
    counts = stratum.value_counts()
    sparse = counts[counts < 5]
    if len(sparse) > 0:
        logger.warning(
            "Sparse strata (< 5 samples): %d of %d strata",
            len(sparse),
            len(counts),
        )
        for s, n in sparse.items():
            logger.warning("  stratum '%s': %d samples", s, n)


def _make_pipeline_compatible(
    matched_df: pd.DataFrame,
    present_comorb: list[str],
) -> pd.DataFrame:
    """Add / rename columns so the output CSV passes MetadataRow validation."""
    out = matched_df.copy()

    # Recording identifier (deterministic hash)
    out["recording_identifier"] = out.apply(
        lambda r: generate_recording_identifier(
            audio_id=int(r["audio_id"]),
            recording_category="poem",
            date=str(r["date"]) if pd.notna(r.get("date")) else "2020-01-01",
        ),
        axis=1,
    )

    # Required pipeline columns
    out["lung_disease_main"] = out["label"].map({1: "copd", 0: "control"})
    out["data_source"] = "uk_covid"
    out["recording_category"] = "poem"
    out["longitudinal"] = 0
    out["temporal_order"] = 1.0

    if "birth_year" not in out.columns:
        out["birth_year"] = (2020 - out["age"]).astype(int)  # UK study year ~2020/21

    if "control" not in out.columns:
        out["control"] = (out["label"] == 0).astype(int)

    if "study_id" not in out.columns and "uid" in out.columns:
        out["study_id"] = out["uid"]

    # Select and order columns for output
    # Keep all original + generated columns; drop internal matching artifacts
    drop_cols = {"stratum", "age_z", "age_z2", "ps", "logit_ps", "label"}
    keep_cols = [c for c in out.columns if c not in drop_cols]
    return out[keep_cols].sort_values("audio_id").reset_index(drop=True)


# ── Quality validation ──────────────────────────────────────────────────────


def validate_matching_quality(
    before_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    smd_df: pd.DataFrame,
    drop_report: dict[str, int],
    smd_threshold: float,
    min_matched_pct: float,
) -> dict:
    """Run all quality checks and return stats dict. Raises on failure."""

    n_cases_before = (before_df.label == 1).sum()
    n_cases_after = (matched_df.label == 1).sum()
    n_controls_after = (matched_df.label == 0).sum()

    case_retention_pct = 100.0 * n_cases_after / max(n_cases_before, 1)
    effective_ratio = n_controls_after / max(n_cases_after, 1)
    max_abs_smd = float(smd_df["smd_after"].abs().max()) if not smd_df.empty else 0.0

    # ── Log summary ──
    logger.info("═══ MATCHING RESULTS ═══")
    logger.info(
        "Cases:    %d → %d  (retention %.1f%%)",
        n_cases_before,
        n_cases_after,
        case_retention_pct,
    )
    logger.info(
        "Controls: %d → %d  (effective ratio %.1f:1)",
        (before_df.label == 0).sum(),
        n_controls_after,
        effective_ratio,
    )
    logger.info("Max |SMD| after matching: %.3f", max_abs_smd)
    logger.info("Drop report: %s", drop_report)

    # ── Log SMD table ──
    logger.info("─── SMD Table ───")
    _log_smd_table(smd_df)

    # ── Build stats dict ──
    smd_before_dict = smd_df["smd_before"].to_dict() if not smd_df.empty else {}
    smd_after_dict = smd_df["smd_after"].to_dict() if not smd_df.empty else {}

    stats = {
        "before": {
            "n_total": len(before_df),
            "n_copd": int(n_cases_before),
            "n_controls": int((before_df.label == 0).sum()),
        },
        "after": {
            "n_total": len(matched_df),
            "n_copd": int(n_cases_after),
            "n_controls": int(n_controls_after),
        },
        "case_retention_pct": round(case_retention_pct, 1),
        "effective_ratio": round(effective_ratio, 1),
        "drop_report": drop_report,
        "smd_before": {k: round(v, 4) for k, v in smd_before_dict.items()},
        "smd_after": {k: round(v, 4) for k, v in smd_after_dict.items()},
        "max_abs_smd_after": round(max_abs_smd, 4),
    }

    # ── Quality gates ──
    failures: list[str] = []

    if max_abs_smd > smd_threshold:
        worst_cov = smd_df["smd_after"].abs().idxmax()
        failures.append(
            f"Max |SMD| after matching = {max_abs_smd:.3f} > threshold {smd_threshold} "
            f"(worst covariate: '{worst_cov}')"
        )

    if case_retention_pct < min_matched_pct:
        failures.append(
            f"Case retention = {case_retention_pct:.1f}% < minimum {min_matched_pct}%"
        )

    if effective_ratio < 2.0:
        failures.append(
            f"Effective ratio = {effective_ratio:.1f}:1 < minimum 2:1"
        )

    stats["quality_passed"] = len(failures) == 0

    if failures:
        for f in failures:
            logger.error("MATCHING QUALITY FAILURE: %s", f)
        raise MatchingQualityError(
            "Matching quality checks failed — pipeline should not continue.\n"
            + "\n".join(f"  • {f}" for f in failures)
        )

    logger.info("✓ All quality checks passed.")
    return stats


# ── Main ────────────────────────────────────────────────────────────────────


def match_uk_controls(
    input_path: Path,
    output_path: Path,
    match_ratio_k: int = 10,
    caliper_coef: float = 0.20,
    age_caliper: int | None = 15,
    exact_keys: list[str] | None = None,
    ps_covariates: list[str] | None = None,
    random_seed: int = 42,
    smd_threshold: float = 0.20,
    min_matched_pct: float = 80.0,
    stats_output: Path | None = None,
    require_audio: bool = True,
) -> Path:
    """Run PSM control matching for UK data and write pipeline-compatible CSV.

    Parameters
    ----------
    input_path : Path to UK metadata CSV.
    output_path : Path for matched output CSV.
    match_ratio_k : Controls per case (default 10).
    caliper_coef : PS caliper on logit-PS scale (default 0.20).
    age_caliper : Max |age_case - age_ctrl| in years, or None.
    exact_keys : Columns for exact-match strata (default: ["sex", "language"]).
    ps_covariates : Columns for PS model (default: all UK comorbidity flags).
    random_seed : Random state for reproducibility.
    smd_threshold : Max tolerable |SMD| after matching (quality gate).
    min_matched_pct : Min % of cases that must be matched (quality gate).
    stats_output : Path for JSON stats report (default: output_path with _stats.json suffix).
    require_audio : If True (default), drop rows whose audio file does not exist
        on disk *before* computing propensity scores. This prevents matched
        controls from being silently lost in downstream audio-quality checks.

    Returns
    -------
    output_path : Path to the written CSV.

    Raises
    ------
    MatchingQualityError : If quality checks fail.
    """
    if exact_keys is None:
        exact_keys = ["sex", "language"]
    if stats_output is None:
        stem = output_path.stem
        stats_output = output_path.with_name(f"{stem}_stats.json")

    # ── 1. Load & validate ──
    df, present_comorb = _load_and_validate(input_path, require_audio=require_audio)

    if ps_covariates is None:
        ps_covariates = present_comorb
    else:
        # Filter to actually present columns
        ps_covariates = [c for c in ps_covariates if c in df.columns]

    logger.info(
        "Config: K=%d, caliper=%.2f, age_caliper=%s, exact=%s, ps_covs=%d cols, seed=%d",
        match_ratio_k,
        caliper_coef,
        age_caliper,
        exact_keys,
        len(ps_covariates),
        random_seed,
    )

    _log_demographics(df, "BEFORE matching")

    # ── 2. Compute propensity scores ──
    use_age = age_caliper is not None
    if ps_covariates:
        df_ps, clf = compute_propensity_scores(
            df, ps_covariates, use_age=use_age, random_state=random_seed
        )
        # Log PS model coefficients
        used_cols = [c for c in ps_covariates if c in df.columns]
        if use_age:
            used_cols = used_cols + ["age_z", "age_z2"]
        logger.info("PS model coefficients:")
        for col, coef in zip(used_cols, clf.coef_.ravel()):
            logger.info("  %-25s  %.4f", col, coef)
        logger.info("  %-25s  %.4f", "intercept", clf.intercept_[0])
    else:
        # No PS covariates — matching on exact keys + age caliper only
        df_ps = df.copy()
        df_ps["ps"] = 0.5
        df_ps["logit_ps"] = 0.0
        logger.info("No PS covariates specified — matching on exact keys and age caliper only.")

    # ── 3. Match ──
    matched, drop_report = match_within_strata(
        df_ps,
        exact_keys=exact_keys,
        k=match_ratio_k,
        caliper_coef=caliper_coef,
        age_caliper_years=age_caliper,
    )

    _log_demographics(matched, "AFTER matching")

    # ── 4. SMD diagnostics ──
    # Broad SMD: all covariates (age + comorbidities)
    bin_cols = present_comorb
    cont_cols = ["age"]
    smd_df = smd_table(df, matched, bin_cols, cont_cols)

    # ── 5. Stratum sparsity warnings ──
    _check_stratum_sparsity(matched, exact_keys)

    # ── 6. Quality validation (raises on failure) ──
    stats = validate_matching_quality(
        before_df=df,
        matched_df=matched,
        smd_df=smd_df,
        drop_report=drop_report,
        smd_threshold=smd_threshold,
        min_matched_pct=min_matched_pct,
    )

    # Add config to stats
    stats["timestamp"] = datetime.now(timezone.utc).isoformat()
    stats["config"] = {
        "input": str(input_path),
        "match_ratio_k": match_ratio_k,
        "caliper_coef": caliper_coef,
        "age_caliper": age_caliper,
        "exact_keys": exact_keys,
        "ps_covariates": ps_covariates,
        "random_seed": random_seed,
        "smd_threshold": smd_threshold,
        "min_matched_pct": min_matched_pct,
    }

    # ── 7. Write outputs ──
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_df = _make_pipeline_compatible(matched, present_comorb)
    pipeline_df.to_csv(output_path, index=False)
    logger.info("Written matched metadata: %s  (%d rows)", output_path, len(pipeline_df))

    # Emit ps_covariates.csv (convention: first col = audio_id, rest = covariates).
    # Only include comorbidity columns with non-zero variance in the matched set,
    # so the PS overlap weighting model doesn't waste capacity on constant features.
    nonzero_comorb = [
        c for c in present_comorb
        if c in pipeline_df.columns and pipeline_df[c].var() > 0
    ]
    if nonzero_comorb:
        ps_cov_path = output_path.with_name(f"{output_path.stem}_ps_covariates.csv")
        ps_cov_df = pipeline_df[["audio_id"] + nonzero_comorb].copy()
        ps_cov_df.to_csv(ps_cov_path, index=False)
        logger.info(
            "Written ps_covariates: %s  (%d covariates: %s)",
            ps_cov_path,
            len(nonzero_comorb),
            nonzero_comorb,
        )
        stats["ps_covariates_file"] = str(ps_cov_path)
        stats["ps_covariates_columns"] = nonzero_comorb

    with open(stats_output, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Written stats report: %s", stats_output)

    return output_path


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="PSM control matching for UK COVID voice biomarker data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to UK metadata CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for matched output CSV (pipeline-compatible).",
    )
    parser.add_argument(
        "--match-ratio-k",
        type=int,
        default=10,
        help="Number of controls to match per case.",
    )
    parser.add_argument(
        "--caliper-coef",
        type=float,
        default=0.20,
        help="PS caliper = coef × local SD of logit-PS.",
    )
    parser.add_argument(
        "--age-caliper",
        type=int,
        default=15,
        help="Max absolute age difference (years). Set to 0 to disable.",
    )
    parser.add_argument(
        "--exact-keys",
        nargs="*",
        default=["sex", "language"],
        help="Columns for exact-match strata.",
    )
    parser.add_argument(
        "--ps-covariates",
        nargs="*",
        default=None,
        help="Columns for PS model. Default: all UK comorbidity flags.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--smd-threshold",
        type=float,
        default=0.20,
        help="Max tolerable |SMD| after matching (quality gate).",
    )
    parser.add_argument(
        "--min-matched-pct",
        type=float,
        default=80.0,
        help="Min %% of cases that must be matched (quality gate).",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Path for JSON stats report. Default: <output>_stats.json.",
    )
    parser.add_argument(
        "--no-require-audio",
        action="store_true",
        default=False,
        help="Skip audio file existence check before PSM (not recommended).",
    )

    args = parser.parse_args()

    age_caliper = args.age_caliper if args.age_caliper > 0 else None

    try:
        match_uk_controls(
            input_path=args.input,
            output_path=args.output,
            match_ratio_k=args.match_ratio_k,
            caliper_coef=args.caliper_coef,
            age_caliper=age_caliper,
            exact_keys=args.exact_keys,
            ps_covariates=args.ps_covariates,
            random_seed=args.random_seed,
            smd_threshold=args.smd_threshold,
            min_matched_pct=args.min_matched_pct,
            stats_output=args.stats_output,
            require_audio=not args.no_require_audio,
        )
    except MatchingQualityError as e:
        logger.error("Pipeline stopped: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
