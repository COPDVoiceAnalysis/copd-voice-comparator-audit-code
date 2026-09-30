"""
Build ps_covariates.csv from Excel sources (no aggregation).

Convention: first column = ID (audio_id), remaining columns = raw disease covariates.
Age and sex are taken from metadata; comorbidity_count_bin is computed in
extract_baseline_features / get_metadata_for_audio_id_sequence, not here.

Reads:
  - comorbidities Excel (Studien-ID + disease columns)
  - voice_audio_id.xlsx (Studien-ID -> Audio-ID)

Maps disease columns to 4 binary groups of pre-COPD baseline confounders:
breathing, metabolic, depression, hypothyreose. Cardiovascular and renal
conditions (KHK, Herzinsuffizienz, VHF, Niereninsuffizienz) are excluded because
they are clinically downstream of COPD (cor pulmonale, pulmonary hypertension,
hypoxia-related CKD) and 0% prevalent in controls here, which would make them
perfect-separation predictors rather than baseline confounders. Dyslipidämie and
Steatosis hepatis are excluded as likely coding artifacts (0/26 controls despite
~40-50% population prevalence). Output is audio_id plus these 4 columns only.

Usage:
  python scripts/build_ps_covariates_csv.py
  python scripts/build_ps_covariates_csv.py --comorbidities path/to/comorbidities.xlsx --voice-id path/to/voice_audio_id.xlsx --output path/to/ps_covariates.csv
"""

from pathlib import Path
import argparse
import pandas as pd


def _normalize_study_id(s: pd.Series) -> pd.Series:
    """Normalize Studien-ID to lowercase, strip whitespace."""
    return s.astype(str).str.strip().str.lower()


def _fillna_zero(s: pd.Series) -> pd.Series:
    """Treat NaN as 0 for binary disease columns."""
    return s.fillna(0).astype(float).clip(0, 1)


def build_ps_covariates_csv(
    comorbidities_path: Path,
    voice_audio_id_path: Path,
    output_path: Path,
) -> None:
    """
    Load Excel files, map to 4 disease columns, join with voice_audio_id, write ps_covariates.csv.
    Output convention: first column = audio_id, rest = covariate columns (header = names).
    """
    # Load Excel files
    comorbidities_df = pd.read_excel(comorbidities_path)
    voice_df = pd.read_excel(voice_audio_id_path)

    # Normalize column names (strip whitespace) for robust access
    comorbidities_df.columns = comorbidities_df.columns.str.strip()
    voice_df.columns = voice_df.columns.str.strip()

    # Studien-ID: normalize to match metadata (voice_005)
    study_id_col = "Studien-ID"
    if study_id_col not in comorbidities_df.columns:
        raise ValueError(f"Expected column '{study_id_col}' in comorbidities file")
    comorbidities_df["study_id_norm"] = _normalize_study_id(pd.Series(comorbidities_df[study_id_col]))

    if study_id_col not in voice_df.columns:
        raise ValueError(f"Expected column '{study_id_col}' in voice_audio_id file")
    voice_df["study_id_norm"] = _normalize_study_id(pd.Series(voice_df[study_id_col]))

    # Build 4 disease columns: binary per group (0 = none, 1 = at least one). NaN -> 0.
    # breathing: at least one of Schlafapnoe, Adipositas
    cols_breathing = ["Schlafapnoe", "Adipositas"]
    breathing_parts: list[pd.Series] = []
    for c in cols_breathing:
        if c in comorbidities_df.columns:
            breathing_parts.append(_fillna_zero(pd.Series(comorbidities_df[c])))
    breathing = (
        (pd.concat(breathing_parts, axis=1) > 0).any(axis=1).astype(int)
        if breathing_parts
        else pd.Series(0, index=comorbidities_df.index)
    )

    # metabolic: at least one of Arterielle Hypertonie, Diabetes II
    # (Dyslipidämie and Steatosis hepatis excluded: likely coding artifacts,
    # 0/26 controls vs ~40-50% pop prevalence.)
    cols_met = ["Arterielle Hypertonie", "Diabetes Mellitus II"]
    met_parts = []
    for c in cols_met:
        if c in comorbidities_df.columns:
            met_parts.append(_fillna_zero(pd.Series(comorbidities_df[c])))
    metabolic = (
        (pd.concat(met_parts, axis=1) > 0).any(axis=1).astype(int)
        if met_parts
        else pd.Series(0, index=comorbidities_df.index)
    )

    # depression: Depression (single column -> binary 0/1)
    col_dep = "Depression"
    depression = (
        (_fillna_zero(pd.Series(comorbidities_df[col_dep])) > 0).astype(int)
        if col_dep in comorbidities_df.columns
        else pd.Series(0, index=comorbidities_df.index)
    )

    # hypothyreose: Hypothyreose (single column -> binary 0/1). Pre-COPD baseline
    # endocrine condition; previously silently dropped.
    col_hypo = "Hypothyreose"
    hypothyreose = (
        (_fillna_zero(pd.Series(comorbidities_df[col_hypo])) > 0).astype(int)
        if col_hypo in comorbidities_df.columns
        else pd.Series(0, index=comorbidities_df.index)
    )

    out_df = pd.DataFrame({
        "study_id_norm": comorbidities_df["study_id_norm"],
        "breathing": breathing,
        "metabolic": metabolic,
        "depression": depression,
        "hypothyreose": hypothyreose,
    })

    # Join with voice_audio_id to get audio_id (left join)
    audio_col = "Audio-ID"
    if audio_col not in voice_df.columns:
        raise ValueError(f"Expected column '{audio_col}' in voice_audio_id file")
    voice_lookup = voice_df[["study_id_norm", audio_col]].drop_duplicates(subset=["study_id_norm"])  # type: ignore[arg-type]
    merged = out_df.merge(voice_lookup, on="study_id_norm", how="left")
    merged = merged.rename(columns={audio_col: "audio_id"})
    merged["audio_id"] = pd.to_numeric(merged["audio_id"], errors="coerce")
    merged["audio_id"] = merged["audio_id"].astype("Int64")

    # Final columns: audio_id (first), then covariate columns (convention for ps_covariates.csv)
    result = merged[["audio_id", "breathing", "metabolic", "depression", "hypothyreose"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    audio_id_series = result["audio_id"]
    print(f"Wrote {len(result)} rows to {output_path}")
    print(f"  audio_id: {int(audio_id_series.notna().sum())} non-null, {int(audio_id_series.isna().sum())} null (no match in voice_audio_id)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ps_covariates.csv from Excel sources. Convention: first column = ID, rest = covariate columns."
    )
    parser.add_argument(
        "--comorbidities",
        type=Path,
        default=Path("data/2025-08-13/comorbidities_2026_01_28.xlsx"),
        help="Path to comorbidities Excel file",
    )
    parser.add_argument(
        "--voice-id",
        type=Path,
        default=Path("data/2025-08-13/voice_audio_id.xlsx"),
        help="Path to voice_audio_id Excel file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/2025-08-13/ps_covariates.csv"),
        help="Output CSV path (default: ps_covariates.csv)",
    )
    args = parser.parse_args()

    build_ps_covariates_csv(
        comorbidities_path=args.comorbidities,
        voice_audio_id_path=args.voice_id,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
