"""
Dataset merging script for COPD voice biomarker project.

Merges the Charité dataset with the UK COVID-19 Sounds dataset.  All
PSM-matched UK participants are kept; class-imbalance handling is deferred
to training-time weighting (propensity overlap weights, strata weights).
"""

import pandas as pd
from pathlib import Path
import logging
import argparse
from src.models.core_models import MetadataRow
from src.utils.recording_identifier import (
    generate_recording_identifier,
    validate_recording_identifiers,
)
from pydub import AudioSegment
from src.utils.logging_config import setup_basic_logging

logger = logging.getLogger(__name__)


def load_charite_metadata(metadata_path: str) -> pd.DataFrame:
    """Load and prepare Charite metadata."""
    logger.info(f"Loading Charite metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)

    # Add source column
    df["data_source"] = "charite"

    # Add language column if not present (Charite data is German)
    if "language" not in df.columns:
        df["language"] = "de"
        logger.info("Added language='de' column for Charite data")

    # Create age from birth_year (assuming current year is 2025)
    current_year = 2025
    df["age"] = current_year - df["birth_year"]

    # Create age groups (10-year bins)
    df["age_group"] = (df["age"] // 10) * 10

    logger.info(f"Loaded {len(df)} Charite records")
    return df


def load_uk_metadata(uk_metadata_path: str) -> pd.DataFrame:
    """Load pre-processed UK metadata."""
    logger.info(f"Loading UK metadata from {uk_metadata_path}")
    uk_df = pd.read_csv(uk_metadata_path)

    # Create age groups (10-year bins) if not already present
    if "age_group" not in uk_df.columns:
        uk_df["age_group"] = (uk_df["age"] // 10) * 10

    # Add data source if not already present
    if "data_source" not in uk_df.columns:
        uk_df["data_source"] = "uk_covid"

    # Rename uid to study_id if needed to match Charite format
    if "uid" in uk_df.columns and "study_id" not in uk_df.columns:
        uk_df = uk_df.rename(columns={"uid": "study_id"})

    logger.info(f"Loaded {len(uk_df)} UK records")
    logger.info(f"UK COPD cases: {len(uk_df[uk_df.lung_disease_main == 'copd'])}")
    logger.info(
        f"UK control cases: {len(uk_df[uk_df.lung_disease_main == 'control'])}"
    )

    return uk_df


def generate_recording_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate recording identifiers for all rows in the merged dataset.

    This ensures that both Charite and UK data have proper recording identifiers
    before any downstream processing (like poem splitting).

    Args:
        df: Merged dataframe that may have missing recording_identifier values

    Returns:
        DataFrame with recording_identifier column populated for all rows
    """
    logger.info("Generating recording identifiers for merged dataset...")

    # Check required columns
    required_columns = ["audio_id", "recording_category", "date"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns for recording identifier generation: {missing_columns}"
        )

    # Generate recording identifiers for all rows
    # Note: At this stage (before poem splitting), chunk_number should be None for all rows
    df["recording_identifier"] = df.apply(
        lambda row: generate_recording_identifier(
            row["audio_id"],
            row["recording_category"],
            row["date"],
            chunk_number=None,  # No chunking at merge stage
        ),
        axis=1,
    )

    validate_recording_identifiers(df)

    logger.info(f"Successfully generated {len(df)} unique recording identifiers")
    return df


def validate_audio_quality(
    df: pd.DataFrame, min_duration_sec: float = 3.0
) -> pd.DataFrame:
    """
    Filter out audio recordings shorter than minimum duration threshold.

    This quality management step ensures that only audio files with sufficient
    duration are included in the pipeline, preventing issues with feature extraction
    and model training on very short recordings.

    Args:
        df: DataFrame with audio file paths in 'audio_sample_path' column
        min_duration_sec: Minimum duration threshold in seconds (default: 3.0)

    Returns:
        Filtered DataFrame containing only recordings that meet quality criteria
    """
    logger.info(
        f"Starting audio quality validation (min duration: {min_duration_sec}s)"
    )

    if "audio_sample_path" not in df.columns:
        logger.warning(
            "No 'audio_sample_path' column found - skipping audio quality validation"
        )
        return df

    initial_count = len(df)
    valid_recordings = []
    invalid_recordings = []
    processing_errors = []

    for idx, row in df.iterrows():
        audio_path = Path(row["audio_sample_path"])

        try:
            # Check if file exists
            if not audio_path.exists():
                invalid_recordings.append(
                    {
                        "audio_id": row.get("audio_id", "unknown"),
                        "path": str(audio_path),
                        "reason": "File not found",
                    }
                )
                continue

            # Load audio and check duration
            try:
                audio = AudioSegment.from_file(audio_path)
                duration_sec = len(audio) / 1000.0

                if duration_sec >= min_duration_sec:
                    valid_recordings.append(idx)
                else:
                    invalid_recordings.append(
                        {
                            "audio_id": row.get("audio_id", "unknown"),
                            "path": str(audio_path),
                            "reason": f"Too short ({duration_sec:.2f}s < {min_duration_sec}s)",
                        }
                    )

            except Exception as audio_error:
                invalid_recordings.append(
                    {
                        "audio_id": row.get("audio_id", "unknown"),
                        "path": str(audio_path),
                        "reason": f"Audio processing error: {str(audio_error)}",
                    }
                )

        except Exception as e:
            processing_errors.append(
                {
                    "audio_id": row.get("audio_id", "unknown"),
                    "path": str(audio_path),
                    "error": str(e),
                }
            )

    # Filter DataFrame to keep only valid recordings
    filtered_df = df.iloc[valid_recordings].copy()

    # Log quality validation results
    valid_count = len(valid_recordings)
    invalid_count = len(invalid_recordings)
    error_count = len(processing_errors)

    logger.info(f"Audio quality validation results:")
    logger.info(f"  Initial recordings: {initial_count}")
    logger.info(f"  Valid recordings: {valid_count}")
    logger.info(f"  Invalid recordings: {invalid_count}")
    logger.info(f"  Processing errors: {error_count}")
    logger.info(
        f"  Filtered out: {initial_count - valid_count} ({((initial_count - valid_count) / initial_count * 100):.1f}%)"
    )

    # Log details about invalid recordings (first 10)
    if invalid_recordings:
        logger.info("Invalid recordings (showing first 10):")
        for i, invalid in enumerate(invalid_recordings[:10]):
            logger.info(
                f"  {i + 1}. Audio ID {invalid['audio_id']}: {invalid['reason']}"
            )
        if len(invalid_recordings) > 10:
            logger.info(f"  ... and {len(invalid_recordings) - 10} more")

    # Log processing errors if any
    if processing_errors:
        logger.warning(f"Processing errors encountered:")
        for error in processing_errors[:5]:  # Show first 5 errors
            logger.warning(f"  Audio ID {error['audio_id']}: {error['error']}")
        if len(processing_errors) > 5:
            logger.warning(f"  ... and {len(processing_errors) - 5} more errors")

    return filtered_df


def validate_metadata_output(df: pd.DataFrame) -> None:
    """Validate the merged metadata using MetadataRow pydantic model."""
    logger.info("Validating merged metadata with MetadataRow model...")

    validation_errors = []
    valid_rows = 0

    for idx, row in df.iterrows():
        try:
            # Convert row to dict and validate with MetadataRow
            row_dict = row.to_dict()
            MetadataRow(**row_dict)
            valid_rows += 1
        except Exception as e:
            validation_errors.append(f"Row {idx}: {str(e)}")
            if len(validation_errors) <= 10:  # Only log first 10 errors to avoid spam
                logger.error(f"Validation error in row {idx}: {str(e)}")

    logger.info(f"Validation complete: {valid_rows}/{len(df)} rows valid")

    if validation_errors:
        logger.error(f"Found {len(validation_errors)} validation errors")
        if len(validation_errors) > 10:
            logger.error("... (showing first 10 errors only)")
        raise ValueError(
            f"Metadata validation failed with {len(validation_errors)} errors. "
            f"First error: {validation_errors[0]}"
        )
    else:
        logger.info("All metadata rows passed validation!")


def merge_cohorts(
    charite_df: pd.DataFrame, uk_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge Charité and UK COVID datasets, keeping all PSM-matched UK participants.

    No sub-sampling is applied — the UK cohort enters the merged dataset exactly
    as produced by the upstream propensity score matching step
    (``scripts/match_uk_controls.py``).  Class-imbalance handling is deferred to
    training-time weighting (propensity overlap weights, strata weights).
    """
    logger.info("Merging Charité and UK COVID datasets")

    uk_copd = uk_df[uk_df.lung_disease_main == "copd"]
    uk_controls = uk_df[uk_df.lung_disease_main == "control"]

    logger.info(f"Charité records: {len(charite_df)}")
    logger.info(f"UK records: {len(uk_df)} (COPD: {len(uk_copd)}, controls: {len(uk_controls)})")

    final_df = pd.concat([charite_df, uk_df], ignore_index=True)

    # Log final statistics
    logger.info(f"Merged dataset: {len(final_df)} total records")

    longitudinal = final_df["longitudinal"] if "longitudinal" in final_df.columns else 0
    copd_final = final_df[
        (final_df.lung_disease_main == "copd") & (longitudinal == 0)
    ]
    control_final = final_df[
        (final_df.lung_disease_main == "control") & (longitudinal == 0)
    ]

    logger.info(f"Non-longitudinal COPD: {len(copd_final)}, controls: {len(control_final)}")
    if len(control_final) > 0:
        logger.info(f"COPD:Control ratio: {len(copd_final) / len(control_final):.2f}:1")

    return final_df


def main():
    """Main function to execute the dataset merging process."""
    parser = argparse.ArgumentParser(
        description="Merge Charite and UK COVID datasets with stratified matching"
    )
    parser.add_argument(
        "--charite_metadata", required=True, help="Path to Charite metadata.csv"
    )
    parser.add_argument("--uk_metadata", required=True, help="Path to UK metadata.csv")
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for merged dataset"
    )
    args = parser.parse_args()

    setup_basic_logging()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    charite_df = load_charite_metadata(args.charite_metadata)
    uk_df = load_uk_metadata(args.uk_metadata)

    # Merge cohorts (all PSM-matched UK participants are kept)
    merged_df = merge_cohorts(charite_df, uk_df)

    # Apply audio quality filtering (remove recordings < 3 seconds)
    merged_df = validate_audio_quality(merged_df, min_duration_sec=3.0)

    # Generate recording identifiers for all merged rows
    merged_df = generate_recording_identifiers(merged_df)

    # Drop chunking columns before validation — they are added later by split_poem
    chunk_cols = ["is_chunked", "chunk_number", "chunk_start_sec", "chunk_duration_sec"]
    merged_df = merged_df.drop(columns=[c for c in chunk_cols if c in merged_df.columns])

    # Validate the merged dataset before saving
    validate_metadata_output(merged_df)

    # Save merged dataset
    output_file = output_path / "metadata.csv"
    merged_df.to_csv(output_file, index=False)

    logger.info(f"Merged dataset saved to {output_file}")
    logger.info("Dataset merging completed successfully!")


if __name__ == "__main__":
    main()
