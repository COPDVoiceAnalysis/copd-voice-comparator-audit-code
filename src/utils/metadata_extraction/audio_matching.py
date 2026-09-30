"""
Audio file discovery and matching functions.
Handles finding audio files and matching them with metadata.
"""

import os
import pandas as pd
from typing import Any
import logging

from src.utils.metadata_extraction.config import AUDIO_FILE_PATTERNS, DATE_FORMATS

logger = logging.getLogger(__name__)


def discover_audio_files(directory: str) -> tuple[list[str], list[str]]:
    """
    Discover audio files in directory, separating single and longitudinal recordings.

    Args:
        directory: Directory path to search for audio files

    Returns:
        Tuple of (single_recording_files, longitudinal_recording_files)
    """
    logger.info(f"Discovering audio files in: {directory}")

    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    single_files = []
    longitudinal_files = []

    # Get patterns
    sr_pattern = AUDIO_FILE_PATTERNS["single_recording"]
    lr_pattern = AUDIO_FILE_PATTERNS["longitudinal_recording"]

    # Scan directory for matching files
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Check patterns
        if sr_pattern.match(filename):
            single_files.append(file_path)
        elif lr_pattern.match(filename):
            longitudinal_files.append(file_path)

    logger.info(
        f"Found {len(single_files)} single recording files and {len(longitudinal_files)} longitudinal recording files"
    )

    return single_files, longitudinal_files


def extract_file_metadata(
    file_path: str, is_longitudinal: bool = False
) -> dict[str, Any]:
    """
    Extract metadata from audio file path.

    Args:
        file_path: Path to audio file
        is_longitudinal: Whether this is a longitudinal recording

    Returns:
        Dictionary with extracted metadata
    """
    filename = os.path.basename(file_path)
    parts = filename.split("_")

    if is_longitudinal:
        # Format: audio_id_category_YYYYMMDD.wav
        if len(parts) < 3:
            raise ValueError(f"Invalid longitudinal filename format: {filename}")

        audio_id = int(parts[0])
        recording_category = parts[1]
        date_part = parts[2].split(".")[0]  # Remove .wav extension

        # Convert YYYYMMDD to YYYY-MM-DD
        try:
            date_obj = pd.to_datetime(date_part, format=DATE_FORMATS["filename_format"])
            date = date_obj.strftime(DATE_FORMATS["output_format"])
        except Exception as e:
            raise ValueError(f"Invalid date format in filename {filename}: {e}")

        return {
            "audio_sample_path": file_path,
            "audio_id": audio_id,
            "recording_category": recording_category,
            "date": date,
        }
    else:
        # Format: audio_id_category.wav
        if len(parts) < 2:
            raise ValueError(f"Invalid single recording filename format: {filename}")

        audio_id = int(parts[0])
        recording_category = parts[1].split(".")[0]  # Remove .wav extension

        return {
            "audio_sample_path": file_path,
            "audio_id": audio_id,
            "recording_category": recording_category,
        }


def create_file_dataframes(
    single_files: list[str], longitudinal_files: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create DataFrames from audio file lists with extracted metadata.

    Args:
        single_files: List of single recording file paths
        longitudinal_files: List of longitudinal recording file paths

    Returns:
        Tuple of (single_files_df, longitudinal_files_df)
    """
    logger.info("Creating file DataFrames with extracted metadata")

    # Process single recording files
    single_records = []
    for file_path in single_files:
        try:
            metadata = extract_file_metadata(file_path, is_longitudinal=False)
            single_records.append(metadata)
        except Exception as e:
            logger.warning(f"Skipping invalid single recording file {file_path}: {e}")

    single_df = pd.DataFrame(single_records)

    # Process longitudinal recording files
    longitudinal_records = []
    for file_path in longitudinal_files:
        try:
            metadata = extract_file_metadata(file_path, is_longitudinal=True)
            longitudinal_records.append(metadata)
        except Exception as e:
            logger.warning(
                f"Skipping invalid longitudinal recording file {file_path}: {e}"
            )

    longitudinal_df = pd.DataFrame(longitudinal_records)

    # Convert audio_id to integer for consistency
    if len(single_df) > 0:
        single_df["audio_id"] = pd.to_numeric(
            single_df["audio_id"], errors="coerce"
        ).astype(int)

    if len(longitudinal_df) > 0:
        longitudinal_df["audio_id"] = pd.to_numeric(
            longitudinal_df["audio_id"], errors="coerce"
        ).astype(int)

    logger.info(
        f"Created file DataFrames: {len(single_df)} single, {len(longitudinal_df)} longitudinal"
    )

    return single_df, longitudinal_df


def match_single_recordings(
    main_df: pd.DataFrame,
    single_recordings_df: pd.DataFrame,
    single_files_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match single recording metadata with audio files.

    Args:
        main_df: Main metadata DataFrame
        single_recordings_df: Single recording dates DataFrame
        single_files_df: Single recording files DataFrame

    Returns:
        Merged DataFrame with single recording information
    """
    logger.info("Matching single recordings with audio files")

    # Filter main data to single recordings only
    main_single = main_df.loc[main_df.longitudinal == 0].reset_index(drop=True).copy()

    # Merge main data with single recording dates
    main_with_dates = pd.merge(
        main_single,
        single_recordings_df,
        on="audio_id",
        how="left",
    )

    # Merge with file information
    result = pd.merge(
        main_with_dates,
        single_files_df[["audio_id", "recording_category", "audio_sample_path"]],
        on="audio_id",
        how="left",
    )

    logger.info(f"Single recordings matching completed: {len(result)} rows")
    return result


def match_longitudinal_recordings(
    main_df: pd.DataFrame,
    longitudinal_recordings_df: pd.DataFrame,
    longitudinal_files_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match longitudinal recording metadata with audio files.

    Args:
        main_df: Main metadata DataFrame
        longitudinal_recordings_df: Longitudinal recording dates DataFrame
        longitudinal_files_df: Longitudinal recording files DataFrame

    Returns:
        Merged DataFrame with longitudinal recording information
    """
    logger.info("Matching longitudinal recordings with audio files")

    # Filter main data to longitudinal recordings only
    main_longitudinal = (
        main_df.loc[main_df.longitudinal == 1].reset_index(drop=True).copy()
    )

    # Merge main data with longitudinal recording dates
    main_with_dates = pd.merge(
        main_longitudinal,
        longitudinal_recordings_df,
        on="audio_id",
        how="left",
    )

    # Merge with file information (match on audio_id and date)
    result = pd.merge(
        main_with_dates,
        longitudinal_files_df[
            ["audio_id", "recording_category", "audio_sample_path", "date"]
        ],
        on=["audio_id", "date"],
        how="left",
    )

    logger.info(f"Longitudinal recordings matching completed: {len(result)} rows")
    return result


def validate_audio_id_overlap(
    single_recordings_df: pd.DataFrame, longitudinal_recordings_df: pd.DataFrame
) -> None:
    """
    Validate that there's no overlap between single and longitudinal audio IDs.

    Args:
        single_recordings_df: Single recording dates DataFrame
        longitudinal_recordings_df: Longitudinal recording dates DataFrame

    Raises:
        ValueError: If overlap is found
    """
    if len(single_recordings_df) == 0 or len(longitudinal_recordings_df) == 0:
        return

    single_audio_ids = set(single_recordings_df["audio_id"].unique())
    longitudinal_audio_ids = set(longitudinal_recordings_df["audio_id"].unique())
    overlap = single_audio_ids.intersection(longitudinal_audio_ids)

    if overlap:
        raise ValueError(
            f"Found {len(overlap)} audio_ids that appear in both single and longitudinal recordings: "
            f"{sorted(list(overlap))}. Each audio_id should be either single OR longitudinal, not both."
        )

    logger.info("Audio ID overlap validation passed")


def combine_matched_recordings(
    single_matched: pd.DataFrame, longitudinal_matched: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine single and longitudinal matched recordings into final dataset.

    Args:
        single_matched: Matched single recordings DataFrame
        longitudinal_matched: Matched longitudinal recordings DataFrame

    Returns:
        Combined final DataFrame
    """
    logger.info("Combining matched recordings")

    # Combine both DataFrames
    final_df = pd.concat(
        [single_matched, longitudinal_matched],
        ignore_index=True,
    )

    logger.info(f"Combined dataset: {len(final_df)} rows")
    return final_df


def match_all_recordings(
    main_df: pd.DataFrame,
    single_recordings_df: pd.DataFrame,
    longitudinal_recordings_df: pd.DataFrame,
    audio_files_directory: str,
) -> pd.DataFrame:
    """
    Complete audio file matching pipeline.

    Args:
        main_df: Main metadata DataFrame
        single_recordings_df: Single recording dates DataFrame
        longitudinal_recordings_df: Longitudinal recording dates DataFrame
        audio_files_directory: Directory containing audio files

    Returns:
        Final matched DataFrame with all recordings
    """
    logger.info("Starting complete audio file matching pipeline")

    # Validate no overlap between single and longitudinal audio IDs
    validate_audio_id_overlap(single_recordings_df, longitudinal_recordings_df)

    # Discover audio files
    single_files, longitudinal_files = discover_audio_files(audio_files_directory)

    # Create file DataFrames
    single_files_df, longitudinal_files_df = create_file_dataframes(
        single_files, longitudinal_files
    )

    # Match recordings
    single_matched = match_single_recordings(
        main_df, single_recordings_df, single_files_df
    )
    longitudinal_matched = match_longitudinal_recordings(
        main_df, longitudinal_recordings_df, longitudinal_files_df
    )

    # Combine results
    final_df = combine_matched_recordings(single_matched, longitudinal_matched)

    logger.info("Audio file matching pipeline completed successfully")
    return final_df
