"""
Excel file loading and processing functions.
Handles loading and processing of different Excel sheets.
"""

import pandas as pd
import numpy as np
from typing import Any, Optional
import logging

from src.utils.metadata_extraction.config import (
    EXCEL_SHEETS,
    MAIN_COLUMN_MAPPING,
    SINGLE_RECORDINGS_COLUMN_MAPPING,
    LONGITUDINAL_COLUMN_MAPPING,
    LONGITUDINAL_DATE_COLUMNS,
    LONGITUDINAL_TIME_COLUMNS,
)
from src.utils.metadata_extraction.data_processing import (
    clean_column_names,
    rename_columns,
    standardize_missing_values,
    convert_data_types,
    clean_string_columns,
    standardize_disease_names,
    validate_data_consistency,
    drop_unnecessary_columns,
    parse_dates_to_standard_format,
)

logger = logging.getLogger(__name__)


def load_main_sheet(excel_path: str) -> pd.DataFrame:
    """
    Load the main data sheet from Excel file.

    Args:
        excel_path: Path to Excel file

    Returns:
        Raw DataFrame from main sheet

    Raises:
        IOError: If sheet cannot be loaded or is empty
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=EXCEL_SHEETS["main"])
    except Exception as e:
        raise OSError(f"Failed to load main sheet '{EXCEL_SHEETS['main']}': {e}")

    if len(df) == 0:
        raise OSError(f"Main sheet '{EXCEL_SHEETS['main']}' is empty")

    logger.info(f"Loaded main sheet with {len(df)} rows")
    return df


def load_single_recordings_sheet(excel_path: str) -> pd.DataFrame:
    """
    Load the single recordings details sheet from Excel file.

    Args:
        excel_path: Path to Excel file

    Returns:
        Raw DataFrame from single recordings sheet

    Raises:
        IOError: If sheet cannot be loaded or is empty
    """
    try:
        df = pd.read_excel(excel_path, sheet_name=EXCEL_SHEETS["single_recordings"])
    except Exception as e:
        raise OSError(
            f"Failed to load single recordings sheet '{EXCEL_SHEETS['single_recordings']}': {e}"
        )

    if len(df) == 0:
        raise OSError(
            f"Single recordings sheet '{EXCEL_SHEETS['single_recordings']}' is empty"
        )

    logger.info(f"Loaded single recordings sheet with {len(df)} rows")
    return df


def load_longitudinal_recordings_sheet(excel_path: str) -> pd.DataFrame:
    """
    Load the longitudinal recordings details sheet from Excel file.

    Args:
        excel_path: Path to Excel file

    Returns:
        Raw DataFrame from longitudinal recordings sheet

    Raises:
        IOError: If sheet cannot be loaded or is empty
    """
    try:
        df = pd.read_excel(
            excel_path, sheet_name=EXCEL_SHEETS["longitudinal_recordings"]
        )
    except Exception as e:
        raise OSError(
            f"Failed to load longitudinal recordings sheet '{EXCEL_SHEETS['longitudinal_recordings']}': {e}"
        )

    if len(df) == 0:
        raise OSError(
            f"Longitudinal recordings sheet '{EXCEL_SHEETS['longitudinal_recordings']}' is empty"
        )

    logger.info(f"Loaded longitudinal recordings sheet with {len(df)} rows")
    return df


def process_main_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process main data sheet through standardized pipeline.

    Args:
        raw_df: Raw DataFrame from main sheet

    Returns:
        Processed and validated main DataFrame
    """
    logger.info("Processing main data sheet")

    # Apply processing pipeline
    df = clean_column_names(raw_df)
    df = rename_columns(df, MAIN_COLUMN_MAPPING)
    df = standardize_missing_values(df)
    df = clean_string_columns(df)
    df = standardize_disease_names(df)
    df = convert_data_types(df)
    df = validate_data_consistency(df)
    df = drop_unnecessary_columns(df)

    logger.info(f"Main data processing completed: {len(df)} rows")
    return df


def process_single_recordings(
    raw_df: pd.DataFrame, main_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process single recordings details sheet.

    Args:
        raw_df: Raw DataFrame from single recordings sheet
        main_df: Processed main DataFrame for merging

    Returns:
        Processed single recordings DataFrame with audio_id and date
    """
    logger.info("Processing single recordings data")

    # Apply basic processing
    df = clean_column_names(raw_df)
    df = rename_columns(df, SINGLE_RECORDINGS_COLUMN_MAPPING)
    df = standardize_missing_values(df)
    df = clean_string_columns(df)

    # Remove rows with missing study_id or date
    df = df.dropna(subset=["study_id", "date"]).reset_index(drop=True)

    # Parse dates to standard format
    df = parse_dates_to_standard_format(df, ["date"])

    # Remove rows where date parsing failed
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    # Merge with main_data to get audio_id mapping
    df = pd.merge(
        df[["study_id", "date"]],
        main_df[["audio_id", "study_id"]],
        on="study_id",
        how="inner",
    )

    # Convert audio_id to integer for consistency
    df["audio_id"] = pd.to_numeric(df["audio_id"], errors="coerce").astype(int)

    # Return only audio_id and date columns
    result = df[["audio_id", "date"]].drop_duplicates().reset_index(drop=True)

    logger.info(f"Single recordings processing completed: {len(result)} rows")
    return result


def process_longitudinal_recordings(
    raw_df: pd.DataFrame, main_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Process longitudinal recordings details sheet.

    Args:
        raw_df: Raw DataFrame from longitudinal recordings sheet
        main_df: Processed main DataFrame for merging

    Returns:
        Processed longitudinal recordings DataFrame with audio_id, date, and temporal_order
    """
    logger.info("Processing longitudinal recordings data")

    # Apply basic processing
    df = clean_column_names(raw_df)
    df = rename_columns(df, LONGITUDINAL_COLUMN_MAPPING)
    df = standardize_missing_values(df)
    df = clean_string_columns(df)

    # Merge with main_data to get complete audio_id mapping
    df = pd.merge(
        df,
        main_df[["audio_id", "study_id"]],
        on="study_id",
        how="inner",
    )

    # Remove time columns (not needed)
    for time_col in LONGITUDINAL_TIME_COLUMNS:
        if time_col in df.columns:
            df.drop(columns=[time_col], inplace=True)

    # Rename date columns to standard format
    date_column_mapping = {
        f"datumstimmaufnahme{i}": f"dt_recording_{i}" for i in range(1, 10)
    }
    df.rename(columns=date_column_mapping, inplace=True)

    # Melt the DataFrame to get all recording dates per audio_id
    date_columns = [f"dt_recording_{i}" for i in range(1, 10)]
    existing_date_columns = [col for col in date_columns if col in df.columns]

    df_melted = df.melt(
        id_vars=["audio_id", "study_id"],
        value_vars=existing_date_columns,
        var_name="recording_number",
        value_name="date",
    )

    # Remove rows with missing dates
    df_melted = df_melted.dropna(subset=["date"]).reset_index(drop=True)

    # Parse dates to standard format
    df_melted = parse_dates_to_standard_format(df_melted, ["date"])

    # Remove rows where date parsing failed
    df_melted = df_melted.dropna(subset=["date"]).reset_index(drop=True)

    # Sort by audio_id and date to establish temporal order
    df_melted = df_melted.sort_values(["audio_id", "date"]).reset_index(drop=True)

    # Add temporal order within each audio_id
    df_melted["temporal_order"] = df_melted.groupby("audio_id").cumcount() + 1

    # Clean up columns
    df_melted.drop(
        columns=["recording_number", "study_id"], inplace=True, errors="ignore"
    )

    # Convert audio_id to integer for consistency
    if "audio_id" in df_melted.columns:
        df_melted["audio_id"] = pd.to_numeric(
            df_melted["audio_id"], errors="coerce"
        ).astype(int)

    logger.info(f"Longitudinal recordings processing completed: {len(df_melted)} rows")
    return df_melted


def load_and_process_all_sheets(
    excel_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process all Excel sheets.

    Args:
        excel_path: Path to Excel file

    Returns:
        Tuple of (main_df, single_recordings_df, longitudinal_recordings_df)
    """
    logger.info(f"Loading and processing all sheets from: {excel_path}")

    # Load raw sheets
    main_raw = load_main_sheet(excel_path)
    single_raw = load_single_recordings_sheet(excel_path)
    longitudinal_raw = load_longitudinal_recordings_sheet(excel_path)

    # Process main data first (needed for merging)
    main_df = process_main_data(main_raw)

    # Process other sheets using main data for merging
    single_df = process_single_recordings(single_raw, main_df)
    longitudinal_df = process_longitudinal_recordings(longitudinal_raw, main_df)

    logger.info("All Excel sheets processed successfully")
    return main_df, single_df, longitudinal_df
