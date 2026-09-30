"""
Core data processing functions for metadata extraction.
Pure functions for data cleaning, validation, and transformation.
"""

import pandas as pd
import numpy as np
import logging

from src.utils.metadata_extraction.config import (
    MISSING_VALUE_REPRESENTATIONS,
    DISEASE_MAPPING,
    DISEASE_COLUMNS,
    COLUMNS_TO_DROP,
    NUMERIC_COLUMNS,
    DATE_FORMATS,
    VALIDATION_RULES,
)

logger = logging.getLogger(__name__)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize column names.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with cleaned column names
    """
    df = df.copy()

    # Remove whitespace and convert to lowercase
    df.columns = df.columns.str.replace(r"\s+", "", regex=True).str.lower()

    return df


def rename_columns(df: pd.DataFrame, column_mapping: dict[str, str]) -> pd.DataFrame:
    """
    Rename columns according to mapping.

    Args:
        df: Input DataFrame
        column_mapping: Dictionary mapping old names to new names

    Returns:
        DataFrame with renamed columns
    """
    df = df.copy()
    df.rename(columns=column_mapping, inplace=True)
    return df


def standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize various missing value representations to NaN.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with standardized missing values
    """
    df = df.copy()

    # Replace missing value representations with NaN
    for col in df.select_dtypes(include=["object"]).columns:
        # Use infer_objects to avoid FutureWarning about downcasting
        df[col] = df[col].replace(MISSING_VALUE_REPRESENTATIONS, np.nan).infer_objects()

    return df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with converted data types
    """
    df = df.copy()

    # Convert numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert audio_id to integer (special handling)
    if "audio_id" in df.columns:
        df["audio_id"] = pd.to_numeric(df["audio_id"], errors="coerce")
        # Remove rows with invalid audio_ids
        df = df.dropna(subset=["audio_id"])
        # Convert to integer
        df["audio_id"] = df["audio_id"].astype(int)

    return df


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean string columns by stripping whitespace and converting to lowercase.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with cleaned string columns
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str, errors="ignore").str.strip().str.lower()

    return df


def standardize_disease_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize disease names in disease columns.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with standardized disease names
    """
    df = df.copy()

    for col in DISEASE_COLUMNS:
        if col in df.columns:
            # Clean whitespace first
            df[col] = df[col].str.strip().str.replace(r"\s+", " ", regex=True)
            df[col] = df[col].str.lower()
            # Apply disease mapping
            df[col] = df[col].replace(DISEASE_MAPPING)

    return df


def filter_ready_to_use(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only ready-to-use records.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame filtered to ready_to_use == 1
    """
    df = df.copy()

    if "ready_to_use" in df.columns:
        df = df.loc[df.ready_to_use == 1].reset_index(drop=True)

    return df


def validate_control_group_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and fix control group consistency issues.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with control group consistency validated
    """
    df = df.copy()

    if "control" not in df.columns or "lung_disease_main" not in df.columns:
        return df

    # Check for inconsistencies in control group
    control_with_disease = (df.control == 1) & (df.lung_disease_main != "control")

    if control_with_disease.any():
        logger.warning(
            f"Found {control_with_disease.sum()} patients in control group with lung disease. "
            f"Setting ready_to_use to 0."
        )
        if "ready_to_use" in df.columns:
            df.loc[control_with_disease, "ready_to_use"] = 0

    # Check for patients with no disease not in control group
    no_disease_not_control = (df.control == 0) & (df.lung_disease_main == "control")

    if no_disease_not_control.any():
        logger.warning(
            f"Found {no_disease_not_control.sum()} patients with no disease not in control group. "
            f"Setting ready_to_use to 0."
        )
        if "ready_to_use" in df.columns:
            df.loc[no_disease_not_control, "ready_to_use"] = 0

    return df


def remove_question_mark_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing question marks in any field.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with question mark rows removed
    """
    df = df.copy()

    if "ready_to_use" in df.columns:
        # Only check specific critical columns
        critical_cols = ["audio_id", "lung_disease_main"]
        for col in critical_cols:
            if col in df.columns:
                mask = df[col].astype(str).str.contains("?", na=False, regex=False)
                df.loc[mask, "ready_to_use"] = 0
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not needed.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with unnecessary columns dropped
    """
    df = df.copy()

    # Drop specified columns
    df.drop(columns=COLUMNS_TO_DROP, inplace=True, errors="ignore")

    # Drop ready_to_use after filtering
    df.drop(columns=["ready_to_use"], inplace=True, errors="ignore")

    return df


def validate_audio_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate audio IDs for consistency and correctness.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with validated audio IDs

    Raises:
        ValueError: If audio ID validation fails
    """
    df = df.copy()

    if "audio_id" not in df.columns:
        return df

    # Check for multiple audio_ids in single field (comma-separated)
    multi_audio_mask = df.audio_id.astype(str).str.contains(",")
    if multi_audio_mask.any():
        raise ValueError(
            f"Found {multi_audio_mask.sum()} cases with multiple audio_ids in single field"
        )

    # Ensure audio_ids are positive
    if (df.audio_id <= 0).any():
        raise ValueError("Found non-positive audio IDs")

    return df


def parse_dates_to_standard_format(
    df: pd.DataFrame, date_columns: list[str]
) -> pd.DataFrame:
    """
    Parse dates to standard YYYY-MM-DD format.

    Args:
        df: Input DataFrame
        date_columns: List of column names containing dates

    Returns:
        DataFrame with standardized date formats
    """
    df = df.copy()

    for col in date_columns:
        if col not in df.columns:
            continue

        try:
            # Check if dates are already datetime objects
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col]).dt.strftime(
                    DATE_FORMATS["output_format"]
                )
            else:
                # Handle string format - extract date part and parse
                df[col] = df[col].astype(str).str.split().str[0]  # Remove time part
                df[col] = pd.to_datetime(
                    df[col], format=DATE_FORMATS["input_format"], errors="coerce"
                ).dt.strftime(DATE_FORMATS["output_format"])
        except Exception as e:
            logger.error(f"Failed to parse dates in column '{col}': {e}")
            raise ValueError(f"Date parsing failed for column '{col}': {e}")

    return df


def validate_data_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform comprehensive data consistency validation.

    Args:
        df: Input DataFrame

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If validation fails
    """
    df = df.copy()

    # Validate audio IDs
    df = validate_audio_ids(df)

    # Validate control group consistency
    df = validate_control_group_consistency(df)

    # Remove question mark rows
    df = remove_question_mark_rows(df)

    # Filter to ready-to-use records
    df = filter_ready_to_use(df)

    logger.info(f"Data validation completed. Final dataset: {len(df)} rows")

    return df
