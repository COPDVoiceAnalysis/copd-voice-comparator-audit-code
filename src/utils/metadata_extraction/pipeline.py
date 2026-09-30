"""
Main pipeline orchestration for metadata extraction.
Coordinates all components to create the final metadata CSV.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from src.utils.metadata_extraction.excel_loaders import load_and_process_all_sheets
from src.utils.metadata_extraction.audio_matching import match_all_recordings
from src.utils.recording_identifier import (
    validate_recording_identifiers,
    generate_recording_identifier,
)

logger = logging.getLogger(__name__)


def localize_excel_file(
    vocal_recordings_dir: str | None = None, excel_file_path: str | None = None
) -> str:
    """
    Locate Excel file either by explicit path or auto-discovery.

    Args:
        vocal_recordings_dir: Optional directory to search for Excel files
        excel_file_path: Optional specific path to Excel file

    Returns:
        Path to the Excel file

    Raises:
        FileNotFoundError: If Excel file not found
        ValueError: If multiple Excel files found in directory
    """
    if vocal_recordings_dir is None and excel_file_path is None:
        raise ValueError(
            "Either vocal_recordings_dir or excel_file_path must be provided"
        )

    # Use explicit path if provided
    if excel_file_path:
        if not os.path.exists(excel_file_path):
            raise FileNotFoundError(f"Excel file not found: {excel_file_path}")
        logger.info(f"Using explicit Excel file: {excel_file_path}")
        return excel_file_path

    # Auto-discover patient documentation Excel file in directory
    if vocal_recordings_dir:
        excel_files = [
            os.path.join(vocal_recordings_dir, f)
            for f in os.listdir(vocal_recordings_dir)
            if f.endswith(".xlsx") and f.startswith("Patienten_Doku")
        ]

        if len(excel_files) == 0:
            raise FileNotFoundError(
                f"No Patienten_Doku*.xlsx file found in directory: {vocal_recordings_dir}"
            )
        elif len(excel_files) > 1:
            raise ValueError(
                f"Multiple Patienten_Doku*.xlsx files found in directory: {vocal_recordings_dir}"
            )

        logger.info(f"Auto-discovered Excel file: {excel_files[0]}")
        return excel_files[0]

    raise FileNotFoundError("No Excel file could be located")


def generate_recording_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate recording identifiers for the final dataset.

    Args:
        df: DataFrame with audio_id, recording_category, and optional date

    Returns:
        DataFrame with recording_identifier column added
    """
    logger.info("Generating recording identifiers")

    df = df.copy()

    # Add language column for Charite data (German)
    if "language" not in df.columns:
        df["language"] = "de"
        logger.info("Added language='de' column for Charite data")

    # Generate recording identifiers
    df["recording_identifier"] = df.apply(
        lambda row: generate_recording_identifier(
            row["audio_id"],
            row["recording_category"],
            row["date"],  # date should already be validated as non-null by this point
        ),
        axis=1,
    )

    return df


def validate_final_metadata(df: pd.DataFrame, output_path: str) -> None:
    """
    Validate final metadata using existing Pydantic infrastructure.

    Args:
        df: Final metadata DataFrame
        output_path: Path where CSV was saved

    Raises:
        RuntimeError: If validation fails
    """
    logger.info("Validating final metadata with Pydantic models")

    try:
        from pathlib import Path
        from src.models.metadata import Metadata

        metadata_handler = Metadata(Path(output_path))
        logger.info(
            f"✓ Metadata validation successful: {len(metadata_handler)} rows validated"
        )

    except Exception as e:
        logger.error(f"✗ Metadata validation failed: {e}")
        logger.error(f"CSV file preserved at {output_path} for debugging")
        raise RuntimeError(
            f"Metadata validation failed. Check {output_path} for issues."
        )


def save_metadata_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save metadata DataFrame to CSV file.

    Args:
        df: Final metadata DataFrame
        output_path: Path to save CSV file
    """
    logger.info(f"Saving metadata CSV to: {output_path}")

    # Ensure output directory exists (only if there's a directory part)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save CSV
    df.to_csv(output_path, index=False)

    logger.info(f"Metadata CSV saved successfully: {len(df)} rows")


def extract_metadata_pipeline(
    vocal_recordings_dir: str,
    excel_file_path: str | None = None,
    output_file: str | None = None,
) -> None:
    """
    Complete metadata extraction pipeline.

    This is the main orchestration function that coordinates all components
    to extract metadata from Excel files and audio files, then creates
    a consolidated metadata CSV file.

    Args:
        vocal_recordings_dir: Directory containing audio files
        excel_file_path: Optional path to Excel file (auto-discovered if not provided)
        output_file: Optional output CSV path (defaults to metadata.csv in vocal_recordings_dir)

    Raises:
        Various exceptions if any step fails
    """
    logger.info("Starting metadata extraction pipeline")
    logger.info(f"Audio directory: {vocal_recordings_dir}")
    logger.info(f"Excel file: {excel_file_path or 'auto-discover'}")
    logger.info(f"Output file: {output_file or 'default'}")

    # Step 1: Locate Excel file
    excel_path = localize_excel_file(
        vocal_recordings_dir=vocal_recordings_dir, excel_file_path=excel_file_path
    )

    # Step 2: Load and process all Excel sheets
    main_df, single_recordings_df, longitudinal_recordings_df = (
        load_and_process_all_sheets(excel_path)
    )

    logger.info(
        f"Loaded Excel data: {len(main_df)} main records, "
        f"{len(single_recordings_df)} single recording dates, "
        f"{len(longitudinal_recordings_df)} longitudinal recording dates"
    )

    # Step 3: Match with audio files
    final_df = match_all_recordings(
        main_df=main_df,
        single_recordings_df=single_recordings_df,
        longitudinal_recordings_df=longitudinal_recordings_df,
        audio_files_directory=vocal_recordings_dir,
    )

    logger.info(f"Audio matching completed: {len(final_df)} final records")

    # Step 4: Generate recording identifiers
    final_df = generate_recording_identifiers(final_df)

    # Step 5: Validate recording identifiers
    validate_recording_identifiers(final_df)

    # Step 6: Determine output path
    if output_file is None:
        output_file = os.path.join(vocal_recordings_dir, "metadata.csv")

    # Step 7: Save CSV
    save_metadata_csv(final_df, output_file)

    # Step 8: Validate final metadata
    validate_final_metadata(final_df, output_file)

    logger.info("✓ Metadata extraction pipeline completed successfully")
    logger.info(f"Final output: {output_file}")
    logger.info(f"Total records: {len(final_df)}")


def extract_metadata_pipeline_safe(
    vocal_recordings_dir: str,
    excel_file_path: str | None = None,
    output_file: str | None = None,
) -> bool:
    """
    Safe wrapper around metadata extraction pipeline with comprehensive error handling.

    Args:
        vocal_recordings_dir: Directory containing audio files
        excel_file_path: Optional path to Excel file
        output_file: Optional output CSV path

    Returns:
        True if successful, False if failed
    """
    try:
        extract_metadata_pipeline(
            vocal_recordings_dir=vocal_recordings_dir,
            excel_file_path=excel_file_path,
            output_file=output_file,
        )
        return True

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return False

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Full traceback:")
        return False
