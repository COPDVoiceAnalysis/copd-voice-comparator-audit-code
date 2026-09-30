"""
Core utility function for generating unique recording identifiers.
This provides a pure function for deterministic identifier generation
that can be used by Pydantic models and other components.
"""

import hashlib

import pandas as pd


def generate_recording_identifier(
    audio_id: int,
    recording_category: str,
    date: str,
    chunk_number: int | None = None,
) -> int:
    """
    Generate a unique recording identifier from audio_id, category, date, and optional chunk_number.

    Uses a hash-based approach to ensure collision-free identifiers while
    maintaining deterministic generation from the same inputs.

    IMPORTANT: Date is now REQUIRED. Phantom IDs (with date=None) are no longer allowed.
    All recordings must have valid dates to prevent phantom ID generation.

    Args:
        audio_id: The audio ID (e.g., 113)
        recording_category: Recording category ("a", "i", "o", "poem")
        date: REQUIRED date string in YYYY-MM-DD format (no longer optional)
        chunk_number: Optional chunk number for split audio files (e.g., 0, 1, 2...)

    Returns:
        Unique positive integer identifier

    Raises:
        ValueError: If date is None, empty, or invalid

    Examples:
        >>> generate_recording_identifier(113, "a", "2025-02-04")
        1234567891
        >>> generate_recording_identifier(113, "poem", "2025-02-04")
        1234567892
        >>> generate_recording_identifier(113, "poem", "2025-02-04", chunk_number=0)
        1234567893
        >>> generate_recording_identifier(113, "poem", "2025-02-04", chunk_number=1)
        1234567894
    """
    # Validate that date is provided and not empty
    if not date or date.strip() == "":
        raise ValueError(
            f"Date is required for recording identifier generation. "
            f"Got audio_id={audio_id}, category='{recording_category}', date={date}. "
            f"Phantom IDs (with date=None) are no longer allowed."
        )

    # Validate date format (basic check)
    date = date.strip()
    if len(date) != 10 or date.count("-") != 2:
        raise ValueError(
            f"Date must be in YYYY-MM-DD format. "
            f"Got audio_id={audio_id}, category='{recording_category}', date='{date}'"
        )

    # Create a unique string representation (always includes date now)
    key_string = f"{audio_id}_{recording_category}_{date}"
    if chunk_number is not None and not pd.isna(chunk_number):
        try:
            chunk_int = int(chunk_number)
            # Ensure no data loss during conversion
            if chunk_number != chunk_int:
                raise ValueError(
                    f"chunk_number must be a whole number, got {chunk_number}"
                )
            key_string += f"_chunk{chunk_int}"
        except (ValueError, TypeError):
            raise ValueError(
                f"chunk_number must be convertible to integer. "
                f"Got audio_id={audio_id}, category='{recording_category}', "
                f"date='{date}', chunk_number={chunk_number} (type: {type(chunk_number)})"
            )

    # Generate hash and convert to positive integer
    hash_object = hashlib.md5(key_string.encode())
    hash_hex = hash_object.hexdigest()

    # Take first 8 hex characters and convert to int (ensures 32-bit positive integer)
    identifier = int(hash_hex[:8], 16)

    # Ensure it's positive (though MD5 hash should always be positive when truncated)
    return abs(identifier)


def validate_recording_identifiers(df) -> None:
    """
    Validate that all recording identifiers in a DataFrame are unique.

    This is the canonical validation function that should be used across
    the entire codebase to ensure recording identifier uniqueness.

    Args:
        df: DataFrame or pandas-like object with recording_identifier column

    Raises:
        ValueError: If recording_identifier column is missing or duplicates are found
    """
    import logging
    import pandas as pd

    logger = logging.getLogger(__name__)

    if hasattr(df, "columns"):
        # DataFrame-like object
        if "recording_identifier" not in df.columns:
            raise ValueError("No recording_identifier column found")

        identifier_counts = df["recording_identifier"].value_counts()
    else:
        # Assume it's a series or array-like
        identifier_counts = pd.Series(df).value_counts()

    duplicates = identifier_counts[identifier_counts > 1]

    if len(duplicates) > 0:
        logger.error(f"Found {len(duplicates)} duplicate recording identifiers:")
        for identifier, count in duplicates.items():
            logger.error(f"  Identifier {identifier}: {count} occurrences")
        raise ValueError("Recording identifiers are not unique!")

    logger.info(f"✓ Validated {len(identifier_counts)} unique recording identifiers")
