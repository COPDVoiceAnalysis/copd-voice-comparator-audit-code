"""
Utility functions for the voice biomarker pipeline models.
Contains helper functions used across different model modules.
"""

from pathlib import Path
import numpy as np


def _save_structured_array(
    file_path: Path, recording_identifiers: np.ndarray, data: np.ndarray, data_name: str
) -> None:
    """
    Save data as an optimized structured numpy array with recording identifiers.

    Uses the same optimized format as Parselmouth features for consistency and performance.

    Args:
        file_path: Output file path
        recording_identifiers: Array of recording identifiers
        data: Data array (means or stds)
        data_name: Name for the data field ("mean" or "std")
    """
    if len(recording_identifiers) != len(data):
        raise ValueError(
            f"Mismatch between recording_identifiers ({len(recording_identifiers)}) and data ({len(data)}) lengths"
        )

    n_records = len(recording_identifiers)

    # Create optimized structured array dtype (same format as Parselmouth)
    dtype = np.dtype(
        [
            ("recording_identifiers", "i8", (n_records,)),  # All IDs as 1D array
            (
                f"{data_name}_matrix",
                "f4",
                (n_records, data.shape[1]),
            ),  # All data as 2D array
        ]
    )

    # Create single structured record containing all data
    structured_array = np.empty(1, dtype=dtype)
    structured_array["recording_identifiers"][0] = recording_identifiers.astype(
        np.int64
    )
    structured_array[f"{data_name}_matrix"][0] = data.astype(np.float32)

    # Save to file
    np.save(file_path, structured_array)
