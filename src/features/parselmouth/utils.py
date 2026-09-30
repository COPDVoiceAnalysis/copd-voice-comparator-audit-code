"""
Utility functions for parselmouth feature extraction.

This module contains utility functions for array creation, I/O operations,
and data processing. Feature names are now generated directly by extraction
functions, eliminating the need for separate name generation.
"""

import os
import json
import math
import numpy as np
import scipy.signal
from numpy import matlib
from typing import Any
from datetime import datetime

import logging

logger = logging.getLogger(__name__)


def remove_zero_variance_features(
    feature_matrix: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Remove features with zero variance from the feature matrix.

    Args:
        feature_matrix: 2D array of shape (n_samples, n_features)
        feature_names: List of feature names corresponding to columns

    Returns:
        Tuple of (filtered_matrix, remaining_feature_names, removed_feature_names)
    """
    if feature_matrix.shape[0] < 2:
        logger.warning(
            "Cannot compute variance with less than 2 samples. Skipping zero-variance filtering."
        )
        return feature_matrix, feature_names, []

    feature_stds = np.std(feature_matrix, axis=0)
    zero_variance_mask = feature_stds == 0
    zero_variance_indices = np.where(zero_variance_mask)[0]

    if len(zero_variance_indices) == 0:
        logger.info("No zero-variance features found.")
        return feature_matrix, feature_names, []

    removed_feature_names = [feature_names[i] for i in zero_variance_indices]

    filtered_matrix = feature_matrix[:, ~zero_variance_mask]
    remaining_feature_names = [
        name for i, name in enumerate(feature_names) if not zero_variance_mask[i]
    ]

    logger.info(
        f"Removed {len(removed_feature_names)} zero-variance features: {removed_feature_names}"
    )
    return filtered_matrix, remaining_feature_names, removed_feature_names


def features_dict_to_array(
    features_dict: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """
    Convert features dictionary to numpy array - no conditionals needed.

    Args:
        features_dict: Flat dictionary with feature names as keys and values as floats

    Returns:
        Tuple of (feature_array, feature_names_list)
    """
    feature_names = list(features_dict.keys())
    feature_values = [features_dict[name] for name in feature_names]
    return np.array(feature_values, dtype=np.float32), feature_names


def cpp(x, fs, pitch_range):
    """
    Computes cepstral peak prominence for a given signal

    Parameters
    -----------
    x: ndarray
        The audio signal
    fs: integer
        The sampling frequency
    pitch_range: list of 2 elements
        The pitch range where a peak is searched for

    Returns
    -----------
    float
        The cepstral peak prominence of the audio signal
    """
    # Quefrency
    frameLen = len(x)
    NFFT = 2 ** (math.ceil(np.log(frameLen) / np.log(2)))
    quef = np.linspace(0, frameLen / 1000, NFFT)
    # Allowed quefrency range
    quef_lim = [int(np.round(fs / pitch_range[1])), int(np.round(fs / pitch_range[0]))]
    quef_seq = range(quef_lim[0] - 1, quef_lim[1])

    # High-pass filtering
    HPfilt_b = [1 - 0.97]
    x = scipy.signal.lfilter(HPfilt_b, 1, x)

    # FrameMat
    frameMat = np.zeros(NFFT)
    frameMat[0:frameLen] = x

    # Hanning
    def hanning(N):
        x = np.array([i / (N + 1) for i in range(1, int(np.ceil(N / 2)) + 1)])
        w = 0.5 - 0.5 * np.cos(2 * np.pi * x)
        w_rev = w[::-1]
        return np.concatenate((w, w_rev[int(np.ceil(N % 2)) :]))

    win = hanning(frameLen)

    winmat = matlib.repmat(win, 1, 1)
    frameMat = frameMat[0:frameLen] * winmat
    frameMat = frameMat[0]

    # Cepstrum
    SpecMat = np.abs(np.fft.fft(frameMat))
    SpecdB = 20 * np.log10(SpecMat)
    ceps = 20 * np.log10(np.abs(np.fft.fft(SpecdB)))

    # Finding the peak
    ceps_lim = ceps[quef_seq]
    ceps_max = np.max(ceps_lim)
    max_index = np.argmax(ceps_lim)

    # Normalisation
    p = np.polyfit(quef_seq, ceps_lim, 1)
    ceps_norm = np.polyval(p, quef_seq[max_index])

    cpp1 = ceps_max - ceps_norm

    return cpp1


def create_feature_array(
    records: list[dict[str, Any]],
    extraction_params: dict[str, Any],
    remove_zero_variance: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Create optimized structured array for memory-mapped access with contiguous 2D feature storage.
    Includes zero-variance filtering and metadata generation.

    Args:
        records: List of records with recording_identifier and features
        extraction_params: Parameters used for feature extraction
        remove_zero_variance: Whether to remove zero-variance features

    Returns:
        Tuple of (structured_array, metadata_dict)
    """
    if not records:
        raise ValueError("No records to process")

    # Get feature names from the first record (since features_dict keys are the names)
    first_features = records[0]["features"]
    assert isinstance(first_features, dict)
    # features is a dict with names as keys
    feature_names = list(first_features.keys())
    n_features_original = len(feature_names)

    # Convert all records to arrays using the same feature order
    feature_arrays = []
    for record in records:
        feature_array = np.array(
            [record["features"][name] for name in feature_names], dtype=np.float32
        )
        feature_arrays.append(feature_array)
    feature_matrix = np.vstack(feature_arrays)

    n_records = len(records)
    logger.info(
        f"Creating optimized feature array with {n_features_original} features for {n_records} records"
    )

    # Create feature matrix
    recording_ids = np.array(
        [record["recording_identifier"] for record in records], dtype=np.int64
    )

    # Apply zero-variance filtering if requested
    removed_features = []
    if remove_zero_variance:
        feature_matrix, feature_names, removed_features = remove_zero_variance_features(
            feature_matrix, feature_names
        )

    n_features_final = feature_matrix.shape[1]

    # Create enhanced structured array with metadata
    dtype = [
        ("recording_identifiers", "i8", (n_records,)),  # All IDs as 1D array
        (
            "feature_matrix",
            "f4",
            (n_records, n_features_final),
        ),  # All features as 2D array
        ("feature_names", "U50", (n_features_final,)),  # Feature names
    ]

    # Create single structured record containing all data
    structured_array = np.empty(1, dtype=dtype)
    structured_array["recording_identifiers"][0] = recording_ids
    structured_array["feature_matrix"][0] = feature_matrix
    # Ensure feature_names matches the final feature count
    final_feature_names = feature_names[:n_features_final]
    structured_array["feature_names"][0] = np.array(final_feature_names, dtype="U50")

    # Create metadata dictionary
    metadata = {
        "extraction_params": extraction_params,
        "feature_names": feature_names,
        "n_features_original": n_features_original,
        "n_features_final": n_features_final,
        "n_records": n_records,
        "zero_variance_filtering_applied": remove_zero_variance,
        "removed_features": removed_features,
        "extraction_date": datetime.now().isoformat(),
        "extractor_version": "2.0",  # Updated version for new format
    }

    logger.info(
        f"Created optimized structured array with shape {structured_array.shape}"
    )
    logger.info(
        f"Recording IDs shape: {recording_ids.shape}, Feature matrix shape: {feature_matrix.shape}"
    )
    logger.info(
        f"Final feature count: {n_features_final} (removed {len(removed_features)} zero-variance features)"
    )

    return structured_array, metadata


def load_features_with_metadata(
    npy_path: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """
    Load features with names and metadata from the enhanced structured array format.

    Args:
        npy_path: Path to the .npy file containing features

    Returns:
        Tuple of (recording_ids, feature_matrix, feature_names, metadata)
    """
    data = np.load(npy_path)

    recording_ids = data["recording_identifiers"][0]
    feature_matrix = data["feature_matrix"][0]
    feature_names = data["feature_names"][0].tolist()

    # Load metadata if available
    metadata_path = npy_path.replace(".npy", "_metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        logger.warning(f"Metadata file not found: {metadata_path}")
        # Create minimal metadata from available info
        metadata = {
            "feature_names": feature_names,
            "n_features_final": len(feature_names),
            "n_records": len(recording_ids),
            "zero_variance_filtering_applied": "unknown",
            "removed_features": [],
        }

    logger.info(
        f"Loaded {len(recording_ids)} recordings with {len(feature_names)} features"
    )
    if metadata.get("removed_features"):
        logger.info(
            f"Zero-variance features were removed: {metadata['removed_features']}"
        )

    return recording_ids, feature_matrix, feature_names, metadata
