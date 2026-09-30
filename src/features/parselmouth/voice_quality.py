"""
Voice quality feature extraction functions for parselmouth.

This module contains functions for extracting voice quality measures
like jitter, shimmer, CPP, and DSI.
"""

import numpy as np
import parselmouth
from parselmouth.praat import call
import logging
from src.features.parselmouth.utils import cpp

logger = logging.getLogger(__name__)


def extract_jitter_shimmer_features(
    sound: parselmouth.Sound,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 300.0,
) -> dict[str, float]:
    """
    Extract jitter and shimmer features using Praat's algorithms.

    Always extracts ALL jitter and shimmer variants regardless of parameters.
    Feature selection is handled during loading, not extraction.

    Args:
        sound: Parselmouth Sound object
        pitch_floor: Minimum pitch in Hz
        pitch_ceiling: Maximum pitch in Hz

    Returns:
        Flat dictionary with feature names as keys and values as floats
    """
    # Create a PointProcess using the periodic cross-correlation method
    point_process = call(
        sound,
        "To PointProcess (periodic, cc)...",
        pitch_floor,
        pitch_ceiling,
    )

    features = {}

    # Basic jitter measures
    features["jitter_local"] = call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )
    features["jitter_rap"] = call(
        point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3
    )
    features["jitter_ppq5"] = call(
        point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3
    )
    features["jitter_ddp"] = call(
        point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3
    )

    # Always extract additional jitter measure (no conditionals)
    features["jitter_local_absolute"] = call(
        point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3
    )

    # Basic shimmer measures
    features["shimmer_local"] = call(
        [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    features["shimmer_local_db"] = call(
        [sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    features["shimmer_apq3"] = call(
        [sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    features["shimmer_apq5"] = call(
        [sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    features["shimmer_apq11"] = call(
        [sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    # Always extract additional shimmer measure (no conditionals)
    features["shimmer_dda"] = call(
        [sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    return features


def extract_cpp_features(
    sound: parselmouth.Sound,
    audio_path: str,
) -> dict[str, float]:
    """
    Extract Cepstral Peak Prominence (CPP) features.

    Args:
        sound: Parselmouth Sound object
        audio_path: Path to audio file (for error logging)

    Returns:
        Dictionary containing CPP features
    """
    features = {}

    try:
        cpp_value = cpp(
            sound.values.flatten(), int(sound.sampling_frequency), [60, 333.3]
        )
        features["cpp_value"] = cpp_value
    except Exception as e:
        logger.warning(f"CPP calculation failed for {audio_path}: {e}")
        features["cpp_value"] = 0.0

    return features


def extract_dsi_features(
    sound: parselmouth.Sound,
    f0_mean: float,
    intensity_mean: float,
    jitter_ddp: float,
) -> dict[str, float]:
    """
    Extract Dysphonia Severity Index (DSI) features.

    Args:
        sound: Parselmouth Sound object
        f0_mean: Mean fundamental frequency
        intensity_mean: Mean intensity
        jitter_ddp: DDP jitter value

    Returns:
        Dictionary containing DSI features
    """
    duration = sound.get_total_duration()

    # DSI formula from notebook: 0.13*duration + 0.0053*meanF0 - 0.26*intensity - 1.18*ddpJitter*100 + 12.4
    dsi_value = (
        0.13 * duration
        + 0.0053 * f0_mean
        - 0.26 * intensity_mean
        - 1.18 * jitter_ddp * 100
        + 12.4
    )

    features = {}
    features["dsi_value"] = dsi_value

    return features
