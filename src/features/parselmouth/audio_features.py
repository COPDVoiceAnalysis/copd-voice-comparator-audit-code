"""
Audio feature extraction functions for parselmouth.

This module contains functions for extracting basic audio features
like MFCCs, pitch, and harmonicity-to-noise ratio.
"""

import numpy as np
import parselmouth
from spafe.features.mfcc import mfcc
from spafe.utils.preprocessing import SlidingWindow

import logging

logger = logging.getLogger(__name__)


def extract_mfcc_features(
    sound: parselmouth.Sound,
    num_mfcc: int = 30,
    time_step: float = 0.005,
    window_length: float = 0.025,
    pitch_floor: float = 75.0,
) -> dict[str, float]:
    """
    Extract MFCC features including delta and delta-delta coefficients.

    Always extracts ALL MFCC variants (base, delta, delta-delta) regardless of parameters.
    Feature selection is handled during loading, not extraction.

    Args:
        sound: Parselmouth Sound object
        num_mfcc: Number of MFCC coefficients
        time_step: Time step for analysis
        pitch_floor: Pitch floor for MFCC calculation

    Returns:
        Flat dictionary with feature names as keys and values as floats
    """
    window = SlidingWindow(win_len=window_length, win_hop=time_step, win_type="hamming")
    sig = sound.values.T.flatten()
    fs = sound.sampling_frequency
    mfcc_array = mfcc(sig, fs, num_ceps=num_mfcc, window=window, nfilts=40, low_freq=pitch_floor).transpose()  # other params default

    if mfcc_array.shape[0] != num_mfcc:
        raise ValueError(f"Expected {num_mfcc} MFCCs, got {mfcc_array.shape[0]} (shape: {mfcc_array.shape})")

    features = {}

    # Basic MFCC means and standard deviations
    mfcc_means = np.mean(mfcc_array, axis=1)
    mfcc_stds = np.std(mfcc_array, axis=1)

    for i in range(num_mfcc):
        features[f"mfcc_{i}_mean"] = float(mfcc_means[i])
        features[f"mfcc_{i}_std"] = float(mfcc_stds[i])

    # Always extract delta MFCCs (no conditionals)
    d_mfcc = np.diff(mfcc_array, axis=1)
    delta_means = np.mean(d_mfcc, axis=1)
    delta_stds = np.std(d_mfcc, axis=1)

    for i in range(num_mfcc):
        features[f"delta_mfcc_{i}_mean"] = float(delta_means[i])
        features[f"delta_mfcc_{i}_std"] = float(delta_stds[i])

    # Always extract delta-delta MFCCs (no conditionals)
    d2_mfcc = np.diff(d_mfcc, axis=1)
    delta2_means = np.mean(d2_mfcc, axis=1)
    delta2_stds = np.std(d2_mfcc, axis=1)

    for i in range(num_mfcc):
        features[f"delta2_mfcc_{i}_mean"] = float(delta2_means[i])
        features[f"delta2_mfcc_{i}_std"] = float(delta2_stds[i])

    return features


def extract_pitch_features(
    sound: parselmouth.Sound,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 300.0,
    nyquist_frequency: float = 7999,
    time_step: float = 0.01,
) -> dict[str, float]:
    """
    Extract pitch-related features including F0 and semitone statistics.

    Args:
        sound: Parselmouth Sound object
        pitch_floor: Minimum pitch in Hz
        pitch_ceiling: Maximum pitch in Hz
        time_step: Time step for analysis

    Returns:
        Dictionary containing pitch features
    """
    pitch = sound.to_pitch(
        time_step=time_step,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
    )
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]  # ignore unvoiced frames

    features = {}

    # Basic pitch statistics
    features["f0_mean"] = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
    features["f0_std"] = np.std(pitch_values) if len(pitch_values) > 0 else 0.0

    # Get pitch without narrow pitch_ceiling
    pitch = sound.to_pitch(
        time_step=time_step,
        pitch_floor=pitch_floor,
        pitch_ceiling=nyquist_frequency,
    )
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]  # ignore unvoiced

    # Semitone conversion (reference = 1 Hz)
    if len(pitch_values) > 0:
        semitones = 12 * np.log2(pitch_values / 1.0)
        st_mean = np.mean(semitones)
        features["semitone_mean"] = st_mean
        features["semitone_CV"] = (
            np.std(semitones) / st_mean * 100 if st_mean != 0 else 0.0
        )

        # Semitone percentiles
        st_pct = np.percentile(semitones, [20, 50, 80])
        features["semitone_pct20"] = st_pct[0]
        features["semitone_pct50"] = st_pct[1]
        features["semitone_pct80"] = st_pct[2]
    else:
        features["semitone_mean"] = 0.0
        features["semitone_CV"] = 0.0
        features["semitone_pct20"] = 0.0
        features["semitone_pct50"] = 0.0
        features["semitone_pct80"] = 0.0

    # Voiced/unvoiced frame statistics
    perc_voiced_frames = float(pitch.count_voiced_frames()) / float(pitch.n_frames)
    features["perc_unvoiced_frames"] = 1.0 - perc_voiced_frames

    return features


def extract_hnr_features(
    sound: parselmouth.Sound,
    pitch_floor: float = 75.0,
    time_step: float = 0.01,
) -> dict[str, float]:
    """
    Extract harmonicity-to-noise ratio features.

    Args:
        sound: Parselmouth Sound object
        pitch_floor: Minimum pitch for HNR calculation
        time_step: Time step for analysis

    Returns:
        Dictionary containing HNR features
    """
    hnr_object = sound.to_harmonicity_cc(
        time_step=time_step,
        minimum_pitch=pitch_floor,
        silence_threshold=0.1,
        periods_per_window=4.5,
    )
    hnr_values = hnr_object.values[hnr_object.values > 0]

    features = {}
    features["hnr_mean"] = np.mean(hnr_values) if len(hnr_values) > 0 else 0.0
    features["hnr_std"] = np.std(hnr_values) if len(hnr_values) > 0 else 0.0

    return features


def extract_intensity_features(
    sound: parselmouth.Sound,
) -> dict[str, float]:
    """
    Extract intensity features.

    Args:
        sound: Parselmouth Sound object

    Returns:
        Dictionary containing intensity features
    """
    intensity_obj = sound.to_intensity()
    intensity_values = intensity_obj.values[intensity_obj.values > 10]

    features = {}
    features["intensity_mean"] = (
        np.mean(intensity_values) if len(intensity_values) > 0 else 0.0
    )

    return features
