"""
Spectral feature extraction functions for parselmouth.

This module contains functions for extracting spectral analysis features
using STFT and frequency domain analysis.
"""

import numpy as np
import librosa
import parselmouth
import logging

logger = logging.getLogger(__name__)


def extract_spectral_features(
    audio_path: str,
    time_step: float = 0.01,
    n_fft: int = 512,
) -> dict[str, float]:
    """
    Extract spectral features via Short-Time Fourier Transform (STFT).

    Args:
        audio_path: Path to the audio file
        time_step: Time step for analysis
        n_fft: FFT size for spectral analysis

    Returns:
        Flat dictionary with feature names as keys and values as floats
    """
    # Load raw signal for spectral features
    y, sr = librosa.load(audio_path, sr=None)

    # Compute STFT
    hop_length = int(time_step * sr)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    features = {}

    # Alpha Ratio: 50–1000 Hz vs. 1000–5000 Hz
    low_mask = (freqs >= 50) & (freqs <= 1000)
    high_mask = (freqs >= 1000) & (freqs <= 5000)
    E_low = S[low_mask, :].sum()
    E_high = S[high_mask, :].sum()
    features["alpha_ratio"] = float(
        10 * np.log10(E_low / E_high) if E_high > 0 else 0.0
    )

    # Hammarberg Index: max energy 0–2kHz vs. 2–5kHz
    m1 = freqs <= 2000
    m2 = (freqs > 2000) & (freqs <= 5000)
    spec_mean = S.mean(axis=1)
    features["hammarberg_index"] = float(
        (spec_mean[m1].max() / spec_mean[m2].max()) if spec_mean[m2].max() > 0 else 0.0
    )

    # Spectral slopes
    s1_mask = freqs <= 500
    s2_mask = (freqs > 500) & (freqs <= 1500)
    features["slope_0_500Hz"] = float(
        np.polyfit(freqs[s1_mask], spec_mean[s1_mask], 1)[0]
    )
    features["slope_500_1500Hz"] = float(
        np.polyfit(freqs[s2_mask], spec_mean[s2_mask], 1)[0]
    )

    # Spectral flux (mean)
    features["spectral_flux"] = float(
        np.mean(np.sqrt(np.sum(np.diff(np.abs(S), axis=1) ** 2, axis=0)))
    )

    return features
