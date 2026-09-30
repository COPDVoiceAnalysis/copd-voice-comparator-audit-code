"""
Formant feature extraction functions for parselmouth.

This module contains functions for extracting formant frequencies
and bandwidths using Burg's method.
"""

import numpy as np
import parselmouth
import logging

logger = logging.getLogger(__name__)


def extract_formant_features(
    sound: parselmouth.Sound,
    time_step: float = 0.01,
) -> dict[str, float]:
    """
    Extract formant frequencies and bandwidths using Burg's method.

    Args:
        sound: Parselmouth Sound object
        time_step: Time step for analysis

    Returns:
        Flat dictionary with feature names as keys and values as floats
    """
    duration = sound.get_total_duration()

    # Extract formants using Burg's method
    formant = sound.to_formant_burg(
        time_step=time_step,
        max_number_of_formants=5.0,
        maximum_formant=5500.0,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )

    # Sample formant values at regular intervals
    times = np.arange(0, duration, time_step)
    f1, f2, f3 = [], [], []
    b1, b2, b3 = [], [], []

    for t in times:
        # Formant frequencies
        v1 = formant.get_value_at_time(1, t)
        v2 = formant.get_value_at_time(2, t)
        v3 = formant.get_value_at_time(3, t)

        if v1 > 0:
            f1.append(v1)
        if v2 > 0:
            f2.append(v2)
        if v3 > 0:
            f3.append(v3)

        # Formant bandwidths
        bw1 = formant.get_bandwidth_at_time(1, t)
        bw2 = formant.get_bandwidth_at_time(2, t)
        bw3 = formant.get_bandwidth_at_time(3, t)

        if bw1 > 0:
            b1.append(bw1)
        if bw2 > 0:
            b2.append(bw2)
        if bw3 > 0:
            b3.append(bw3)

    # Calculate mean values
    features = {}
    features["f1_mean"] = float(np.mean(f1) if f1 else 0.0)
    features["f2_mean"] = float(np.mean(f2) if f2 else 0.0)
    features["f3_mean"] = float(np.mean(f3) if f3 else 0.0)
    features["bw1_mean"] = float(np.mean(b1) if b1 else 0.0)
    features["bw2_mean"] = float(np.mean(b2) if b2 else 0.0)
    features["bw3_mean"] = float(np.mean(b3) if b3 else 0.0)

    return features
