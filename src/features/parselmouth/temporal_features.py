"""
Temporal feature extraction functions for parselmouth.

This module contains functions for extracting temporal analysis features
like voiced/unvoiced durations and syllable rates.
"""

import numpy as np
import parselmouth
import logging

logger = logging.getLogger(__name__)


def extract_temporal_features(
    sound: parselmouth.Sound,
    pitch_values: np.ndarray,
    time_step: float = 0.01,
) -> dict[str, float]:
    """
    Extract temporal features related to voiced/unvoiced segments and timing.

    Args:
        sound: Parselmouth Sound object
        pitch_values: Array of pitch values (with zeros for unvoiced frames)
        time_step: Time step for analysis

    Returns:
        Flat dictionary with feature names as keys and values as floats
    """
    duration = sound.get_total_duration()

    # Create boolean array for voiced frames
    voiced_flags = pitch_values > 0

    features = {}

    if len(voiced_flags) > 0:
        # Calculate durations of voiced and unvoiced segments
        durations_voiced, durations_unvoiced = [], []

        curr = voiced_flags[0]
        count = 1

        for v in voiced_flags[1:]:
            if v == curr:
                count += 1
            else:
                dur = count * time_step
                if curr:
                    durations_voiced.append(dur)
                else:
                    durations_unvoiced.append(dur)
                curr, count = v, 1

        # Handle final run
        dur = count * time_step
        if curr:
            durations_voiced.append(dur)
        else:
            durations_unvoiced.append(dur)

        # Calculate statistics for voiced durations
        features["voiced_dur_mean"] = float(
            np.mean(durations_voiced) if durations_voiced else 0.0
        )
        features["voiced_dur_std"] = float(
            np.std(durations_voiced) if durations_voiced else 0.0
        )

        # Calculate statistics for unvoiced durations
        features["unvoiced_dur_mean"] = float(
            np.mean(durations_unvoiced) if durations_unvoiced else 0.0
        )
        features["unvoiced_dur_std"] = float(
            np.std(durations_unvoiced) if durations_unvoiced else 0.0
        )

        # Pseudo-syllable rate (number of voiced segments per second)
        features["pseudo_syll_rate"] = float(
            len(durations_voiced) / duration if duration > 0 else 0.0
        )
    else:
        features["voiced_dur_mean"] = 0.0
        features["voiced_dur_std"] = 0.0
        features["unvoiced_dur_mean"] = 0.0
        features["unvoiced_dur_std"] = 0.0
        features["pseudo_syll_rate"] = 0.0

    return features
