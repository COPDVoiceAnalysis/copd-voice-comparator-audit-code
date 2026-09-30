"""
OpenSMILE eGeMAPS feature extraction functions.

This module provides extraction of the complete eGeMAPS v02 feature set
using the OpenSMILE library, which will replace MFCC extraction and
provide additional standardized acoustic features.
"""

import opensmile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def extract_egemaps_features(audio_path: str) -> dict[str, float]:
    """
    Extract complete eGeMAPS v02 feature set using OpenSMILE.

    The eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set) v02
    contains 88 features including:
    - Frequency features (F0, formants)
    - Energy features (loudness, shimmer)
    - Spectral features (MFCC 1-13, spectral centroid, flux, rolloff)
    - Voice quality features (jitter, HNR, harmonicity)
    - Temporal features (timing and rhythm measures)

    Args:
        audio_path: Path to the audio file

    Returns:
        Flat dictionary with feature names as keys and values as floats
    """
    logger.debug(f"Extracting eGeMAPS features from: {Path(audio_path).name}")

    try:
        # Initialize OpenSMILE with eGeMAPS v02 feature set
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        # Extract features - returns a pandas DataFrame
        features_df = smile.process_file(audio_path)

        # Convert to flat dictionary
        features_dict = {}

        # The DataFrame has a MultiIndex with (file, start, end) as index
        # and feature names as columns. We want just the feature values.
        if len(features_df) > 0:
            # Get the first (and typically only) row of features
            feature_row = features_df.iloc[0]

            # Convert to dictionary with clean feature names
            for feature_name, value in feature_row.items():
                # Clean up feature names - remove any problematic characters
                clean_name = str(feature_name).replace(" ", "_").replace("-", "_")
                clean_name = clean_name.replace("(", "").replace(")", "")
                clean_name = clean_name.replace("[", "").replace("]", "")
                clean_name = clean_name.replace("%", "pct")

                # Ensure the value is a float
                features_dict[f"egemaps_{clean_name}"] = float(value)
        else:
            logger.warning(f"No features extracted from {audio_path}")
            # Return empty dict - will be handled by calling function
            return {}

        logger.debug(f"Extracted {len(features_dict)} eGeMAPS features")
        return features_dict

    except Exception as e:
        logger.error(f"Error extracting eGeMAPS features from {audio_path}: {str(e)}")
        raise


def get_egemaps_feature_names() -> list[str]:
    """
    Get the list of all eGeMAPS feature names that will be extracted.

    This is useful for feature selection and validation.

    Returns:
        List of feature names that will be extracted
    """
    try:
        # Initialize OpenSMILE to get feature names
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        # Get feature names from the feature set
        feature_names = smile.feature_names

        # Clean up names and add prefix
        clean_names = []
        for name in feature_names:
            clean_name = str(name).replace(" ", "_").replace("-", "_")
            clean_name = clean_name.replace("(", "").replace(")", "")
            clean_name = clean_name.replace("[", "").replace("]", "")
            clean_name = clean_name.replace("%", "pct")
            clean_names.append(f"egemaps_{clean_name}")

        return clean_names

    except Exception as e:
        logger.error(f"Error getting eGeMAPS feature names: {str(e)}")
        return []


def validate_egemaps_installation() -> bool:
    """
    Validate that OpenSMILE is properly installed and can extract eGeMAPS features.

    Returns:
        True if OpenSMILE is working correctly, False otherwise
    """
    try:
        # Try to initialize OpenSMILE
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        # Check that we can get feature names
        feature_names = smile.feature_names

        if len(feature_names) > 0:
            logger.info(
                f"OpenSMILE eGeMAPS validation successful: {len(feature_names)} features available"
            )
            return True
        else:
            logger.error("OpenSMILE eGeMAPS validation failed: no features available")
            return False

    except Exception as e:
        logger.error(f"OpenSMILE eGeMAPS validation failed: {str(e)}")
        return False
