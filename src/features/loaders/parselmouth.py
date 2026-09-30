"""
Parselmouth feature loader.

Handles Parselmouth feature loading and processing while maintaining 3D output structure.
"""

import numpy as np
from pathlib import Path
import logging
from src.models.data_models import ParselmouthFeatures, ParselmouthConfig
from src.features.loaders.base import BaseFeatureLoader

logger = logging.getLogger(__name__)


class ParselmouthFeatureLoader(BaseFeatureLoader):
    """
    Handles only Parselmouth feature loading and processing.

    Maintains 3D output structure: (n_samples, n_recordings_per_sample, n_parsel_features)
    """

    def __init__(
        self,
        parsel_feature_file: Path,
        recordings_parsel: tuple[str, ...],
        parselmouth_config: ParselmouthConfig | None = None,
    ):
        """
        Initialize Parselmouth feature extractor.

        Args:
            parsel_feature_file: Path to parselmouth features file
            recordings_parsel: Tuple of recording categories to process
            parselmouth_config: ParselmouthConfig for feature filtering
        """
        self.parsel_feature_file = parsel_feature_file
        self.recordings_parsel = recordings_parsel
        self.parselmouth_config = parselmouth_config
        self._parsel_features: ParselmouthFeatures | None = None

        logger.debug(
            f"ParselmouthFeatureExtractor initialized for categories: {recordings_parsel}"
        )

    def fit(self, X, y=None):
        """Load Parselmouth features from file."""
        if self.recordings_parsel and self.parsel_feature_file:
            logger.debug(
                f"Loading Parselmouth features from {self.parsel_feature_file}"
            )
            self._parsel_features = ParselmouthFeatures.from_file(
                self.parsel_feature_file
            )

            # Apply feature filtering if config is provided
            if self.parselmouth_config is not None:
                logger.debug("Applying feature filtering based on ParselmouthConfig")
                self._parsel_features = self._parsel_features.filter_by_config(
                    self.parselmouth_config
                )

            logger.debug("Parselmouth features loaded successfully")
        else:
            logger.debug(
                "No Parselmouth categories specified, skipping feature loading"
            )

        return self

    def transform(self, recording_ids_list: list[list[int]]) -> np.ndarray:
        """
        Extract Parselmouth features preserving 2D structure.

        Args:
            recording_ids_list: Shape (n_samples, n_recordings_per_sample)

        Returns:
            np.ndarray: Shape (n_samples, n_recordings_per_sample, n_parsel_features)
        """
        if not self._parsel_features or not self.recordings_parsel:
            n_samples = len(recording_ids_list)
            n_recordings = len(recording_ids_list[0]) if recording_ids_list else 0
            logger.debug(
                f"No Parselmouth features available, returning empty array ({n_samples}, {n_recordings}, 0)"
            )
            return np.empty((n_samples, n_recordings, 0))

        if not recording_ids_list:
            raise ValueError("ParselmouthFeatureLoader: recording_ids_list is empty")

        # Convert to rectangular array for batch processing
        recording_ids_array = np.array(recording_ids_list)  # (n_samples, n_recordings)
        all_ids = recording_ids_array.flatten()

        logger.debug(f"Processing {len(all_ids)} Parselmouth recording IDs")

        try:
            # Batch lookup
            batch_features = self._parsel_features.get_features_for_ids(
                all_ids.tolist()
            )
            # batch_features shape: (n_samples * n_recordings, n_features)

            # Reshape to maintain dimensionality
            n_samples, n_recordings = recording_ids_array.shape
            n_features = batch_features.shape[1]

            result = batch_features.reshape(n_samples, n_recordings, n_features)
            logger.debug(f"Parselmouth features extracted: {result.shape}")
            return result

        except KeyError as e:
            raise KeyError(
                f"Missing Parselmouth features for recording IDs: {e}. "
                f"Check that all recording identifiers exist in the feature file."
            ) from e

    def get_feature_dimension(self) -> int:
        """Return the number of Parselmouth features per recording."""
        if self._parsel_features is not None:
            return self._parsel_features.feature_matrix.shape[1]
        return 0
