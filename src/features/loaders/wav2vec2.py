"""
Wav2Vec2 feature loader.

Handles Wav2Vec2 feature loading and processing while maintaining 3D output structure.
"""

import numpy as np
from pathlib import Path
import logging
from src.models.data_models import Wav2VecFeatures
from src.features.loaders.base import BaseFeatureLoader

logger = logging.getLogger(__name__)


class Wav2Vec2FeatureLoader(BaseFeatureLoader):
    """
    Handles only Wav2Vec2 feature loading and processing.

    Maintains 3D output structure: (n_samples, n_recordings_per_sample, n_wav2vec_features)
    """

    def __init__(
        self,
        embeddings_dir: str,
        recordings_wav2vec2: tuple[str, ...],
        wav2vec2_layer: int,
        wav2vec2_embedding_statistics: str,
    ):
        """
        Initialize Wav2Vec2 feature extractor.

        Args:
            embeddings_dir: Directory containing wav2vec2 embeddings
            recordings_wav2vec2: Tuple of recording categories to process
            wav2vec2_layer: Layer index to extract features from
            wav2vec2_embedding_statistics: Statistics to use ("mean" or "mean_std")
        """
        self.embeddings_dir = embeddings_dir
        self.recordings_wav2vec2 = recordings_wav2vec2
        self.wav2vec2_layer = wav2vec2_layer
        self.wav2vec2_embedding_statistics = wav2vec2_embedding_statistics
        self._wav2vec_features_cache = {}

        logger.debug(
            f"Wav2Vec2FeatureExtractor initialized for categories: {recordings_wav2vec2}, "
            f"layer: {wav2vec2_layer}, statistics: {wav2vec2_embedding_statistics}"
        )

    def fit(self, X, y=None):
        """Load Wav2Vec2 features for all required combinations."""
        if not self.recordings_wav2vec2:
            logger.debug("No Wav2Vec2 categories specified, skipping feature loading")
            return self

        for category in self.recordings_wav2vec2:
            # Load mean features
            mean_key = (self.wav2vec2_layer, "mean", category)
            logger.debug(
                f"Loading Wav2Vec2 mean features for layer {self.wav2vec2_layer}, category {category}"
            )

            self._wav2vec_features_cache[mean_key] = Wav2VecFeatures.from_directory(
                embeddings_dir=Path(self.embeddings_dir),
                layer=self.wav2vec2_layer,
                statistics="mean",
                recording_category=category,
            )

            # Load std features if needed
            if self.wav2vec2_embedding_statistics == "mean_std":
                std_key = (self.wav2vec2_layer, "std", category)
                logger.debug(
                    f"Loading Wav2Vec2 std features for layer {self.wav2vec2_layer}, category {category}"
                )

                self._wav2vec_features_cache[std_key] = Wav2VecFeatures.from_directory(
                    embeddings_dir=Path(self.embeddings_dir),
                    layer=self.wav2vec2_layer,
                    statistics="std",
                    recording_category=category,
                )

        logger.debug("Wav2Vec2 features loaded successfully")
        return self

    def transform(self, recording_ids_list: list[list[int]]) -> np.ndarray:
        """
        Extract Wav2Vec2 features preserving 3D structure.

        Each sample's recording_ids are ordered to match self.recordings_wav2vec2 categories.
        For example, if recordings_wav2vec2 = ('a', 'i', 'o'), then
        recording_ids_list[sample] = [id_for_a, id_for_i, id_for_o].

        Each category's IDs are looked up only in that category's embedding file.

        Args:
            recording_ids_list: Shape (n_samples, n_recordings_per_sample)
                Column k contains recording IDs for category self.recordings_wav2vec2[k].

        Returns:
            np.ndarray: Shape (n_samples, n_recordings_per_sample, n_wav2vec_features)
        """
        if not self.recordings_wav2vec2 or not self._wav2vec_features_cache:
            n_samples = len(recording_ids_list)
            n_recordings = len(recording_ids_list[0]) if recording_ids_list else 0
            logger.debug(
                f"No Wav2Vec2 features available, returning empty array ({n_samples}, {n_recordings}, 0)"
            )
            return np.empty((n_samples, n_recordings, 0))

        if not recording_ids_list:
            return np.empty((0, 0, 0))

        # Convert to rectangular array: (n_samples, n_categories)
        recording_ids_array = np.array(recording_ids_list)
        n_samples, n_recordings = recording_ids_array.shape

        if n_recordings != len(self.recordings_wav2vec2):
            raise ValueError(
                f"Number of recording IDs per sample ({n_recordings}) does not match "
                f"number of wav2vec2 categories ({len(self.recordings_wav2vec2)}): "
                f"{self.recordings_wav2vec2}"
            )

        logger.debug(
            f"Processing {n_samples} samples × {n_recordings} categories for Wav2Vec2"
        )

        # Process each category using only the IDs belonging to that category
        category_feature_arrays = []

        for cat_idx, category in enumerate(self.recordings_wav2vec2):
            # Extract only the IDs for this category (column cat_idx)
            category_ids = recording_ids_array[:, cat_idx]

            # Batch lookup for mean features
            mean_key = (self.wav2vec2_layer, "mean", category)
            mean_features = self._wav2vec_features_cache[mean_key]
            mean_batch = mean_features.get_features_for_ids(category_ids.tolist())

            if mean_batch.size == 0:
                raise ValueError(
                    f"No mean features found for category {category}"
                )

            if self.wav2vec2_embedding_statistics == "mean":
                category_features = mean_batch
            elif self.wav2vec2_embedding_statistics == "mean_std":
                # Batch lookup for std features and concatenate
                std_key = (self.wav2vec2_layer, "std", category)
                std_features = self._wav2vec_features_cache[std_key]
                std_batch = std_features.get_features_for_ids(category_ids.tolist())

                if std_batch.size == 0:
                    raise ValueError(
                        f"No std features found for category {category}"
                    )
                category_features = np.hstack([mean_batch, std_batch])
            else:
                raise ValueError(
                    f"Unknown embedding_statistics: {self.wav2vec2_embedding_statistics}"
                )

            # category_features shape: (n_samples, n_features_per_category)
            category_feature_arrays.append(category_features)

        # Stack features: each array is (n_samples, n_features_per_category)
        # Result: (n_samples, n_categories, n_features_per_category)
        result = np.stack(category_feature_arrays, axis=1)
        logger.debug(f"Wav2Vec2 features extracted: {result.shape}")
        return result

    def get_feature_dimension(self) -> int:
        """Return the number of Wav2Vec2 features per recording."""
        if not self._wav2vec_features_cache:
            return 0

        # Calculate total feature dimension across all categories
        total_features = 0
        for category in self.recordings_wav2vec2:
            mean_key = (self.wav2vec2_layer, "mean", category)
            if mean_key in self._wav2vec_features_cache:
                mean_dim = self._wav2vec_features_cache[mean_key].feature_matrix.shape[
                    1
                ]
                if self.wav2vec2_embedding_statistics == "mean_std":
                    total_features += mean_dim * 2  # mean + std
                else:
                    total_features += mean_dim

        return total_features
