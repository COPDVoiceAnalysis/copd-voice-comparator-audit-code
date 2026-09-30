"""
Feature preloader for efficient one-time data loading.

This module handles loading all required features once before gridsearch,
eliminating redundant I/O operations during cross-validation.
"""

import numpy as np
import logging
from src.models.data_models import (
    FeatureConfig,
    RecordingIdentifierSet,
    ParselmouthFeatures,
)
from src.features.loaders.parselmouth import ParselmouthFeatureLoader
from src.features.loaders.wav2vec2 import Wav2Vec2FeatureLoader

logger = logging.getLogger(__name__)


class FeaturePreloader:
    """
    Preloads all required features once and provides them as numpy arrays.

    This replaces the CompositeFeatureLoader approach by loading features
    once at the beginning of training rather than repeatedly during CV.
    """

    def __init__(self, feature_config: FeatureConfig):
        """
        Initialize the feature preloader.

        Args:
            feature_config: Configuration specifying which features to load
        """
        self.feature_config = feature_config
        self.parsel_features: np.ndarray | None = None
        self.wav2vec_features: np.ndarray | None = None
        self.feature_names: list[str] = []

        logger.info(f"FeaturePreloader initialized with config: {feature_config}")

    def load_features(
        self,
        recording_identifier_sets: list[RecordingIdentifierSet],
        demographic_features: np.ndarray | None = None,
        include_voice_features: bool = True,
    ) -> tuple[np.ndarray, int, int]:
        """
        Load all features for the given recording identifier sets.

        Args:
            recording_identifier_sets: List of RecordingIdentifierSet objects
            demographic_features: Optional demographic features array of shape (n_samples, n_demographic_features)
            include_voice_features: If False, load only baseline (demographic) features; demographic_features must be provided.

        Returns:
            Tuple of (concatenated_features, parsel_feature_count, demographic_feature_count)
            - concatenated_features: Combined parselmouth, wav2vec, and demographic features
            - parsel_feature_count: Number of parselmouth features (for splitting)
            - demographic_feature_count: Number of demographic features (for splitting)
            Shape: (n_samples, total_features)
        """
        logger.info(f"Loading features for {len(recording_identifier_sets)} samples")

        if not include_voice_features:
            if demographic_features is None:
                raise ValueError(
                    "include_voice_features=False requires demographic_features (baseline features)"
                )
            n = len(recording_identifier_sets)
            if demographic_features.shape[0] != n:
                raise ValueError(
                    f"demographic_features shape mismatch: expected {n}, got {demographic_features.shape[0]}"
                )
            logger.info(
                f"Baseline-only features (include_voice_features=False): shape={demographic_features.shape}"
            )
            return demographic_features, 0, demographic_features.shape[1]

        # Load parselmouth features if configured
        if self.feature_config.recordings_parsel:
            logger.info("Loading Parselmouth features...")
            self.parsel_features = self._load_parselmouth_features(
                recording_identifier_sets
            )
            logger.info(f"Parselmouth features shape: {self.parsel_features.shape}")
        else:
            self.parsel_features = np.empty((len(recording_identifier_sets), 0))
            logger.info("No Parselmouth features configured")

        # Load wav2vec2 features if configured
        if self.feature_config.recordings_wav2vec2:
            logger.info("Loading Wav2Vec2 features...")
            self.wav2vec_features = self._load_wav2vec_features(
                recording_identifier_sets
            )
            logger.info(f"Wav2Vec2 features shape: {self.wav2vec_features.shape}")
        else:
            self.wav2vec_features = np.empty((len(recording_identifier_sets), 0))
            logger.info("No Wav2Vec2 features configured")

        # Handle demographic features
        demographic_feature_count = 0
        if demographic_features is not None:
            if demographic_features.shape[0] != len(recording_identifier_sets):
                raise ValueError(
                    f"Demographic features shape mismatch: expected {len(recording_identifier_sets)} samples, got {demographic_features.shape[0]}"
                )
            demographic_feature_count = demographic_features.shape[1]
            logger.info(f"Demographic features shape: {demographic_features.shape}")
        else:
            demographic_features = np.empty((len(recording_identifier_sets), 0))
            logger.info("No demographic features provided")

        # Validate that we have some features
        if (
            self.parsel_features.shape[1] == 0
            and self.wav2vec_features.shape[1] == 0
            and demographic_feature_count == 0
        ):
            raise ValueError(
                "No features were loaded! Check your feature configuration."
            )

        # Concatenate all features for sklearn compatibility: [parselmouth | wav2vec | demographic]
        concatenated_features = np.hstack(
            [self.parsel_features, self.wav2vec_features, demographic_features]
        )
        parsel_feature_count = self.parsel_features.shape[1]

        logger.info(
            f"Feature loading complete. Parselmouth: {self.parsel_features.shape}, Wav2Vec2: {self.wav2vec_features.shape}, Demographic: {demographic_features.shape}"
        )
        logger.info(
            f"Concatenated features shape: {concatenated_features.shape}, Parselmouth feature count: {parsel_feature_count}, Demographic feature count: {demographic_feature_count}"
        )

        return concatenated_features, parsel_feature_count, demographic_feature_count

    def _load_parselmouth_features(
        self, recording_identifier_sets: list[RecordingIdentifierSet]
    ) -> np.ndarray:
        """Load and process Parselmouth features."""
        # Create parselmouth loader
        parsel_loader = ParselmouthFeatureLoader(
            self.feature_config.parsel_feature_file,
            self.feature_config.recordings_parsel,
            self.feature_config.parselmouth_config,
        )

        # Fit the loader (loads features from disk)
        parsel_loader.fit(recording_identifier_sets)

        # Extract parselmouth IDs for all samples
        parsel_ids_list = [
            identifier_set.parselmouth_ids
            for identifier_set in recording_identifier_sets
        ]

        # Transform to get 3D features: (n_samples, n_recordings_per_sample, n_features)
        parsel_3d = parsel_loader.transform(parsel_ids_list)

        # Flatten recordings per sample: (n_samples, n_recordings_per_sample * n_features)
        parsel_features = self._stack_recordings(parsel_3d)

        return parsel_features

    def _load_wav2vec_features(
        self, recording_identifier_sets: list[RecordingIdentifierSet]
    ) -> np.ndarray:
        """Load and process Wav2Vec2 features."""
        # Create wav2vec loader
        wav2vec_loader = Wav2Vec2FeatureLoader(
            str(self.feature_config.embeddings_dir),
            self.feature_config.recordings_wav2vec2,
            self.feature_config.wav2vec2_layer,
            self.feature_config.wav2vec2_embedding_statistics,
        )

        # Fit the loader (loads features from disk)
        wav2vec_loader.fit(recording_identifier_sets)

        # Extract wav2vec IDs for all samples
        wav2vec_ids_list = [
            identifier_set.wav2vec_ids for identifier_set in recording_identifier_sets
        ]

        # Transform to get 3D features: (n_samples, n_recordings_per_sample, n_features)
        wav2vec_3d = wav2vec_loader.transform(wav2vec_ids_list)

        # Flatten recordings per sample: (n_samples, n_recordings_per_sample * n_features)
        wav2vec_features = self._stack_recordings(wav2vec_3d)

        return wav2vec_features

    def _stack_recordings(self, features_3d: np.ndarray) -> np.ndarray:
        """
        Stack features from multiple recordings per sample.

        Args:
            features_3d: Shape (n_samples, n_recordings_per_sample, n_features_per_recording)

        Returns:
            np.ndarray: Shape (n_samples, n_recordings_per_sample * n_features_per_recording)
        """
        if features_3d.size == 0:
            return features_3d.reshape(features_3d.shape[0], -1)

        n_samples, n_recordings, n_features = features_3d.shape

        # Simple flattening (stacking in order)
        result = features_3d.reshape(n_samples, n_recordings * n_features)

        logger.debug(f"Stacked recordings: {features_3d.shape} -> {result.shape}")
        return result

    def generate_feature_names(
        self,
        demographic_feature_count: int = 0,
        demographic_feature_columns: list[str] | None = None,
    ) -> list[str]:
        """
        Generate descriptive feature names for all loaded features.

        Args:
            demographic_feature_count: Number of demographic features
            demographic_feature_columns: Actual column names used (falls back to generic names)

        Returns:
            List of feature names in the same order as concatenated features
        """
        feature_names = []

        # 1. Parselmouth feature names
        if self.parsel_features is not None and self.parsel_features.shape[1] > 0:
            try:
                # Load parselmouth features to get the feature names
                parsel_features_obj = ParselmouthFeatures.from_file(
                    self.feature_config.parsel_feature_file
                )

                # Apply the same filtering as used during loading
                if self.feature_config.parselmouth_config is not None:
                    parsel_features_obj = parsel_features_obj.filter_by_config(
                        self.feature_config.parselmouth_config
                    )

                base_parsel_names = parsel_features_obj.feature_names

                # Handle multiple recordings per sample (stacking)
                n_recordings = len(self.feature_config.recordings_parsel)
                if n_recordings > 1:
                    # Create names for stacked recordings: feature_name_recording0, feature_name_recording1, etc.
                    for recording_category in self.feature_config.recordings_parsel:
                        for base_name in base_parsel_names:
                            feature_names.append(f"{base_name}_{recording_category}")
                else:
                    # Single recording, use base names
                    feature_names.extend(base_parsel_names)

                logger.debug(
                    f"Generated {len(feature_names)} Parselmouth feature names"
                )

            except Exception as e:
                logger.warning(f"Could not load Parselmouth feature names: {e}")
                # Fallback: generate generic parselmouth names
                n_parsel_features = self.parsel_features.shape[1]
                for i in range(n_parsel_features):
                    feature_names.append(f"parsel_feature_{i}")

        # 2. Wav2Vec2 feature names
        if self.wav2vec_features is not None and self.wav2vec_features.shape[1] > 0:
            layer = self.feature_config.wav2vec2_layer
            statistics = self.feature_config.wav2vec2_embedding_statistics
            n_recordings = len(self.feature_config.recordings_wav2vec2)

            # Calculate dimensions per recording (total features / number of recordings)
            total_wav2vec_features = self.wav2vec_features.shape[1]
            dims_per_recording = (
                total_wav2vec_features // n_recordings if n_recordings > 0 else 0
            )

            for recording_category in self.feature_config.recordings_wav2vec2:
                for dim in range(dims_per_recording):
                    feature_name = f"wav2vec_layer{layer}_{recording_category}_{statistics}_dim{dim}"
                    feature_names.append(feature_name)

            logger.debug(
                f"Generated {len(self.feature_config.recordings_wav2vec2) * dims_per_recording} Wav2Vec2 feature names"
            )

        # 3. Demographic feature names
        if demographic_feature_count > 0:
            if demographic_feature_columns and len(demographic_feature_columns) >= demographic_feature_count:
                demographic_names = demographic_feature_columns[:demographic_feature_count]
            else:
                demographic_names = [f"demographic_{i}" for i in range(demographic_feature_count)]
            feature_names.extend(demographic_names)
            logger.debug(
                f"Added {min(demographic_feature_count, len(demographic_names))} demographic feature names"
            )

        logger.info(f"Generated {len(feature_names)} total feature names")
        logger.debug(
            f"Feature name breakdown: Parselmouth={self.parsel_features.shape[1] if self.parsel_features is not None else 0}, "
            f"Wav2Vec={self.wav2vec_features.shape[1] if self.wav2vec_features is not None else 0}, "
            f"Demographic={demographic_feature_count}"
        )

        return feature_names

    def get_feature_info(self) -> dict:
        """Get information about loaded features."""
        info = {
            "parselmouth_features": self.parsel_features.shape[1]
            if self.parsel_features is not None
            else 0,
            "wav2vec_features": self.wav2vec_features.shape[1]
            if self.wav2vec_features is not None
            else 0,
            "total_features": 0,
            "parselmouth_categories": self.feature_config.recordings_parsel,
            "wav2vec_categories": self.feature_config.recordings_wav2vec2,
            "wav2vec_layer": self.feature_config.wav2vec2_layer,
            "wav2vec_statistics": self.feature_config.wav2vec2_embedding_statistics,
        }

        if self.parsel_features is not None and self.wav2vec_features is not None:
            info["total_features"] = (
                self.parsel_features.shape[1] + self.wav2vec_features.shape[1]
            )

        return info
