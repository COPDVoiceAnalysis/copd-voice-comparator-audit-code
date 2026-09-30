"""
Feature models for the voice biomarker pipeline.
Contains models for handling Parselmouth and Wav2Vec2 features.
"""

from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
import logging
from src.models.core_models import RECORDING_CATEGORY

logger = logging.getLogger(__name__)


class LayerEmbedding(BaseModel):
    """Single layer embedding with validation for wav2vec2 processing."""

    layer_idx: int = Field(
        ge=0, le=24, description="Layer index (0-24 for wav2vec2-large)"
    )
    mean: np.ndarray = Field(..., description="Mean embedding vector")
    std: np.ndarray = Field(..., description="Standard deviation embedding vector")

    class Config:
        arbitrary_types_allowed = True

    @field_validator("mean", "std")
    @classmethod
    def validate_embedding_array(cls, v):
        """Validate embedding arrays for proper shape and type."""
        if not isinstance(v, np.ndarray):
            raise ValueError("Must be numpy array")
        if v.ndim != 1:
            raise ValueError("Must be 1D array")
        if v.shape[0] != 1024:  # wav2vec2-large-robust has 1024 dimensions
            raise ValueError(f"Must have 1024 dimensions, got {v.shape[0]}")
        if v.dtype not in [np.float32, np.float64]:
            raise ValueError("Must be float32 or float64")
        return v


class RecordingEmbedding(BaseModel):
    """Complete embedding data for one recording with validation."""

    recording_identifier: int = Field(..., description="Unique recording identifier")
    recording_category: RECORDING_CATEGORY = Field(
        ..., description="Recording category"
    )
    layer_embeddings: dict[int, LayerEmbedding] = Field(
        ..., description="Embeddings by layer"
    )

    @field_validator("layer_embeddings")
    @classmethod
    def validate_layer_consistency(cls, v):
        """Ensure layer embeddings are consistent."""
        if not v:
            raise ValueError("Must have at least one layer embedding")

        # Check that all mean/std arrays have same shape
        shapes = [(emb.mean.shape, emb.std.shape) for emb in v.values()]
        if len(set(shapes)) > 1:
            raise ValueError("All layer embeddings must have consistent shapes")
        return v


class EmbeddingCollection(BaseModel):
    """Collection of all recording embeddings with reorganization methods."""

    recordings: dict[int, RecordingEmbedding] = Field(
        ..., description="Embeddings by recording ID"
    )

    def reorganize_by_layer_category(
        self,
    ) -> dict[int, dict[str, list[RecordingEmbedding]]]:
        """Type-safe reorganization method for efficient saving."""
        from collections import defaultdict

        result = defaultdict(lambda: defaultdict(list))

        for recording in self.recordings.values():
            for layer_idx in recording.layer_embeddings.keys():
                result[layer_idx][recording.recording_category].append(recording)

        return dict(result)

    def get_layer_category_samples(
        self, layer_idx: int, category: str
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """Extract samples for a specific layer and category."""
        samples = []
        for recording in self.recordings.values():
            if (
                recording.recording_category == category
                and layer_idx in recording.layer_embeddings
            ):
                layer_emb = recording.layer_embeddings[layer_idx]
                samples.append(
                    (recording.recording_identifier, layer_emb.mean, layer_emb.std)
                )
        return samples

    def save_layer_category_to_files(
        self, layer_idx: int, category: str, embeddings_dir: str
    ) -> None:
        """Save specific layer-category combination to files."""
        from pathlib import Path
        from src.models.utils import _save_structured_array

        # Use existing method to extract samples directly
        samples = self.get_layer_category_samples(layer_idx, category)

        if not samples:
            logger.warning(f"No embeddings for layer {layer_idx}, category {category}")
            return

        # Simple extraction - no validation needed since Pydantic already validated the data
        recording_identifiers, means, stds = zip(*samples)
        recording_identifiers = np.array(recording_identifiers)
        means = np.stack(means)
        stds = np.stack(stds)

        # Create output directory
        output_dir = Path(embeddings_dir) / f"layer_{layer_idx}" / category
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save both mean and std files using private utility function
        _save_structured_array(
            output_dir / "mean.npy", recording_identifiers, means, "mean"
        )
        _save_structured_array(
            output_dir / "std.npy", recording_identifiers, stds, "std"
        )

        logger.info(
            f"Saved aggregated files for layer {layer_idx}, recording category {category} ({len(samples)} recordings)"
        )

    def save_to_directory(self, embeddings_dir: str) -> None:
        """Save all embeddings as structured numpy arrays organized by recording category."""
        logger.info("Saving aggregated embeddings as structured arrays...")

        if not self.recordings:
            logger.warning("No recordings to save")
            return

        # Use the type-safe reorganization method
        layer_category_data = self.reorganize_by_layer_category()

        logger.info(
            f"Creating embeddings structure for {len(layer_category_data)} layers..."
        )

        # Process each layer-category combination
        total_files_saved = 0
        for layer_idx, category_data in layer_category_data.items():
            for recording_category, recordings in category_data.items():
                self.save_layer_category_to_files(
                    layer_idx, recording_category, embeddings_dir
                )
                total_files_saved += 2  # mean.npy and std.npy

        logger.info(
            f"Temporal mean/std embeddings saved successfully ({total_files_saved} files)"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingCollection":
        """
        Create EmbeddingCollection from legacy dictionary format for backward compatibility.

        Args:
            data: Legacy format {recording_identifier: {category: {layer: (mean, std)}}}

        Returns:
            EmbeddingCollection with validated data
        """
        recordings = {}

        for recording_identifier, category_data in data.items():
            for recording_category, layer_data in category_data.items():
                # Create LayerEmbedding objects
                layer_embeddings = {}
                for layer_idx, (mean_embedding, std_embedding) in layer_data.items():
                    layer_embeddings[layer_idx] = LayerEmbedding(
                        layer_idx=layer_idx, mean=mean_embedding, std=std_embedding
                    )

                # Create RecordingEmbedding
                recording_embedding = RecordingEmbedding(
                    recording_identifier=recording_identifier,
                    recording_category=recording_category,
                    layer_embeddings=layer_embeddings,
                )

                recordings[recording_identifier] = recording_embedding

        return cls(recordings=recordings)


class FeatureData(BaseModel):
    """Model for extracted features."""

    parsel_features: np.ndarray = Field(..., description="Parselmouth features")
    wav2vec_features: np.ndarray = Field(..., description="Wav2Vec2 features")

    class Config:
        arbitrary_types_allowed = True

    @field_validator("parsel_features", "wav2vec_features")
    @classmethod
    def validate_numpy_arrays(cls, v):
        """Ensure features are numpy arrays."""
        if not isinstance(v, np.ndarray):
            raise ValueError("Features must be numpy arrays")
        return v

    def combine_features(self) -> np.ndarray:
        """Combine Parselmouth and Wav2Vec2 features."""
        if self.parsel_features.shape[1] > 0 and self.wav2vec_features.shape[1] > 0:
            return np.concatenate([self.parsel_features, self.wav2vec_features], axis=1)
        elif self.parsel_features.shape[1] > 0:
            return self.parsel_features
        elif self.wav2vec_features.shape[1] > 0:
            return self.wav2vec_features
        else:
            raise ValueError("No features available")


class ParselmouthFeatures(BaseModel):
    """Pydantic model that loads and validates Parselmouth features."""

    recording_identifiers: np.ndarray = Field(
        ..., description="Array of recording identifiers"
    )
    feature_matrix: np.ndarray = Field(
        ..., description="Feature matrix (n_samples, n_features)"
    )
    feature_names: list[str] = Field(..., description="Names of features")
    file_path: Path = Field(..., description="Source file path")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._id_to_index_cache: dict[int, int] | None = None

    @field_validator("recording_identifiers")
    @classmethod
    def validate_recording_ids(cls, v):
        """Validate recording identifiers array."""
        if not isinstance(v, np.ndarray):
            raise ValueError("recording_identifiers must be numpy array")
        if v.ndim != 1:
            raise ValueError("recording_identifiers must be 1D array")
        if len(v) == 0:
            raise ValueError("recording_identifiers cannot be empty")
        return v

    @field_validator("feature_matrix")
    @classmethod
    def validate_feature_matrix(cls, v):
        """Validate feature matrix array."""
        if not isinstance(v, np.ndarray):
            raise ValueError("feature_matrix must be numpy array")
        if v.ndim != 2:
            raise ValueError("feature_matrix must be 2D array")
        if len(v) == 0:
            raise ValueError("feature_matrix cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_consistent_lengths(self):
        """Ensure recording_identifiers and feature_matrix have consistent lengths."""
        if len(self.recording_identifiers) != len(self.feature_matrix):
            raise ValueError(
                f"recording_identifiers length ({len(self.recording_identifiers)}) "
                f"must match feature_matrix length ({len(self.feature_matrix)})"
            )
        return self

    def get_features_for_ids(self, recording_ids: list[int]) -> np.ndarray:
        """
        Get features for specific recording IDs in the requested order.

        Args:
            recording_ids: List of recording identifiers to extract features for

        Returns:
            Feature array with shape (len(recording_ids), n_features) in requested order

        Raises:
            KeyError: If any recording ID is not found
        """
        # Lazy initialization - build cache only on first call
        if self._id_to_index_cache is None:
            self._id_to_index_cache = {
                int(rid): idx for idx, rid in enumerate(self.recording_identifiers)
            }

        # Get indices in the requested order, preserving order
        try:
            indices = [self._id_to_index_cache[rid] for rid in recording_ids]
        except KeyError as e:
            raise KeyError(f"Recording ID not found: {e}")

        # Return features in requested order
        return self.feature_matrix[indices]

    @staticmethod
    def generate_feature_names(
        num_mfcc: int = 14,
        include_basic_acoustics: bool = True,
        include_advanced_voice_quality: bool = True,
        include_spectral_analysis: bool = True,
        include_mfcc_derivatives: bool = True,
        include_temporal_features: bool = True,
    ) -> list[str]:
        """
        Generate feature names matching the new clinical relevance grouping structure.

        Always generates ALL available features (except EMD-DWT) - filtering is done during loading.

        Args:
            num_mfcc: Number of MFCC coefficients
            include_basic_acoustics: F0, intensity, HNR, all jitter/shimmer
            include_advanced_voice_quality: CPP, DSI
            include_spectral_analysis: MFCCs (base only), formants, spectral features
            include_mfcc_derivatives: Delta/delta-delta MFCCs
            include_temporal_features: Temporal durations, syllable rate

        Returns:
            List of feature names in the same order as extracted features
        """
        feat_names = []

        # Spectral Analysis Group: Base MFCCs, formants, spectral features
        if include_spectral_analysis:
            # Base MFCC means & stds (using new naming convention: mfcc_0_mean, etc.)
            for i in range(num_mfcc):
                feat_names.append(f"mfcc_{i}_mean")
            for i in range(num_mfcc):
                feat_names.append(f"mfcc_{i}_std")

            # Formant frequencies and bandwidths
            feat_names += [
                "f1_mean",
                "f2_mean",
                "f3_mean",
                "bw1_mean",
                "bw2_mean",
                "bw3_mean",
            ]

            # Spectral features
            feat_names += [
                "alpha_ratio",
                "hammarberg_index",
                "slope_0_500Hz",
                "slope_500_1500Hz",
                "spectral_flux",
            ]

        # MFCC Derivatives Group: Delta and delta-delta MFCCs
        if include_mfcc_derivatives:
            # Delta MFCCs
            for i in range(num_mfcc):
                feat_names.append(f"delta_mfcc_{i}_mean")
            for i in range(num_mfcc):
                feat_names.append(f"delta_mfcc_{i}_std")

            # Delta-delta MFCCs
            for i in range(num_mfcc):
                feat_names.append(f"delta2_mfcc_{i}_mean")
            for i in range(num_mfcc):
                feat_names.append(f"delta2_mfcc_{i}_std")

        # Basic Acoustics Group: Core voice measures
        if include_basic_acoustics:
            # Pitch features
            feat_names += ["f0_mean", "f0_std"]

            # Semitone features
            feat_names += [
                "semitone_mean",
                "semitone_CV",
                "semitone_pct20",
                "semitone_pct50",
                "semitone_pct80",
            ]

            # HNR features
            feat_names += ["hnr_mean", "hnr_std"]

            # All jitter features (basic + additional)
            feat_names += [
                "jitter_local",
                "jitter_rap",
                "jitter_ppq5",
                "jitter_ddp",
                "jitter_local_absolute",  # Additional jitter
            ]

            # All shimmer features (basic + additional)
            feat_names += [
                "shimmer_local",
                "shimmer_local_db",
                "shimmer_apq3",
                "shimmer_apq5",
                "shimmer_apq11",
                "shimmer_dda",  # Additional shimmer
            ]

            # Intensity
            feat_names.append("intensity_mean")

        # Temporal Features Group
        if include_temporal_features:
            feat_names += [
                "voiced_dur_mean",
                "voiced_dur_std",
                "unvoiced_dur_mean",
                "unvoiced_dur_std",
                "pseudo_syll_rate",
                "perc_unvoiced_frames",
            ]

        # Advanced Voice Quality Group: Specialized clinical measures
        if include_advanced_voice_quality:
            feat_names.append("cpp_value")
            feat_names.append("dsi_value")

        # Note: EMD-DWT features are excluded entirely as requested

        return feat_names

    def filter_by_config(self, config) -> "ParselmouthFeatures":
        """Filter features based on ParselmouthConfig (exact list or grouping structure)."""

        # Step 1: Get traditional features
        traditional_features = self._get_traditional_features(config)

        # Step 2: Add eGeMAPS features if enabled
        if getattr(config, "include_egemaps_features", True):
            egemaps_features = self._get_egemaps_features()

            if egemaps_features["matrix"].shape[1] > 0:
                # Combine traditional + eGeMAPS features
                combined_matrix = np.hstack(
                    [traditional_features["matrix"], egemaps_features["matrix"]]
                )
                combined_names = (
                    traditional_features["names"] + egemaps_features["names"]
                )

                logger.debug(
                    f"Combined features: {len(traditional_features['names'])} traditional + "
                    f"{len(egemaps_features['names'])} eGeMAPS = {len(combined_names)} total"
                )
            else:
                # No eGeMAPS features found, use traditional only
                combined_matrix = traditional_features["matrix"]
                combined_names = traditional_features["names"]
                logger.warning(
                    "eGeMAPS features requested but none found, using traditional features only"
                )
        else:
            # Traditional features only
            combined_matrix = traditional_features["matrix"]
            combined_names = traditional_features["names"]
            logger.debug(
                f"Using traditional features only: {len(combined_names)} features"
            )

        return ParselmouthFeatures(
            recording_identifiers=self.recording_identifiers,
            feature_matrix=combined_matrix,
            feature_names=combined_names,
            file_path=self.file_path,
        )

    def _get_traditional_features(self, config) -> dict:
        """Extract traditional Parselmouth features based on config."""
        # Check if exact feature list is provided (overrides grouping)
        if (
            hasattr(config, "exact_feature_list")
            and config.exact_feature_list is not None
        ):
            desired_feature_names = config.exact_feature_list
            logger.debug(
                f"Using exact feature list with {len(desired_feature_names)} features"
            )
        else:
            # Generate desired feature names based on grouping structure
            desired_feature_names = self.generate_feature_names(
                num_mfcc=14,  # Always use 14 MFCCs for consistency
                include_basic_acoustics=getattr(
                    config, "include_basic_acoustics", True
                ),
                include_advanced_voice_quality=getattr(
                    config, "include_advanced_voice_quality", True
                ),
                include_spectral_analysis=getattr(
                    config, "include_spectral_analysis", True
                ),
                include_mfcc_derivatives=getattr(
                    config, "include_mfcc_derivatives", True
                ),
                include_temporal_features=getattr(
                    config, "include_temporal_features", True
                ),
            )

        # Find indices of desired features in current feature list
        keep_indices = []
        kept_feature_names = []
        missing_features = []

        for desired_name in desired_feature_names:
            if desired_name in self.feature_names:
                idx = self.feature_names.index(desired_name)
                keep_indices.append(idx)
                kept_feature_names.append(desired_name)
            else:
                missing_features.append(desired_name)

        if missing_features:
            logger.warning(
                f"Missing traditional features not found in extracted features: {missing_features}"
            )

        if not keep_indices:
            logger.warning("No matching traditional features found after filtering")
            # Return empty features
            return {
                "matrix": np.empty((len(self.recording_identifiers), 0)),
                "names": [],
            }

        # Filter feature matrix and names
        filtered_matrix = self.feature_matrix[:, keep_indices]

        logger.debug(
            f"Traditional features: filtered from {len(self.feature_names)} to {len(kept_feature_names)}"
        )

        return {"matrix": filtered_matrix, "names": kept_feature_names}

    def _get_egemaps_features(self) -> dict:
        """Extract all eGeMAPS features (features starting with 'egemaps_')."""
        egemaps_indices = []
        egemaps_names = []

        for i, feature_name in enumerate(self.feature_names):
            if feature_name.startswith("egemaps_"):
                egemaps_indices.append(i)
                egemaps_names.append(feature_name)

        if egemaps_indices:
            egemaps_matrix = self.feature_matrix[:, egemaps_indices]
            logger.debug(f"Found {len(egemaps_names)} eGeMAPS features")
        else:
            egemaps_matrix = np.empty((len(self.recording_identifiers), 0))
            logger.warning(
                "No eGeMAPS features found (features starting with 'egemaps_')"
            )

        return {"matrix": egemaps_matrix, "names": egemaps_names}

    @classmethod
    def from_file(cls, file_path: Path) -> "ParselmouthFeatures":
        """Load and validate Parselmouth features from optimized structured array file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Parselmouth feature file not found: {file_path}")

        try:
            data = np.load(file_path)

            if (
                "recording_identifiers" not in data.dtype.names
                or "feature_matrix" not in data.dtype.names
            ):
                raise ValueError(
                    f"Invalid Parselmouth feature file format: {file_path}. "
                    f"Expected optimized format with 'recording_identifiers' and 'feature_matrix' fields. "
                    f"Please regenerate features using the current parselmouth_extractor.py."
                )

            # Optimized format: direct access to 2D arrays
            recording_identifiers = data["recording_identifiers"][0]  # 1D array
            feature_matrix = data["feature_matrix"][0]  # 2D array
            feature_names = data["feature_names"][
                0
            ].tolist()  # Convert from numpy array to list

            return cls(
                recording_identifiers=recording_identifiers,
                feature_matrix=feature_matrix,
                feature_names=feature_names,
                file_path=file_path,
            )

        except Exception as e:
            logger.error(f"Failed to load Parselmouth features from {file_path}: {e}")
            raise


class Wav2VecFeatures(BaseModel):
    """Pydantic model that loads and validates Wav2Vec features using recording identifiers."""

    recording_identifiers: np.ndarray = Field(
        ..., description="Array of recording identifiers"
    )
    feature_matrix: np.ndarray = Field(
        ..., description="Feature matrix (n_samples, n_features)"
    )
    layer: int = Field(..., description="Wav2Vec layer")
    statistics: str = Field(..., description="Statistics used (mean/std)")
    recording_category: str = Field(..., description="Recording category")
    file_path: Path = Field(..., description="Source file path")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._id_to_index_cache: dict[int, int] | None = None

    @field_validator("recording_identifiers")
    @classmethod
    def validate_recording_ids(cls, v):
        """Validate recording identifiers array."""
        if not isinstance(v, np.ndarray):
            raise ValueError("recording_identifiers must be numpy array")
        if v.ndim != 1:
            raise ValueError("recording_identifiers must be 1D array")
        if len(v) == 0:
            raise ValueError("recording_identifiers cannot be empty")
        return v

    @field_validator("feature_matrix")
    @classmethod
    def validate_feature_matrix(cls, v):
        """Validate feature matrix array."""
        if not isinstance(v, np.ndarray):
            raise ValueError("feature_matrix must be numpy array")
        if v.ndim != 2:
            raise ValueError("feature_matrix must be 2D array")
        if len(v) == 0:
            raise ValueError("feature_matrix cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_consistent_lengths(self):
        """Ensure recording_identifiers and feature_matrix have consistent lengths."""
        if len(self.recording_identifiers) != len(self.feature_matrix):
            raise ValueError(
                f"recording_identifiers length ({len(self.recording_identifiers)}) "
                f"must match feature_matrix length ({len(self.feature_matrix)})"
            )
        return self

    def get_features_for_ids(self, recording_ids: list[int]) -> np.ndarray:
        """
        Get features for specific recording IDs in the requested order.

        Args:
            recording_ids: List of recording identifiers to extract features for

        Returns:
            Feature array with shape (len(recording_ids), n_features) in requested order

        Raises:
            KeyError: If any recording ID is not found
        """
        # Lazy initialization - build cache only on first call
        if self._id_to_index_cache is None:
            self._id_to_index_cache = {
                int(rid): idx for idx, rid in enumerate(self.recording_identifiers)
            }

        # Get indices in the requested order, preserving order
        try:
            indices = [self._id_to_index_cache[rid] for rid in recording_ids]
        except KeyError as e:
            raise KeyError(f"Recording ID not found: {e}")

        # Return features in requested order
        return self.feature_matrix[indices]

    @classmethod
    def from_directory(
        cls,
        embeddings_dir: Path,
        layer: int = 4,
        statistics: str = "mean",
        recording_category: str = "poem",
    ) -> "Wav2VecFeatures":
        """Load and validate Wav2Vec features from optimized structured array files."""
        logger.info(
            f"Loading Wav2Vec features from {embeddings_dir}, layer {layer}, {statistics}, category {recording_category}"
        )

        if not embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

        # Navigate directory structure: embeddings_dir/layer_X/category/statistics.npy
        layer_dir = embeddings_dir / f"layer_{layer}"
        if not layer_dir.exists():
            raise FileNotFoundError(f"Layer directory not found: {layer_dir}")

        category_dir = layer_dir / recording_category
        if not category_dir.exists():
            raise FileNotFoundError(
                f"Recording category directory not found: {category_dir}"
            )

        feature_file = category_dir / f"{statistics}.npy"
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")

        try:
            data = np.load(feature_file, mmap_mode="r")  # Memory-mapped for efficiency

            # Only support optimized format - no legacy support
            expected_matrix_field = f"{statistics}_matrix"
            if (
                "recording_identifiers" not in data.dtype.names
                or expected_matrix_field not in data.dtype.names
            ):
                raise ValueError(
                    f"Invalid Wav2Vec feature file format: {feature_file}. "
                    f"Expected optimized format with 'recording_identifiers' and '{expected_matrix_field}' fields. "
                    f"Please regenerate embeddings using the current get_embeddings.py."
                )

            # Optimized format: direct access to 2D arrays
            recording_identifiers = data["recording_identifiers"][0]  # 1D array
            feature_matrix = data[expected_matrix_field][0]  # 2D array

            logger.info(
                f"Loaded {len(recording_identifiers)} Wav2Vec features"
            )

            return cls(
                recording_identifiers=recording_identifiers,
                feature_matrix=feature_matrix,
                layer=layer,
                statistics=statistics,
                recording_category=recording_category,
                file_path=feature_file,
            )

        except Exception as e:
            logger.error(f"Failed to load Wav2Vec features from {feature_file}: {e}")
            raise
