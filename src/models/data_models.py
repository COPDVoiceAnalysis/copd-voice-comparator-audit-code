"""
Pydantic data models for the voice biomarker pipeline.
These models replace manual validation and provide type safety throughout the pipeline.

This file maintains backward compatibility by re-exporting all models from the new modular structure.
For new code, consider importing directly from the specific modules:
- src.models.core_models
- src.models.config_models
- src.models.feature_models
- src.models.pipeline_models
"""

# Re-export all models from the new modular structure for backward compatibility
from src.models.core_models import (
    LUNG_DISEASE,
    RECORDING_CATEGORY,
    DATA_SOURCE,
    DATE_STR_FORMAT,
    MetadataRow,
    RecordingIdentifierSet,
)

from src.models.config_models import (
    ParselmouthConfig,
    FeatureConfig,
    TaskConfig,
    TrainingConfig,
    TemporalConfig,
    AutoGluonConfig,
    ExperimentConfig,
)

from src.models.feature_models import (
    LayerEmbedding,
    RecordingEmbedding,
    EmbeddingCollection,
    FeatureData,
    ParselmouthFeatures,
    Wav2VecFeatures,
)

from src.models.pipeline_models import (
    MetadataSummary,
    PipelineResults,
)

from src.models.utils import _save_structured_array

# Maintain the same exports as the original file
__all__ = [
    # Type definitions
    "LUNG_DISEASE",
    "RECORDING_CATEGORY",
    "DATA_SOURCE",
    "DATE_STR_FORMAT",
    # Core models
    "MetadataRow",
    "RecordingIdentifierSet",
    # Configuration models
    "ParselmouthConfig",
    "FeatureConfig",
    "TaskConfig",
    "TrainingConfig",
    "TemporalConfig",
    "AutoGluonConfig",
    "ExperimentConfig",
    # Feature models
    "LayerEmbedding",
    "RecordingEmbedding",
    "EmbeddingCollection",
    "FeatureData",
    "ParselmouthFeatures",
    "Wav2VecFeatures",
    # Pipeline models
    "MetadataSummary",
    "PipelineResults",
    # Utility functions
    "_save_structured_array",
]
