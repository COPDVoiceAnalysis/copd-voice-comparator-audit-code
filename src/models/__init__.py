"""
Voice biomarker pipeline data models.

This module provides all Pydantic models for the voice biomarker pipeline,
maintaining backward compatibility while organizing models into logical modules.
"""

# Core models and type definitions
from src.models.core_models import (
    LUNG_DISEASE,
    RECORDING_CATEGORY,
    DATE_STR_FORMAT,
    MetadataRow,
)

# Configuration models
from src.models.config_models import (
    FeatureConfig,
    TaskConfig,
    TrainingConfig,
    TemporalConfig,
    AutoGluonConfig,
)

# Feature models
from src.models.feature_models import (
    LayerEmbedding,
    RecordingEmbedding,
    EmbeddingCollection,
    FeatureData,
    ParselmouthFeatures,
    Wav2VecFeatures,
)

# Pipeline models
from src.models.pipeline_models import (
    MetadataSummary,
    PipelineResults,
)

# Utility functions
from src.models.utils import _save_structured_array

# Export all models for backward compatibility
__all__ = [
    # Type definitions
    "LUNG_DISEASE",
    "RECORDING_CATEGORY",
    "DATE_STR_FORMAT",
    # Core models
    "MetadataRow",
    # Configuration models
    "FeatureConfig",
    "TaskConfig",
    "TrainingConfig",
    "TemporalConfig",
    "AutoGluonConfig",
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
