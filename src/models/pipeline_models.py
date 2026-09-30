"""
Pipeline models for the voice biomarker pipeline.
Contains models for pipeline execution, validation, and temporal processing.
"""

from typing import Any
from pydantic import BaseModel, Field


class MetadataSummary(BaseModel):
    """Summary of metadata state."""

    total_rows: int = Field(..., ge=0, description="Total number of rows")
    original_rows: int = Field(..., ge=0, description="Original number of rows")
    unique_audio_ids: int = Field(..., ge=0, description="Number of unique audio IDs")
    columns: list[str] = Field(default_factory=list, description="Column names")
    class_distribution: dict[str, int] | None = Field(
        None, description="Distribution of classes"
    )
    missing_values: dict[str, int] = Field(
        default_factory=dict, description="Missing values per column"
    )
    filtering_history: list[dict[str, Any]] = Field(
        default_factory=list, description="History of filtering operations"
    )


class PipelineResults(BaseModel):
    """Results from pipeline execution."""

    cv_results: dict[str, Any] = Field(
        default_factory=dict, description="Cross-validation results"
    )
    final_results: dict[str, Any] = Field(
        default_factory=dict, description="Final test results"
    )
    dataset_info: dict[str, Any] = Field(
        default_factory=dict, description="Dataset information"
    )
    config_used: dict[str, Any] = Field(
        default_factory=dict, description="Configuration used"
    )

    class Config:
        arbitrary_types_allowed = True
