"""
Core data models for the voice biomarker pipeline.
Contains fundamental data structures and type definitions.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, computed_field
from src.utils.recording_identifier import generate_recording_identifier

# Type definitions
LUNG_DISEASE = Literal[
    "asthma",
    "control",
    "restriction",
    "copd",
    "fibrosis",
    "ohs",
    "ph",
    "cteph",
    "sarcoidosis",
    "hypersensitivity_pneumonitis",
    "granulomatosis_w_polyangiitis",
    "influenza_a_pneumonia",
    "bronchiectasis",
    "emphysema",
    "post_mycoplasma_infection",
    "lung_cancer",
]

RECORDING_CATEGORY = Literal["a", "i", "o", "poem"]

DATA_SOURCE = Literal["charite", "uk_covid"]

DATE_STR_FORMAT = "%Y-%m-%d"


class MetadataRow(BaseModel):
    """Pydantic model for strict metadata row validation."""

    model_config = {"extra": "allow"}

    audio_id: int = Field(gt=0, description="Positive integer audio ID")
    recording_identifier: int = Field(..., description="Unique recording identifier")
    audio_sample_path: str = Field(..., description="Path to audio sample file")

    birth_year: int | None = Field(None, ge=1900, le=2030, description="Birth year")
    height_cm: float | None = Field(None, gt=0, le=300, description="Height in cm")
    weight_kg: float | None = Field(None, gt=0, le=500, description="Weight in kg")

    lung_disease_main: LUNG_DISEASE = Field(..., description="Primary lung disease")
    recording_category: RECORDING_CATEGORY = Field(
        ..., description="Recording category"
    )
    data_source: DATA_SOURCE | None = Field(None, description="Data source identifier")

    longitudinal: int = Field(..., ge=0, le=1, description="Longitudinal flag (0 or 1)")
    date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Date string in YYYY-MM-DD format (REQUIRED - no recordings without dates allowed)",
    )

    # Chunking-related fields
    chunk_number: int | None = Field(
        None,
        ge=1,
        description="Chunk number for split audio files (None for original recordings, 1+ for chunks)",
    )
    is_chunked: bool = Field(
        False,
        description="Flag indicating whether this is a chunked entry (True) or original entry (False)",
    )
    chunk_start_sec: float | None = Field(
        None,
        ge=0,
        description="Start time of chunk in seconds (None for original recordings)",
    )
    chunk_duration_sec: float | None = Field(
        None,
        gt=0,
        description="Duration of chunk in seconds (None for original recordings)",
    )

    @field_validator(
        "height_cm",
        "weight_kg",
        "birth_year",
        "chunk_number",
        "chunk_start_sec",
        "chunk_duration_sec",
        mode="before",
    )
    @classmethod
    def convert_nan_to_none(cls, v):
        """Convert NaN values to None for proper Optional field handling."""
        if pd.isna(v) or (isinstance(v, float) and np.isnan(v)):
            return None
        return v

    @computed_field
    @property
    def computed_recording_identifier(self) -> int:
        """Automatically compute recording identifier from audio_id, category, date, and chunk_number."""
        if self.recording_category is None:
            raise ValueError(
                "recording_category is required to compute recording_identifier"
            )
        return generate_recording_identifier(
            self.audio_id, self.recording_category, self.date, self.chunk_number
        )

    @model_validator(mode="before")
    @classmethod
    def auto_populate_recording_identifier(cls, values):
        """Auto-populate recording_identifier if missing, before validation."""
        if isinstance(values, dict):
            # If recording_identifier is missing or None, compute it
            if values.get("recording_identifier") is None:
                # Need to compute it here since we don't have access to the computed property yet
                audio_id = values.get("audio_id")
                recording_category = values.get("recording_category")
                date = values.get("date")
                chunk_number = values.get("chunk_number")

                if (
                    audio_id is not None
                    and recording_category is not None
                    and date is not None
                ):
                    values["recording_identifier"] = generate_recording_identifier(
                        audio_id, recording_category, date, chunk_number
                    )
        return values

    @model_validator(mode="after")
    def validate_recording_identifier(self):
        """Validate that recording_identifier matches computed value."""
        computed = self.computed_recording_identifier
        if self.recording_identifier != computed:
            raise ValueError(
                f"recording_identifier {self.recording_identifier} doesn't match "
                f"computed value {computed} for audio_id={self.audio_id}, "
                f"category={self.recording_category}, date={self.date}"
            )
        return self

    # # validate that audio_sample_path is a valid file path and not empty
    # @field_validator("audio_sample_path")
    # @classmethod
    # def validate_audio_sample_path(cls, v):
    #     """Ensure audio_sample_path is a valid file path."""
    #     path = Path(v)
    #     if not path.is_file():
    #         raise ValueError(f"audio_sample_path must be a valid file, got {v}")
    #     if path.stat().st_size == 0:
    #         raise ValueError(f"audio_sample_path cannot be an empty file: {v}")
    #     if not str(path).lower().endswith(".wav"):
    #         raise ValueError(f"Unsupported audio file format: {path}")
    #     return str(path)  # Return as string for consistency with other fields


class RecordingIdentifierSet(BaseModel):
    """
    Set of recording identifiers for one audio sample in the feature extraction pipeline.

    Groups the recording identifiers needed to extract features for a single audio_id,
    organized by feature type (Parselmouth vs Wav2Vec). This replaces the cryptic
    nested list structure with a clear, type-safe model.
    """

    audio_id: int = Field(..., description="Source audio ID for traceability")
    parselmouth_ids: list[int] = Field(
        default_factory=list,
        description="Recording identifiers for Parselmouth features",
    )
    wav2vec_ids: list[int] = Field(
        default_factory=list, description="Recording identifiers for Wav2Vec features"
    )

    @property
    def total_recordings(self) -> int:
        """Total number of recording identifiers in this set."""
        return len(self.parselmouth_ids) + len(self.wav2vec_ids)

    @property
    def has_parselmouth(self) -> bool:
        """Whether this set has Parselmouth recording identifiers."""
        return len(self.parselmouth_ids) > 0

    @property
    def has_wav2vec(self) -> bool:
        """Whether this set has Wav2Vec recording identifiers."""
        return len(self.wav2vec_ids) > 0

    def get_all_ids(self) -> list[int]:
        """Get all recording identifiers as a flat list."""
        return self.parselmouth_ids + self.wav2vec_ids
