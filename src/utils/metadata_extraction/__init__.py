"""
Metadata extraction package.
Provides a clean, modular approach to extracting metadata from Excel files and audio recordings.
"""

from src.utils.metadata_extraction.pipeline import (
    extract_metadata_pipeline,
    extract_metadata_pipeline_safe,
)

# Public API - only expose the main pipeline functions
__all__ = [
    "extract_metadata_pipeline",
    "extract_metadata_pipeline_safe",
]
