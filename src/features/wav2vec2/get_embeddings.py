"""
Wav2Vec2 embeddings extraction orchestrator.

This module coordinates the overall workflow for extracting wav2vec2 embeddings
from audio files using GPU processing. It imports specialized modules for
audio transforms, model inference, and embedding storage.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import logging
from src.utils.logging_config import setup_basic_logging
from src.models.metadata import Metadata
from src.features.wav2vec2.wav2vec_model import (
    process_single_recording,
    load_wav2vec_model,
)
from src.models.data_models import EmbeddingCollection

logger = logging.getLogger(__name__)


def process_embeddings_gpu(
    metadata_handler: Metadata, embeddings_dir: str, sample_rate: int
) -> int:
    """
    Memory-efficient GPU processing that aggregates embeddings in memory and saves only mean/std files.
    Uses Pydantic models for type safety and validation.

    Args:
        metadata_handler: Metadata handler for the dataset
        embeddings_dir: Output directory for embeddings
        sample_rate: Audio sample rate

    Returns:
        Number of recordings successfully processed.
    """
    model_name = "facebook/wav2vec2-large-robust"

    logger.info("Starting GPU processing for wav2vec2 embeddings")

    feature_extractor, model = load_wav2vec_model(model_name)

    recordings = {}
    total_samples = len(metadata_handler)
    processed_count = 0

    logger.info(f"Starting processing: {total_samples} samples")

    # Process each metadata row individually
    for sample_idx, metadata_row in enumerate(metadata_handler.iter_metadata_rows()):
        try:
            if (
                sample_idx % 10 == 0 or sample_idx < 5
            ):  # Log first few and every 10th sample
                logger.info(
                    f"Processing sample {sample_idx + 1}/{total_samples} ({(sample_idx + 1) / total_samples * 100:.1f}%)"
                )

            recording_embedding = process_single_recording(
                metadata_row, feature_extractor, model, sample_rate
            )

            # Store results in aggregation structure
            recordings[recording_embedding.recording_identifier] = recording_embedding
            processed_count += 1

        except Exception as e:
            logger.error(
                f"Error processing sample {sample_idx}: {e}", exc_info=True
            )
            continue

    logger.info(
        f"GPU processing completed: {processed_count} samples processed successfully"
    )

    embedding_collection = EmbeddingCollection(recordings=recordings)
    embedding_collection.save_to_directory(embeddings_dir)
    return processed_count


def get_wav2vec2_embeddings(
    embeddings_dir: str,
    metadata_file: str,
    sample_rate: int = 16000,
) -> None:
    """
    Extract wav2vec2 embeddings from audio files using validated metadata on GPU.

    Args:
        embeddings_dir: Output directory for embeddings
        metadata_file: Path to metadata CSV file
        sample_rate: Audio sample rate (default: 16000)
    """
    logger = logging.getLogger(__name__)

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This function requires GPU processing.")
        raise RuntimeError(
            "GPU not available. Please ensure CUDA is installed and a GPU is accessible."
        )

    os.makedirs(embeddings_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING WAV2VEC2 EMBEDDINGS EXTRACTION (GPU-ONLY)")
    logger.info("=" * 80)

    metadata_handler = Metadata(Path(metadata_file))

    logger.info(f"GPU available: {torch.cuda.get_device_name()}")

    total_samples = len(metadata_handler)
    logger.info(f"Total samples to process: {total_samples}")

    if total_samples == 0:
        logger.warning("No samples found in metadata")
        raise ValueError("Metadata is empty. Please check your metadata file.")

    processed_count = process_embeddings_gpu(
        metadata_handler,
        embeddings_dir,
        sample_rate,
    )

    if processed_count == 0:
        raise RuntimeError(
            "No recordings were processed successfully. All samples failed. "
            "Check logs for per-sample errors (e.g. missing audio, CUDA, paths)."
        )

    logger.info("=" * 80)
    logger.info("EMBEDDINGS EXTRACTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    setup_basic_logging()

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Extract wav2vec2 embeddings from audio files using GPU."
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Directory to save extracted embedding files.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate of the audio files. Default: 16000 Hz.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Path to the metadata CSV file.",
    )

    args = parser.parse_args()

    try:
        get_wav2vec2_embeddings(
            embeddings_dir=args.embeddings_dir,
            sample_rate=args.sample_rate,
            metadata_file=args.metadata_file,
        )
    except Exception as e:
        logger.critical(f"Fatal error during embeddings extraction: {e}", exc_info=True)
        sys.exit(1)
