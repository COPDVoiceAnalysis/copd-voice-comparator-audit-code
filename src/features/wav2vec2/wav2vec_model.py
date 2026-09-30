"""
Wav2Vec2 model management and inference.

This module handles model loading, GPU setup, and the core inference logic
for extracting embeddings from audio signals using wav2vec2.
"""

import logging
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from src.models.data_models import MetadataRow, RecordingEmbedding, LayerEmbedding
from src.features.wav2vec2.audio_transforms import preprocess_audio_signal

logger = logging.getLogger(__name__)


def process_single_recording(
    metadata_row: MetadataRow,
    feature_extractor: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
    sample_rate: int,
) -> RecordingEmbedding:
    """
    Core processing logic for a single recording on GPU using Pydantic models.

    Args:
        metadata_row: Validated metadata for the recording
        feature_extractor: Wav2Vec2 feature extractor
        model: Wav2Vec2 model
        sample_rate: Target sample rate

    Returns:
        RecordingEmbedding with validated layer embeddings
    """
    device = "cuda"

    # Load and process audio file
    signal = preprocess_audio_signal(
        metadata_row.audio_sample_path,
        target_sample_rate=sample_rate,
        device="cpu",  # Keep on CPU for feature extractor compatibility
        normalize=False,
    )

    # Extract metadata
    recording_identifier = metadata_row.recording_identifier
    recording_category = metadata_row.recording_category

    # Prepare input for feature extractor
    input_values = feature_extractor(
        signal.numpy(),  # Feature extractor expects numpy on CPU
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).input_values.to(device)

    # Model inference
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    # Process each layer's embeddings using Pydantic models
    layer_embeddings = {}

    for layer_idx, hidden_state in enumerate(hidden_states):
        # hidden_state shape: [1, sequence_length, hidden_size]
        # Compute temporal mean and std over sequence dimension
        layer_mean = hidden_state.mean(dim=1).squeeze(0).cpu().numpy()  # [hidden_size]
        layer_std = hidden_state.std(dim=1).squeeze(0).cpu().numpy()  # [hidden_size]

        # Create validated LayerEmbedding
        layer_embeddings[layer_idx] = LayerEmbedding(
            layer_idx=layer_idx, mean=layer_mean, std=layer_std
        )

    # Create and return validated RecordingEmbedding
    return RecordingEmbedding(
        recording_identifier=recording_identifier,
        recording_category=recording_category,
        layer_embeddings=layer_embeddings,
    )


def load_wav2vec_model(
    model_name: str = "facebook/wav2vec2-large-robust",
) -> tuple[Wav2Vec2FeatureExtractor, Wav2Vec2Model]:
    """
    Load wav2vec2 model and feature extractor on GPU.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Tuple of (feature_extractor, model)
    """
    device = "cuda"

    logger.info(f"Loading model {model_name} on GPU...")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model_instance = Wav2Vec2Model.from_pretrained(model_name)
    model = model_instance.to(device)  # type: ignore[arg-type]
    model.eval()

    logger.info(f"Model loaded successfully on {device}")

    return feature_extractor, model
