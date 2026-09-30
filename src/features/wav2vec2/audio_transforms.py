"""
Real-time audio transformations for model inference.

This module provides audio processing functions specifically designed for preparing
audio signals for wav2vec2 model inference. These are lightweight, tensor-based
operations that differ from the batch preprocessing pipeline in utils/preprocessing.py.
"""

from typing import Optional

import soundfile as sf
import torch
import torchaudio
from torch import Tensor


def _load_audio_tensor(path: str) -> tuple[Tensor, int]:
    """Load audio file as (channels, samples) tensor and sample rate. Uses soundfile to avoid torchcodec."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    # soundfile: mono -> (samples,); stereo -> (samples, channels)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    else:
        data = data.T  # (samples, channels) -> (channels, samples)
    signal = torch.from_numpy(data)
    return signal, sr


def preprocess_audio_signal(
    audio_sample_path: str,
    target_sample_rate: int,
    device: str = "cpu",
    num_samples: int | None = None,
    normalize: bool = False,
) -> Tensor:
    """
    Process audio signal with robust loading, resampling, and normalization.

    Args:
        audio_sample_path: Path to audio file
        target_sample_rate: Target sample rate
        device: Device for processing
        num_samples: Optional fixed length for audio
        normalize: Whether to normalize the signal

    Returns:
        Processed audio tensor
    """
    signal, sr = _load_audio_tensor(audio_sample_path)

    transformed_signal = resample_if_necessary(signal, sr, target_sample_rate, device)
    transformed_signal = mix_down_if_necessary(transformed_signal)

    if num_samples:
        transformed_signal = cut_if_necessary(transformed_signal, num_samples)
        transformed_signal = right_pad_if_necessary(transformed_signal, num_samples)

    if transformed_signal.shape[0] == 1:
        transformed_signal = transformed_signal.squeeze(0)

    if normalize:
        transformed_signal = normalize_signal(transformed_signal)

    transformed_signal = transformed_signal.to(device)
    return transformed_signal


def normalize_signal(signal: Tensor) -> Tensor:
    """Normalize signal to [-1, 1] range."""
    max_val: float = float(signal.abs().max())
    if max_val > 0:
        return signal / max_val
    return signal


def cut_if_necessary(signal: Tensor, num_samples: int) -> Tensor:
    """Cut signal to specified length if too long."""
    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    return signal


def right_pad_if_necessary(signal: Tensor, num_samples: int) -> Tensor:
    """Pad signal with zeros if too short."""
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        signal = torch.nn.functional.pad(signal, (0, num_missing_samples))
    return signal


def resample_if_necessary(
    signal: Tensor, sr: int, target_sample_rate: int, device: str
) -> Tensor:
    """Resample signal to target sample rate if necessary."""
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate).to(device)
        signal = resampler(signal)
    return signal


def mix_down_if_necessary(signal: Tensor) -> Tensor:
    """Convert stereo to mono if necessary."""
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal
