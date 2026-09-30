"""
Main parselmouth feature extractor orchestrator.

This module provides both single-file and batch processing capabilities
for parselmouth feature extraction with multiprocessing support.
"""

import os
import json
import time
import argparse
import multiprocessing
import numpy as np
import parselmouth
from pathlib import Path
from typing import Any

# Per-file retry configuration for transient failures (I/O, memory pressure, etc.)
_MAX_RETRIES = 3
_RETRY_DELAY_SEC = 2.0

# Per-file timeout in seconds for parselmouth feature extraction.
# Protects against hangs on problematic audio where pitch detection
# can loop indefinitely. Set to 600s to accommodate long recordings
# (poems up to ~160s) that require multiple expensive extraction steps.
_EXTRACTION_TIMEOUT_SEC = 600

from src.features.parselmouth.opensmile_features import extract_egemaps_features
import logging
from src.models.metadata import Metadata
from src.features.parselmouth.audio_features import (
    extract_mfcc_features,
    extract_pitch_features,
    extract_hnr_features,
    extract_intensity_features,
)
from src.features.parselmouth.voice_quality import (
    extract_jitter_shimmer_features,
    extract_cpp_features,
    extract_dsi_features,
)
from src.features.parselmouth.spectral_features import extract_spectral_features
from src.features.parselmouth.temporal_features import extract_temporal_features
from src.features.parselmouth.formant_features import extract_formant_features
from src.features.parselmouth.utils import create_feature_array, features_dict_to_array

logger = logging.getLogger(__name__)


def extract_parselmouth_features(
    audio_path: str,
    pitch_floor: float = 40.0,
    pitch_ceiling: float = 650.0,
    time_step: float = 0.01,
    num_mfcc: int = 30,
    n_fft: int = 512,
) -> dict[str, float]:
    """
    Extract a comprehensive set of acoustic features combining eGeMAPS and custom Parselmouth features.

    This function extracts ALL available features from both eGeMAPS and custom Parselmouth implementations.
    Feature selection is handled during the loading process using ParselmouthConfig.

    Features extracted:
      - eGeMAPS v02 feature set (88 features including MFCCs, F0, HNR, jitter, shimmer, formants, etc.)
      - Custom semitone pitch statistics not in eGeMAPS
      - Custom spectral features (alpha ratio, Hammarberg index, slopes, flux)
      - Custom temporal features (voiced durations, syllable rate)
      - Voice quality measures (CPP, DSI)

    Args:
        audio_path: Path to the audio file
        pitch_floor: Minimum pitch in Hz
        pitch_ceiling: Maximum pitch in Hz
        time_step: Time step for analysis
        num_mfcc: Number of MFCC coefficients (legacy parameter, now handled by eGeMAPS)
        n_fft: FFT size for spectral analysis

    Returns:
        Flat dictionary with feature names as keys and values as floats
    """
    logger.debug(f"Processing: {os.path.basename(audio_path)}")

    try:
        sound = parselmouth.Sound(audio_path)

        # Extract all feature groups - no conditionals, always extract everything
        all_features = {}

        # 1. eGeMAPS features (complete 88-feature set including MFCCs)
        egemaps_features = extract_egemaps_features(audio_path)
        all_features.update(egemaps_features)

        # Note: F0, HNR, jitter, shimmer, intensity, and formant features are now
        # included in eGeMAPS, so we skip the overlapping parselmouth extractions

        # Keep only unique parselmouth features not covered by eGeMAPS:
        # - Custom pitch statistics (semitone features, percentiles)
        # - Custom formant bandwidths (if not in eGeMAPS)

        # Extract minimal pitch features for DSI calculation and custom statistics
        pitch_features = extract_pitch_features(
            sound, pitch_floor, pitch_ceiling, 7999, time_step
        )
        # Only keep custom semitone features not in eGeMAPS
        custom_pitch_features = {
            k: v
            for k, v in pitch_features.items()
            if k.startswith("semitone_") or k == "perc_unvoiced_frames"
        }
        all_features.update(custom_pitch_features)

        # Keep f0_mean for DSI calculation (will be overridden by eGeMAPS version)
        all_features["f0_mean"] = pitch_features.get("f0_mean", 0.0)

        # Extract intensity for DSI calculation (will be overridden by eGeMAPS version)
        intensity_features = extract_intensity_features(sound)
        all_features["intensity_mean"] = intensity_features.get("intensity_mean", 0.0)

        # Extract jitter for DSI calculation (will be overridden by eGeMAPS version)
        jitter_shimmer_features = extract_jitter_shimmer_features(
            sound, pitch_floor, pitch_ceiling
        )
        all_features["jitter_ddp"] = jitter_shimmer_features.get("jitter_ddp", 0.0)

        # Spectral features
        spectral_features = extract_spectral_features(audio_path, time_step, n_fft)
        all_features.update(spectral_features)

        # Temporal features (needs pitch values)
        pitch = sound.to_pitch(
            time_step=time_step,
            pitch_floor=pitch_floor,
            pitch_ceiling=pitch_ceiling,
        )
        pitch_values = pitch.selected_array["frequency"]
        temporal_features = extract_temporal_features(sound, pitch_values, time_step)
        all_features.update(temporal_features)

        # CPP features (always extract)
        cpp_features = extract_cpp_features(sound, audio_path)
        all_features.update(cpp_features)

        # DSI features (always extract)
        dsi_features = extract_dsi_features(
            sound,
            all_features["f0_mean"],
            all_features.get("intensity_mean", 0.0),
            all_features["jitter_ddp"],
        )
        all_features.update(dsi_features)

        # custom MFCCs
        mfcc_features = extract_mfcc_features(sound, num_mfcc)
        all_features.update(mfcc_features)

        # Return the complete feature dictionary - feature names are the keys!
        return all_features

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {str(e)}")
        raise


import signal


class _ExtractionTimeout(Exception):
    """Raised when SIGALRM fires during feature extraction."""
    pass


def _sigalrm_handler(signum, frame):
    raise _ExtractionTimeout("SIGALRM timeout")


def _extract_single_direct(args_tuple, timeout_sec):
    """Extract features directly in the current process with SIGALRM timeout.

    Fast path: no child process overhead. SIGALRM works for pure Python and
    most C extensions. If a C extension blocks signals, the alarm fires after
    the C call returns — the file just takes longer than timeout_sec but the
    worker is NOT stuck forever.

    Returns a result dict with recording_identifier, features, success, error.
    """
    audio_path, recording_identifier, pitch_floor, pitch_ceiling, time_step, num_mfcc, n_fft = args_tuple

    old_handler = signal.signal(signal.SIGALRM, _sigalrm_handler)
    signal.alarm(timeout_sec)
    try:
        features = extract_parselmouth_features(
            audio_path, pitch_floor, pitch_ceiling, time_step, num_mfcc, n_fft
        )
        signal.alarm(0)  # cancel alarm
        return {
            "recording_identifier": recording_identifier,
            "features": features,
            "success": True,
        }
    except _ExtractionTimeout:
        return {
            "recording_identifier": recording_identifier,
            "features": None,
            "success": False,
            "error": f"timed out after {timeout_sec}s (SIGALRM)",
        }
    except Exception as e:
        signal.alarm(0)
        return {
            "recording_identifier": recording_identifier,
            "features": None,
            "success": False,
            "error": str(e),
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _extract_file_worker(args_tuple, result_queue):
    """Target function for per-file child process. Must be top-level for pickling.

    Used only in retry pass for files that failed with SIGALRM timeout,
    where the C extension may be blocking signals.
    """
    audio_path, _rec_id, pitch_floor, pitch_ceiling, time_step, num_mfcc, n_fft = args_tuple
    try:
        features = extract_parselmouth_features(
            audio_path, pitch_floor, pitch_ceiling, time_step, num_mfcc, n_fft
        )
        result_queue.put({"features": features, "success": True})
    except Exception as e:
        result_queue.put({"features": None, "success": False, "error": str(e)})


def _extract_one_with_process_timeout(args_tuple, timeout_sec):
    """Extract features via a child process with hard kill timeout.

    Expensive (full process spawn), but can kill C-extension hangs.
    Used only for retries of files that failed the SIGALRM timeout.
    """
    recording_identifier = args_tuple[1]
    result_queue = multiprocessing.Queue(maxsize=1)

    proc = multiprocessing.Process(
        target=_extract_file_worker,
        args=(args_tuple, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        return {
            "recording_identifier": recording_identifier,
            "features": None,
            "success": False,
            "error": f"timed out after {timeout_sec}s (hard kill)",
        }

    try:
        result = result_queue.get_nowait()
        result["recording_identifier"] = recording_identifier
        return result
    except Exception:
        return {
            "recording_identifier": recording_identifier,
            "features": None,
            "success": False,
            "error": "worker process died without returning a result",
        }


def _extract_batch(batch_args_tuple):
    """Worker function for ProcessPoolExecutor. Processes a batch of files.

    Uses fast SIGALRM timeout (no child process overhead). The worker stays
    alive across the whole batch, keeping expensive imports loaded.

    Args:
        batch_args_tuple: (list_of_file_args, timeout_sec)

    Returns:
        List of result dicts, one per file in the batch.
    """
    file_args_list, timeout_sec = batch_args_tuple
    pid = os.getpid()
    results = []
    for i, file_args in enumerate(file_args_list):
        rec_id = file_args[1]
        t0 = time.monotonic()
        result = _extract_single_direct(file_args, timeout_sec)
        elapsed = time.monotonic() - t0
        status = "ok" if result["success"] else f"FAILED: {result.get('error', '?')}"
        print(
            f"[worker {pid}] [{i+1}/{len(file_args_list)}] {rec_id}: {status} ({elapsed:.0f}s)",
            flush=True,
        )
        results.append(result)
    return results


def _extract_batch_with_process_timeout(batch_args_tuple):
    """Retry worker: processes a batch using child-process timeout per file.

    Slower but can hard-kill C-extension hangs. Used only for retries.
    """
    file_args_list, timeout_sec = batch_args_tuple
    pid = os.getpid()
    results = []
    for i, file_args in enumerate(file_args_list):
        rec_id = file_args[1]
        t0 = time.monotonic()
        result = _extract_one_with_process_timeout(file_args, timeout_sec)
        elapsed = time.monotonic() - t0
        status = "ok" if result["success"] else f"FAILED: {result.get('error', '?')}"
        print(
            f"[retry worker {pid}] [{i+1}/{len(file_args_list)}] {rec_id}: {status} ({elapsed:.0f}s)",
            flush=True,
        )
        results.append(result)
    return results


def process_parselmouth_features_batch(
    output_file: str,
    metadata_file: str,
    pitch_floor: float = 40.0,
    pitch_ceiling: float = 650.0,
    time_step: float = 0.01,
    num_mfcc: int = 30,
    n_fft: int = 512,
    remove_zero_variance: bool = True,
    max_files: int | None = None,
    max_workers: int | None = None,
    batch_size: int = 25,
) -> dict[str, Any]:
    """
    Extract parselmouth features from multiple audio files in parallel.

    Architecture: ProcessPoolExecutor with batched workers.
    - Files are split into batches of `batch_size` (default 25).
    - Each worker processes one batch sequentially, keeping expensive imports
      (parselmouth, opensmile, librosa) loaded across all files in the batch.
    - Within each batch, individual files get a hard per-file timeout via a
      child Process, so C-extension hangs can be killed reliably.
    - Failed files are collected and retried in a separate pass.

    Args:
        output_file: Path to save the feature array (.npy file)
        metadata_file: Path to metadata CSV file
        pitch_floor: Minimum pitch in Hz
        pitch_ceiling: Maximum pitch in Hz
        time_step: Time step for analysis
        num_mfcc: Number of MFCC coefficients
        n_fft: FFT size for spectral analysis
        remove_zero_variance: Whether to remove zero-variance features
        max_files: Maximum number of files to process (for testing)
        max_workers: Maximum number of parallel workers (defaults to half of CPU count)
        batch_size: Number of files per worker batch (default 25)

    Returns:
        Dictionary with processing results
    """
    # Use Metadata class for validated metadata loading
    metadata_handler = Metadata(Path(metadata_file))
    logger.info(f"Loaded and validated metadata with {len(metadata_handler)} rows")

    # Collect extraction parameters (no selection parameters - always extract all)
    extraction_params = {
        "pitch_floor": pitch_floor,
        "pitch_ceiling": pitch_ceiling,
        "time_step": time_step,
        "num_mfcc": num_mfcc,
        "n_fft": n_fft,
    }

    # Collect all files
    all_files = []
    for i, metadata_row in enumerate(metadata_handler.iter_metadata_rows()):
        if max_files and i >= max_files:
            logger.info(f"Processing {i} files, stopping after {max_files} for testing")
            break
        all_files.append(
            (metadata_row.audio_sample_path, metadata_row.recording_identifier)
        )

    if max_workers is None:
        # Respect SLURM allocation: cpu_count() sees all node CPUs, but we
        # may only have a subset allocated. Check SLURM env vars for actual
        # allocation (try multiple vars for compatibility).
        available_cpus = None
        for env_var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
            val = os.environ.get(env_var)
            if val is not None:
                available_cpus = int(val)
                logger.info(f"Detected {available_cpus} CPUs from {env_var}")
                break
        if available_cpus is None:
            available_cpus = multiprocessing.cpu_count()
            logger.info(f"No SLURM CPU env vars found, using cpu_count()={available_cpus}")
        # Main pass uses SIGALRM (no child processes), so workers = CPUs.
        # Retry pass spawns child processes but has very few files.
        max_workers = max(1, available_cpus)

    # Build task args for each file
    task_args = [
        (audio_path, rec_id, pitch_floor, pitch_ceiling, time_step, num_mfcc, n_fft)
        for audio_path, rec_id in all_files
    ]

    # Split into batches
    batches = []
    for i in range(0, len(task_args), batch_size):
        batch = task_args[i : i + batch_size]
        batches.append((batch, _EXTRACTION_TIMEOUT_SEC))

    logger.info(
        f"Processing {len(all_files)} files in {len(batches)} batches "
        f"(batch_size={batch_size}) with {max_workers} workers, "
        f"timeout={_EXTRACTION_TIMEOUT_SEC}s per file, "
        f"retries={_MAX_RETRIES}"
    )

    records = []
    failed_files = []

    def _run_batches(batch_list, attempt, worker_fn=_extract_batch):
        """Run a list of batches through the pool and collect results."""
        batch_results = []
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Each batch takes up to batch_size * timeout seconds worst case.
        # Set a generous future timeout to avoid blocking forever.
        max_batch_files = max(len(b[0]) for b in batch_list)
        future_timeout = (max_batch_files * _EXTRACTION_TIMEOUT_SEC) + 120

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch_idx = {}
            for batch_idx, batch_tuple in enumerate(batch_list):
                future = executor.submit(worker_fn, batch_tuple)
                future_to_batch_idx[future] = batch_idx

            completed = 0
            for future in as_completed(future_to_batch_idx, timeout=future_timeout):
                batch_idx = future_to_batch_idx[future]
                completed += 1
                try:
                    results = future.result(timeout=10)
                    batch_results.extend(results)
                except Exception as e:
                    # Entire batch failed — mark all files in batch as failed
                    batch_files = batch_list[batch_idx][0]
                    for file_args in batch_files:
                        batch_results.append({
                            "recording_identifier": file_args[1],
                            "features": None,
                            "success": False,
                            "error": f"batch-level error: {e}",
                        })

                n_ok = sum(1 for r in batch_results if r["success"])
                n_fail = len(batch_results) - n_ok
                logger.info(
                    f"  Attempt {attempt}: batch {completed}/{len(batch_list)} done "
                    f"(total: {n_ok} ok, {n_fail} failed)"
                )

        return batch_results

    # --- Main extraction pass ---
    all_results = _run_batches(batches, attempt=1)

    for result in all_results:
        if result["success"]:
            records.append({
                "recording_identifier": result["recording_identifier"],
                "features": result["features"],
            })
        else:
            failed_files.append(
                (result["recording_identifier"], result.get("error", "unknown"))
            )

    logger.info(
        f"Pass 1 complete: {len(records)} ok, {len(failed_files)} failed"
    )

    # --- Retry passes for failed files ---
    for attempt in range(2, _MAX_RETRIES + 1):
        if not failed_files:
            break

        logger.info(
            f"Retry pass {attempt}/{_MAX_RETRIES}: "
            f"retrying {len(failed_files)} failed files..."
        )

        # Build retry task args — lookup by rec_id
        rec_id_to_args = {args[1]: args for args in task_args}
        retry_args = [
            rec_id_to_args[rec_id]
            for rec_id, _error in failed_files
            if rec_id in rec_id_to_args
        ]

        # Re-batch the retries (smaller batches for retries)
        retry_batch_size = max(1, batch_size // 5)
        retry_batches = []
        for i in range(0, len(retry_args), retry_batch_size):
            batch = retry_args[i : i + retry_batch_size]
            retry_batches.append((batch, _EXTRACTION_TIMEOUT_SEC))

        # Retries use child-process timeout that can hard-kill C-extension hangs
        retry_results = _run_batches(
            retry_batches, attempt=attempt,
            worker_fn=_extract_batch_with_process_timeout,
        )

        # Partition into successes and failures
        still_failed = []
        for result in retry_results:
            if result["success"]:
                records.append({
                    "recording_identifier": result["recording_identifier"],
                    "features": result["features"],
                })
                logger.info(
                    f"Retry succeeded for {result['recording_identifier']} "
                    f"on attempt {attempt}"
                )
            else:
                still_failed.append(
                    (result["recording_identifier"], result.get("error", "unknown"))
                )

        failed_files = still_failed
        logger.info(
            f"Retry pass {attempt} complete: "
            f"{len(records)} total ok, {len(failed_files)} still failed"
        )

    # Convert failed_files from list of tuples to list of rec_ids for reporting
    final_failed_rec_ids = [rec_id for rec_id, _error in failed_files]
    if failed_files:
        logger.error(
            f"Permanently failed recordings: "
            + ", ".join(f"{rec_id} ({error})" for rec_id, error in failed_files)
        )

    logger.info(
        f"Processing complete: {len(records)} successful, "
        f"{len(final_failed_rec_ids)} failed"
    )

    if not records:
        raise ValueError("No files were successfully processed")

    # Fail-fast: abort if any files failed after retries.
    # Silently skipping failures leads to missing recording IDs in the feature
    # file, which causes downstream KeyErrors during training.
    if final_failed_rec_ids:
        raise ValueError(
            f"Feature extraction failed for {len(final_failed_rec_ids)} recording(s) "
            f"after {_MAX_RETRIES} retries each: {final_failed_rec_ids}. "
            f"Total expected: {len(all_files)}, successful: {len(records)}. "
            f"Aborting to prevent incomplete feature files."
        )

    # Create feature array with zero-variance filtering and metadata
    feature_array, metadata = create_feature_array(
        records, extraction_params, remove_zero_variance
    )

    # Persist failure/success info into metadata for debuggability
    metadata["total_expected"] = len(all_files)
    metadata["total_processed"] = len(records)
    metadata["total_failed"] = len(final_failed_rec_ids)
    metadata["failed_recordings"] = final_failed_rec_ids

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save feature array
    np.save(output_file, feature_array)
    logger.info(f"Saved {len(records)} feature records to {output_file}")

    # Save metadata as JSON
    metadata_output_file = output_file.replace(".npy", "_metadata.json")
    with open(metadata_output_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_output_file}")

    return {
        "output_file": output_file,
        "metadata_file": metadata_output_file,
        "total_processed": len(records),
        "total_failed": len(final_failed_rec_ids),
        "feature_dimension": metadata["n_features_final"],
        "features_removed": len(metadata["removed_features"]),
        "failed_recordings": final_failed_rec_ids,
    }


def main():
    """Command-line interface for batch parselmouth feature extraction."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Extract Parselmouth features in batch with multiprocessing."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output .npy file path to save features.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Path to the metadata CSV file.",
    )
    parser.add_argument(
        "--pitch_floor",
        type=float,
        default=40.0,
        help="Pitch floor in Hz. Default: 40.0",
    )
    parser.add_argument(
        "--pitch_ceiling",
        type=float,
        default=650.0,
        help="Pitch ceiling in Hz. Default: 650.0",
    )
    parser.add_argument(
        "--time_step",
        type=float,
        default=0.01,
        help="Time step for analysis. Default: 0.01",
    )
    parser.add_argument(
        "--num_mfcc",
        type=int,
        default=14,
        help="Number of MFCC coefficients. Default: 14",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=512,
        help="FFT size for spectral analysis. Default: 512",
    )
    parser.add_argument(
        "--no_zero_variance_filtering",
        action="store_false",
        dest="remove_zero_variance",
        help="Disable zero-variance feature filtering.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers. Default: half of CPU count",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=25,
        help="Number of files per worker batch. Default: 25",
    )

    parser.set_defaults(remove_zero_variance=True)

    args = parser.parse_args()

    result = process_parselmouth_features_batch(
        output_file=args.output_file,
        metadata_file=args.metadata_file,
        pitch_floor=args.pitch_floor,
        pitch_ceiling=args.pitch_ceiling,
        time_step=args.time_step,
        num_mfcc=args.num_mfcc,
        n_fft=args.n_fft,
        remove_zero_variance=args.remove_zero_variance,
        max_files=args.max_files,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )

    logger.info(f"Feature extraction completed: {result}")


if __name__ == "__main__":
    main()
