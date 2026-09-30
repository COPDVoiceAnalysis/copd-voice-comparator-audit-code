import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, lfilter, resample_poly
import noisereduce as nr
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.logging_config import setup_basic_logging

# Setup basic logging when module is imported
setup_basic_logging()


class Preprocessor:
    """
    Handles audio preprocessing tasks including loading, noise reduction, normalization, and saving processed audio files.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        preemp_coef: float | None = None,
        apply_noise_reduction: bool = False,
        sample_rate: int = 16000,
        normalize_audio: bool = True,
        lowcut_freq: float = 40,
        highcut_freq: float | None = None,  # Will be set to sample_rate/2 - 1 if None
        filter_order: int = 5,
        trim_top_db: int = 20,
        noise_decrease_static: float = 0.0,
        noise_decrease_dynamic: float = 0.0,
    ) -> None:
        # Read parameters from kwargs
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.preemp_coef = preemp_coef
        self.apply_noise_reduction = apply_noise_reduction
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.lowcut_freq = lowcut_freq
        # Set highcut_freq to sample_rate/2 - 1 if None
        self.highcut_freq = (
            highcut_freq if highcut_freq is not None else (sample_rate / 2 - 1)
        )
        self.filter_order = filter_order
        self.trim_top_db = trim_top_db
        self.noise_decrease_static = noise_decrease_static
        self.noise_decrease_dynamic = noise_decrease_dynamic

        if not self.input_dir or not self.output_dir:
            raise ValueError("Config must specify input_dir and output_dir.")
        if not self.apply_noise_reduction and any(
            [
                self.noise_decrease_static != 0.0,
                self.noise_decrease_dynamic != 0.0,
            ]
        ):
            raise ValueError(
                "Noise reduction parameters are set but apply_noise_reduction is False."
            )
        if self.noise_decrease_static < 0 or self.noise_decrease_dynamic < 0:
            raise ValueError("Noise reduction parameters must be non-negative.")
        if self.lowcut_freq < 0 or self.highcut_freq < 0:
            raise ValueError("Lowcut and highcut frequencies must be non-negative.")

        # Validate filter frequencies against Nyquist frequency
        nyquist_freq = self.sample_rate / 2
        if self.lowcut_freq >= nyquist_freq:
            raise ValueError(
                f"Lowcut frequency ({self.lowcut_freq}) must be less than Nyquist frequency ({nyquist_freq})."
            )
        if self.highcut_freq >= nyquist_freq:
            raise ValueError(
                f"Highcut frequency ({self.highcut_freq}) must be less than Nyquist frequency ({nyquist_freq})."
            )
        if self.lowcut_freq >= self.highcut_freq:
            raise ValueError("Lowcut frequency must be less than highcut frequency.")
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_audio(
        self, path: str, target_sr: int, dtype: type = np.float32
    ) -> tuple[np.ndarray, int]:
        """Load audio with soundfile and resample to target_sr if needed. Returns (audio, sr)."""
        data, sr = sf.read(path, dtype=dtype)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = resample_poly(data, target_sr, sr).astype(dtype)
            sr = target_sr
        return data, sr

    def noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply noise reduction to an audio signal using the Wiener filter.

        This method reduces noise in an audio signal by applying the reduce_noise() method from the noisereduce library for varying and static noise consecutively.

        Args:
            audio (numpy.ndarray): The input audio signal to be processed. It should
                be a 1-dimensional array representing the audio waveform.
            sr (int): The sampling rate.

        Returns:
            numpy.ndarray: The noise-reduced audio signal. The returned signal is
                also a 1-dimensional array with the same shape as the input.
        """
        if self.noise_decrease_static == 0 and self.noise_decrease_dynamic == 0:
            logging.debug("No noise reduction applied.")
            return audio

        # Pre-emphasis-filter / Highpassfilter to elevate high frequency signals.
        if self.preemp_coef:
            audio = lfilter([1, -self.preemp_coef], [1], audio).astype(np.float32)

        # adaptive noise reduction by utilizing short-time fourier transform and masking of the noise spektrum.
        # Retransformation by inverse short-time fourier transrofm
        # -> non-stationary noise reduction
        reduced_noise = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=self.noise_decrease_dynamic,
            stationary=False,
            n_fft=4096,
            win_length=1024,
            hop_length=256,
            n_std_thresh_stationary=1.5,
        )

        # same type of noise reduction as above but for stationary noise, therefore stationary=True.
        reduced_noise = nr.reduce_noise(
            y=reduced_noise,
            sr=sr,
            prop_decrease=self.noise_decrease_static,
            stationary=True,
            n_fft=2048,
            win_length=1024,
            hop_length=256,
            n_std_thresh_stationary=1.0,
        )
        return reduced_noise

    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim leading/trailing silence using frame RMS (no librosa)."""
        top_db, frame_length, hop_length = self.trim_top_db, 2048, 512
        if len(audio) == 0:
            return audio
        ref = np.max(np.abs(audio))
        if ref <= 0:
            return audio
        threshold = ref * (10 ** (-top_db / 20.0))
        n_frames = max(0, (len(audio) - frame_length) // hop_length + 1)
        first, last = None, None
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end > len(audio):
                break
            frame = audio[start:end]
            rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
            if rms >= threshold:
                if first is None:
                    first = i
                last = i
        if first is None or last is None:
            return audio
        start_sample = first * hop_length
        end_sample = last * hop_length + frame_length
        audio = audio[start_sample:end_sample]
        return audio

    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self.normalize_audio:
            peak = np.max(np.abs(audio))
            audio = (audio / peak) if peak > 0 else audio

        audio = self.trim_silence(audio)

        # Butterworth via SOS-Filter (Second-Order Sections)
        # The order of a filter indicatess how steep the filter is at the transition form passband to stopband.
        sos = None
        if self.lowcut_freq and self.highcut_freq:
            sos = butter(
                self.filter_order,
                [self.lowcut_freq, self.highcut_freq],
                btype="band",
                fs=sr,
                output="sos",
            )
        elif self.lowcut_freq and not self.highcut_freq:
            sos = butter(
                self.filter_order,
                self.lowcut_freq,
                btype="highpass",
                fs=sr,
                output="sos",
            )
        elif self.highcut_freq and not self.lowcut_freq:
            raise ValueError(
                "Either lowcut or bandpass filter should be used must be specified. Highcut is not supported."
            )
        else:
            pass
        if sos is not None:
            result = sosfilt(sos, audio)
            # to prevent type errors, check if result is a tuple
            audio = result[0] if isinstance(result, tuple) else result

        if self.noise_decrease_dynamic != 0 or self.noise_decrease_static != 0:
            audio = self.noise_reduction(audio, sr)

        if self.normalize_audio:
            peak = np.max(np.abs(audio))
            audio = (audio / peak) if peak > 0 else audio
        return audio

    def _process_file(self, file_name: str) -> str:
        """Worker for one .wav file: load → preprocess_audio → write."""
        in_path = os.path.join(self.input_dir, file_name)
        out_path = os.path.join(self.output_dir, file_name)

        try:
            # Check file size first
            file_size = os.path.getsize(in_path)
            if file_size == 0:
                raise ValueError(f"Empty file (0 bytes): {file_name}")

            # Load audio
            audio, sr = self._load_audio(in_path, self.sample_rate, np.float32)

            # Check if audio data is valid
            if len(audio) == 0:
                raise ValueError(f"No audio data found in: {file_name}")

            # Process and save
            processed = self.preprocess_audio(audio, int(sr))
            sf.write(out_path, processed, sr)
            return file_name

        except Exception as e:
            # Log specific error for this file
            logging.warning(f"Skipping corrupted file {file_name}: {str(e)[:100]}")
            # Re-raise so the main loop can handle it consistently
            raise

    def preprocess_directory(self, max_workers: int | None = None) -> None:
        """
        Parallel bulk processing of all .wav files in input_dir.
        `max_workers=None` defaults to os.cpu_count().
        """
        wavs: list[str] = [
            f for f in os.listdir(self.input_dir) if f.lower().endswith(".wav")
        ]

        logging.info(f"Found {len(wavs)} .wav files to process")

        successful_files: list[str] = []
        failed_files: list[tuple[str, str]] = []

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(self._process_file, fn): fn for fn in wavs}
            for fut in as_completed(futures):
                fn = futures[fut]
                try:
                    done = fut.result()
                    logging.debug(f"✔ {done}")
                    successful_files.append(done)
                except Exception as e:
                    logging.error(f"✘ {fn} failed: {e}")
                    failed_files.append((fn, str(e)))
                    exe.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(
                        f"Processing failed for {fn}: {e}. Aborting on first error."
                    )

        self.update_metadata()

        # Summary
        logging.info("\nProcessing Summary:")
        logging.info(f"   Successful: {len(successful_files)}")
        logging.info(f"   Failed: {len(failed_files)}")

        if failed_files:
            logging.warning("\nFailed files:")
            for filename, error in failed_files:
                logging.warning(f"   {filename}: {error[:100]}")
            raise RuntimeError(
                f"Processing failed for {len(failed_files)} files. See details above."
            )

    def preprocess_from_metadata(
        self, metadata_file: str, max_workers: int | None = None
    ) -> None:
        """
        Process audio files listed in a metadata CSV file.
        Handles both relative and absolute paths in the metadata.
        """
        import pandas as pd

        logging.info(f"Loading metadata from {metadata_file}")
        metadata_df = pd.read_csv(metadata_file)

        if "audio_sample_path" not in metadata_df.columns:
            raise ValueError("Metadata file must contain 'audio_sample_path' column")

        # Get list of audio files to process
        audio_files = []
        for _, row in metadata_df.iterrows():
            audio_path = row["audio_sample_path"]

            # Only accept absolute paths
            if not os.path.isabs(audio_path):
                raise ValueError(
                    f"Audio path must be absolute, got relative path: {audio_path}"
                )

            input_path = audio_path
            filename = os.path.basename(audio_path)

            if os.path.exists(input_path):
                audio_files.append((input_path, filename))
            else:
                logging.warning(f"Audio file not found: {input_path}")

        logging.info(f"Found {len(audio_files)} audio files to process from metadata")

        successful_files: list[str] = []
        failed_files: list[tuple[str, str]] = []

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {
                exe.submit(self._process_metadata_file, input_path, filename): filename
                for input_path, filename in audio_files
            }
            for fut in as_completed(futures):
                fn = futures[fut]
                try:
                    done = fut.result()
                    logging.debug(f"✔ {done}")
                    successful_files.append(done)
                except Exception as e:
                    logging.error(f"✘ {fn} failed: {e}")
                    failed_files.append((fn, str(e)))
                    exe.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(
                        f"Processing failed for {fn}: {e}. Aborting on first error."
                    )

        # Copy and update metadata
        self.copy_and_update_metadata(metadata_file)

        # Summary
        logging.info("\nProcessing Summary:")
        logging.info(f"   Successful: {len(successful_files)}")
        logging.info(f"   Failed: {len(failed_files)}")

        if failed_files:
            logging.warning("\nFailed files:")
            for filename, error in failed_files:
                logging.warning(f"   {filename}: {error[:100]}")
            raise RuntimeError(
                f"Processing failed for {len(failed_files)} files. See details above."
            )

    def _process_metadata_file(self, input_path: str, filename: str) -> str:
        """Worker for processing a file from metadata with custom input path."""
        output_filename = os.path.basename(filename)
        out_path = os.path.join(self.output_dir, output_filename)

        try:
            # Check file size first
            file_size = os.path.getsize(input_path)
            if file_size == 0:
                raise ValueError(f"Empty file (0 bytes): {filename}")

            # Load audio
            audio, sr = self._load_audio(input_path, self.sample_rate, np.float32)

            # Check if audio data is valid
            if len(audio) == 0:
                raise ValueError(f"No audio data found in: {filename}")

            # Process and save
            processed = self.preprocess_audio(audio, int(sr))
            sf.write(out_path, processed, sr)
            return filename

        except Exception as e:
            # Log specific error for this file
            logging.warning(f"Skipping corrupted file {filename}: {str(e)[:100]}")
            # Re-raise so the main loop can handle it consistently
            raise

    def copy_and_update_metadata(self, metadata_file: str) -> None:
        """
        Copy metadata file to output directory and update paths.
        """
        import pandas as pd

        metadata_new = os.path.join(self.output_dir, "metadata.csv")

        # Load metadata
        df = pd.read_csv(metadata_file)

        # Update audio_sample_path to point to processed files
        if "audio_sample_path" in df.columns:
            df["audio_sample_path"] = df["audio_sample_path"].apply(
                lambda x: os.path.join(self.output_dir, os.path.basename(x))
            )

        # Save updated metadata
        df.to_csv(metadata_new, index=False)
        logging.info(f"Metadata updated and saved to {metadata_new}")

    def update_metadata(self) -> None:
        """
        Update metadata for the processed audio files.
        """
        # replace the old directory with the new one in all file paths in the metadata.csv
        metadata_old = os.path.join(self.input_dir, "metadata.csv")
        metadata_new = os.path.join(self.output_dir, "metadata.csv")
        if os.path.exists(metadata_old):
            with open(metadata_old) as f:
                lines = f.readlines()

            with open(metadata_new, "w") as f:
                for line in lines:
                    # Replace the old directory path with the new one
                    new_line = line.replace(self.input_dir, self.output_dir)
                    f.write(new_line)
            logging.info(f"Metadata updated and saved to {metadata_new}")
        else:
            raise FileNotFoundError(f"Metadata file {metadata_old} not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run standalone preprocessing.")

    # Add individual arguments for each parameter
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory for unprocessed audio files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for processed audio files.",
    )
    parser.add_argument(
        "--preemp_coef",
        type=float,
        help="Pre-emphasis coefficient. (class default: None)",
    )
    parser.add_argument(
        "--apply_noise_reduction",
        action="store_true",
        help="Apply noise reduction. If this flag is present, noise reduction is applied. (class default: False)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize audio. (class default: True)",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable audio normalization.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        help="Sample rate for audio processing. (class default: 16000)",
    )
    parser.add_argument(
        "--lowcut_freq",
        type=float,
        help="Lowcut frequency for bandpass/highpass filter. (class default: 40)",
    )
    parser.add_argument(
        "--highcut_freq",
        type=float,
        help="Highcut frequency for bandpass filter. (class default: sample_rate/2 - 1)",
    )
    parser.add_argument(
        "--filter_order",
        type=int,
        help="Order of the Butterworth filter. (class default: 5)",
    )
    parser.add_argument(
        "--trim_top_db",
        type=int,
        help="The threshold (in dB) below reference to consider silent for trimming. (class default: 20)",
    )
    parser.add_argument(
        "--noise_decrease_static",
        type=float,
        help="Proportion to decrease static noise. (class default: 0.0)",
    )
    parser.add_argument(
        "--noise_decrease_dynamic",
        type=float,
        help="Proportion to decrease dynamic noise. (class default: 0.0)",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        help="Path to metadata CSV file. If provided, only files listed in this metadata will be processed.",
    )

    args = parser.parse_args()

    init_kwargs = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "preemp_coef": args.preemp_coef,
        "apply_noise_reduction": args.apply_noise_reduction,
        "normalize_audio": args.normalize,
        "sample_rate": args.sample_rate,
        "lowcut_freq": args.lowcut_freq,
        "highcut_freq": args.highcut_freq,
        "filter_order": args.filter_order,
        "trim_top_db": args.trim_top_db,
        "noise_decrease_static": args.noise_decrease_static,
        "noise_decrease_dynamic": args.noise_decrease_dynamic,
    }

    # Filter out None values to use class defaults
    filtered_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    preprocessor = Preprocessor(**filtered_kwargs)

    # Use metadata-based processing if metadata file is provided
    if args.metadata_file:
        preprocessor.preprocess_from_metadata(args.metadata_file)
    else:
        # Deprecated
        preprocessor.preprocess_directory()
