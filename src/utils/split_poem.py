"""
Auth: Florian Zwicker
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import cast
from pydub import AudioSegment
from src.utils.recording_identifier import generate_recording_identifier


def create_chunk_metadata(
    original_metadata_path: Path, output_dir: Path, chunk_info: list[dict]
) -> None:
    """
    Create updated metadata CSV that includes both original and chunk entries.

    This function keeps original poem entries unchanged and adds new chunk entries,
    allowing for side-by-side comparison between original and chunked audio files.
    Distinguishing fields are added to make filtering easy.

    Args:
        original_metadata_path: Path to the original metadata.csv file
        output_dir: Directory where chunks are stored
        chunk_info: List of dictionaries containing chunk information
    """
    # Load original metadata
    if original_metadata_path.exists():
        original_df: pd.DataFrame = pd.read_csv(original_metadata_path)
        print(f"Loaded original metadata with {len(original_df)} rows")
    else:
        print(f"Warning: Original metadata file not found at {original_metadata_path}")
        original_df = pd.DataFrame()

    # Add distinguishing fields to original metadata if not present
    if not original_df.empty:
        # Add is_chunked flag for original entries
        if "is_chunked" not in original_df.columns:
            original_df["is_chunked"] = False

        # Ensure chunk_number is None for original entries (will be handled by MetadataRow validation)
        if "chunk_number" not in original_df.columns:
            original_df["chunk_number"] = None

        # Add chunk timing fields for consistency
        if "chunk_start_sec" not in original_df.columns:
            original_df["chunk_start_sec"] = None
        if "chunk_duration_sec" not in original_df.columns:
            original_df["chunk_duration_sec"] = None

    # Create new metadata entries for chunks
    chunk_rows: list[dict] = []
    processed_audio_ids: set[int] = set()

    for chunk in chunk_info:
        processed_audio_ids.add(chunk["audio_id"])

        # Find original metadata row for this audio_id and poem recording
        if not original_df.empty:
            original_row = original_df[
                (original_df["audio_id"] == chunk["audio_id"])
                & (original_df["recording_category"] == "poem")
            ]

            if not original_row.empty:
                # Use the first matching row as template to preserve all metadata fields
                base_row = original_row.iloc[0].to_dict()

                # Update with chunk-specific information
                base_row["audio_sample_path"] = str(chunk["chunk_path"])
                base_row["recording_category"] = (
                    "poem"  # Keep as "poem" for consistency
                )
                base_row["chunk_start_sec"] = chunk.get("start_time_sec", None)
                base_row["chunk_duration_sec"] = chunk.get("duration_sec", None)
                base_row["chunk_number"] = chunk["chunk_number"]
                base_row["is_chunked"] = True  # Mark as chunked entry

                # Generate new recording identifier with chunk number
                base_row["recording_identifier"] = generate_recording_identifier(
                    chunk["audio_id"],
                    "poem",
                    base_row["date"],
                    chunk_number=chunk["chunk_number"],
                )

                chunk_rows.append(base_row)
            else:
                print(
                    f"Warning: No original metadata found for audio_id {chunk['audio_id']}"
                )
                # Create minimal metadata entry with required fields
                chunk_rows.append(
                    {
                        "audio_id": chunk["audio_id"],
                        "audio_sample_path": str(chunk["chunk_path"]),
                        "recording_category": "poem",
                        "chunk_start_sec": chunk.get("start_time_sec", None),
                        "chunk_duration_sec": chunk.get("duration_sec", None),
                        "chunk_number": chunk["chunk_number"],
                        "is_chunked": True,
                        "date": chunk.get("date", "1900-01-01"),  # Fallback date
                        "recording_identifier": generate_recording_identifier(
                            chunk["audio_id"],
                            "poem",
                            chunk.get("date", "1900-01-01"),
                            chunk_number=chunk["chunk_number"],
                        ),
                    }
                )
        else:
            # No original metadata available, create minimal entries
            chunk_rows.append(
                {
                    "audio_id": chunk["audio_id"],
                    "audio_sample_path": str(chunk["chunk_path"]),
                    "recording_category": "poem",
                    "chunk_start_sec": chunk.get("start_time_sec", None),
                    "chunk_duration_sec": chunk.get("duration_sec", None),
                    "chunk_number": chunk["chunk_number"],
                    "is_chunked": True,
                    "date": chunk.get("date", "1900-01-01"),  # Fallback date
                    "recording_identifier": generate_recording_identifier(
                        chunk["audio_id"],
                        "poem",
                        chunk.get("date", "1900-01-01"),
                        chunk_number=chunk["chunk_number"],
                    ),
                }
            )

    # Create DataFrame from chunk rows
    chunk_df: pd.DataFrame = pd.DataFrame(chunk_rows)

    # Combine original metadata with chunk entries (keep ALL original entries)
    if not original_df.empty:
        combined_df = pd.concat([original_df, chunk_df], ignore_index=True)
        print(f"Kept all {len(original_df)} original entries")
    else:
        combined_df = chunk_df

    # Deduplicate poem recordings based on key fields to prevent duplicate chunk generation
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(
        subset=["audio_id", "recording_category", "date", "chunk_number"]
    )
    deduplicated_count = len(combined_df)

    if initial_count != deduplicated_count:
        print(
            f"Deduplicated poem recordings: {initial_count} -> {deduplicated_count} (removed {initial_count - deduplicated_count} duplicates)"
        )
    else:
        print(f"No duplicate poem recordings found")

    # Save updated metadata
    output_metadata_path: Path = output_dir / "metadata.csv"
    combined_df.to_csv(output_metadata_path, index=False)
    print(f"Updated metadata saved to: {output_metadata_path}")
    print(f"Total rows in updated metadata: {len(combined_df)}")
    print(f"Original entries: {len(original_df) if not original_df.empty else 0}")
    print(f"Chunk entries added: {len(chunk_df)}")
    print(f"Processed audio_ids: {sorted(processed_audio_ids)}")


def handle_short_audio(
    audio: AudioSegment,
    audio_path: Path,
    output_dir: Path,
    audio_id: int,
) -> tuple[int, list[dict]]:
    """
    Handle audio files that are shorter than the target segment duration.
    Creates a single chunk from the entire audio file.

    Args:
        audio: AudioSegment object
        audio_path: Path to input audio file
        output_dir: Directory where the chunk will be saved
        audio_id: Audio ID from metadata

    Returns:
        count: Number of segments created (always 1)
        chunk_info: List with single chunk information
    """
    base_name: str = audio_path.stem
    total_duration_ms: int = len(audio)

    # Create single chunk from entire audio
    output_file: Path = output_dir / f"{base_name}_01.wav"
    audio.export(output_file, format="wav")

    chunk_info = [
        {
            "audio_id": audio_id,
            "chunk_number": 1,
            "chunk_path": output_file,
            "start_time_sec": 0.0,
            "duration_sec": total_duration_ms / 1000.0,
        }
    ]

    print(f" → Created 1 chunk from short audio ({total_duration_ms / 1000.0:.1f}s)")
    return 1, chunk_info


def split_audio_fixed_interval(
    audio_path: Path,
    output_dir: Path,
    audio_id: int,
    segment_duration_sec: float = 10.0,
    overlap: float = 0.0,
) -> tuple[int, list[dict]]:
    """
    Split audio file into fixed-length segments with smart handling of short audio.

    Args:
        audio_path: Path to input audio file
        output_dir: Directory where the snippets will be saved
        audio_id: Audio ID from metadata (preserves original ID)
        segment_duration_sec: Segment duration in seconds
        overlap: Overlap between segments in seconds

    Returns:
        count: Number of segments created
        chunk_info: List of dictionaries containing chunk information

    Saves the cuts in the output directory with proper naming.

    Smart handling:
    - If entire audio < target duration: creates single chunk
    - If final chunk < 5s: merges with previous chunk
    - Otherwise: normal chunking with 5s minimum threshold
    """
    audio: AudioSegment = AudioSegment.from_file(audio_path)
    base_name: str = audio_path.stem

    total_duration_ms: int = len(audio)
    segment_duration_ms: int = int(segment_duration_sec * 1000)
    overlap_ms: int = int(overlap * 1000)

    # Calculate step size accounting for overlap
    step_size_ms: int = segment_duration_ms - overlap_ms
    if step_size_ms <= 0:
        raise ValueError("Overlap cannot be greater than or equal to segment duration")

    # Use the audio_id passed from metadata instead of extracting from filename
    print(f"Processing audio_id {audio_id} from file: {base_name}")

    # Handle very short audio files (shorter than target segment duration)
    if total_duration_ms < segment_duration_ms:
        return handle_short_audio(audio, audio_path, output_dir, audio_id)

    # Normal chunking process
    count: int = 0
    start_ms: int = 0
    chunk_info: list[dict] = []
    segments_data: list[dict] = []  # Store segment data for potential merging

    while start_ms < total_duration_ms:
        end_ms: int = min(start_ms + segment_duration_ms, total_duration_ms)
        actual_segment_length: int = end_ms - start_ms

        # Check if this is the last segment
        is_last_segment = start_ms + step_size_ms >= total_duration_ms

        # Store segment data
        segment_data = {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "length_ms": actual_segment_length,
            "is_last": is_last_segment,
        }
        segments_data.append(segment_data)

        # Move to next segment position
        start_ms += step_size_ms

    # Process segments with smart final chunk handling
    for i, seg_data in enumerate(segments_data):
        start_ms = seg_data["start_ms"]
        end_ms = seg_data["end_ms"]
        actual_segment_length = seg_data["length_ms"]
        is_last = seg_data["is_last"]

        # Check if segment meets minimum length requirement
        if actual_segment_length >= (segment_duration_ms / 2):
            # Normal chunk - create it
            segment = cast(AudioSegment, audio[start_ms:end_ms])
            output_file = output_dir / f"{base_name}_{count + 1:02d}.wav"
            segment.export(output_file, format="wav")

            chunk_info.append(
                {
                    "audio_id": audio_id,
                    "chunk_number": count + 1,
                    "chunk_path": output_file,
                    "start_time_sec": start_ms / 1000.0,
                    "duration_sec": actual_segment_length / 1000.0,
                }
            )
            count += 1

        elif is_last and count > 0:
            # Short final chunk - merge with previous chunk
            print(
                f"Merging short final chunk ({actual_segment_length / 1000.0:.1f}s) with previous chunk"
            )

            # Remove the last saved chunk info and file
            last_chunk = chunk_info.pop()
            last_file = last_chunk["chunk_path"]
            if last_file.exists():
                last_file.unlink()  # Delete the file

            # Create extended segment from previous chunk start to current end
            extended_start_ms = int(last_chunk["start_time_sec"] * 1000)
            extended_segment = cast(AudioSegment, audio[extended_start_ms:end_ms])

            # Save extended chunk with same filename as the removed chunk
            extended_segment.export(last_file, format="wav")

            # Update chunk info with extended duration
            chunk_info.append(
                {
                    "audio_id": audio_id,
                    "chunk_number": last_chunk["chunk_number"],
                    "chunk_path": last_file,
                    "start_time_sec": extended_start_ms / 1000.0,
                    "duration_sec": (end_ms - extended_start_ms) / 1000.0,
                }
            )

        else:
            # Short intermediate chunk - this shouldn't happen with fixed intervals
            print(
                f"Skipping short intermediate segment ({actual_segment_length / 1000.0:.1f}s)"
            )

    return count, chunk_info


def main() -> None:
    """
    CLI that processes poem recordings based on metadata filtering:
        uv run split_poems.py --metadata /path/to/metadata.csv --output_dir /path/to/output --duration 10.0

    Uses metadata to identify poem recordings instead of filename patterns.
    Saves the splits in output_dir and creates updated metadata CSV with both original and chunk entries.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Split poem audio files into fixed-length segments based on metadata filtering."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to metadata.csv file containing recording information",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to the output directory where segments will be saved.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Segment duration in seconds (default: 6.0)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap between a segment and the prior segment in seconds (default: 0.0)",
    )

    args: argparse.Namespace = parser.parse_args()

    # Validate metadata file exists
    if not args.metadata.exists():
        print(f"Error: Metadata file '{args.metadata}' does not exist.")
        exit(1)

    # Create output_dir if necessary
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata and filter for poem recordings
    print(f"Loading metadata from: {args.metadata}")
    metadata_df = pd.read_csv(args.metadata)
    print(f"Loaded metadata with {len(metadata_df)} total rows")

    # Filter for poem recordings and deduplicate
    poem_recordings = metadata_df[metadata_df["recording_category"] == "poem"].copy()
    print(f"Found {len(poem_recordings)} poem recordings before deduplication")

    if poem_recordings.empty:
        print("No poem recordings found in metadata.")
        return

    # Collect all chunk information
    all_chunk_info: list[dict] = []
    processed_files = 0
    skipped_files = 0

    for _, row in poem_recordings.iterrows():
        audio_path = Path(row["audio_sample_path"])

        # Check if audio file exists
        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            skipped_files += 1
            continue

        print(f"Processing: {audio_path.name}")
        try:
            count, chunk_info = split_audio_fixed_interval(
                audio_path,
                args.output_dir,
                row["audio_id"],
                args.duration,
                args.overlap,
            )

            # Add date information to chunk_info for identifier generation
            for chunk in chunk_info:
                chunk["date"] = row["date"]

            print(f" → Created {count} segment(s)")
            all_chunk_info.extend(chunk_info)
            processed_files += 1

        except Exception as e:
            print(f"Error processing {audio_path.name}: {e}")
            skipped_files += 1
            continue

    print(f"\nProcessing complete:")
    print(f"  Files processed: {processed_files}")
    print(f"  Files skipped: {skipped_files}")
    print(f"  Total chunks created: {len(all_chunk_info)}")

    # Create updated metadata with chunk information
    if all_chunk_info:
        print(
            f"\nCreating updated metadata with {len(all_chunk_info)} chunk entries..."
        )
        create_chunk_metadata(args.metadata, args.output_dir, all_chunk_info)
    else:
        print("No chunks were created, skipping metadata generation.")


if __name__ == "__main__":
    main()
