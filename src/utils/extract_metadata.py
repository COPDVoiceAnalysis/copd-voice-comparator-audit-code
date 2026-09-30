"""
Metadata extraction CLI tool.
Refactored to use the modular metadata_extraction package.

This module maintains the same command-line interface while delegating
all processing to the well-structured metadata_extraction package.
"""

import argparse
import sys

from src.utils.metadata_extraction import extract_metadata_pipeline_safe
import logging

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for metadata extraction.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process vocal recordings and create a metadata file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover Excel file in directory
  python -m src.utils.extract_metadata -d /path/to/recordings

  # Use specific Excel file
  python -m src.utils.extract_metadata -d /path/to/recordings -e /path/to/metadata.xlsx

  # Specify custom output file
  python -m src.utils.extract_metadata -d /path/to/recordings -o /path/to/output.csv
        """,
    )

    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        required=True,
        help="The directory containing the vocal recordings and audio files.",
    )

    parser.add_argument(
        "--excel_file",
        "-e",
        type=str,
        default=None,
        help="Path to the Excel file containing metadata (optional, will auto-discover if not provided).",
    )

    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default=None,
        help="Path to the output CSV file (optional, defaults to metadata.csv in the directory).",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for metadata extraction CLI.
    """
    args = parse_arguments()

    # Configure logging level
    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

    # Log input parameters
    logger.info("=" * 60)
    logger.info("METADATA EXTRACTION STARTING")
    logger.info("=" * 60)
    logger.info(f"Audio recordings directory: {args.directory}")
    logger.info(f"Excel file: {args.excel_file or 'auto-discover'}")
    logger.info(f"Output file: {args.output_file or 'default (metadata.csv)'}")

    # Run the extraction pipeline
    success = extract_metadata_pipeline_safe(
        vocal_recordings_dir=args.directory,
        excel_file_path=args.excel_file,
        output_file=args.output_file,
    )

    # Report results
    if success:
        logger.info("=" * 60)
        logger.info("✓ METADATA EXTRACTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        output_path = args.output_file or f"{args.directory}/metadata.csv"
        logger.info(f"Metadata file created: {output_path}")
        sys.exit(0)
    else:
        logger.error("=" * 60)
        logger.error("✗ METADATA EXTRACTION FAILED")
        logger.error("=" * 60)
        logger.error("Check the error messages above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
