"""
Simplified metadata handling using Pydantic models.
Replaces complex manual validation in metadata.py.
"""

from itertools import zip_longest
import pandas as pd
import numpy as np
from typing import Any
from collections.abc import Iterator
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from src.models.data_models import (
    MetadataRow,
    TaskConfig,
    MetadataSummary,
)
from src.models.core_models import RecordingIdentifierSet
import logging

logger = logging.getLogger(__name__)

# Reference year for age calculation (birth_year -> age).
REFERENCE_YEAR = 2025


class Metadata:
    """
    Simplified metadata handler using Pydantic models.
    Replaces the complex Metadata class with cleaner validation and processing.
    """

    def __init__(
        self,
        csv_path: Path,
        ps_covariates_file: Path | None = None,
    ):
        """
        Initialize metadata with hardcoded, stable requirements.

        Args:
            csv_path: Path to metadata CSV file
            ps_covariates_file: Optional path to ps_covariates.csv; first column = ID, rest = covariate
                columns (header = names). Merged into metadata (only columns not already present).
                Missing file or merge errors are logged and skipped.
        """
        self.csv_path = Path(csv_path)
        self.ps_covariates_file = (
            Path(ps_covariates_file) if ps_covariates_file else None
        )
        self.df: pd.DataFrame = pd.DataFrame()
        self.original_row_count = 0
        self.filtering_history: list[dict[str, Any]] = []
        self.le: LabelEncoder | None = None
        self._ps_covariate_columns: list[str] = []

        logger.info(f"Loading metadata from: {csv_path}")
        self._load_and_process()

    def _load_and_process(self):
        """Load and process metadata with improved validation approach."""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.original_row_count = len(self.df)
            logger.info(f"Loaded {self.original_row_count} rows")

            # Sort by audio_id, recording_category, date, and chunk_number to ensure
            # consistent ordering for feature vector construction (grouping by date).
            # This ensures that when multiple recording types are used, recordings from
            # the same date are grouped together during transposition in get_recording_identifier_sets.
            sort_columns = ["audio_id", "recording_category", "date"]
            if "chunk_number" in self.df.columns:
                sort_columns.append("chunk_number")
            self.df = self.df.sort_values(sort_columns, na_position="last").reset_index(drop=True)
            logger.debug("Sorted DataFrame by audio_id, recording_category, date, chunk_number")

            self._standardize_missing_values()
            self._clean_and_convert_types()
            self._merge_ps_covariates_if_requested()
            self._validate_data_quality()

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise

    def _standardize_missing_values(self):
        """Standardize missing value representations."""
        # Replace various missing value representations
        missing_representations = ["nan", "NaN", "NAN", "", " ", "null", "NULL"]

        for col in self.df.select_dtypes(include=["object"]).columns:
            self.df[col] = self.df[col].replace(missing_representations, np.nan)

        logger.debug("Standardized missing value representations")

    def _clean_and_convert_types(self):
        """Clean and convert data types, failing immediately on problems."""
        logger.debug("Cleaning and converting data types")

        # Convert audio_id to integer (handle float representations like 123.0)
        try:
            # Convert to numeric first, which handles string representations
            self.df["audio_id"] = pd.to_numeric(self.df["audio_id"], errors="coerce")

            # Check for NaN values after conversion (invalid audio IDs)
            invalid_mask = self.df["audio_id"].isna()
            if invalid_mask.any():
                invalid_count = invalid_mask.sum()
                invalid_rows = self.df[invalid_mask].index.tolist()[:5]
                raise ValueError(
                    f"Found {invalid_count} invalid audio IDs that cannot be converted to numbers. "
                    f"First few problematic rows: {invalid_rows}"
                )

            # Check for non-integer values (e.g., 123.5)
            non_integer_mask = self.df["audio_id"] % 1 != 0
            if non_integer_mask.any():
                non_integer_count = non_integer_mask.sum()
                non_integer_rows = self.df[non_integer_mask].index.tolist()[:5]
                raise ValueError(
                    f"Found {non_integer_count} non-integer audio IDs. "
                    f"First few problematic rows: {non_integer_rows}"
                )

            # Convert to integer
            self.df["audio_id"] = self.df["audio_id"].astype(int)

            # Check for non-positive audio IDs
            non_positive_mask = self.df["audio_id"] <= 0
            if non_positive_mask.any():
                non_positive_count = non_positive_mask.sum()
                non_positive_rows = self.df[non_positive_mask].index.tolist()[:5]
                raise ValueError(
                    f"Found {non_positive_count} non-positive audio IDs. "
                    f"First few problematic rows: {non_positive_rows}"
                )

            logger.debug("Successfully converted audio_id to integers")

        except Exception as e:
            raise ValueError(f"Failed to process audio IDs: {e}")

        numeric_columns = ["birth_year", "height_cm", "weight_kg", "longitudinal"]
        for col in numeric_columns:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                except Exception as e:
                    raise ValueError(f"Failed to process column '{col}': {e}")

    def _merge_ps_covariates_if_requested(self) -> None:
        """Load optional ps_covariates CSV and left-merge on ID. Only columns not already in metadata are merged.
        Convention: first column = ID, rest = covariate columns (header = names). On missing file or error, log warning and skip."""
        if self.ps_covariates_file is None:
            return
        path = Path(self.ps_covariates_file)
        if not path.exists():
            logger.warning(
                "ps_covariates_file set but file does not exist: %s; continuing without ps_covariates.",
                path,
            )
            return
        try:
            ps_df = pd.read_csv(path)
            if len(ps_df.columns) < 2:
                logger.warning(
                    "ps_covariates file %s has fewer than 2 columns; skipping merge.",
                    path,
                )
                return
            id_col = ps_df.columns[0]
            covariate_cols = list(ps_df.columns[1:])
            # Only merge columns that are not already in metadata (avoid overwriting age, sex, bmi, etc.)
            merge_cols = [c for c in covariate_cols if c not in self.df.columns]
            # Columns already present in metadata still count as PS covariates
            already_present = [c for c in covariate_cols if c in self.df.columns]
            if not merge_cols:
                # All covariate columns already in metadata (e.g. UK comorbidities
                # are inline in the matched metadata CSV).  Register them as PS
                # covariate columns so overlap_covariate_columns_simplified picks
                # them up — no merge needed.
                self._ps_covariate_columns = already_present
                logger.info(
                    "ps_covariates from %s: all %d covariate columns already in metadata; registered as PS covariates (no merge needed).",
                    path,
                    len(already_present),
                )
                return
            # Resolve merge key: file's first column must match audio_id or study_id in metadata
            merge_on = None
            if id_col == "audio_id" and "audio_id" in self.df.columns:
                merge_on = "audio_id"
            elif id_col == "study_id" and "study_id" in self.df.columns:
                merge_on = "study_id"
            elif "audio_id" in self.df.columns:
                ps_df = ps_df.rename(columns={id_col: "audio_id"})
                id_col = "audio_id"
                merge_on = "audio_id"
            elif "study_id" in self.df.columns:
                ps_df = ps_df.rename(columns={id_col: "study_id"})
                id_col = "study_id"
                merge_on = "study_id"
            if merge_on is None:
                logger.warning(
                    "ps_covariates file first column %r and metadata have no common ID (audio_id or study_id); skipping merge.",
                    id_col,
                )
                return
            use = [merge_on] + merge_cols
            ps_sub = ps_df[use].copy()
            self.df = self.df.merge(ps_sub, on=merge_on, how="left", suffixes=("", "_ps"))
            for c in list(self.df.columns):
                if c.endswith("_ps"):
                    self.df.drop(columns=[c], inplace=True)
            for c in merge_cols:
                if c in self.df.columns:
                    self.df[c] = pd.to_numeric(self.df[c], errors="coerce").fillna(0)
            self._ps_covariate_columns = already_present + merge_cols
            logger.info(
                "Merged ps_covariates from %s (merged: %s, already present: %s).",
                path,
                merge_cols,
                already_present,
            )
        except Exception as e:
            logger.warning(
                "Failed to load or merge ps_covariates from %s: %s; continuing without ps_covariates.",
                path,
                e,
                exc_info=False,
            )

    @property
    def overlap_covariate_columns(self) -> list[str]:
        """Column names used for overlap/PS weighting: age and sex (from metadata) plus merged ps_covariates columns."""
        return ["age", "sex"] + self._ps_covariate_columns

    @property
    def overlap_covariate_columns_simplified(self) -> list[str]:
        """Covariates for PS overlap weighting: age, sex, plus individual ps_covariate columns.

        Uses the individual covariate columns (e.g. breathing, metabolic, ...
        for Charité; angina, asthma, diabetes, ... for UK) rather than an
        aggregated comorbidity_count_bin.  This avoids cross-dataset aggregation
        incompatibilities while keeping ESS stable (verified empirically).
        """
        return ["age", "sex"] + self._ps_covariate_columns

    def _validate_data_quality(self):
        """Validate data quality using Pydantic models - fail immediately on errors."""
        logger.debug("Validating data quality with Pydantic")

        validation_errors = []

        for idx, row in self.df.iterrows():
            try:
                # Try to validate with Pydantic
                MetadataRow(**row.to_dict())
            except Exception as e:
                validation_errors.append(f"Row {idx}: {str(e)}")

        if validation_errors:
            raise ValueError(
                f"Pydantic validation failed for {len(validation_errors)} sample rows:\n"
                + "\n".join(validation_errors[:3])
            )

        logger.debug("Data quality validation passed")

        # Ensure recording identifiers exist before validation
        self._ensure_recording_identifiers()

        # Validate recording identifier uniqueness
        if not self.validate_identifier_uniqueness():
            raise ValueError("Recording identifier uniqueness validation failed")

    def _ensure_recording_identifiers(self):
        """
        Ensure recording_identifier column exists, generating it if missing.
        This provides backward compatibility for CSV files without identifiers.
        """
        if "recording_identifier" not in self.df.columns:
            logger.warning(
                "No recording_identifier column found, generating identifiers..."
            )

            # Check if we have the required columns to generate identifiers
            required_columns = ["audio_id", "recording_category", "date"]
            missing_columns = [
                col for col in required_columns if col not in self.df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Cannot generate recording identifiers: missing required columns {missing_columns}"
                )

            # Generate identifiers using the existing method
            self.add_recording_identifiers()
        else:
            logger.debug("Recording identifier column found")

    def _record_filtering(self, operation: str, before: int, after: int):
        """Record filtering operation for history tracking."""
        self.filtering_history.append(
            {
                "operation": operation,
                "rows_before": before,
                "rows_after": after,
                "rows_removed": before - after,
            }
        )

    def filter_complete_recording_categories(
        self, required_categories: list[str]
    ) -> "Metadata":
        """
        Filter to audio IDs that have all required recording categories.

        Args:
            required_categories: List of recording categories that must be present

        Returns:
            Self for method chaining
        """
        if not required_categories:
            logger.warning(
                "No required categories specified, skipping complete recordings filter"
            )
            return self

        logger.info(
            f"Filtering for complete recordings with categories: {required_categories}"
        )
        initial_count = len(self.df)

        if "recording_category" not in self.df.columns:
            raise ValueError(
                "recording_category column required for complete recordings filter"
            )

        # Group by audio_id and check which categories are available
        audio_categories = self.df.groupby("audio_id")["recording_category"].apply(set)

        # Find audio IDs that have all required categories
        complete_audio_ids = []
        for audio_id, available_categories in audio_categories.items():
            if set(required_categories).issubset(available_categories):
                complete_audio_ids.append(audio_id)

        # Filter to only complete audio IDs
        self.df = self.df[self.df["audio_id"].isin(complete_audio_ids)]

        removed_count = initial_count - len(self.df)
        logger.info(
            f"Filtered for complete recordings: {initial_count} -> {len(self.df)} (removed {removed_count})"
        )
        self._record_filtering(
            "filter_complete_recordings", initial_count, len(self.df)
        )
        return self

    def filter_by_data_sources(self, included_sources: list) -> "Metadata":
        """
        Filter metadata to include only specified data sources.

        Args:
            included_sources: List of data source identifiers to include

        Returns:
            Self for method chaining
        """
        if not included_sources:
            logger.warning("No data sources specified, keeping all data")
            return self

        logger.info(f"Filtering by data sources: {included_sources}")
        initial_count = len(self.df)

        if "data_source" not in self.df.columns:
            logger.warning(
                "No data_source column found in metadata. "
                "All data will be included regardless of included_sources setting."
            )
            return self

        # Log current data source distribution
        current_distribution = self.df["data_source"].value_counts().to_dict()
        logger.info(f"Current data source distribution: {current_distribution}")

        # Filter to included sources
        self.df = self.df[self.df["data_source"].isin(included_sources)]

        # Log filtering results
        final_count = len(self.df)
        removed_count = initial_count - final_count
        
        if final_count > 0:
            final_distribution = self.df["data_source"].value_counts().to_dict()
            logger.info(f"Final data source distribution: {final_distribution}")
        
        logger.info(
            f"Data source filtering complete: {initial_count} -> {final_count} "
            f"(removed {removed_count} rows, {removed_count / initial_count * 100:.1f}%)"
        )

        self._record_filtering(
            f"filter_by_data_sources({included_sources})",
            initial_count,
            final_count,
        )

        return self

    def filter_by_task(self, task_config: TaskConfig) -> "Metadata":
        """
        Apply task-specific filtering using validated TaskConfig.

        Args:
            task_config: Validated TaskConfig instance

        Returns:
            Self for method chaining
        """
        logger.info(f"Applying task filtering: {task_config}")
        initial_count = len(self.df)

        # Apply data source filtering first
        self.filter_by_data_sources(task_config.included_data_sources)

        # Apply longitudinal filtering
        if task_config.exclude_longitudinal:
            self.df = self.df[self.df["longitudinal"] == 0]
            logger.info(
                f"Excluded longitudinal recordings: {initial_count} -> {len(self.df)}"
            )
        elif task_config.only_longitudinal:
            self.df = self.df[self.df["longitudinal"] == 1]
            logger.info(
                f"Filtered to longitudinal only: {initial_count} -> {len(self.df)}"
            )

        # Apply splitting configuration (integrated into task filtering)
        self.filter_by_splitting_config(task_config.use_split_poems)

        # Handle temporal vs standard classification
        if not task_config.use_temporal_pairs:
            # Standard classification - set class labels
            if task_config.target_column not in self.df.columns:
                raise ValueError(
                    f"Target column '{task_config.target_column}' not found"
                )

            self.df["class_label"] = self.df[task_config.target_column]

            # Filter to target classes
            if task_config.target_classes:
                before_class_filter = len(self.df)
                self.df = self.df[
                    self.df["class_label"].isin(task_config.target_classes)
                ]
                logger.info(
                    f"Filtered to target classes: {before_class_filter} -> {len(self.df)}"
                )

                # Validate we have all target classes
                available_classes = set(self.df["class_label"].unique())
                missing_classes = set(task_config.target_classes) - available_classes
                if missing_classes:
                    raise ValueError(f"Missing target classes: {missing_classes}")

        self._record_filtering("filter_by_task", initial_count, len(self.df))

        # Encode labels if we have class_label column
        if "class_label" in self.df.columns:
            self._encode_class_labels()

        return self

    def _encode_class_labels(self):
        """Encode class labels using LabelEncoder."""
        self.le = LabelEncoder()
        self.df["class_idx"] = self.le.fit_transform(self.df["class_label"])
        logger.info(
            f"Encoded class labels: {dict(zip(self.le.classes_, range(len(self.le.classes_))))}"
        )

    def get_unique_audio_ids(self) -> np.ndarray:
        """Get unique audio IDs as validated integers."""
        if len(self.df) == 0:
            return np.array([], dtype=int)

        unique_ids = self.df["audio_id"].sort_values().unique()
        return unique_ids.astype(int)

    def get_metadata_for_audio_id_sequence(
        self,
        audio_ids: np.ndarray,
        columns: list[str],
    ) -> pd.DataFrame:
        """
        Return a DataFrame with one row per audio_id (same order as audio_ids),
        containing only the requested columns. Used for overlap-weight covariates.

        - Row i is the first metadata row with audio_id == audio_ids[i].
        - 'age' is derived from birth_year (REFERENCE_YEAR - birth_year).
        - 'sex' is numeric 0 (f/w) / 1 (m).
        - Missing columns or NaN are filled: 0 for discrete, median for continuous (age, bmi).
        """
        if len(audio_ids) == 0:
            return pd.DataFrame(columns=columns)

        ref_year = REFERENCE_YEAR
        rows = []
        for aid in audio_ids:
            match = self.df[self.df["audio_id"] == aid]
            if len(match) == 0:
                rows.append({c: np.nan for c in columns})
                continue
            row = match.iloc[0].to_dict()
            out = {}
            for c in columns:
                if c == "age":
                    by = row.get("birth_year")
                    out[c] = (
                        (ref_year - float(by))
                        if pd.notna(by)
                        else np.nan
                    )
                elif c == "sex":
                    s = row.get("sex")
                    if pd.isna(s):
                        out[c] = np.nan
                    else:
                        out[c] = 1 if str(s).strip().lower() in ("m", "male") else 0
                elif c == "comorbidity_count_bin" and self._ps_covariate_columns:
                    raw_sum = sum(
                        float(row.get(x, 0) or 0) for x in self._ps_covariate_columns
                    )
                    out[c] = 0 if raw_sum == 0 else (1 if raw_sum <= 2 else 2)
                elif c in row and c in self.df.columns:
                    out[c] = row[c]
                else:
                    out[c] = np.nan
            rows.append(out)
        meta = pd.DataFrame(rows, columns=columns)
        # Fill NaN: median for continuous (age, bmi), 0 for discrete
        continuous = ["age", "bmi"]
        for c in columns:
            if c not in meta.columns:
                continue
            if c in continuous and meta[c].notna().any():
                meta[c] = pd.to_numeric(meta[c], errors="coerce")
                meta[c] = meta[c].fillna(meta[c].median())
            meta[c] = pd.to_numeric(meta[c], errors="coerce")
            meta[c] = meta[c].fillna(0)
        return meta

    def get_class_labels(self) -> np.ndarray:
        """Get class labels with optional encoding."""
        if "class_label" not in self.df.columns:
            raise ValueError("No class_label column found")

        # Deduplicate by audio_id
        deduplicated_df = self.df.drop_duplicates("audio_id").sort_values("audio_id")

        return deduplicated_df["class_label"].to_numpy()

    def get_class_labels_encoded(self) -> tuple[np.ndarray, LabelEncoder]:
        """Get class labels with optional encoding."""
        if "class_label" not in self.df.columns:
            raise ValueError("No class_label column found")
        if self.le is None:
            raise ValueError(
                "LabelEncoder not fitted - call _encode_class_labels() first"
            )

        # Deduplicate by audio_id
        deduplicated_df = self.df.drop_duplicates("audio_id").sort_values("audio_id")

        return deduplicated_df["class_idx"].to_numpy(), self.le

    def get_class_labels_for_sets(self, recording_identifier_sets_list: list) -> tuple[np.ndarray, LabelEncoder]:
        """
        Get class labels for RecordingIdentifierSets without deduplication.
        
        Args:
            recording_identifier_sets_list: List of RecordingIdentifierSet objects
            
        Returns:
            Tuple of (class_labels_encoded, label_encoder)
            - class_labels_encoded: Array of encoded class labels, one per RecordingIdentifierSet
            - label_encoder: The fitted LabelEncoder
        """
        if "class_label" not in self.df.columns:
            raise ValueError("No class_label column found")
        if self.le is None:
            raise ValueError(
                "LabelEncoder not fitted - call _encode_class_labels() first"
            )

        # Create mapping from audio_id to class_idx
        df = self.to_dataframe()
        audio_id_to_class_idx = df.drop_duplicates("audio_id").set_index("audio_id")["class_idx"].to_dict()
        
        # Map to RecordingIdentifierSets (duplicating for chunks)
        class_labels = np.array([audio_id_to_class_idx[rs.audio_id] for rs in recording_identifier_sets_list])
        
        return class_labels, self.le

    def summary(self) -> MetadataSummary:
        """Get metadata summary using Pydantic model."""
        if len(self.df) == 0:
            return MetadataSummary(
                total_rows=0,
                original_rows=self.original_row_count,
                unique_audio_ids=0,
                columns=[],
                class_distribution=None,
                missing_values={},
                filtering_history=self.filtering_history,
            )

        # Class distribution
        class_distribution = None
        if "class_label" in self.df.columns:
            value_counts = self.df["class_label"].value_counts()
            class_distribution = {str(k): int(v) for k, v in value_counts.items()}

        # Missing values for key columns
        key_columns = [
            "audio_id",
            "birth_year",
            "height_cm",
            "weight_kg",
            "lung_disease_main",
        ]
        missing_values = {}
        for col in key_columns:
            if col in self.df.columns:
                missing_values[col] = int(self.df[col].isna().sum())

        return MetadataSummary(
            total_rows=len(self.df),
            original_rows=self.original_row_count,
            unique_audio_ids=len(self.df["audio_id"].unique()),
            columns=list(self.df.columns),
            class_distribution=class_distribution,
            missing_values=missing_values,
            filtering_history=self.filtering_history,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return copy of current dataframe."""
        return self.df.copy()

    def iter_metadata_rows(self) -> Iterator[MetadataRow]:
        """
        Iterate over metadata as validated MetadataRow objects.

        Yields:
            MetadataRow: Validated Pydantic model for each row
        """
        for _, row in self.df.iterrows():
            yield MetadataRow(**row.to_dict())

    def __len__(self) -> int:
        """Return number of rows."""
        return len(self.df)

    def validate_identifier_uniqueness(self) -> bool:
        """
        Validate that all recording identifiers in the metadata are unique.

        Returns:
            True if all identifiers are unique, False otherwise
        """
        from src.utils.recording_identifier import validate_recording_identifiers

        try:
            validate_recording_identifiers(self.df)
            return True
        except ValueError:
            return False

    def add_recording_identifiers(self) -> "Metadata":
        """
        Add recording_identifier column to metadata DataFrame.

        Returns:
            Self for method chaining
        """
        from src.utils.recording_identifier import generate_recording_identifier

        logger.debug("Adding recording identifiers to metadata")

        # Check that all rows have valid dates
        if "date" not in self.df.columns:
            raise ValueError(
                "Date column is required to generate recording identifiers. "
                "All recordings must have dates - phantom IDs not allowed."
            )

        # Generate recording identifiers (all rows now guaranteed to have dates)
        self.df["recording_identifier"] = self.df.apply(
            lambda row: generate_recording_identifier(
                row["audio_id"],
                row["recording_category"],
                row["date"],
            ),
            axis=1,
        )

        # Validate uniqueness
        if not self.validate_identifier_uniqueness():
            raise ValueError("Generated recording identifiers are not unique!")

        logger.debug("Successfully added recording identifiers")
        return self

    def decode_identifier(self, recording_identifier: int) -> dict[str, Any] | None:
        """
        Decode recording identifier back to its components using metadata lookup.

        This is a reverse lookup function that finds the metadata row matching
        the recording identifier and returns the component information.

        Args:
            recording_identifier: The unique recording identifier

        Returns:
            Dictionary with audio_id, recording_category, and date, or None if not found
        """
        try:
            # Find the row with matching recording_identifier
            matching_rows = self.df[
                self.df["recording_identifier"] == recording_identifier
            ]

            if len(matching_rows) == 0:
                return None

            row = matching_rows.iloc[0]

            return {
                "audio_id": row["audio_id"],
                "recording_category": row["recording_category"],
                "date": row.get("date", None),
                "longitudinal": row.get("longitudinal", 0),
            }

        except Exception:
            return None

    def get_recording_identifier_sets(
        self, feature_config
    ) -> list[RecordingIdentifierSet]:
        """
        Generate RecordingIdentifierSet objects for all unique audio IDs after filtering.

        This is the main method for getting recording identifiers in a type-safe format.

        Args:
            feature_config: FeatureConfig specifying which categories to use for each feature type

        Returns:
            List of RecordingIdentifierSet objects, one per unique audio_id
        """
        from src.utils.recording_identifier import generate_recording_identifier

        # Get all unique audio IDs after filtering
        audio_ids = self.get_unique_audio_ids()

        result = []

        for audio_id in audio_ids:
            # Generate parselmouth identifiers
            parsel_ids_by_category = []  # each entry is a list of parsel IDs for one RecordingIdentifierSet, e.g. [["a1", "a2"], ["poem1", "poem2"]]
            if feature_config.recordings_parsel:
                for category in feature_config.recordings_parsel:
                    cat_rec_parsel_ids = []
                    # Look up the date for this audio_id and category
                    matching_rows = self.df[
                        (self.df["audio_id"] == audio_id)
                        & (self.df["recording_category"] == category)
                    ]
                    for _, row in matching_rows.iterrows():
                        date = str(row["date"])
                        chunk_number = row.get("chunk_number", None)
                        cat_rec_id = generate_recording_identifier(
                            audio_id, category, date, chunk_number
                        )
                        cat_rec_parsel_ids.append(cat_rec_id)
                    parsel_ids_by_category.append(cat_rec_parsel_ids)

            # Generate wav2vec identifiers
            wav2vec_ids_by_category = []  # each entry is a list of parsel IDs for one RecordingIdentifierSet, e.g. [["a1", "a2"], ["poem1", "poem2"]]
            if feature_config.recordings_wav2vec2:
                for category in feature_config.recordings_wav2vec2:
                    cat_rec_w2v_ids = []
                    # Look up the date for this audio_id and category
                    matching_rows = self.df[
                        (self.df["audio_id"] == audio_id)
                        & (self.df["recording_category"] == category)
                    ]
                    # if len(matching_rows) > 1:
                    #     print("stop")
                    for _, row in matching_rows.iterrows():
                        date = str(row["date"])
                        chunk_number = row.get("chunk_number", None)
                        recording_id = generate_recording_identifier(
                            audio_id, category, date, chunk_number
                        )
                        cat_rec_w2v_ids.append(recording_id)
                    wav2vec_ids_by_category.append(cat_rec_w2v_ids)

            # for parsel and wav2vec separately do [["p1", "p2"], ["a1", "a2"]] -> [("p1", "a1"), ("p2", "a2")]
            parsel_ids_transposed = list(zip(*parsel_ids_by_category)) if parsel_ids_by_category else []
            wav2vec_ids_transposed = list(zip(*wav2vec_ids_by_category)) if wav2vec_ids_by_category else []

            # Log if parsel and wav2vec have different recording counts.
            # This is expected for mixed-feature configs (e.g. parsel vowel + wav2vec poem)
            # with split poems — poems expand to multiple chunks while vowels stay single.
            # zip_longest below handles the asymmetry by filling with empty tuples.
            if parsel_ids_transposed and wav2vec_ids_transposed and len(parsel_ids_transposed) != len(wav2vec_ids_transposed):
                logger.debug(
                    "Different recording counts for audio_id=%s: "
                    "%d parselmouth vs %d wav2vec2 (expected for mixed-feature split configs)",
                    audio_id, len(parsel_ids_transposed), len(wav2vec_ids_transposed),
                )

            # Demographics-only: no parsel/wav2vec categories → one RecordingIdentifierSet per audio_id
            # so the preloader gets one sample per person (demographic features only).
            if not parsel_ids_transposed and not wav2vec_ids_transposed:
                result.append(
                    RecordingIdentifierSet(
                        audio_id=int(audio_id),
                        parselmouth_ids=[],
                        wav2vec_ids=[],
                    )
                )
            else:
                # Create one RecordingIdentifierSet per recording instance (chunk)
                for parsel_tuple, wav2vec_tuple in zip_longest(
                    parsel_ids_transposed, wav2vec_ids_transposed, fillvalue=()
                ):
                    identifier_set = RecordingIdentifierSet(
                        audio_id=int(audio_id),
                        parselmouth_ids=list(parsel_tuple) if parsel_tuple else [],
                        wav2vec_ids=list(wav2vec_tuple) if wav2vec_tuple else [],
                    )
                    result.append(identifier_set)

        return result

    def filter_by_splitting_config(self, use_split_poems: bool) -> "Metadata":
        """
        Filter metadata based on poem splitting configuration.

        This method allows the training pipeline to choose between:
        - Original poem recordings (use_split_poems=False)
        - Split poem chunks (use_split_poems=True)

        Uses the chunk_number field to distinguish between originals (chunk_number=None)
        and chunks (chunk_number>=1).

        Args:
            use_split_poems: If True, keep only split poem chunks.
                           If False, keep only original poem recordings.

        Returns:
            Self for method chaining
        """
        logger.info(f"Filtering by splitting config: use_split_poems={use_split_poems}")
        initial_count = len(self.df)

        if "recording_category" not in self.df.columns:
            logger.warning(
                "No recording_category column found, skipping splitting filter"
            )
            return self

        # Get all poem recordings
        poem_mask = self.df["recording_category"] == "poem"
        poem_recordings = self.df[poem_mask].copy()
        non_poem_recordings = self.df[~poem_mask].copy()

        if len(poem_recordings) == 0:
            logger.warning("No poem recordings found, skipping splitting filter")
            return self

        # Check if chunk_number column exists
        if "chunk_number" not in poem_recordings.columns:
            # No chunked data exists - all poems are originals
            if use_split_poems:
                # User wants chunks but none exist - return empty
                filtered_poems = poem_recordings.iloc[
                    0:0
                ]  # Empty DataFrame with same structure
                logger.warning(
                    f"No poem chunks found (no chunk_number column), returning empty set"
                )
            else:
                # User wants originals and all poems are originals
                filtered_poems = poem_recordings
                logger.warning(
                    f"No chunk_number column found - treating all {len(filtered_poems)} poems as originals"
                )
        else:
            # chunk_number column exists - filter based on it
            if use_split_poems:
                filtered_poems = poem_recordings[
                    poem_recordings["chunk_number"].notna()
                ]
                logger.info(
                    f"Keeping {len(filtered_poems)} poem chunks out of {len(poem_recordings)} poem recordings"
                )
            else:
                filtered_poems = poem_recordings[poem_recordings["chunk_number"].isna()]
                logger.info(
                    f"Keeping {len(filtered_poems)} original poems out of {len(poem_recordings)} poem recordings"
                )

        # Combine filtered poems with non-poem recordings
        self.df = pd.concat([filtered_poems, non_poem_recordings], ignore_index=True)

        self._record_filtering(
            f"filter_by_splitting_config(use_split_poems={use_split_poems})",
            initial_count,
            len(self.df),
        )

        return self

    def filter_for_experiment(self, task_config, feature_config) -> "Metadata":
        """
        Apply all experiment-specific filtering based on task and feature requirements.

        This unified method handles:
        - Task-specific filtering (longitudinal, target classes, splitting)
        - Complete recording requirements (based on feature needs)
        - Class label encoding

        Args:
            task_config: TaskConfig with task-specific requirements
            feature_config: FeatureConfig with feature extraction requirements

        Returns:
            Self for method chaining

        Raises:
            ValueError: If dataset becomes empty after filtering or requirements cannot be met
        """
        logger.info("=== APPLYING EXPERIMENT FILTERING ===")
        initial_count = len(self.df)

        logger.info("Applying task-specific filtering")
        self.filter_by_task(task_config)

        if len(self.df) == 0:
            raise ValueError("Dataset is empty after task filtering")

        # If require_all_recording_categories is set, filter to participants who have ALL
        # listed categories. This ensures identical participant sets across all recording
        # types (e.g., a, i, o, poem), enabling fully paired comparisons.
        if task_config.require_all_recording_categories:
            logger.info(
                f"Applying all-recording-categories filter: {task_config.require_all_recording_categories}"
            )
            self.filter_complete_recording_categories(
                task_config.require_all_recording_categories
            )
            if len(self.df) == 0:
                raise ValueError(
                    f"Dataset is empty after filtering for all recording categories: "
                    f"{task_config.require_all_recording_categories}"
                )

        logger.info("Applying complete recordings filtering")
        if required_categories := feature_config.get_required_recording_categories():
            self.filter_complete_recording_categories(required_categories)

        after_complete_count = len(self.df)
        if after_complete_count == 0:
            raise ValueError(
                f"Dataset is empty after filtering for complete recordings. "
                f"Required categories: {required_categories}. "
            )

        unique_audio_ids = self.get_unique_audio_ids()
        logger.info(
            f"Final experiment dataset: {len(unique_audio_ids)} unique audio IDs"
        )

        # Log filtering summary
        total_removed = initial_count - len(self.df)
        logger.info(
            f"Experiment filtering complete: {initial_count} -> {len(self.df)} "
            f"(removed {total_removed} rows, {total_removed / initial_count * 100:.1f}%)"
        )

        self._record_filtering(
            "filter_for_experiment(comprehensive)",
            initial_count,
            len(self.df),
        )

        return self

    def extract_demographic_features(
        self, recording_identifier_sets_list: list
    ) -> np.ndarray:
        """
        Extract demographic features from metadata for given recording identifier sets.

        Args:
            recording_identifier_sets_list: List of RecordingIdentifierSet objects

        Returns:
            np.ndarray: Demographic features array of shape (n_samples, n_demographic_features)
                       Features: [age, sex_encoded, bmi]
        """
        import numpy as np
        import pandas as pd

        logger = logging.getLogger(__name__)

        # Get audio IDs from recording identifier sets
        audio_ids = [rs.audio_id for rs in recording_identifier_sets_list]

        # Get metadata dataframe
        df = self.to_dataframe()

        # Create demographic features array
        n_samples = len(audio_ids)
        demographic_features = np.zeros((n_samples, 3))  # age, sex, bmi

        ref_year = REFERENCE_YEAR

        for i, audio_id in enumerate(audio_ids):
            # Get metadata for this audio_id (take first row since demographic info is same across recordings)
            audio_metadata = df[df["audio_id"] == audio_id].iloc[0]

            # Extract and process features
            # 1. Age (from birth_year, reference year 2025)
            if pd.notna(audio_metadata.get("birth_year")):
                age = ref_year - audio_metadata["birth_year"]
                demographic_features[i, 0] = age
            else:
                # Use median age for imputation (will be calculated below)
                demographic_features[i, 0] = np.nan

            # 2. Sex (binary encoding: w=0, m=1)
            sex = audio_metadata.get("sex")
            if pd.notna(sex):
                demographic_features[i, 1] = 1 if sex == "m" else 0
            else:
                demographic_features[i, 1] = np.nan

            # 3. BMI
            bmi = audio_metadata.get("bmi")
            if pd.notna(bmi):
                # Convert to numeric, handling string values like 'nicht vorhanden'
                bmi_numeric = pd.to_numeric(bmi, errors="coerce")
                if pd.notna(bmi_numeric):
                    demographic_features[i, 2] = bmi_numeric
                else:
                    # BMI is a non-numeric string, treat as missing
                    demographic_features[i, 2] = np.nan
            else:
                # Try to calculate from height and weight
                height_cm = audio_metadata.get("height_cm")
                weight_kg = audio_metadata.get("weight_kg")
                if pd.notna(height_cm) and pd.notna(weight_kg) and height_cm > 0:
                    height_m = height_cm / 100
                    calculated_bmi = weight_kg / (height_m**2)
                    demographic_features[i, 2] = calculated_bmi
                else:
                    demographic_features[i, 2] = np.nan

        logger.info(
            f"Extracted demographic features: shape={demographic_features.shape}"
        )
        age_col = demographic_features[:, 0]
        logger.info(
            f"Age range: {np.nanmin(age_col):.1f} - {np.nanmax(age_col):.1f}"
        )
        sex_col = demographic_features[:, 1]
        sex_valid = sex_col[~np.isnan(sex_col)]
        if len(sex_valid) > 0:
            logger.info(
                f"Sex distribution: {np.bincount(sex_valid.astype(int))} ({len(sex_col) - len(sex_valid)} missing)"
            )
        else:
            logger.info("Sex distribution: all values missing")
        bmi_col = demographic_features[:, 2]
        logger.info(
            f"BMI range: {np.nanmin(bmi_col):.1f} - {np.nanmax(bmi_col):.1f}"
        )

        return demographic_features

    def extract_baseline_features(
        self,
        recording_identifier_sets_list: list,
        scope: str,
    ) -> np.ndarray:
        """
        Extract baseline (non-voice) features for baseline experiments.
        scope: "age_sex" (age, sex), "comorbidities" (comorbidity_count_bin 0/1/2),
               "age_sex_comorbidities" (age, sex, comorbidity_count_bin).
        Returns array of shape (n_samples, n_features).
        """
        audio_ids = [rs.audio_id for rs in recording_identifier_sets_list]
        n_samples = len(audio_ids)
        df = self.df
        if scope == "age_sex":
            dem = self.extract_demographic_features(recording_identifier_sets_list)
            return dem[:, :2].astype(float)
        def _comorbidity_bin_for_row(row) -> float:
            raw_cols = [c for c in self._ps_covariate_columns if c in row.index]
            if not raw_cols:
                return 0.0
            raw_sum = row[raw_cols].fillna(0).astype(float).sum()
            return 0.0 if raw_sum == 0 else (1.0 if raw_sum <= 2 else 2.0)

        if scope == "comorbidities":
            out = np.zeros((n_samples, 1), dtype=float)
            for i, aid in enumerate(audio_ids):
                rows = df[df["audio_id"] == aid]
                if len(rows) > 0:
                    out[i, 0] = _comorbidity_bin_for_row(rows.iloc[0])
            return out
        if scope == "age_sex_comorbidities":
            dem = self.extract_demographic_features(recording_identifier_sets_list)
            age_sex = dem[:, :2].astype(float)
            comorb = np.zeros((n_samples, 1), dtype=float)
            for i, aid in enumerate(audio_ids):
                rows = df[df["audio_id"] == aid]
                if len(rows) > 0:
                    comorb[i, 0] = _comorbidity_bin_for_row(rows.iloc[0])
            return np.hstack([age_sex, comorb])
        raise ValueError(f"Unknown baseline scope: {scope}")

    def get_demographic_features(
        self,
        recording_identifier_sets_list: list,
        columns: list[str],
    ) -> np.ndarray:
        """
        Return demographic/non-voice features for the given recording sets as an array.

        Uses the merged metadata (base + ps_covariates) so that columns like
        age, sex, bmi, comorbidity_count_bin (and any ps_covariate columns) are available.
        Derived columns (e.g. comorbidity_count_bin) are computed as in
        get_metadata_for_audio_id_sequence. Missing columns yield NaN, then filled
        (median for continuous, 0 for discrete) in get_metadata_for_audio_id_sequence.

        Args:
            recording_identifier_sets_list: List of RecordingIdentifierSet objects
            columns: List of metadata column names in desired order (e.g. ["age", "sex", "bmi", "comorbidity_count_bin"])

        Returns:
            np.ndarray of shape (n_samples, len(columns)), same row order as recording_identifier_sets_list.
            If columns is empty, returns shape (n_samples, 0).
        """
        audio_ids = np.array([rs.audio_id for rs in recording_identifier_sets_list])
        if len(columns) == 0:
            return np.empty((len(audio_ids), 0), dtype=np.float64)
        meta = self.get_metadata_for_audio_id_sequence(audio_ids, columns)
        return meta.values.astype(np.float64)

    def extract_audio_ids(self, recording_identifier_sets_list: list) -> np.ndarray:
        """
        Extract audio IDs from recording identifier sets as a numpy array.

        Args:
            recording_identifier_sets_list: List of RecordingIdentifierSet objects

        Returns:
            np.ndarray: Array of audio IDs as integers
        """
        return np.array([rs.audio_id for rs in recording_identifier_sets_list])

    def get_original_recording_identifier_sets_subset(
        self, audio_ids: list[int], feature_config
    ) -> list[RecordingIdentifierSet]:
        """
        Generate recording identifier sets for original (unchunked) recordings
        for a specific subset of audio IDs.

        This method is used for lazy loading of original features during CV evaluation,
        ensuring the same patients are used for both chunked training and original evaluation.

        Args:
            audio_ids: Specific list of audio IDs (patient identifiers) to process
            feature_config: FeatureConfig specifying which categories to use for each feature type

        Returns:
            List of RecordingIdentifierSet objects for original recordings of specified patients
        """
        from src.utils.recording_identifier import generate_recording_identifier

        result = []

        for audio_id in audio_ids:
            # Generate original recording identifiers (without chunk_number)
            original_parsel_ids = []
            if feature_config.recordings_parsel:
                for category in feature_config.recordings_parsel:
                    # Look up the date for this audio_id and category
                    matching_rows = self.df[
                        (self.df["audio_id"] == audio_id)
                        & (self.df["recording_category"] == category)
                    ]
                    if len(matching_rows) == 0:
                        raise ValueError(
                            f"No original recording found for {audio_id=}, {category=}. "
                            f"Expected one original recording per category."
                        )
                    else:
                        date = str(matching_rows.iloc[0]["date"])
                        # Generate original identifier (no chunk_number = original recording)
                        recording_id = generate_recording_identifier(
                            audio_id, category, date
                        )
                        original_parsel_ids.append(recording_id)

            # Generate wav2vec identifiers
            original_wav2vec_ids = []
            if feature_config.recordings_wav2vec2:
                for category in feature_config.recordings_wav2vec2:
                    # Look up the date for this audio_id and category
                    matching_rows = self.df[
                        (self.df["audio_id"] == audio_id)
                        & (self.df["recording_category"] == category)
                    ]
                    if len(matching_rows) == 0:
                        raise ValueError(
                            f"No original recording found for {audio_id=}, {category=}. "
                            f"Expected one original recording per category."
                        )
                    else:
                        date = str(matching_rows.iloc[0]["date"])
                        recording_id = generate_recording_identifier(
                            audio_id, category, date
                        )
                        original_wav2vec_ids.append(recording_id)

            # Create RecordingIdentifierSet object for original recordings
            identifier_set = RecordingIdentifierSet(
                audio_id=int(audio_id),
                parselmouth_ids=original_parsel_ids,
                wav2vec_ids=original_wav2vec_ids,
            )
            result.append(identifier_set)

        logger.info(
            f"Generated {len(result)} original recording identifier sets for {len(audio_ids)} patients"
        )
        return result

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SimplifiedMetadata({len(self.df)} rows, {len(self.df.columns)} columns)"
        )
