"""
Configuration models for the voice biomarker pipeline.
Contains all configuration classes for different pipeline components.
"""

from typing import Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator

import logging
from src.models.core_models import DATA_SOURCE

logger = logging.getLogger(__name__)


class ParselmouthConfig(BaseModel):
    """Configuration for Parselmouth feature selection using clinical relevance grouping."""

    # Clinical relevance feature groups
    include_basic_acoustics: bool = Field(
        default=True, description="F0, intensity, HNR, all jitter/shimmer"
    )
    include_advanced_voice_quality: bool = Field(default=True, description="CPP, DSI")
    include_spectral_analysis: bool = Field(
        default=True, description="MFCCs (base only), formants, spectral features"
    )
    include_mfcc_derivatives: bool = Field(
        default=True, description="Delta/delta-delta MFCCs (usually excluded)"
    )
    include_temporal_features: bool = Field(
        default=True, description="Temporal durations, syllable rate"
    )

    # eGeMAPS feature integration
    include_egemaps_features: bool = Field(
        default=False,
        description="Include eGeMAPS features (88 features) as a feature group",
    )

    # Exact feature list override (bypasses grouping logic)
    exact_feature_list: list[str] | None = Field(
        default=None,
        description="Exact list of feature names to include (overrides group settings)",
    )

    @model_validator(mode="after")
    def validate_at_least_one_selection(self):
        """Ensure at least one feature group is selected or exact list is provided."""
        if self.exact_feature_list is not None:
            if len(self.exact_feature_list) == 0:
                raise ValueError("exact_feature_list cannot be empty")
            return self

        if not any(
            [
                self.include_basic_acoustics,
                self.include_advanced_voice_quality,
                self.include_spectral_analysis,
                self.include_mfcc_derivatives,
                self.include_temporal_features,
                self.include_egemaps_features,
            ]
        ):
            raise ValueError("At least one feature group must be selected")
        return self


class FeatureConfig(BaseModel):
    """Configuration for feature extraction."""

    recordings_parsel: tuple[str, ...] = Field(
        default=(), description="Parselmouth recording categories"
    )
    recordings_wav2vec2: tuple[str, ...] = Field(
        default=(), description="Wav2Vec2 recording categories"
    )
    demographics_only: bool = Field(
        default=False,
        description="If True, no voice features are loaded (baseline: demographics only). recordings_parsel and recordings_wav2vec2 may both be empty.",
    )
    wav2vec2_layer: int = Field(
        default=4, ge=0, le=24, description="Wav2Vec2 layer to extract"
    )
    wav2vec2_embedding_statistics: Literal["mean", "mean_std"] = Field(
        default="mean", description="Embedding statistics to use"
    )
    parsel_feature_file: Path = Field(
        ..., description="Path to Parselmouth features .npy file"
    )
    embeddings_dir: Path = Field(
        ..., description="Directory containing Wav2Vec2 embeddings"
    )
    parselmouth_config: ParselmouthConfig = Field(
        default_factory=ParselmouthConfig,
        description="Parselmouth feature extraction configuration",
    )

    @field_validator("parsel_feature_file")
    @classmethod
    def validate_parsel_feature_file(cls, v):
        """Validate Parselmouth feature file exists and has correct optimized format."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Parselmouth feature file does not exist: {path}")
        if not path.suffix == ".npy":
            raise ValueError(
                f"Parselmouth feature file must have .npy extension: {path}"
            )

        # Validate optimized file structure only
        try:
            import numpy as np

            data = np.load(path)
            if "recording_identifiers" not in data.dtype.names:
                raise ValueError(
                    f"Parselmouth feature file missing 'recording_identifiers' field: {path}"
                )
            if "feature_matrix" not in data.dtype.names:
                raise ValueError(
                    f"Parselmouth feature file missing 'feature_matrix' field: {path}"
                )
        except Exception as e:
            raise ValueError(
                f"Invalid Parselmouth feature file format: {path}. Error: {e}"
            )

        return path

    @field_validator("embeddings_dir")
    @classmethod
    def validate_embeddings_dir_exists(cls, v):
        """Ensure embeddings directory exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Embeddings directory does not exist: {path}")
        return path

    @model_validator(mode="after")
    def validate_at_least_one_feature_type(self):
        """Ensure at least one feature type is specified, unless demographics_only is True."""
        if self.demographics_only:
            return self
        if not self.recordings_parsel and not self.recordings_wav2vec2:
            raise ValueError(
                "At least one of recordings_parsel or recordings_wav2vec2 must be specified (or set demographics_only=True for baseline configs)"
            )
        return self

    # Private attributes to store metadata for validation
    _metadata_recording_identifiers: set[int] | None = None

    def set_recording_identifiers_for_validation(
        self, metadata_recording_identifiers: set[int]
    ) -> None:
        """Set metadata recording identifiers for automatic validation."""
        self._metadata_recording_identifiers = metadata_recording_identifiers

    def get_required_recording_categories(self) -> list[str]:
        """Get all recording categories required for this feature configuration."""
        required = []
        if self.recordings_parsel:
            required.extend(self.recordings_parsel)
        if self.recordings_wav2vec2:
            required.extend(self.recordings_wav2vec2)
        return list(set(required))  # Remove duplicates


class TaskConfig(BaseModel):
    """Configuration for different classification tasks."""

    target_classes: list[str] = Field(
        default_factory=list, description="Target classes for classification"
    )
    target_column: str = Field(..., description="Column name containing target labels")
    only_longitudinal: bool = Field(
        default=False, description="Filter to longitudinal cases only"
    )
    exclude_longitudinal: bool = Field(
        default=False, description="Exclude longitudinal recordings"
    )
    use_temporal_pairs: bool = Field(
        default=False, description="Use temporal pair logic"
    )
    use_split_poems: bool = Field(
        default=False, description="Use split poem chunks instead of original poems"
    )
    included_data_sources: list[DATA_SOURCE] = Field(
        default_factory=lambda: ["charite", "uk_covid"],
        description="List of data sources to include in training"
    )
    require_all_recording_categories: list[str] = Field(
        default_factory=list,
        description="If non-empty, only keep participants who have ALL listed recording categories. "
        "Used to ensure identical participant sets across recording types for paired comparisons.",
    )
    ps_covariates_file: str | None = Field(
        default=None,
        description="Task-specific ps_covariates file path. Overrides the global ps_covariates_file "
        "from pipeline_config when set. Supports {date} placeholder.",
    )

    @field_validator("included_data_sources")
    @classmethod
    def validate_included_data_sources_not_empty(cls, v):
        """Ensure included_data_sources is not empty."""
        if not v:
            raise ValueError("included_data_sources cannot be empty")
        return v

    @field_validator("target_classes")
    @classmethod
    def validate_target_classes_lowercase(cls, v):
        """Ensure all target classes are lowercase."""
        for class_name in v:
            if class_name != class_name.lower():
                raise ValueError(f"Target class '{class_name}' must be lowercase")
        return v

    @field_validator("target_column")
    @classmethod
    def validate_target_column_lowercase(cls, v):
        """Ensure target_column is lowercase."""
        if v != v.lower():
            raise ValueError(f"Target column '{v}' must be lowercase")
        return v

    @classmethod
    def for_task(
        cls,
        task_type: Literal[
            "copd_classification",
            "temporal_classification",
            "first_last_classification",
        ],
    ) -> "TaskConfig":
        """Create task configuration for specific task types."""
        if task_type == "copd_classification":
            return cls(
                target_classes=["copd", "control"],
                target_column="lung_disease_main",
                only_longitudinal=False,
                exclude_longitudinal=True,
                use_temporal_pairs=False,
            )
        elif task_type in ["temporal_classification", "first_last_classification"]:
            return cls(
                target_classes=[],
                target_column="date",
                only_longitudinal=True,
                exclude_longitudinal=False,
                use_temporal_pairs=True,
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")


class TrainingConfig(BaseModel):
    """Unified training configuration with clear separation of fixed vs hyperparameter values."""

    # FIXED TRAINING PARAMETERS (not searched over)
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    n_splits_outer: int = Field(default=5, ge=2, description="Outer CV splits")
    n_repeats_outer: int = Field(default=10, ge=1, description="Outer CV repeats")
    n_splits_inner: int = Field(default=5, ge=2, description="Inner CV splits")
    n_repeats_inner: int = Field(default=3, ge=1, description="Inner CV repeats")
    scoring_metric: str = Field(
        default="auroc", description="Scoring metric for CV (primary: auroc for signal existence)"
    )
    n_jobs: int = Field(default=-1, description="Number of parallel jobs")
    sample_weight_strategy: Literal[
        "class_only",
        "ps_overlap",
        "overlap_ps_and_deploy",
        "strata_only",
        "flat",
    ] = Field(
        default="ps_overlap",
        description="Sample weight strategy; ps_overlap: PS only (simplified, ESS gatekeeper); strata_only: coarse age×sex strata only; flat: uniform weights of 1.0 (no reweighting at all)",
    )
    pi_deploy: float | None = Field(
        default=0.2,
        description="Target prevalence for deployment.",
    )
    min_controls_per_stratum: int = Field(
        default=2,
        ge=1,
        description="Minimum controls per (age_band, sex) stratum; bands are merged until satisfied (strata strategy)",
    )
    weight_clip_low_percentile: float = Field(
        default=1.0,
        description="Lower percentile for weight clipping (strata strategy)",
    )
    weight_clip_high_percentile: float = Field(
        default=99.0,
        description="Upper percentile for weight clipping (strata strategy)",
    )
    overlap_covariate_columns: list[str] | None = Field(
        default=None,
        description=(
            "Explicit covariate columns for the ps_overlap / "
            "overlap_ps_and_deploy propensity model. If None (default), the "
            "propensity model uses metadata.overlap_covariate_columns_simplified "
            "(age + sex + merged ps_covariates). Set e.g. ['age'] for an "
            "age-only propensity sensitivity arm that balances on the single "
            "genuine confounder instead of recruitment-artefact comorbidity flags."
        ),
    )
    demographic_feature_columns: list[str] = Field(
        default_factory=lambda: ["age", "sex", "bmi"],
        description="Metadata column names to use as demographic/non-voice features, in order (e.g. age, sex, bmi, comorbidity_count_bin). Empty = no demographics.",
    )
    include_voice_features: bool = Field(
        default=True,
        description="If True, load voice (Parsel + Wav2Vec) and append demographics; if False, baseline-only (demographics only).",
    )

    # HYPERPARAMETER SEARCH SPACES (grid search parameters)
    classifier_name: str = Field(..., description="Single classifier to use")
    classifier_hyperparams: dict[str, list] = Field(
        default_factory=dict, description="Classifier hyperparameter grid"
    )
    pca_hyperparams: dict[str, list] = Field(
        default_factory=dict, description="PCA hyperparameter grid"
    )

    @field_validator("classifier_name")
    @classmethod
    def validate_classifier_name(cls, v):
        """Validate classifier name is supported."""
        supported_classifiers = {
            "LogisticRegression", "XGBoost", "SVC",
            "RandomForestClassifier", "XGBClassifier"
        }
        if v not in supported_classifiers:
            raise ValueError(f"Unsupported classifier: {v}. Must be one of {supported_classifiers}")
        return v

class TemporalConfig(BaseModel):
    """Configuration for temporal order classification."""

    required_categories: list[str] = Field(
        default_factory=list, description="Required recording categories"
    )
    min_recordings_for_test: int = Field(
        default=4, ge=2, description="Minimum recordings needed for test set"
    )
    feature_combination: Literal["single_category", "multi_category"] = Field(
        default="single_category", description="How to combine features"
    )


class AutoGluonConfig(BaseModel):
    """Configuration for AutoGluon training."""

    time_limit: int = Field(
        default=300, ge=1, description="Training time limit in seconds"
    )
    presets: Literal[
        "best_quality",
        "high_quality",
        "good_quality",
        "medium_quality",
        "optimize_for_deployment",
    ] = Field(default="medium_quality", description="AutoGluon preset")
    eval_metric: str = Field(default="accuracy", description="Evaluation metric")
    verbosity: int = Field(default=2, ge=0, le=4, description="Verbosity level")


class ExperimentConfig(BaseModel):
    """Unified experiment configuration containing all components."""

    # EXPERIMENT METADATA
    experiment_name: str = Field(..., description="Name of the experiment")
    description: str = Field(..., description="Description of the experiment")
    tags: list[str] = Field(default_factory=list, description="Experiment tags")
    date: str | None = Field(
        default=None,
        description="Dataset/experiment version date from snakemake_matrix.yml; used for MLFlow experiment grouping",
    )
    experiment_version: str | None = Field(
        default=None,
        description="Optional version suffix (e.g. 'v2') appended to date in MLFlow experiment names; data paths still use date alone",
    )
    classification_task: str | None = Field(
        default=None,
        description="Classification task name from snakemake_matrix (e.g. 'copd_classification_charite_only'); used for MLFlow experiment naming",
    )
    
    # CORE CONFIGURATION COMPONENTS
    task_config: TaskConfig = Field(..., description="Task-specific configuration")
    training_config: TrainingConfig = Field(..., description="Training parameters")
    feature_config: FeatureConfig = Field(..., description="Feature extraction configuration")
    
    # FILE PATHS
    metadata_file: Path = Field(..., description="Path to metadata CSV file")
    output_dir: Path = Field(..., description="Output directory for results")
    ps_covariates_file: Path | None = Field(
        default=None,
        description="Optional path to ps_covariates.csv; first column = ID, rest = covariate columns (header = names); merged into metadata (only columns not already present)",
    )

    @field_validator("metadata_file")
    @classmethod
    def validate_metadata_file_exists(cls, v):
        """Ensure metadata file exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Metadata file does not exist: {path}")
        return path

    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            config_data = yaml.safe_load(f)

        # Backward compatibility: map legacy disease_comorbidities_file to ps_covariates_file
        if config_data and "disease_comorbidities_file" in config_data and "ps_covariates_file" not in config_data:
            config_data["ps_covariates_file"] = config_data.pop("disease_comorbidities_file")

        return cls(**config_data)
