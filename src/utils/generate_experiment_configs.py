#!/usr/bin/env python3
"""
Generate experiment configuration files for the voice biomarker pipeline.

This script handles both single configuration generation (for Snakemake integration)
and bulk generation of all valid configurations. It replaces the separate
generate_configs.py script to eliminate code duplication.
"""

import argparse
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.config_models import (
    ExperimentConfig,
    TaskConfig,
    TrainingConfig,
    FeatureConfig,
    ParselmouthConfig,
)
import logging


# Scopes that use only demographics (no voice); one config per scope with feature=demographics_only
DEMOGRAPHICS_ONLY_SCOPES = (
    "age_only",
    "age_sex_only",
    "comorbidity_only",
    "age_sex_comorbidity",
    "age_sex_comorbidity_individual",
)

# Sentinel in pipeline_config.yml:scopes.*.demographic_feature_columns meaning
# "age + sex + every PS-covariate column in the task's ps_covariates_file"
# (the ps_covariates_file's first column is the ID and is skipped).
PS_COVARIATES_SENTINEL = "from_ps_covariates"


@dataclass
class ExperimentParams:
    """Parameters defining an experiment configuration."""

    exp: str
    scope: str  # age_sex_only, comorbidity_only, age_sex_comorbidity, voice_only, voice_plus_age_sex, voice_plus_age_sex_comorbidity
    feature: str
    hyperparam: str  # basic or extended
    task: str
    classifier: str
    splitting: str

    def to_identifier(self) -> str:
        """Generate experiment identifier string (used in config filename).
        Demographics-only scopes omit exp (norm_off/norm_on) and splitting from the name.
        """
        if self.scope in DEMOGRAPHICS_ONLY_SCOPES:
            return f"{self.scope}__{self.feature}__{self.hyperparam}__{self.task}__{self.classifier}"
        return f"{self.exp}__{self.scope}__{self.feature}__{self.hyperparam}__{self.task}__{self.splitting}__{self.classifier}"


@dataclass
class DirectoryConfig:
    """Directory and file paths for experiment generation."""

    parsel_dir: str
    embeddings_dir: str
    metadata_file: str
    output_file: str


class ConfigurationGenerator:
    """Main class for generating experiment configurations."""

    def __init__(self, experiment_version: str | None = None):
        self.logger = logging.getLogger(__name__)
        self.pipeline_config = load_pipeline_config()
        self.matrix_config = load_snakemake_matrix()
        self.experiment_version = experiment_version

    def generate_single_config(
        self, params: ExperimentParams, dirs: DirectoryConfig
    ) -> str:
        """Generate a single experiment configuration file."""
        # Validate and get feature variant
        feature_variant = self._validate_and_get_feature_variant(params)

        # Get task config for skip check
        task_config_dict = self.pipeline_config["tasks"][params.task]

        # Check if this combination should be skipped
        if should_skip_training(
            feature_variant,
            params.splitting,
            params.hyperparam,
            params.classifier,
            task_config_dict,
            scope=params.scope,
        ):
            self.logger.warning(
                f"Skipping invalid combination: {params.feature} + {params.splitting} + {params.hyperparam} + {params.classifier} + {params.task}"
            )
            return ""

        # Build and save configuration
        config = self._build_experiment_config(params, dirs, feature_variant)
        self._save_config(config, dirs.output_file, params)
        return dirs.output_file

    def generate_all_configs(
        self,
        output_dir: str,
        parsel_dirs: list[str],
        embeddings_dirs: list[str],
        metadata_files: list[str],
    ) -> list[str]:
        """Generate all valid experiment configurations.

        Demographics-only scopes are emitted only once (using the first preproc variant
        for directory resolution); their config filenames do not include norm_off/norm_on.
        """
        self.logger.info("=== GENERATING EXPERIMENT CONFIGURATIONS ===")
        self.logger.info(f"Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # Get all variant combinations
        variants = self._get_all_matrix_variants()
        created_configs, skipped_configs = [], []
        combination_count = 0

        # Demographics-only configs are generated only once; use first preproc for dir resolution
        first_preproc_name = variants["preproc"][0]["name"] if variants["preproc"] else ""

        for preproc_variant in variants["preproc"]:
            exp_name = preproc_variant["name"]
            feature_dirs = find_feature_directories(
                parsel_dirs + embeddings_dirs, exp_name
            )
            metadata_file = self._find_metadata_file(metadata_files, exp_name)

            if not feature_dirs["parsel_dir"] or not feature_dirs["embeddings_dir"]:
                raise ValueError(
                    f"Missing feature directories for experiment {exp_name}"
                )

            for (
                scope,
                feature,
                hyperparam,
                task,
                classifier,
                splitting,
            ) in self._iter_combinations(variants):
                # Emit demographics-only only for the first preproc variant (filename has no exp)
                if scope in DEMOGRAPHICS_ONLY_SCOPES and exp_name != first_preproc_name:
                    continue
                combination_count += 1
                try:
                    result = self._process_combination(
                        exp_name,
                        scope,
                        feature,
                        hyperparam,
                        task,
                        classifier,
                        splitting,
                        feature_dirs,
                        metadata_file,
                        output_dir,
                    )
                    if result:
                        created_configs.append(result)
                        self.logger.info(
                            f"Created {combination_count}: {os.path.basename(result)}"
                        )
                    else:
                        params = ExperimentParams(
                            exp_name, scope, feature, hyperparam, task, classifier, splitting
                        )
                        skipped_configs.append(params.to_identifier())
                except Exception as e:
                    params = ExperimentParams(
                        exp_name, scope, feature, hyperparam, task, classifier, splitting
                    )
                    raise ValueError(
                        f"Failed to create config for {params.to_identifier()}: {e}"
                    )

        self._log_summary(combination_count, created_configs, skipped_configs)
        return created_configs

    def _validate_and_get_feature_variant(self, params: ExperimentParams) -> dict:
        """Validate parameters and return feature variant config."""
        # Validate scope
        scope_variants = self.matrix_config.get("scope_variants", [])
        if scope_variants and params.scope not in scope_variants:
            raise ValueError(f"{params.scope} not found in scope_variants")

        # Validate feature (for demographics_only scopes feature is demographics_only; for voice scopes feature is in feature_variants)
        if params.scope in DEMOGRAPHICS_ONLY_SCOPES:
            if params.feature != "demographics_only":
                raise ValueError(
                    f"For scope {params.scope} feature must be demographics_only, got {params.feature}"
                )
        else:
            if params.feature not in self.matrix_config.get("feature_variants", []):
                raise ValueError(f"{params.feature} not found in feature_variants")

        validations = [
            (params.exp, "preproc_variants", lambda p: p["name"] == params.exp),
            (
                params.hyperparam,
                "hyperparam_variants",
                lambda h: h == params.hyperparam,
            ),
            (params.task, "classification_tasks", lambda t: t == params.task),
        ]
        for value, config_key, check_func in validations:
            if not any(check_func(item) for item in self.matrix_config[config_key]):
                raise ValueError(f"{value} not found in {config_key}")

        feature_variant = self.pipeline_config["feature_variant_configs"].get(
            params.feature
        )
        if not feature_variant:
            raise ValueError(
                f"Feature variant config '{params.feature}' not found in pipeline_config"
            )
        return feature_variant

    def _build_experiment_config(
        self, params: ExperimentParams, dirs: DirectoryConfig, feature_variant: dict
    ) -> ExperimentConfig:
        """Build complete experiment configuration."""
        # Build experiment metadata
        name = params.to_identifier()
        poem_suffix = f" ({params.splitting} poems)" if params.scope not in DEMOGRAPHICS_ONLY_SCOPES else ""
        description = f"Training experiment for {params.task} using {params.hyperparam} hyperparameters, scope={params.scope}, {params.feature} features{poem_suffix}"
        tags = [params.exp, params.scope, params.feature, params.hyperparam, params.task]
        if params.scope not in DEMOGRAPHICS_ONLY_SCOPES:
            tags.append(params.splitting)
        training_mode = "basic" if params.hyperparam == "basic" else "extended"

        # Create FeatureConfig with direct unpacking
        feature_variant_clean = feature_variant.copy()
        parselmouth_config_dict = feature_variant_clean.pop("parselmouth_config")

        feature_config = FeatureConfig(
            **feature_variant_clean,
            parsel_feature_file=Path(f"{dirs.parsel_dir}/acoustic_features.npy"),
            embeddings_dir=Path(dirs.embeddings_dir),
            parselmouth_config=ParselmouthConfig(**parselmouth_config_dict),
        )

        # Demographics-only: skip recording-type (split vs unsplit) filtering
        use_split_poems = (
            (params.splitting == "split") if params.scope not in DEMOGRAPHICS_ONLY_SCOPES else False
        )
        # Tasks may define a `metadata_file` override (e.g. a PSM-matched cohort CSV
        # produced by an earlier pipeline rule). This is not a TaskConfig field; we
        # pop it, apply {date} substitution, and use it as the final metadata path.
        task_raw = dict(self.pipeline_config["tasks"][params.task])
        metadata_override_raw = task_raw.pop("metadata_file", None)
        task_config = TaskConfig(
            **task_raw,
            use_split_poems=use_split_poems,
        )

        # Direct hyperparameter access
        try:
            classifier_hyperparams = self.pipeline_config["clf_hyperparameters"][
                training_mode
            ][params.classifier]
        except KeyError:
            raise ValueError(
                f"Classifier '{params.classifier}' not found in hyperparameter config '{training_mode}'"
            )

        pca_hyperparams = self.pipeline_config["pca_hyperparameters"][training_mode]
        if params.scope in DEMOGRAPHICS_ONLY_SCOPES:
            # Demographics-only: no voice features → single PCA config to avoid useless grid search
            pca_hyperparams = {k: [v[0]] for k, v in pca_hyperparams.items()}

        # Resolve ps_covariates path first (may be needed to expand the
        # from_ps_covariates sentinel below, and is also passed through to the
        # ExperimentConfig). Task-level ps_covariates_file overrides the
        # global one, allowing different covariate files per dataset.
        date = self.matrix_config.get("date") or None
        ps_covariates_path: Path | None = None
        task_raw_config = self.pipeline_config["tasks"][params.task]
        raw_path = (
            task_raw_config.get("ps_covariates_file")
            or self.pipeline_config.get("ps_covariates_file")
        )
        if raw_path and isinstance(raw_path, str):
            replaced = raw_path.replace("{date}", date or "").strip()
            if replaced:
                ps_covariates_path = Path(replaced)
                # Resolve cluster-absolute paths to the project-local mirror
                # for laptop runs, mirroring the metadata_file behaviour
                # below. Without this, non-sentinel demographics scopes
                # (e.g. age_only / age_sex_only) write the unreadable
                # /data/cephfs-1/... path straight into the config.
                resolved_ps = self._resolve_cluster_path_locally(
                    ps_covariates_path
                )
                if resolved_ps is not None:
                    ps_covariates_path = resolved_ps

        # Create TrainingConfig; override demographic/voice from scopes[scope]
        training_base = dict(self.pipeline_config["training"][training_mode])
        scopes_config = self.pipeline_config.get("scopes", {})
        if params.scope and params.scope in scopes_config:
            demo_cols = scopes_config[params.scope].get(
                "demographic_feature_columns",
                training_base.get("demographic_feature_columns", []),
            )
            if demo_cols == PS_COVARIATES_SENTINEL:
                demo_cols = self._resolve_ps_covariate_columns(
                    ps_covariates_path, params.scope, params.task
                )
            training_base["demographic_feature_columns"] = demo_cols
            training_base["include_voice_features"] = scopes_config[params.scope].get(
                "include_voice_features", training_base.get("include_voice_features", True)
            )
        training_config = TrainingConfig(
            **training_base,
            classifier_name=params.classifier,
            classifier_hyperparams=classifier_hyperparams,
            pca_hyperparams=pca_hyperparams,
        )

        # Apply task-level metadata_file override if declared (e.g. PSM cohort).
        # The override path is usually a cluster-absolute location produced by
        # an earlier Snakemake rule. When running the generator locally, fall
        # back to the project-local mirror (data/{date}/<basename>).
        final_metadata_file = dirs.metadata_file
        if metadata_override_raw:
            replaced = metadata_override_raw.replace("{date}", date or "").strip()
            if replaced:
                resolved_local = self._resolve_cluster_path_locally(Path(replaced))
                final_metadata_file = str(resolved_local) if resolved_local else replaced

        return ExperimentConfig(
            experiment_name=name,
            description=description,
            tags=tags,
            date=date,
            experiment_version=self.experiment_version,
            classification_task=params.task,
            task_config=task_config,
            training_config=training_config,
            feature_config=feature_config,
            metadata_file=Path(final_metadata_file),
            output_dir=Path(dirs.output_file),
            ps_covariates_file=ps_covariates_path,
        )

    def _resolve_ps_covariate_columns(
        self,
        ps_covariates_path: Path | None,
        scope: str,
        task: str,
    ) -> list[str]:
        """Resolve the `from_ps_covariates` sentinel to [age, sex, <covariate columns>].

        Reads the task's ps_covariates CSV header (first column = ID, rest =
        covariate indicators) and prepends [age, sex]. Cached per path so the
        CSV is only read once per bulk generation.

        ``ps_covariates_path`` in pipeline_config.yml is usually a cluster-
        absolute path (/data/cephfs-1/...). When running the generator
        locally for a smoke test, fall back to the project-relative mirror
        at ``data/{date}/<basename>`` if it exists.
        """
        if ps_covariates_path is None:
            raise ValueError(
                f"Scope '{scope}' for task '{task}' uses "
                f"demographic_feature_columns={PS_COVARIATES_SENTINEL!r} but "
                f"no ps_covariates_file is configured."
            )
        resolved = self._resolve_cluster_path_locally(ps_covariates_path)
        if resolved is None:
            raise FileNotFoundError(
                f"Scope '{scope}' for task '{task}': ps_covariates_file not "
                f"readable (tried {ps_covariates_path} and local fallback). "
                f"Can't resolve {PS_COVARIATES_SENTINEL!r} sentinel."
            )
        if not hasattr(self, "_ps_covariate_columns_cache"):
            self._ps_covariate_columns_cache: dict[Path, list[str]] = {}
        cache = self._ps_covariate_columns_cache
        if resolved not in cache:
            with open(resolved) as f:
                header_line = f.readline().strip()
            header = [c.strip() for c in header_line.split(",")]
            if len(header) < 2:
                raise ValueError(
                    f"ps_covariates file {resolved} has fewer than 2 "
                    f"columns; cannot resolve {PS_COVARIATES_SENTINEL!r}."
                )
            # First column is the join ID (audio_id or study_id); skip it.
            covariate_cols = header[1:]
            cache[resolved] = ["age", "sex"] + covariate_cols
            self.logger.info(
                "Resolved %s for task=%s scope=%s via %s -> %s",
                PS_COVARIATES_SENTINEL, task, scope, resolved, cache[resolved],
            )
        return list(cache[resolved])

    def _resolve_cluster_path_locally(self, path: Path) -> Path | None:
        """Return ``path`` if it exists, else try project-local mirrors.

        Cluster-absolute paths in pipeline_config.yml (``/data/cephfs-1/...``)
        don't resolve on a developer laptop. This helper falls back to the
        local mirror at ``data/{date}/<basename>`` and ``data/<basename>``
        so the config generator can run in both environments without
        cluster-only branches.
        """
        if path.exists():
            return path
        basename = path.name
        if not basename:
            return None
        date = self.matrix_config.get("date") or ""
        candidates = []
        if date:
            candidates.append(Path("data") / date / basename)
        candidates.append(Path("data") / basename)
        for cand in candidates:
            if cand.exists():
                return cand
        return None

    def _get_all_matrix_variants(self) -> dict:
        """Get all variant configurations."""
        return {
            "preproc": self.matrix_config["preproc_variants"],
            "scope": self.matrix_config.get("scope_variants", []),
            "feature": self.matrix_config["feature_variants"],
            "hyperparam": self.matrix_config["hyperparam_variants"],
            "task": self.matrix_config["classification_tasks"],
            "classifier": self.matrix_config["classifier_variants"],
            "splitting": self.matrix_config["splitting_variants"],
        }

    def _iter_combinations(self, variants: dict):
        """Iterate over (scope, feature, hyperparam, task, classifier, splitting).
        For demographics-only scopes: one tuple per scope with feature=demographics_only.
        For voice scopes: one tuple per feature variant.
        """
        for scope in variants.get("scope", []):
            if scope in DEMOGRAPHICS_ONLY_SCOPES:
                for hyperparam in variants["hyperparam"]:
                    for task in variants["task"]:
                        for classifier in variants["classifier"]:
                            # One config per demographics-only combo; no split/unsplit in filename
                            yield scope, "demographics_only", hyperparam, task, classifier, "unsplit"
            else:
                for feature in variants["feature"]:
                    for hyperparam in variants["hyperparam"]:
                        for task in variants["task"]:
                            for classifier in variants["classifier"]:
                                for splitting in variants["splitting"]:
                                    yield scope, feature, hyperparam, task, classifier, splitting

    def _process_combination(
        self,
        exp_name: str,
        scope: str,
        feature: str,
        hyperparam: str,
        task: str,
        classifier: str,
        splitting: str,
        feature_dirs: dict,
        metadata_file: str,
        output_dir: str,
    ) -> str | None:
        """Process a single experiment combination."""
        params = ExperimentParams(
            exp_name, scope, feature, hyperparam, task, classifier, splitting
        )
        config_filename = f"{params.to_identifier()}__config.yml"
        config_path = os.path.join(output_dir, config_filename)

        dirs = DirectoryConfig(
            parsel_dir=feature_dirs["parsel_dir"],
            embeddings_dir=feature_dirs["embeddings_dir"],
            metadata_file=metadata_file,
            output_file=config_path,
        )

        return self.generate_single_config(params, dirs)

    def _find_metadata_file(self, metadata_files: list[str], exp_name: str) -> str:
        """Find metadata file for experiment."""
        for mf in metadata_files:
            if exp_name in mf:
                return mf
        raise ValueError(f"No metadata file found for experiment {exp_name}")

    def _save_config(
        self, config: ExperimentConfig, output_file: str, params: ExperimentParams
    ) -> None:
        """Save configuration to file with metadata."""
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "w") as f:
            yaml.dump(config.model_dump(mode="json"), f, default_flow_style=False)

        self.logger.info(f"Configuration saved to: {output_file}")

    def _log_summary(self, total: int, created: list, skipped: list) -> None:
        """Log generation summary."""
        self.logger.info("=== CONFIGURATION GENERATION SUMMARY ===")
        self.logger.info(f"Total combinations processed: {total}")
        self.logger.info(f"Valid configurations created: {len(created)}")
        self.logger.info(f"Invalid combinations skipped: {len(skipped)}")

        if skipped:
            self.logger.info("Skipped combinations:")
            for combo in skipped:
                self.logger.info(f"  - {combo}")


def load_pipeline_config(
    config_path: Path = Path("config/pipeline_config.yml"),
) -> dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_snakemake_matrix(
    config_path: Path = Path("config/snakemake_matrix.yml"),
) -> dict[str, Any]:
    """Load Snakemake matrix configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Snakemake matrix config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def should_skip_training(
    feature_variant: dict,
    splitting_variant: str,
    hyperparam_variant: str,
    classifier_variant: str,
    task_config: dict,
    scope: str | None = None,
) -> bool:
    """
    Determine if this feature+splitting+hyperparam+classifier+task combination should be skipped.

    For demographics-only scopes (age_sex_only, comorbidity_only, age_sex_comorbidity): do not skip
    based on voice/split conditions. For voice scopes, apply existing skip logic.
    """
    # Demographics-only scopes: no voice/split-based skips
    if scope and scope in DEMOGRAPHICS_ONLY_SCOPES:
        if hyperparam_variant == "basic" and classifier_variant != "RandomForestClassifier":
            return True
        return False

    # Skip basic hyperparameter with non-RandomForest classifiers
    if hyperparam_variant == "basic" and classifier_variant != "RandomForestClassifier":
        return True

    # Skip if non-poem recordings are used but ONLY uk_covid is included
    # UK COVID data only has poem recordings, so vowel features would result in empty datasets
    # But if both charite and uk_covid are included, it's valid (charite provides vowels, uk_covid provides poems)
    uses_non_poem_recordings = (
        any(cat != "poem" for cat in feature_variant.get("recordings_parsel", []))  # Any non-poem parselmouth
        or any(cat != "poem" for cat in feature_variant.get("recordings_wav2vec2", []))  # Any non-poem wav2vec
    )
    
    included_sources = task_config.get("included_data_sources", [])
    if uses_non_poem_recordings and "uk_covid" in included_sources and "charite" not in included_sources:
        return True

    # Splitting-based skip conditions: "split" is only valid for poem recordings
    if splitting_variant != "split":
        return False

    parsel_recordings = feature_variant.get("recordings_parsel", [])
    wav2vec_recordings = feature_variant.get("recordings_wav2vec2", [])
    has_poem_parsel = parsel_recordings == ["poem"]
    has_poem_wav2vec2 = "poem" in wav2vec_recordings
    has_non_poem_parsel = len(parsel_recordings) > 0 and not has_poem_parsel

    # Skip if non-poem parsel recordings are used (vowels cannot be split)
    # This includes mixed variants (e.g. wav2vec poem + parsel vowel): splitting
    # creates asymmetric recording counts (1 vowel vs N poem chunks per person),
    # which is not meaningful for training.
    if has_non_poem_parsel:
        return True
    # Skip if no poem recordings at all (nothing to split)
    if not has_poem_parsel and not has_poem_wav2vec2:
        return True
    return False


def find_feature_directories(base_dirs: list[str], exp_name: str) -> dict[str, str]:
    """Find the actual feature directories for a given experiment."""
    parsel_dir = embeddings_dir = ""

    for base_dir in base_dirs:
        if f"parselmouth__{exp_name}" in base_dir:
            parsel_dir = base_dir
        elif f"wav2vec2__{exp_name}" in base_dir:
            embeddings_dir = base_dir

    return {"parsel_dir": parsel_dir, "embeddings_dir": embeddings_dir}


def main():
    """Main entry point for config generation."""
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Generate experiment configuration files",
    )

    subparsers = parser.add_subparsers(dest="mode", help="Generation mode")

    # Single config generation
    single_parser = subparsers.add_parser(
        "single", help="Generate single configuration"
    )
    single_parser.add_argument("--exp", required=True, help="Experiment name")
    single_parser.add_argument(
        "--scope",
        required=True,
        help="Scope variant (e.g. age_sex_comorbidity, voice_plus_age_sex)",
    )
    single_parser.add_argument("--feature", required=True, help="Feature variant")
    single_parser.add_argument(
        "--hyperparam", required=True, help="Hyperparameter variant"
    )
    single_parser.add_argument("--task", required=True, help="Task variant")
    single_parser.add_argument("--classifier", required=True, help="Classifier variant")
    single_parser.add_argument("--splitting", required=True, help="Splitting variant")
    single_parser.add_argument(
        "--parsel-dir", required=True, help="Parselmouth feature directory"
    )
    single_parser.add_argument(
        "--embeddings-dir", required=True, help="Wav2Vec2 embeddings directory"
    )
    single_parser.add_argument(
        "--metadata-file", required=True, help="Metadata CSV file"
    )
    single_parser.add_argument(
        "--output", required=True, help="Output configuration file"
    )

    # Bulk config generation
    bulk_parser = subparsers.add_parser(
        "bulk", help="Generate all valid configurations"
    )
    bulk_parser.add_argument(
        "--output-dir", required=True, help="Output directory for config files"
    )
    bulk_parser.add_argument(
        "--parsel-dirs",
        nargs="+",
        required=True,
        help="Parselmouth feature directories",
    )
    bulk_parser.add_argument(
        "--embeddings-dirs",
        nargs="+",
        required=True,
        help="Wav2Vec2 embeddings directories",
    )
    bulk_parser.add_argument(
        "--metadata-files", nargs="+", required=True, help="Metadata CSV files"
    )
    bulk_parser.add_argument(
        "--experiment-version",
        default=None,
        help="Optional version suffix for MLFlow experiment names (e.g. 'v2')",
    )

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    try:
        if args.mode == "single":
            params = ExperimentParams(
                args.exp,
                args.scope,
                args.feature,
                args.hyperparam,
                args.task,
                args.classifier,
                args.splitting,
            )
            dirs = DirectoryConfig(
                args.parsel_dir, args.embeddings_dir, args.metadata_file, args.output
            )
            generator = ConfigurationGenerator()
            result = generator.generate_single_config(params, dirs)
            if result:
                logger.info(f"Successfully generated configuration: {result}")
            else:
                logger.warning("Configuration was skipped (invalid combination)")

        elif args.mode == "bulk":
            generator = ConfigurationGenerator(
                experiment_version=getattr(args, "experiment_version", None),
            )
            created_configs = generator.generate_all_configs(
                args.output_dir,
                args.parsel_dirs,
                args.embeddings_dirs,
                args.metadata_files,
            )
            logger.info(
                f"Successfully generated {len(created_configs)} valid configurations"
            )
            logger.info(f"Configurations saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Failed to generate configurations: {e}")
        raise


if __name__ == "__main__":
    main()
