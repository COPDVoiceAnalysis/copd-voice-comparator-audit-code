import os
import datetime
import yaml
import json
import stat
import subprocess

# Load pipeline constants
configfile: "config/snakemake_matrix.yml"

with open("config/pipeline_config.yml") as f:
    pipeline_config = yaml.safe_load(f)

DATE = config[
    "date"
]  # This can be either set in the config file or passed as a command-line argument
# Optional version suffix for MLFlow experiment names (e.g. "v2").
# Data directories still use DATE; only the experiment name gets the suffix.
EXPERIMENT_VERSION = config.get("experiment_version", None)
experiment_version_flag = f"--experiment-version {EXPERIMENT_VERSION}" if EXPERIMENT_VERSION else ""
exp = config["preproc_variants"]
feature_variants = config["feature_variants"]
hyperparam = config["hyperparam_variants"]
tasks = config["classification_tasks"]
classifiers = config["classifier_variants"]
splitting = config["splitting_variants"]

# Experiment, hyperparameter, and task configurations
EXPS = [e["name"] for e in exp]
FEATURE_VARIANTS = feature_variants
HYPERPARAM_CONFIGS = hyperparam
TASKS = tasks
CLASSIFIERS = classifiers
SPLITTING_VARIANTS = splitting

# Raw data directory - PERSISTENT storage for original recordings
RAW_DIR = f"/data/cephfs-1/work/groups/mittermaier/stimmaufnahmen/{DATE}"

# Scratch directory base - TEMPORARY storage. NOTE: cubi cluster auto-purges
# files unmodified for >14 days, with only 3-day snapshot retention. We
# previously stored intermediate features here and lost norm_off twice when
# the run paused for >14 days. Features are now in FEATURES_BASE under /work.
SCRATCH_BASE = os.path.join(os.path.expanduser("~/scratch"), "pipeline_results", DATE)

# Work directory base - PERSISTENT storage for important results
WORK_BASE = os.path.join(os.path.expanduser("~/work"), "pipeline_results", DATE)

# Feature cache directory - PERSISTENT (under /work) so cluster auto-cleanup
# never wipes preprocessed audio / parselmouth features / wav2vec embeddings.
# Migrated from SCRATCH_BASE on 2026-05-07 with mtime preservation in the
# YAML configs to avoid re-triggering the train rules.
FEATURES_BASE = os.path.join(os.path.expanduser("~/work"), "pipeline_features", DATE)

# Intermediate processing directories (Work - Persistent)
PROCESSED_DIR_TPL = os.path.join(FEATURES_BASE, "preprocessed__{exp}")
PARSEL_FEATURE_DIR_TPL = os.path.join(FEATURES_BASE, "parselmouth__{exp}")
WAV2VEC2_DIR_TPL = os.path.join(FEATURES_BASE, "wav2vec2__{exp}")

# Workflow metadata directories (Work - Persistent)
PARAM_DIR = os.path.join(WORK_BASE, "params")
LOG_DIR = os.path.join(WORK_BASE, "logs")

# Note: Individual parameter files removed - all parameters now captured in complete experiment configs

os.makedirs(WORK_BASE, exist_ok=True)
os.makedirs(PARAM_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FEATURES_BASE, exist_ok=True)
os.makedirs(SCRATCH_BASE, exist_ok=True)

# Prevent config_id from greedily consuming the __config.yml suffix, and
# prevent {exp} from over-capturing path components (e.g. when match_charite_psm
# requests <dir(exp)>/metadata.csv, snakemake would otherwise match with
# exp="norm_on/metadata.csv" because rule preprocess outputs a directory).
wildcard_constraints:
    config_id="[^/.]+",
    exp="[^/]+",

# =======================================================
# Dynamic functions for checkpoint-based workflow
# =======================================================
import glob


def get_training_outputs(wildcards):
    """Get training output directories for generated configs."""
    # This function is called after the checkpoint completes
    checkpoint_output = checkpoints.generate_training_configs.get()

    # Find all generated config files
    config_files = glob.glob(os.path.join(PARAM_DIR, "*__config.yml"))

    # Create training output targets for each config
    # config_id includes scope (e.g. norm_off__age_sex_comorbidity__demographics_only__basic__... or norm_off__voice_plus_age_sex__parsel_poem_...__basic__...)
    training_outputs = []
    for config_file in config_files:
        config_id = os.path.basename(config_file).replace("__config.yml", "")
        training_output = os.path.join(WORK_BASE, f"{config_id}")
        training_outputs.append(training_output)

    return training_outputs


# =======================================================
# Rules
# =======================================================
rule all:
    input:
        training_outputs=get_training_outputs,
        cross_dataset_results=os.path.join(
            WORK_BASE, "cross_dataset_psm", "cross_dataset_results.json"
        ),


rule sync:
    output:
        sync_flag=os.path.join(RAW_DIR, ".sync_done"),
    resources:
        runtime="30m",
    retries: 0
    shell:
        """
        mkdir -p {RAW_DIR}
        bash shell_scripts/sync_sharepoint.sh -t {RAW_DIR}
        chmod -R 755 {RAW_DIR}
        touch {output.sync_flag}
        """


rule build_comorbidities:
    input:
        sync_flag=os.path.join(RAW_DIR, ".sync_done"),
        comorbidities_xlsx=os.path.join(
            RAW_DIR,
            pipeline_config.get("comorbidities_xlsx_basename", "comorbidities.xlsx"),
        ),
        voice_id_xlsx=os.path.join(
            RAW_DIR,
            pipeline_config.get("voice_audio_id_xlsx_basename", "voice_audio_id.xlsx"),
        ),
    output:
        csv=os.path.join(RAW_DIR, "comorbidities.csv"),
    resources:
        runtime="5m",
    retries: 0
    shell:
        """
        uv run python scripts/build_ps_covariates_csv.py \
            --comorbidities {input.comorbidities_xlsx} \
            --voice-id {input.voice_id_xlsx} \
            --output {output.csv}
        """


rule extract_metadata:
    input:
        os.path.join(RAW_DIR, ".sync_done"),
    output:
        os.path.join(RAW_DIR, "metadata.csv"),
    resources:
        runtime="15m",
    retries: 0
    shell:
        """
        uv run -m src.utils.extract_metadata \
            --directory {RAW_DIR} \
            --output_file {output}
        """


rule merge_datasets:
    input:
        charite_metadata=os.path.join(RAW_DIR, "metadata.csv"),
    output:
        merged_metadata=os.path.join(RAW_DIR, "merged_metadata.csv"),
    params:
        uk_metadata=os.path.join(RAW_DIR, "uk_matched_metadata.csv"),
        output_dir=os.path.join(RAW_DIR, "merged_metadata_temp"),
    resources:
        runtime="15m"
    retries: 0
    shell:
        """
        # Create temporary output directory
        mkdir -p {params.output_dir}
        
        # Run the merging script
        uv run -m src.utils.merge_datasets \
            --charite_metadata {input.charite_metadata} \
            --uk_metadata {params.uk_metadata} \
            --output_dir {params.output_dir}
        
        # Move the merged metadata to the expected location
        mv {params.output_dir}/metadata.csv {output.merged_metadata}
        
        # Clean up temporary directory
        rm -rf {params.output_dir}
        """


rule preprocess:
    input:
        merged_metadata=os.path.join(RAW_DIR, "merged_metadata.csv"),
    output:
        processed_dir=directory(PROCESSED_DIR_TPL),
    params:
        normalize_flag=lambda wc: (
            "--normalize"
            if next(e["normalize"] for e in exp if e["name"] == wc.exp)
            else "--no-normalize"
        ),
    resources:
        cpus=16,
        mem="32G",
        runtime="1h",
    retries: 0
    run:
        shell(""" mkdir -p {output.processed_dir} """)
        shell(
            """ uv run -m src.utils.recording_preprocessing \
            --input_dir {RAW_DIR} \
            --output_dir {output.processed_dir} \
            --metadata_file {input.merged_metadata} \
            {params.normalize_flag}
        """
        )
        shell(
            """
        uv run -m src.utils.split_poem \
            --metadata {output.processed_dir}/metadata.csv \
            --output_dir {output.processed_dir}
        """
        )


rule parselmouth:
    input:
        processed_dir=PROCESSED_DIR_TPL,
    output:
        parsel_dir=directory(PARSEL_FEATURE_DIR_TPL),
    params:
        metadata_file=lambda wc: f"{PROCESSED_DIR_TPL.format(exp=wc.exp)}/metadata.csv",
        output_file=lambda wc: f"{PARSEL_FEATURE_DIR_TPL.format(exp=wc.exp)}/acoustic_features.npy",
    resources:
        cpus=48,
        mem="128G",
        runtime="8h",
    retries: 1
    run:
        shell(f""" mkdir -p {output.parsel_dir}""")
        shell(
            f""" uv run -m src.features.parselmouth.extractor \
            --output_file {params.output_file} \
            --metadata_file {params.metadata_file} \
            --max_workers {resources.cpus}
        """
        )


rule wav2vec_embeddings:
    input:
        processed_dir=PROCESSED_DIR_TPL,
    output:
        embeddings_dir=directory(WAV2VEC2_DIR_TPL),
    params:
        metadata_file=lambda wc: f"{PROCESSED_DIR_TPL.format(exp=wc.exp)}/metadata.csv",
    resources:
        cpus=16,
        mem="16G",
        runtime="3h",
        slurm_partition="gpu",
        gpu=1,
        gpu_model="tesla",
    retries: 0
    run:
        shell(f""" mkdir -p {output.embeddings_dir}""")
        shell(f""" export SLURM_CPUS_PER_TASK={resources.cpus}""")
        shell(f""" export AVAILABLE_MEMORY="{resources.mem}" """)
        shell(
            f""" uv run -m src.features.wav2vec2.get_embeddings \
            --embeddings_dir {output.embeddings_dir} \
            --metadata_file {params.metadata_file}
        """
        )


rule match_charite_psm:
    """Propensity-score match the Charité cohort.

    Emits a metadata CSV (same schema as the preprocessed metadata, filtered to
    the PSM-matched persons and their recording-level rows) plus a balance
    report. Consumed by task ``copd_classification_charite_psm`` via its
    ``metadata_file`` override in pipeline_config.yml.

    Matching parameters (K, caliper, propensity covariates, seed) come from
    ``psm_matching.charite`` in pipeline_config.yml so the output is fully
    reproducible from a fresh checkout.
    """
    input:
        metadata=os.path.join(
            PROCESSED_DIR_TPL.format(exp=EXPS[0]),
            "metadata.csv",
        ),
        comorbidities=os.path.join(RAW_DIR, "comorbidities.csv"),
        pipeline_config="config/pipeline_config.yml",
    output:
        matched_csv=os.path.join(RAW_DIR, "psm_matched_charite.csv"),
        balance_report=os.path.join(RAW_DIR, "psm_matched_charite_balance_report.csv"),
    resources:
        runtime="10m",
    retries: 0
    shell:
        """
        uv run python -m src.utils.generate_psm_cohort \
            --metadata {input.metadata} \
            --comorbidities {input.comorbidities} \
            --config {input.pipeline_config} \
            --output-csv {output.matched_csv} \
            --balance-report {output.balance_report} \
            --require-recording-categories a i o poem \
            --exclude-longitudinal \
            --cohort charite
        """


checkpoint generate_training_configs:
    input:
        parsel_dirs=expand(PARSEL_FEATURE_DIR_TPL, exp=EXPS),
        embeddings_dirs=expand(WAV2VEC2_DIR_TPL, exp=EXPS),
        comorbidities_csv=os.path.join(RAW_DIR, "comorbidities.csv"),
        psm_matched_charite=os.path.join(RAW_DIR, "psm_matched_charite.csv"),
    output:
        flag=os.path.join(PARAM_DIR, ".config_creation_done"),
        mlflow_flag=os.path.join(PARAM_DIR, ".mlflow_experiments_created"),
    params:
        metadata_files=lambda wc: [
            f"{PROCESSED_DIR_TPL.format(exp=exp)}/metadata.csv" for exp in EXPS
        ],
    resources:
        runtime="30m",
    retries: 0
    shell:
        """
        echo "Starting experiment config generation..."
        echo "Output directory: {PARAM_DIR}"
        
        # Preserve old configs so we can restore mtimes for unchanged ones
        # (prevents Snakemake from re-triggering training jobs whose config
        # content hasn't actually changed)
        echo "Backing up old config files..."
        mkdir -p {PARAM_DIR}/.old_configs
        mv {PARAM_DIR}/*__config.yml {PARAM_DIR}/.old_configs/ 2>/dev/null || true
        rm -f {PARAM_DIR}/.config_creation_done
        rm -f {PARAM_DIR}/.mlflow_experiments_created

        echo "Parselmouth directories: {input.parsel_dirs}"
        echo "Embeddings directories: {input.embeddings_dirs}"
        echo "Metadata files: {params.metadata_files}"

        uv run -m src.utils.generate_experiment_configs bulk \
            --output-dir {PARAM_DIR} \
            --parsel-dirs {input.parsel_dirs} \
            --embeddings-dirs {input.embeddings_dirs} \
            --metadata-files {params.metadata_files} \
            {experiment_version_flag}

        # Restore mtimes for configs whose content hasn't changed
        echo "Restoring timestamps for unchanged configs..."
        for f in {PARAM_DIR}/*__config.yml; do
            old="{PARAM_DIR}/.old_configs/$(basename "$f")"
            if [ -f "$old" ] && cmp -s "$f" "$old"; then
                touch -r "$old" "$f"
            fi
        done
        rm -rf {PARAM_DIR}/.old_configs

        echo "Config generation completed successfully"
        touch {output.flag}
        
        echo "Pre-creating MLFlow experiments to prevent race conditions..."
        uv run -m src.utils.create_mlflow_experiments \
            --config-dir {PARAM_DIR} \
            --verbose
        
        echo "MLFlow experiment pre-creation completed"
        touch {output.mlflow_flag}
        """


rule train:
    input:
        config_file=os.path.join(PARAM_DIR, "{config_id}__config.yml"),
    output:
        output_dir=directory(os.path.join(WORK_BASE, "{config_id}")),
    resources:
        mem="32G",
        cpus_per_task=32,
        runtime="96h",
    retries: 0
    shell:
        """
        echo "Starting training for config: {wildcards.config_id}"
        echo "Config file: {input.config_file}"
        echo "Output directory: {output.output_dir}"

        mkdir -p {output.output_dir}
        uv run -m src.train \
            --config {input.config_file} \
            --output_dir {output.output_dir}

        echo "Training completed for config: {wildcards.config_id}"
        """


rule cross_dataset_validation:
    """Cross-cohort generalisation (train on one cohort, test on the other).

    Primary analysis uses the PSM-matched Charité cohort (produced by rule
    match_charite_psm) as the training/test side; the UK COVID-19 Sounds cohort
    loads from the standard preprocessed metadata. Evaluates all three
    classifiers in both directions, across splitting variants, using the
    extended hyperparameter grid (see src/train_cross_dataset.py). Output is a
    single JSON with bootstrapped AUROC / wBA_opt CIs per direction × classifier
    × splitting, which downstream notebooks aggregate into the canonical
    cross-dataset parquet.
    """
    input:
        psm_matched_charite=os.path.join(RAW_DIR, "psm_matched_charite.csv"),
        processed_dir=PROCESSED_DIR_TPL.format(exp="norm_on"),
        parsel_dir=PARSEL_FEATURE_DIR_TPL.format(exp="norm_on"),
        embeddings_dir=WAV2VEC2_DIR_TPL.format(exp="norm_on"),
    output:
        results_json=os.path.join(
            WORK_BASE, "cross_dataset_psm", "cross_dataset_results.json"
        ),
    params:
        output_dir=os.path.join(WORK_BASE, "cross_dataset_psm"),
        experiment_version_flag=(
            f"--experiment_version {EXPERIMENT_VERSION}" if EXPERIMENT_VERSION else ""
        ),
    resources:
        mem="32G",
        cpus=32,
        runtime="12h",
    retries: 0
    shell:
        """
        echo "Starting cross-dataset validation (PSM-matched Charité primary)"
        echo "Charité metadata: {input.psm_matched_charite}"
        echo "Output directory: {params.output_dir}"

        mkdir -p {params.output_dir}
        uv run python -m src.train_cross_dataset \
            --date {DATE} \
            --preproc norm_on \
            --output_dir {params.output_dir} \
            --charite_metadata_csv {input.psm_matched_charite} \
            --n_jobs {resources.cpus} \
            {params.experiment_version_flag}

        echo "Cross-dataset validation completed"
        """




# rule train_temporal:
#    input:
#        parsel_dir = PARSEL_FEATURE_DIR_TPL,
#        embeddings_dir = WAV2VEC2_DIR_TPL
#    output:
#        output_dir = directory(os.path.join(WORK_BASE, "temporal__{exp}")),
#        params_file = os.path.join(PARAM_DIR, "temporal__{exp}__params.yml"),
#        temporal_train_done = os.path.join(WORK_BASE, "temporal__{exp}", ".train_done")
#    params:
#        metadata_file = lambda wc: f"{PROCESSED_DIR_TPL.format(exp=wc.exp)}/metadata.csv",
#        config_file = "config/hyperparameters_temporal.yml"
#    resources:
#        mem = "32G",
#        cpus = 32,
#        runtime = "2h",
#        slurm_extra = lambda wc: f"--output={LOG_DIR}/temporal_{wc.exp}.log --error={LOG_DIR}/temporal_{wc.exp}.err"
#    retries: 0
#    run:
#        import yaml
#        from datetime import datetime
#
#        # Load temporal config
#        with open(params.config_file, "r") as f:
#            temporal_config = yaml.safe_load(f)
#
#        # Update paths in config
#        temporal_config["parsel_feature_dir"] = input.parsel_dir
#        temporal_config["embeddings_dir"] = input.embeddings_dir
#
#        # Save updated config
#        temp_config_path = f"{output.output_dir}/temporal_config.yml"
#        os.makedirs(output.output_dir, exist_ok=True)
#        with open(temp_config_path, "w") as f:
#            yaml.dump(temporal_config, f, default_flow_style=False)
#
#        # Save parameters for reproducibility
#        temporal_params = {
#            "rule": "train_temporal",
#            "experiment": wildcards.exp,
#            "timestamp": datetime.now().isoformat(),
#            "parameters": {
#                "metadata_file": params.metadata_file,
#                "embeddings_dir": input.embeddings_dir,
#                "parsel_feature_dir": input.parsel_dir,
#                "output_dir": output.output_dir,
#                "config_file": params.config_file
#            },
#            "temporal_config": temporal_config
#        }
#        os.makedirs(os.path.dirname(output.params_file), exist_ok=True)
#        with open(output.params_file, "w") as f:
#            yaml.dump(temporal_params, f, default_flow_style=False)
#
#        # Find the raw data directory for this experiment
#        metadata_path = f"{PROCESSED_DIR_TPL.format(exp=wildcards.exp)}/metadata.csv"
#
#        shell(f""" uv run -m src.train_temporal \
#            --config {temp_config_path} \
#            --metadata {metadata_path} \
#            --output_dir {output.output_dir}
#        """)
#        shell(f"""touch {output.temporal_train_done}""")


