"""Retrain the demographic comparators with and without covariate weighting.

The published comparators (age+sex, age+sex+comorbidity) were trained with
sample_weight_strategy "ps_overlap", whose propensity model uses age, sex and
the comorbidity indicators -- the comparators' own predictors. This script
reruns the comparator scopes under "ps_overlap" and "flat", all else equal, so
the effect of the training weights can be read off directly.

Cohorts:
  charite  copd_classification_charite_complete (N = 99), three scopes.
  charite_psm  copd_classification_charite_psm (N = 55), age only. The
           published PSM "age only" run (8f8cfddc...) resolved no scope and fell
           back to TrainingConfig defaults: 193 Parselmouth /a/ features plus
           age, sex and BMI. This is the age-only model it was meant to be.
  uk       copd_classification_uk_only, age+sex and age+sex+comorbidity. The
           local preprocessed metadata predates the cluster's, so the cohort
           is rebuilt from data/uk_matched_metadata.csv restricted to the 902
           participants of the published voice_uk run and written to
           outputs/comparator_weighting/uk_metadata_902.csv.

Grids:
  extended     LogisticRegression C in {0.01, 0.1, 0.5}, l1/l2, saga -- the
               grid of the published age+sex and age+sex+comorbidity runs.
  lr_baseline  LogisticRegression C in {0.01, 0.1, 1, 10}, l2, lbfgs -- the
               grid of the published age-only run (seed-sweep protocol).

Every (scope, weighting, grid) tuple is one seed-sweep config run through
src.train_seed_sweep, which reuses src.train.run_training (outer 10x5, inner
3x5, person-aware). One MLflow run per seed lands in
outputs/comparator_weighting/mlruns; scripts/paper_comparator_weighting_analysis.py
reads them. n_jobs is 1 per run and tuples run in parallel instead; results
do not depend on n_jobs.

Run from the repository root:
    PYTHONPATH=. uv run python scripts/comparator_weighting_sensitivity.py
    PYTHONPATH=. uv run python scripts/comparator_weighting_sensitivity.py \
        --cohort uk --grids extended --seeds 42
    PYTHONPATH=. uv run python scripts/comparator_weighting_sensitivity.py \
        --cohort charite_psm --weightings flat --seeds 42
"""
import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

OUT = Path("outputs/comparator_weighting")
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
UK_METADATA = OUT / "uk_metadata_902.csv"
UK_VOICE = ("data/2025-08-13/mlflow_cluster/271009020022062271/"
            "de720542f5974ee8998d966ca75a536f/artifacts/fold_predictions/*.parquet")
UK_COMORB = ["angina", "asthma", "cancer", "diabetes", "hbp", "heart", "hiv",
             "longterm", "organ", "otherheart", "stroke", "valvular"]

COHORTS = {
    "charite": {
        "task": "copd_classification_charite_complete",
        "metadata_file": "data/2025-08-13/preprocessed__norm_on/metadata.csv",
        "ps_covariates_file": "data/2025-08-13/comorbidities.csv",
        "included_data_sources": ["charite"],
        "require_all_recording_categories": ["a", "i", "o", "poem"],
        "scopes": {
            "age_only": ["age"],
            "age_sex_only": ["age", "sex"],
            # from_ps_covariates on the Charité task: age, sex + the four clusters
            "age_sex_comorbidity_individual": ["age", "sex", "breathing", "metabolic",
                                               "depression", "hypothyreose"],
        },
    },
    "charite_psm": {
        "task": "copd_classification_charite_psm",
        "metadata_file": "data/2025-08-13/psm_matched_charite.csv",
        "ps_covariates_file": "data/2025-08-13/comorbidities.csv",
        "included_data_sources": ["charite"],
        "require_all_recording_categories": ["a", "i", "o", "poem"],
        "scopes": {"age_only": ["age"]},
    },
    "uk": {
        "task": "copd_classification_uk_only",
        "metadata_file": str(UK_METADATA),
        "ps_covariates_file": "data/uk_matched_metadata_ps_covariates.csv",
        "included_data_sources": ["uk_covid"],
        "require_all_recording_categories": [],
        "scopes": {
            "age_sex_only": ["age", "sex"],
            "age_sex_comorbidity_individual": ["age", "sex", *UK_COMORB],
        },
    },
}

GRIDS = {
    "extended": {"clf__C": [0.01, 0.1, 0.5], "clf__penalty": ["l1", "l2"],
                 "clf__solver": ["saga"]},
    "lr_baseline": {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l2"],
                    "clf__solver": ["lbfgs"]},
}
WEIGHTINGS = ["ps_overlap", "flat"]


def config_id(task, scope, grid, weighting):
    return f"{scope}__{grid}__{task}__{weighting}"


def make_config(cohort, scope, grid, weighting):
    c = COHORTS[cohort]
    task = c["task"]
    cid = config_id(task, scope, grid, weighting)
    return {
        "classification_task": task,
        "date": "2025-08-13",
        "description": f"Comparator weighting sensitivity: scope={scope}, "
                       f"grid={grid}, weighting={weighting}",
        "experiment_name": f"comparator_weighting__{task}",
        "experiment_version": "comparator_weighting",
        "feature_config": {
            "demographics_only": True,
            "embeddings_dir": "data/2025-08-13/wav2vec2__norm_on",
            "parsel_feature_file": "data/2025-08-13/parselmouth__norm_on/acoustic_features.npy",
            "parselmouth_config": {
                "exact_feature_list": None,
                "include_advanced_voice_quality": False,
                "include_basic_acoustics": False,
                "include_egemaps_features": True,
                "include_mfcc_derivatives": False,
                "include_spectral_analysis": False,
                "include_temporal_features": False,
            },
            "recordings_parsel": [],
            "recordings_wav2vec2": [],
            "wav2vec2_embedding_statistics": "mean",
            "wav2vec2_layer": 4,
        },
        "metadata_file": c["metadata_file"],
        "output_dir": str(OUT / "runs" / cid),
        "ps_covariates_file": c["ps_covariates_file"],
        "tags": [scope, grid, task, f"weighting={weighting}", "comparator_weighting"],
        "task_config": {
            "exclude_longitudinal": True,
            "included_data_sources": c["included_data_sources"],
            "only_longitudinal": False,
            "ps_covariates_file": None,
            "require_all_recording_categories": c["require_all_recording_categories"],
            "target_classes": ["copd", "control"],
            "target_column": "lung_disease_main",
            "use_split_poems": False,
            "use_temporal_pairs": False,
        },
        "training_config": {
            "classifier_hyperparams": GRIDS[grid],
            "classifier_name": "LogisticRegression",
            "demographic_feature_columns": c["scopes"][scope],
            "include_voice_features": False,
            "min_controls_per_stratum": 2,
            "n_jobs": 1,
            "n_repeats_inner": 3,
            "n_repeats_outer": 10,
            "n_splits_inner": 5,
            "n_splits_outer": 5,
            "pca_hyperparams": {"PCA__parsel_n_components": [10],
                                "PCA__wav2vec_n_components": [15]},
            "pi_deploy": 0.2,
            "random_seed": SEEDS[0],
            "sample_weight_strategy": weighting,
            "scoring_metric": "auroc",
            "weight_clip_high_percentile": 99.0,
            "weight_clip_low_percentile": 1.0,
        },
    }


def write_uk_metadata():
    """UK cohort of the published comparator runs: the 902 voice_uk participants."""
    ids = pd.concat([pd.read_parquet(f) for f in glob.glob(UK_VOICE)])["audio_id"].unique()
    meta = pd.read_csv("data/uk_matched_metadata.csv")
    meta[meta["audio_id"].isin(ids)].to_csv(UK_METADATA, index=False)


def run(cid, cfg_path, seeds):
    env = dict(os.environ)
    uri = f"file://{(OUT / 'mlruns').resolve()}"
    env.update(MLFLOW_TRACKING_URI=uri, MLFLOW_ARTIFACT_URI=uri, PYTHONPATH=".")
    log = OUT / "runs" / f"{cid}.log"
    cmd = [sys.executable, "-m", "src.train_seed_sweep", "--config", str(cfg_path),
           "--output_dir", str(OUT / "runs" / cid), "--seeds", *map(str, seeds)]
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    print(f"{'OK  ' if rc == 0 else 'FAIL'} {cid}", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=list(COHORTS), default="charite")
    ap.add_argument("--scopes", nargs="+", default=None)
    ap.add_argument("--grids", nargs="+", default=list(GRIDS))
    ap.add_argument("--weightings", nargs="+", default=WEIGHTINGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    c = COHORTS[a.cohort]

    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    if a.cohort == "uk":
        write_uk_metadata()
    jobs = []
    for grid in a.grids:
        for scope in a.scopes or list(c["scopes"]):
            for w in a.weightings:
                cid = config_id(c["task"], scope, grid, w)
                path = OUT / "configs" / f"{cid}__seed_sweep_config.yml"
                path.write_text(yaml.safe_dump(make_config(a.cohort, scope, grid, w),
                                               sort_keys=False))
                jobs.append((cid, path))
    with ThreadPoolExecutor(a.workers) as ex:
        rcs = list(ex.map(lambda j: run(*j, a.seeds), jobs))
    sys.exit(max(rcs))


if __name__ == "__main__":
    main()
