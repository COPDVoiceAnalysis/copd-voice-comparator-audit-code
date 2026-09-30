# Code for "Demographic Comparator Specification Shapes the Apparent Advantage of Voice-Based COPD Screening: A Multi-Cohort Audit"

Analysis code for the article by J. K. Lieberwirth and M. Mittermaier (submitted to *Sensors*). The repository contains the code that produced the results reported in the article and its Supplementary material, and nothing else.

## Data

No data are included.

- **Charité cohort**: recordings and clinical data are not publicly available owing to participant privacy and the terms of the ethics approval (Ethics Committee of Charité – Universitätsmedizin Berlin, EA4/152/24).
- **COVID-19 Sounds**: available from its originating project (University of Cambridge) under its data-sharing conditions.
- **COPDVD**: publicly released by Idrisoglu et al. (2024); the scripts expect the spreadsheet at `data/copdvd/AnonymDataSet_ForBinaryClassificationOf_COPD.xlsx`.

The per-study codings of the literature scan (Section 2.8) are part of the article's Supplementary material (`literature_scan_codings.csv`).

## Environment

Python 3.11 with [uv](https://docs.astral.sh/uv/); `uv.lock` pins every dependency.

```bash
uv sync
```

The within-cohort and cross-cohort experiments ran on a SLURM cluster through Snakemake. Absolute paths in `Snakefile`, `config/pipeline_config.yml`, `src/train_cross_dataset.py` and a few scripts point to that cluster and must be adapted to the local data location. The Snakemake rule `sync`, which copies raw audio from institutional storage, is site-specific; its script is not included.

## Where each result comes from

| Article | Code |
|---|---|
| Pipeline: preprocessing, features, nested person-aware CV, MLflow tracking (§2.2–2.4) | `Snakefile`, `config/pipeline_config.yml`, `config/snakemake_matrix.yml`, `src/` |
| Comorbidity covariates (§2.1, §2.4) | `scripts/build_ps_covariates_csv.py` (Snakemake rule `build_comorbidities`) |
| COVID-19 Sounds analysis cohort: never-smoker controls, propensity matching (§2.1, §2.5) | `notebooks/UK/uk_metadata.ipynb`, `scripts/match_uk_controls.py` |
| Charité propensity-score matching, age-only linear propensity model, K = 5 (§2.5, §3.5) | `src/utils/generate_psm_cohort.py`, `src/utils/psm_matching.py` (Snakemake rule `match_charite_psm`) |
| K sweep and covariate balance (Tables S1, S2) | `scripts/reconstruct_psm_artifacts.py` |
| Table 1, Charité | `scripts/build_charite_demographics_table.py` |
| Table 1, COVID-19 Sounds and COPDVD | `scripts/paper_table1_uk_copdvd.py` |
| Export and aggregation of experiment runs | `scripts/export_v2_results.py`, `scripts/build_canonical_parquets.py` |
| Patient-level bootstrap, Table 2 (§2.6, §3.1–3.3, §3.5) | `scripts/paper_bootstrap_charite.py`, `scripts/paper_bootstrap_uk.py`, `scripts/paper_comparator_weighting_analysis.py` |
| Demographic comparators trained with and without covariate weights (§2.4, §3.2, Table S6) | `scripts/comparator_weighting_sensitivity.py` (reruns through `src/train_seed_sweep.py`), `scripts/paper_comparator_weighting_analysis.py` |
| Cross-dataset transfer, Table 3 (§2.7, §3.4) | `src/train_cross_dataset.py` (Snakemake rule `cross_dataset_validation`), `scripts/merge_xds_full_results.py` |
| COPDVD hold-out simulation (§3.6) | `scripts/copdvd_holdout_sim.py` |
| Age–score correlation among controls (§4.3) | `scripts/compute_age_score_correlation.py` |
| Design-factor decomposition, linear mixed model (Table S3) | `notebooks/lung_disease_runs_analysis.py`, `notebooks/lung_disease_runs_analysis_psm.py`, `scripts/lmm_logit_sensitivity.py` |
| COVID-19 Sounds configuration grid | `notebooks/uk_results_analysis.py` |
| Figures 1–3 | `scripts/paper_fig1_participant_flow.py`, `scripts/paper_fig2_voice_vs_baseline.py`, `scripts/copdvd_sim_figure.py`, shared style in `scripts/paper_fig_style.py` |

The notebooks are [marimo](https://marimo.io) notebooks (`uv run marimo edit <file>`), except `uk_metadata.ipynb` (Jupyter, outputs removed). Scripts are run from the repository root, e.g.

```bash
uv run snakemake --profile <slurm-profile> --jobs 200 --config date=2025-08-13
uv run python scripts/paper_bootstrap_charite.py
uv run python scripts/copdvd_holdout_sim.py pair48
```

The figure scripts write to the directory given as their first argument, or to `$COPDVD_AUDIT_FIGURES`.

## Citation

If you use this code, please cite the article.

## Acknowledgements

The poem segmentation (`src/utils/split_poem.py`) and parts of the PCA step were first written by Florian Zwicker.

## License

PolyForm Noncommercial License 1.0.0; see `LICENSE`. The code may be used, changed and shared for any noncommercial purpose, including research, teaching and personal use, and by noncommercial organisations; commercial use is not permitted.
