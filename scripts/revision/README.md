# Sensors revision (sensors-4632517): cluster runbook

This folder holds the analyses added for the major revision of *Demographic
Comparator Specification Shapes the Apparent Advantage of Voice-Based COPD
Screening*. Some of them need the COVID-19 Sounds (UK) features and the
`norm_on` features of the published run, which exist only on the BIH cluster.
This runbook is written so that someone who did not write the code (or their
coding assistant) can run everything there, check it, and send back the
results.

**In short:** clone the repo on the cluster, check the paths in one file, run
three commands, wait, run one more command, push `results/revision_r1/`.

```bash
git clone -b revision-sensors-r1 git@github.com:COPDVoiceAnalysis/copd-voice-comparator-audit-code.git
cd copd-voice-comparator-audit-code
$EDITOR scripts/revision/cluster/revision.env          # check every path
bash scripts/revision/cluster/00_preflight.sh          # read-only check, ~1 min
bash scripts/revision/cluster/01_setup.sh              # env + data links, login node
bash scripts/revision/cluster/run_all.sh               # submits all Slurm jobs
# ... wait until `squeue --me` is empty, then:
cat ~/work/revision_sensors_r1/analysis/STATUS.txt     # every line should say OK
bash scripts/revision/cluster/50_collect.sh            # copies aggregate results into the repo
git add results/revision_r1 && git commit -m "revision: cluster results" && git push
```

## 1. Data rules (read first)

- Patient data, recordings, metadata tables and fold predictions **stay on the
  cluster**. Nothing under `$OUT` (`~/work/revision_sensors_r1`) or `data/` is
  copied anywhere, and `data/` and `outputs/` are git-ignored.
- Only what `50_collect.sh` puts into `results/revision_r1/` leaves the cluster:
  aggregate JSON/CSV tables, figures and the preflight report. The collector
  refuses any file that has a participant or recording identifier field
  (`audio_id`, `study_id`, `uid`, `recording_identifier`, `birth_year`,
  `audio_sample_path`), any CSV over 500 rows and any file over 2 MB; refused
  files are listed in `results/revision_r1/MANIFEST.txt` and the script exits
  with code 1.
- Do not loosen these checks to make a file pass. If something needed is
  refused, ask the corresponding author.

## 2. What is computed, and why

| Reviewer point | What | Script | Output in `results/revision_r1/` |
|---|---|---|---|
| R1.1 incremental value | voice vs voice+age+sex vs voice+age+sex+comorbidities vs demographic comparators (weighted as published, and unweighted), paired participant bootstrap | `paired_bootstrap.py` | `paired_bootstrap_{charite,uk}.json` |
| R1.7 clinical operating points | sensitivity at specificity 0.80/0.90, specificity at sensitivity 0.80/0.90, with CIs | `operating_points.py` | `operating_points_{charite,uk}.json` |
| R2.1 comparator mechanism | participant-level predictions and per-fold weighted AUROC, voice vs age+sex (weighted / unweighted) | `fold_prediction_figure.py` | `fold_predictions_{charite,uk}.{pdf,png}` |
| R1.2 demographic confounding | age and sex decoded from the voice embeddings (whole cohort, matched subset, COPD only); voice score vs age within class; likelihood-ratio test of the score beyond age+sex | `demographics_from_voice.py` | `demographics_from_voice_{charite,uk}.json` |
| R1.5 cross-cohort, same language | PSM-matched Charité <-> UK, once with all UK participants and once with German-language UK participants only | `src/train_cross_dataset.py --uk_languages de`, `summarize_crossdataset.py` | `crossdataset_summary.csv` |
| R2.3 device / sample rate | sample rate, channels and duration of the original poem recordings per cohort and class | `recording_summary.py` | `recording_summary.csv` |
| R1.6 layer choice | Charité voice model with wav2vec2 layers 0-24 (mean pooling) and layer 4 with mean+std pooling | `arms.py make-configs --layers`, `paired_bootstrap.py` | `paired_bootstrap_layers.json` |
| paper numbers | if the published MLflow runs are readable: the same bootstrap and operating points on the exact published runs | `arms.py published` | `published_*.json` |

All model settings are the published ones: same selected configurations
(Charité: wav2vec2 layer 4, poem split into chunks, SVC; UK: wav2vec2 layer 4,
whole poem, logistic regression), same nested person-aware CV (5 outer splits ×
10 repeats, inner 5 × 3), same seed 42, same `ps_overlap` training weights and
strata-weighted evaluation. **Do not change seeds, grids, CV sizes or
weightings**: the point of the reruns is that they match the paper.

The model runs ("arms") are defined in one place, `arms.py`; the MLflow run
name of each arm is its config file name, so every result can be traced to its
config in `$OUT/configs/`.

## 3. The steps in detail

### 3.0 Edit `scripts/revision/cluster/revision.env`

It is the only file that should need editing. Check:

- `RAW_DIR`: group directory of the 2025-08-13 data snapshot. It must contain
  `merged_metadata.csv`, `comorbidities.csv`, `psm_matched_charite.csv`,
  `uk_matched_metadata.csv`, `uk_matched_metadata_ps_covariates.csv` and the
  raw recordings.
- `OLD_FEATURES_BASE`, `OLD_MLFLOW`: features and MLflow runs of the published
  analysis under the former account of the corresponding author (`joli13_c`). They are used only if
  readable; otherwise the features are rebuilt (see 3.3) and the "paper
  numbers" step is skipped.
- `NEW_FEATURES_BASE`, `OUT`: your own persistent storage (under `~/work`, not
  `~/scratch`, which is purged after 14 days).
- `UV_CACHE_DIR`, `HF_HOME`: caches on `~/work` because the home quota is small.
- `SBATCH_CPU_ARGS`, `SBATCH_GPU_ARGS`: partition names (`sinfo -s`). The GPU is
  needed only if the features must be rebuilt.

`uv` must be installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

### 3.1 `00_preflight.sh` (read-only)

Checks every input and prints one line per check (`OK`, `INFO`, `WARN`,
`FAIL`), and writes `$OUT/preflight_report.txt`. It ends with two verdicts:

- `VERDICT features`: which feature directory will be used: your own (already
  built), the published one (readable), or `BUILD NEEDED`.
- `VERDICT published runs`: how many of the six published runs are readable.

Fix every `FAIL` before going on (usually a path in `revision.env`). A `WARN`
about raw audio matters only if the features must be rebuilt, and for the
sample-rate table.

### 3.2 `01_setup.sh` (login node, needs internet)

Installs the locked Python environment (`uv sync --frozen`), downloads the
wav2vec2 model if the features must be rebuilt, and creates **symlinks** in
`data/` inside the repo (`data/2025-08-13/...` and `data/uk_matched_*.csv`)
pointing at the chosen feature directory and at `RAW_DIR`. The analysis scripts
read these relative paths. Nothing is copied.

### 3.3 `run_all.sh` (submits everything)

Submits the jobs below with Slurm dependencies and writes their ids to
`$OUT/state/jobs.txt`. Options: `--no-layers` skips the layer sweep, `--only
train,analysis` submits only some stages, `--local` runs everything in the
current shell instead of Slurm (for an interactive node or testing).

| Job | When | Resources | Rough time | What |
|---|---|---|---|---|
| `rev-preprocess` | only if BUILD NEEDED | 16 CPU, 32 GB | 1-2 h | resample/normalise recordings, split poems (rule `preprocess`) |
| `rev-parselmouth` | only if BUILD NEEDED | 48 CPU, 128 GB | up to 8 h | Parselmouth features (rule `parselmouth`); needed by config validation only |
| `rev-wav2vec` | only if BUILD NEEDED | 1 GPU | 1-3 h | wav2vec2 embeddings, all layers (rule `wav2vec_embeddings`) |
| `rev-configs` | after features | 1 CPU | 1 min | writes `$OUT/configs/*__config.yml` and `train_list.txt` |
| `rev-train` (array 0-31) | after configs | 16 CPU, 32 GB each | 0.5-3 h each | 6 voice arms + 26 layer-sweep runs (`src.train`) |
| `rev-comparators-charite` | after features | 8 CPU | < 1 h | demographic comparators, weighted and unweighted |
| `rev-comparators-uk` | after the train array | 8 CPU | < 1 h | same for UK; its cohort is the UK voice run's participants |
| `rev-xds-all`, `rev-xds-de` | after features | 32 CPU | up to 12 h | cross-cohort transfer, all UK / German UK |
| `rev-analysis` | after all of the above | 4 CPU | < 1 h | all analyses, aggregate outputs to `$OUT/analysis/` |

Logs: `$OUT/logs/<job-name>-<jobid>_<task>.out`, and per analysis step
`$OUT/logs/analysis_<step>.log`.

### 3.4 Check and collect

`$OUT/analysis/STATUS.txt` lists every analysis step as `OK` or `FAIL`. Then
`50_collect.sh` copies the aggregate files into `results/revision_r1/` with the
identifier guard and writes `MANIFEST.txt` (code commit, date, feature
directory, file list). Commit and push that folder to the branch
`revision-sensors-r1`. If pushing is not possible, send the folder as an
archive: it contains no participant data.

## 4. Testing the chain first (recommended)

Set `SMOKE=1` and a different `OUT` (e.g. `~/work/revision_sensors_r1_smoke`)
in `revision.env`, then run steps 00, 01 and `run_all.sh --no-layers`. SMOKE
shrinks every CV and grid so the whole chain finishes in well under an hour.
Its numbers are meaningless; `50_collect.sh` writes them to
`results/revision_r1_SMOKE/`, which must not be committed. Set `SMOKE=0` and
the real `OUT` back afterwards.

## 5. Re-running and fixing

- A finished training run leaves `$OUT/runs/<config_id>.done` and is skipped
  when the array is resubmitted, so `run_all.sh --only train,analysis` after a
  fix redoes only what failed. To redo one run, delete its `.done` file.
- `rev-analysis` is cheap; rerun it alone with `run_all.sh --only analysis`.
- The analysis picks the newest **finished** MLflow run per name, so failed or
  cancelled runs are ignored.

Common problems:

| Symptom | Likely cause | Fix |
|---|---|---|
| preflight `FAIL raw input ...` | `RAW_DIR` wrong or not readable | fix `RAW_DIR`; check group membership (`id`) |
| `BUILD NEEDED` although the old features exist | old account's files not readable for you | ask for read access, or let the build run (results are equivalent; the published `norm_on` pipeline is deterministic up to library versions, which `uv.lock` pins) |
| `rev-wav2vec` fails with no GPU / CUDA | GPU partition or gres wrong | fix `SBATCH_GPU_ARGS` |
| `rev-wav2vec` tries to download the model | `HF_HOME` differs between login and compute node | keep `HF_HOME` on shared storage and rerun `01_setup.sh` |
| `rev-comparators-uk` "UK voice run ... not finished" | the UK voice training failed | read its log in `$OUT/logs/rev-train-*`, fix, resubmit `--only train,comparators,analysis` |
| `Parselmouth feature file does not exist` | parselmouth job failed or skipped | rerun with `--only features` (or set `FEATURES_BASE_OVERRIDE`) |
| `data/2025-08-13 is a symlink` | an old `data/` link in this checkout | remove that link (it is not data, only a pointer), rerun `01_setup.sh` |
| `uv` re-resolves or downloads on a compute node | environment not installed | run `01_setup.sh` on the login node first |

Guidance for a coding assistant fixing a failure: change paths and Slurm
settings in `revision.env` first; do not edit the analysis settings (seeds,
grids, CV, weightings, metrics) or the identifier guard in
`collect_results.py`; keep fixes in separate commits on this branch with a
message that says what failed and why.

## 6. Sanity values

From the laptop rerun with the `norm_off` features (Charité, N = 99: 77 COPD /
22 controls), the `norm_on` rerun should land close to these:

| Quantity | Laptop (`norm_off`) | Paper (`norm_on`) |
|---|---|---|
| voice wAUROC, participant bootstrap | 0.787 [0.68, 0.88] | 0.80 [0.69, 0.90] |
| age / age+sex / age+sex+comorb, unweighted | 0.67 / 0.62 / 0.58 | 0.67 / 0.62 / 0.58 |
| age+sex, weighted (`ps_overlap`) | 0.39 | near or below 0.5 |
| voice+age+sex minus voice | -0.004 [-0.02, 0.01] | (new) |

Values far from these (e.g. a voice wAUROC near 0.5) point to a data or path
problem rather than a result; check `MANIFEST.txt` and the preflight report.

## 7. Files

```
scripts/revision/
  README.md                    this runbook
  arms.py                      arm definitions; make-configs / resolve / published
  paired_bootstrap.py          R1.1  paired participant bootstrap of AUROC differences
  operating_points.py          R1.7  sensitivity/specificity at fixed operating points
  fold_prediction_figure.py    R2.1  participant predictions and per-fold wAUROC
  simulate_comparator_collapse.py  R2.1  simulation (no data; already run on a laptop)
  demographics_from_voice.py   R1.2  age/sex from voice embeddings, age-proxy check
  recording_summary.py         R2.3  sample rates and durations
  summarize_crossdataset.py    R1.5  one table from the cross-cohort runs
  collect_results.py           copy aggregate outputs into results/ with the guard
  cluster/
    revision.env               the only file to edit
    common.sh                  shared shell setup (sourced)
    preflight.py, 00_preflight.sh
    01_setup.sh
    10/11/12_features_*.sbatch feature rebuild (only if needed)
    19_configs.sbatch, 20_train.sbatch
    21_comparators.sbatch, 30_crossdataset.sbatch
    40_analysis.sbatch
    50_collect.sh
    run_all.sh
```

Changed shared code (old behaviour unchanged by default):
`src/train_cross_dataset.py` (`--features_base`, `--raw_dir`, `--uk_languages`)
and `scripts/comparator_weighting_sensitivity.py` (`--out`, `--uk-voice-glob`,
`--preproc`).

Contact: the corresponding author of the paper.
