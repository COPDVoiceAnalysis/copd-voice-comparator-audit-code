#!/bin/bash
#SBATCH --job-name=rev-xds
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# Cross-cohort transfer (Reviewer 1, point 5), PSM-matched Charité <-> UK, as
# rule `cross_dataset_validation`. Argument: all | de
#   all  every UK participant (reruns the published analysis with this code)
#   de   only UK participants whose recording language is German
source "${REV_HERE:?submit through run_all.sh}/common.sh"
VARIANT=${1:?usage: 30_crossdataset.sbatch all|de}
FB=$(features_base)
ARGS=(--date "$DATE" --preproc "$PREPROC" --output_dir "$OUT/crossdataset/$VARIANT"
      --charite_metadata_csv "data/$DATE/psm_matched_charite.csv"
      --features_base "$FB" --raw_dir "$RAW_DIR" --n_jobs "${SLURM_CPUS_PER_TASK:-4}")
[ "$VARIANT" = de ] && ARGS+=(--uk_languages de)
[ "$SMOKE" = 1 ] && ARGS+=(--classifiers LogisticRegression --splittings unsplit)
log "cross-dataset $VARIANT"
uv run --frozen -m src.train_cross_dataset "${ARGS[@]}"
log "done"
