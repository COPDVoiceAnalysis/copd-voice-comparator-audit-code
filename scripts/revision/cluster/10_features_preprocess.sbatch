#!/bin/bash
#SBATCH --job-name=rev-preprocess
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# Feature build 1/3 (only when the published features are not readable):
# resample/normalise the raw recordings and split the poem, as rule `preprocess`
# in the Snakefile. Writes $NEW_FEATURES_BASE/preprocessed__$PREPROC.
source "${REV_HERE:?submit through run_all.sh}/common.sh"
FB=$(features_base)
[ "$FB" = "$NEW_FEATURES_BASE" ] || die "refusing to build features outside NEW_FEATURES_BASE ($FB)"
NORM=$([ "$PREPROC" = norm_on ] && echo --normalize || echo --no-normalize)
OUTD="$FB/preprocessed__$PREPROC"
mkdir -p "$OUTD"
log "preprocess $RAW_DIR -> $OUTD ($NORM)"
uv run --frozen -m src.utils.recording_preprocessing --input_dir "$RAW_DIR" --output_dir "$OUTD" \
  --metadata_file "$RAW_DIR/merged_metadata.csv" $NORM
uv run --frozen -m src.utils.split_poem --metadata "$OUTD/metadata.csv" --output_dir "$OUTD"
log "done"
