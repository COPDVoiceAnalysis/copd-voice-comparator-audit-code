#!/bin/bash
#SBATCH --job-name=rev-xds
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# Cross-cohort transfer (Reviewer 1, point 5), Charité <-> UK.
# Arguments: <all|de> [psm|full]
#   all   every UK participant (reruns the published analysis with this code)
#   de    only UK participants whose recording language is German
#   psm   Charité side = PSM-matched cohort (N = 55), as rule `cross_dataset_validation` (default)
#   full  Charité side = the analysed full cohort (N = 99, as in Table 3): exactly the
#         participants scored by the selected Charité voice run, so it needs a finished
#         rev-analysis ($OUT/arms/charite.json)
# Output: $OUT/crossdataset/<all|de>[_full]/; afterwards the summary table
# $OUT/analysis/crossdataset_summary.csv is rebuilt from every finished variant.
source "${REV_HERE:?submit through run_all.sh}/common.sh"
VARIANT=${1:?usage: 30_crossdataset.sbatch all|de [psm|full]}
COHORT=${2:-psm}
FB=$(features_base)
A="$OUT/analysis"; mkdir -p "$A"
TAG=$VARIANT; [ "$COHORT" = full ] && TAG=${VARIANT}_full
case $COHORT in
  psm)  CMETA="data/$DATE/psm_matched_charite.csv" ;;
  full)
    CMETA="$OUT/crossdataset/charite_complete_metadata_$VARIANT.csv"
    mkdir -p "$OUT/crossdataset"
    VC=$(py -c "import json,sys; print(json.load(open(sys.argv[1]))['arms']['voice'])" "$OUT/arms/charite.json") \
      || die "no Charité voice run in $OUT/arms/charite.json (run rev-analysis first)"
    py scripts/revision/charite_complete_metadata.py --voice-run "$VC" \
      --metadata "data/$DATE/preprocessed__$PREPROC/metadata.csv" --out "$CMETA" \
      || { echo "FAIL  crossdataset_$TAG (Charité cohort filter)" >> "$A/STATUS.txt"; die "Charité cohort filter failed"; } ;;
  *) die "unknown cohort '$COHORT' (psm|full)" ;;
esac
ARGS=(--date "$DATE" --preproc "$PREPROC" --output_dir "$OUT/crossdataset/$TAG"
      --charite_metadata_csv "$CMETA"
      --features_base "$FB" --raw_dir "$RAW_DIR" --n_jobs "${SLURM_CPUS_PER_TASK:-4}")
[ "$VARIANT" = de ] && ARGS+=(--uk_languages de)
[ "$SMOKE" = 1 ] && ARGS+=(--classifiers LogisticRegression --splittings unsplit)
log "cross-dataset $TAG"
if uv run --frozen -m src.train_cross_dataset "${ARGS[@]}"; then
  echo "OK    crossdataset_$TAG" >> "$A/STATUS.txt"
else
  echo "FAIL  crossdataset_$TAG (see the job log)" >> "$A/STATUS.txt"; die "transfer $TAG failed"
fi
XDS=("$OUT"/crossdataset/*/cross_dataset_results.json)
py scripts/revision/summarize_crossdataset.py "$A/crossdataset_summary.csv.tmp.$TAG" "${XDS[@]}" \
  && mv "$A/crossdataset_summary.csv.tmp.$TAG" "$A/crossdataset_summary.csv"
log "done"
