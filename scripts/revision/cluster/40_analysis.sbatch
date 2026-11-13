#!/bin/bash
#SBATCH --job-name=rev-analysis
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
# All revision analyses on the finished runs. Each step runs on its own: a
# failing step is logged in $OUT/analysis/STATUS.txt and the next one still runs.
# Outputs in $OUT/analysis are aggregate (no participant rows).
source "${REV_HERE:?submit through run_all.sh}/common.sh"
set +e
A="$OUT/analysis"; ARMS="$OUT/arms"
mkdir -p "$A" "$ARMS"
: > "$A/STATUS.txt"
step() {  # step <name> <command...>
  local name=$1; shift
  log "step $name"
  if "$@" > "$OUT/logs/analysis_$name.log" 2>&1; then
    echo "OK    $name" >> "$A/STATUS.txt"
  else
    echo "FAIL  $name (see $OUT/logs/analysis_$name.log)" >> "$A/STATUS.txt"
    log "FAILED $name"
  fi
}
armrun() { py -c "import json,sys; print(json.load(open(sys.argv[1]))['arms'][sys.argv[2]])" "$@"; }

step resolve py scripts/revision/arms.py resolve --preproc "$PREPROC" --mlruns "$OUT/mlruns" \
  --comparator-mlruns "$OUT/comparators/mlruns" --out-dir "$ARMS"

for c in charite uk; do
  [ -f "$ARMS/$c.json" ] || { echo "FAIL  $c (no arm file)" >> "$A/STATUS.txt"; continue; }
  cp "$ARMS/$c.json" "$A/arms_$c.json"
  step "paired_bootstrap_$c" sh -c "uv run --frozen python scripts/revision/paired_bootstrap.py '$ARMS/$c.json' > '$A/paired_bootstrap_$c.json'"
  step "operating_points_$c" sh -c "uv run --frozen python scripts/revision/operating_points.py '$ARMS/$c.json' > '$A/operating_points_$c.json'"
  step "figure_$c" py scripts/revision/fold_prediction_figure.py "$ARMS/${c}_figure.json" "$A/fold_predictions_$c.pdf"
done

VC=$(armrun "$ARMS/charite.json" voice 2>/dev/null)
VU=$(armrun "$ARMS/uk.json" voice 2>/dev/null)
[ -n "$VC" ] && step demographics_charite py scripts/revision/demographics_from_voice.py --cohort charite \
  --data-dir "data/$DATE" --preproc "$PREPROC" --split --voice-run "$VC" \
  --matched-ids-csv "data/$DATE/psm_matched_charite.csv" --out "$A/demographics_from_voice_charite.json"
[ -n "$VU" ] && step demographics_uk py scripts/revision/demographics_from_voice.py --cohort uk \
  --data-dir "data/$DATE" --preproc "$PREPROC" --no-split --voice-run "$VU" \
  --out "$A/demographics_from_voice_uk.json"

COH=(); [ -n "$VC" ] && COH+=(--cohort "charite=$VC"); [ -n "$VU" ] && COH+=(--cohort "uk_covid=$VU")
[ ${#COH[@]} -gt 0 ] && step recording_summary py scripts/revision/recording_summary.py \
  --metadata "$RAW_DIR/merged_metadata.csv" --audio-dir "$RAW_DIR" "${COH[@]}" --out "$A/recording_summary.csv"

XDS=("$OUT"/crossdataset/*/cross_dataset_results.json)
[ -e "${XDS[0]}" ] && step crossdataset py scripts/revision/summarize_crossdataset.py "$A/crossdataset_summary.csv" "${XDS[@]}"

if [ -f "$ARMS/layers.json" ]; then
  cp "$ARMS/layers.json" "$A/arms_layers.json"
  step layers sh -c "uv run --frozen python scripts/revision/paired_bootstrap.py '$ARMS/layers.json' > '$A/paired_bootstrap_layers.json'"
fi

if [ "$(cat "$OUT/state/published_available" 2>/dev/null)" = 1 ]; then
  step published_arms py scripts/revision/arms.py published --mlflow "$OLD_MLFLOW" --out-dir "$ARMS"
  for c in charite uk; do
    [ -f "$ARMS/published_$c.json" ] || continue
    step "published_paired_bootstrap_$c" sh -c "uv run --frozen python scripts/revision/paired_bootstrap.py '$ARMS/published_$c.json' > '$A/published_paired_bootstrap_$c.json'"
    step "published_operating_points_$c" sh -c "uv run --frozen python scripts/revision/operating_points.py '$ARMS/published_$c.json' > '$A/published_operating_points_$c.json'"
  done
fi

log "analysis finished:"; cat "$A/STATUS.txt"
