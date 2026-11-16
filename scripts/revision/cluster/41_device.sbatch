#!/bin/bash
#SBATCH --job-name=rev-device
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Sample rate / device check (Reviewer 2, point 3): does the voice score hold
# when COPD and controls are compared only within the same sample rate?
# Analysis only (no training); reads the finished runs via $OUT/arms/*.json,
# so it needs a finished rev-analysis (or `arms.py resolve`) first.
source "${REV_HERE:?submit through run_all.sh}/common.sh"
set +e
A="$OUT/analysis"; ARMS="$OUT/arms"
mkdir -p "$A"
armrun() { py -c "import json,sys; print(json.load(open(sys.argv[1]))['arms'].get(sys.argv[2], ''))" "$@"; }
COMMON=(--metadata "$OUT/merged_metadata_remapped.csv" --audio-dir "$RAW_DIR"
        --rate-metadata "data/$DATE/preprocessed__$PREPROC/metadata.csv" --data-dir "data/$DATE" --preproc "$PREPROC")
for c in charite uk; do
  split=--split; [ "$c" = uk ] && split=--no-split
  for arm in "voice" "voice (unweighted)"; do
    RUN=$(armrun "$ARMS/$c.json" "$arm" 2>/dev/null)
    [ -n "$RUN" ] || { echo "FAIL  device_${c} ($arm: run not found)" >> "$A/STATUS.txt"; continue; }
    tag=$c; [ "$arm" = "voice (unweighted)" ] && tag=${c}_unweighted
    log "device check $tag"
    if py scripts/revision/device_confound.py --cohort "$c" --voice-run "$RUN" $split "${COMMON[@]}" \
         --out "$A/device_confound_$tag.json" > "$OUT/logs/analysis_device_$tag.log" 2>&1; then
      echo "OK    device_$tag" >> "$A/STATUS.txt"
    else
      echo "FAIL  device_$tag (see $OUT/logs/analysis_device_$tag.log)" >> "$A/STATUS.txt"
    fi
  done
done
log "device check finished"; grep device "$A/STATUS.txt"
