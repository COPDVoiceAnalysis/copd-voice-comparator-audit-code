#!/bin/bash
#SBATCH --job-name=rev-smoking
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Voice score by smoking history (Reviewer 1, point 4; Reviewer 3, point 2).
# Analysis only (no training); reads the finished voice runs via $OUT/arms/*.json,
# so it needs a finished rev-analysis (or `arms.py resolve`) first.
source "${REV_HERE:?submit through run_all.sh}/common.sh"
set +e
A="$OUT/analysis"; ARMS="$OUT/arms"
mkdir -p "$A" "$OUT/logs"
armrun() { py -c "import json,sys; print(json.load(open(sys.argv[1]))['arms'].get(sys.argv[2], ''))" "$@"; }
for c in charite uk; do
  RUN=$(armrun "$ARMS/$c.json" voice 2>/dev/null)
  [ -n "$RUN" ] || { echo "FAIL  smoking_$c (voice run not found)" >> "$A/STATUS.txt"; continue; }
  log "smoking subgroup $c"
  if py scripts/revision/smoking_subgroup.py --cohort "$c" --voice-run "$RUN" \
       --metadata "$OUT/merged_metadata_remapped.csv" --metadata "data/$DATE/preprocessed__$PREPROC/metadata.csv" \
       --out "$A/smoking_$c.json" > "$OUT/logs/analysis_smoking_$c.log" 2>&1; then
    echo "OK    smoking_$c" >> "$A/STATUS.txt"
  else
    echo "FAIL  smoking_$c (see $OUT/logs/analysis_smoking_$c.log)" >> "$A/STATUS.txt"
  fi
done
log "smoking subgroup finished"; grep smoking "$A/STATUS.txt"
