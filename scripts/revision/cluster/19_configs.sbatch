#!/bin/bash
#SBATCH --job-name=rev-configs
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
# Write the training configs ($OUT/configs/*__config.yml) and train_list.txt.
# Runs as a job because the config validation needs the feature directories.
source "${REV_HERE:?submit through run_all.sh}/common.sh"
EXTRA=()
[ "${REV_LAYERS:-1}" = 1 ] && EXTRA+=(--layers)
[ "$SMOKE" = 1 ] && EXTRA+=(--smoke)
py scripts/revision/arms.py make-configs --preproc "$PREPROC" --data-dir "data/$DATE" \
  --out-dir "$OUT/configs" "${EXTRA[@]}"
