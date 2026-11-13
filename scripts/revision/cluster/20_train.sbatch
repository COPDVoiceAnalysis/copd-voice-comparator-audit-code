#!/bin/bash
#SBATCH --job-name=rev-train
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
# One training run per array task: line (task id + 1) of $OUT/configs/train_list.txt.
# Same entry point as rule `train` in the Snakefile. A finished run leaves
# $OUT/runs/<config_id>.done and is skipped on resubmission.
source "${REV_HERE:?submit through run_all.sh}/common.sh"
IDX=${SLURM_ARRAY_TASK_ID:?array task id missing}
CFG=$(sed -n "$((IDX + 1))p" "$OUT/configs/train_list.txt")
[ -n "$CFG" ] || die "no config on line $((IDX + 1)) of $OUT/configs/train_list.txt"
ID=$(basename "$CFG" __config.yml)
if [ -e "$OUT/runs/$ID.done" ]; then log "already done: $ID"; exit 0; fi
log "train $ID"
uv run --frozen -m src.train --config "$CFG" --output_dir "$OUT/runs/$ID"
touch "$OUT/runs/$ID.done"
log "done $ID"
