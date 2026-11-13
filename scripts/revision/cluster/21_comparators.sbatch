#!/bin/bash
#SBATCH --job-name=rev-comparators
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
# Demographic comparators, trained with the published ps_overlap weights and
# without weights ("flat"), seed 42, extended grid. Argument: charite | uk.
# The UK cohort is the participant set of the new UK voice run, so `uk` must
# run after the training array.
source "${REV_HERE:?submit through run_all.sh}/common.sh"
COHORT=${1:?usage: 21_comparators.sbatch charite|uk}
ARGS=(--cohort "$COHORT" --grids extended --seeds 42 --workers "${SLURM_CPUS_PER_TASK:-4}"
      --out "$OUT/comparators" --preproc "$PREPROC")
if [ "$COHORT" = uk ]; then
  NAME=$(py scripts/revision/arms.py voice-run-name --preproc "$PREPROC" --cohort uk)
  RUN=$(py scripts/revision/arms.py run-dir --mlruns "$OUT/mlruns" --name "$NAME") \
    || die "UK voice run $NAME not finished; check the training array logs"
  ARGS+=(--uk-voice-glob "$RUN/artifacts/fold_predictions/*.parquet")
fi
log "comparators $COHORT"
py scripts/comparator_weighting_sensitivity.py "${ARGS[@]}"
log "done"
