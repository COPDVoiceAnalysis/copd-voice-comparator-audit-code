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
# A run whose folds all fail is still marked FINISHED by MLflow, without fold
# predictions; treat that as a failure here so it is not missed.
py - "$OUT/comparators/mlruns" "$COHORT" <<'PY'
import sys
from pathlib import Path
task = {"charite": "charite_complete", "uk": "uk_only"}[sys.argv[2]]
bad = [n.read_text().strip() for n in Path(sys.argv[1]).glob("*/*/tags/mlflow.runName")
       if task in n.read_text() and not list((n.parent.parent / "artifacts" / "fold_predictions").glob("*.parquet"))]
if bad:
    sys.exit("comparator runs without fold predictions (see $OUT/comparators/runs/*.log):\n  " + "\n  ".join(bad))
print("all comparator runs have fold predictions")
PY
log "done"
