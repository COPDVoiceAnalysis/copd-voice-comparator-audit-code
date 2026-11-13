#!/bin/bash
# Step 2: submit every revision job to Slurm with the right dependencies.
# Run on the login node after 00_preflight.sh and 01_setup.sh:
#   bash scripts/revision/cluster/run_all.sh [--no-layers] [--only STAGES]
# Options
#   --no-layers     skip the 26-run layer sweep (Reviewer 1, point 6)
#   --only STAGES   comma list of stages to submit: features,train,comparators,xds,analysis
#                   (default: all; features only if preflight said a build is needed)
#   --local         run the stages one after another in this shell instead of
#                   sbatch (for testing on a laptop or an interactive node)
# Job ids are written to $OUT/state/jobs.txt. Watch with: squeue --me
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
LAYERS=1; ONLY="features,train,comparators,xds,analysis"; LOCAL=0
while [ $# -gt 0 ]; do
  case $1 in
    --no-layers) LAYERS=0 ;;
    --only) ONLY=$2; shift ;;
    --local) LOCAL=1 ;;
    *) die "unknown option $1" ;;
  esac
  shift
done
export REV_LAYERS=$LAYERS
want() { [[ ",$ONLY," == *",$1,"* ]]; }
[ -s "$OUT/preflight_report.txt" ] || die "run 00_preflight.sh and 01_setup.sh first"
[ -d "data/$DATE" ] || die "data/$DATE missing: run 01_setup.sh first"
N_TRAIN=$((6 + 26 * LAYERS))
JOBS="$OUT/state/jobs.txt"
echo "# submitted $(date '+%F %T %Z') SMOKE=$SMOKE LAYERS=$LAYERS ONLY=$ONLY" >> "$JOBS"

# sub <name> <dependency or ""> <extra sbatch args> <script> [args...]; prints the job id
sub() {
  local name=$1 dep=$2 extra=$3; shift 3
  if [ "$LOCAL" = 1 ]; then
    log "LOCAL $name: $*" >&2
    local script=$1; shift
    if [[ "$extra" == *--array=* ]]; then
      local n=${extra##*--array=0-}; n=${n%% *}
      for i in $(seq 0 "$n"); do SLURM_ARRAY_TASK_ID=$i bash "$HERE/$script" "$@" >&2 || log "LOCAL $name task $i FAILED" >&2; done
    else
      bash "$HERE/$script" "$@" >&2 || log "LOCAL $name FAILED" >&2
    fi
    echo "local"
    return
  fi
  local id
  # shellcheck disable=SC2086
  id=$(sbatch --parsable --export=ALL --job-name="rev-$name" \
       --output="$OUT/logs/%x-%A_%a.out" ${dep:+--dependency=$dep} $extra "$HERE/$@")
  echo "$name $id" >> "$JOBS"
  echo "$id"
}
after() {  # after <afterok|afterany> <job ids...>; prints "" when there is nothing to wait for
  local kind=$1 d=""; shift
  for j in "$@"; do if [ -n "$j" ] && [ "$j" != local ]; then d="$d:$j"; fi; done
  if [ -n "$d" ]; then echo "$kind$d"; fi
}

FEAT_DEP=""
if want features && [ "$(cat "$OUT/state/features_need_build")" = 1 ]; then
  J1=$(sub preprocess "" "$SBATCH_CPU_ARGS" 10_features_preprocess.sbatch)
  J2=$(sub parselmouth "$(after afterok "$J1")" "$SBATCH_CPU_ARGS" 11_features_parselmouth.sbatch)
  J3=$(sub wav2vec "$(after afterok "$J1")" "$SBATCH_GPU_ARGS" 12_features_wav2vec.sbatch)
  FEAT_DEP=$(after afterok "$J2" "$J3")
fi

JT=""; JCC=""; JCU=""; JX1=""; JX2=""
if want train; then
  JC=$(sub configs "$FEAT_DEP" "$SBATCH_CPU_ARGS" 19_configs.sbatch)
  JT=$(sub train "$(after afterok "$JC")" "$SBATCH_CPU_ARGS --array=0-$((N_TRAIN - 1))" 20_train.sbatch)
fi
if want comparators; then
  JCC=$(sub comparators-charite "$FEAT_DEP" "$SBATCH_CPU_ARGS" 21_comparators.sbatch charite)
  JCU=$(sub comparators-uk "$(after afterany "$JT")" "$SBATCH_CPU_ARGS" 21_comparators.sbatch uk)
fi
if want xds; then
  JX1=$(sub xds-all "$FEAT_DEP" "$SBATCH_CPU_ARGS" 30_crossdataset.sbatch all)
  JX2=$(sub xds-de "$FEAT_DEP" "$SBATCH_CPU_ARGS" 30_crossdataset.sbatch de)
fi
if want analysis; then
  sub analysis "$(after afterany "$JT" "$JCC" "$JCU" "$JX1" "$JX2")" "$SBATCH_CPU_ARGS" 40_analysis.sbatch > /dev/null
fi
log "submitted; job ids in $JOBS"
[ "$LOCAL" = 1 ] || squeue --me
