#!/bin/bash
#SBATCH --job-name=rev-parselmouth
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=10:00:00
# Feature build 2/3: Parselmouth features, as rule `parselmouth`. The revision
# models do not use them, but the training config validation requires the file.
source "${REV_HERE:?submit through run_all.sh}/common.sh"
FB=$(features_base)
mkdir -p "$FB/parselmouth__$PREPROC"
log "parselmouth -> $FB/parselmouth__$PREPROC"
uv run --frozen -m src.features.parselmouth.extractor \
  --output_file "$FB/parselmouth__$PREPROC/acoustic_features.npy" \
  --metadata_file "$FB/preprocessed__$PREPROC/metadata.csv" --max_workers "${SLURM_CPUS_PER_TASK:-8}"
log "done"
