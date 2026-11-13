#!/bin/bash
#SBATCH --job-name=rev-wav2vec
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
# Feature build 3/3: wav2vec2-large-robust embeddings of all 25 layers, as rule
# `wav2vec_embeddings`. Needs one GPU (SBATCH_GPU_ARGS in revision.env).
source "${REV_HERE:?submit through run_all.sh}/common.sh"
FB=$(features_base)
export AVAILABLE_MEMORY=16G
export HF_HUB_OFFLINE=1   # model was downloaded by 01_setup.sh; compute nodes may have no internet
mkdir -p "$FB/wav2vec2__$PREPROC"
log "wav2vec2 -> $FB/wav2vec2__$PREPROC"
uv run --frozen -m src.features.wav2vec2.get_embeddings \
  --embeddings_dir "$FB/wav2vec2__$PREPROC" --metadata_file "$FB/preprocessed__$PREPROC/metadata.csv"
log "done"
