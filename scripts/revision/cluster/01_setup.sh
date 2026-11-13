#!/bin/bash
# Step 1: install the Python environment, download the wav2vec2 model if the
# features must be rebuilt, and link the data into data/ inside the repo.
# Run on the login node (needs internet):  bash scripts/revision/cluster/01_setup.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
FB=$(features_base)

log "uv sync (locked versions from uv.lock)"
uv sync --frozen

if [ "$(cat "$OUT/state/features_need_build")" = 1 ]; then
  log "features will be built in $FB; downloading facebook/wav2vec2-large-robust to $HF_HOME"
  py -c "from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor as F; \
n='facebook/wav2vec2-large-robust'; F.from_pretrained(n); Wav2Vec2Model.from_pretrained(n); print('model cached')"
fi

# data/ view: the scripts read data/$DATE/... and data/uk_matched_*.csv relative
# to the repo root (as on the authors' laptops). Only symlinks are created.
if [ -L "data/$DATE" ]; then
  die "data/$DATE is a symlink ($(readlink "data/$DATE")); remove it so the setup does not write into another directory"
fi
mkdir -p "data/$DATE"
for d in preprocessed parselmouth wav2vec2; do
  ln -sfn "$FB/${d}__$PREPROC" "data/$DATE/${d}__$PREPROC"
done
ln -sfn "$RAW_DIR/comorbidities.csv" "data/$DATE/comorbidities.csv"
ln -sfn "$RAW_DIR/psm_matched_charite.csv" "data/$DATE/psm_matched_charite.csv"
ln -sfn "$RAW_DIR/uk_matched_metadata.csv" "data/uk_matched_metadata.csv"
ln -sfn "$RAW_DIR/uk_matched_metadata_ps_covariates.csv" "data/uk_matched_metadata_ps_covariates.csv"
log "data view:"; ls -l "data/$DATE" data/uk_matched_metadata*.csv
log "setup done; next: bash scripts/revision/cluster/run_all.sh"
