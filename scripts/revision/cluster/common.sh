# shellcheck shell=bash
# Sourced by every stage script. Not meant to be run directly.
set -euo pipefail
HERE="${REV_HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export REV_HERE="$HERE"
REPO="$(cd "$HERE/../../.." && pwd)"
export REPO
# shellcheck source=revision.env
source "$HERE/revision.env"
cd "$REPO"
mkdir -p "$OUT/logs" "$OUT/state"
export MLFLOW_TRACKING_URI="file://$OUT/mlruns"
export MLFLOW_ARTIFACT_URI="file://$OUT/mlruns"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
log() { echo "[$(date '+%F %T %Z')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
features_base() {
  [ -s "$OUT/state/features_base" ] || die "run 00_preflight.sh first (no $OUT/state/features_base)"
  cat "$OUT/state/features_base"
}
# uv run without re-resolving the environment on compute nodes
py() { uv run --frozen python "$@"; }
