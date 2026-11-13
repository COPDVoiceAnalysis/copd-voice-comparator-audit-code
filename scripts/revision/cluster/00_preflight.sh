#!/bin/bash
# Step 0: check all inputs and decide where the features come from. Read-only.
# Run on the login node:  bash scripts/revision/cluster/00_preflight.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
log "preflight (repo $REPO)"
uv run --frozen python scripts/revision/cluster/preflight.py
