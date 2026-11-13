#!/bin/bash
# Step 5: copy the aggregate results into results/revision_r1/ in the repo
# (with an identifier guard), ready to commit and push. Run on the login node.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
DEST="$REPO/results/revision_r1"
[ "$SMOKE" = 1 ] && DEST="$REPO/results/revision_r1_SMOKE"
py scripts/revision/collect_results.py "$OUT" "$DEST"
log "collected into $DEST"
[ "$SMOKE" = 1 ] && log "SMOKE run: do not commit these numbers"
log "next: git add results/revision_r1 && git commit -m 'revision: cluster results' && git push"
