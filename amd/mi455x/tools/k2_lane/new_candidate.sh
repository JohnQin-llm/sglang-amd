#!/bin/bash
# Create a K2 candidate tree: a copy of the kept tree with every hard-coded tree path rewritten to the copy.
# Usage: new_candidate.sh <name> [source tree, default /data/muse455/k2/ep1]   -> /data/muse455/k2/cand/<name>
set -euo pipefail
NAME=$1
SRC=${2:-/data/muse455/k2/ep1}
DST=/data/muse455/k2/cand/$NAME
[ -e "$DST" ] && { echo "exists: $DST"; exit 1; }
mkdir -p "$(dirname "$DST")"
rsync -a --exclude __pycache__ --exclude 'backup_*' --exclude asm --exclude '*.log' "$SRC/" "$DST/"
grep -rIl --include='*.py' --include='*.sh' "$SRC" "$DST" | xargs -r sed -i "s#$SRC#$DST#g" || true
left=$( (grep -rIl --include="*.py" --include="*.sh" "$SRC" "$DST" || true) | wc -l)
echo "$DST (files still naming $SRC: $left)"
