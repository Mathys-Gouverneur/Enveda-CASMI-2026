#!/usr/bin/env bash
# Vendor a public Kaggle kernel into notebooks/external/.
#
#   scripts/pull_external_notebook.sh <owner>/<kernel-slug> [local-dir-name]
#
# Leaves <dir>/{<kernel>.ipynb,kernel-metadata.json}; the README.md beside them is
# written by hand, because the thing worth recording about a public notebook is why we
# kept it, and no API returns that.
#
# Public kernels are revised in place and sometimes deleted outright, so the copy that
# counts is the one in git, not the one on Kaggle. Re-running this on an existing
# directory overwrites it — which is how you take a deliberate new snapshot.
set -euo pipefail

REF="${1:?usage: $0 <owner>/<kernel-slug> [local-dir-name]}"
DIR="${2:-$(basename "$REF")}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE="$ROOT/.venv/bin/kaggle"
[ -x "$KAGGLE" ] || KAGGLE=kaggle

DEST="$ROOT/notebooks/external/$DIR"
mkdir -p "$DEST"
"$KAGGLE" kernels pull "$REF" -p "$DEST" -m
echo "-> $DEST"
