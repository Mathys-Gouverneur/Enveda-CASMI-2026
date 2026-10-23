#!/usr/bin/env bash
# Fetch the three competition files into /data/mgr/enveda-casmi/raw.
#
#   scripts/download_data.sh
#
# Blocked until the competition rules are accepted in a browser:
#   https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules
# Without that the API answers 403 on every data endpoint, and the error does not say
# why — so this script checks first and says so itself.
set -euo pipefail

COMP=enveda-CASMI26-molecule-id-mass-spectra
DEST=/data/mgr/enveda-casmi/raw

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE="$ROOT/.venv/bin/kaggle"
[ -x "$KAGGLE" ] || KAGGLE=kaggle

entered=$("$KAGGLE" competitions list -s "$COMP" 2>/dev/null | awk 'NR==3 {print $NF}')
if [ "$entered" != "True" ]; then
  echo "Rules not accepted yet (userHasEntered=${entered:-unknown})." >&2
  echo "Accept them here, then re-run:" >&2
  echo "  https://www.kaggle.com/competitions/$COMP/rules" >&2
  exit 1
fi

mkdir -p "$DEST"
# One file at a time rather than the whole-competition zip: train.parquet is 3.03 GB and
# the zip would mean holding it twice on disk before unpacking.
for f in sample_submission.csv test.parquet train.parquet; do
  if [ -f "$DEST/$f" ]; then
    echo "have   $f"
    continue
  fi
  echo "fetch  $f"
  "$KAGGLE" competitions download -c "$COMP" -f "$f" -p "$DEST"
  # Single-file downloads arrive zipped when Kaggle decides to compress them.
  [ -f "$DEST/$f.zip" ] && { unzip -o -q "$DEST/$f.zip" -d "$DEST" && rm "$DEST/$f.zip"; }
done

chmod -R go-rwx /data/mgr/enveda-casmi
ls -la "$DEST"
