#!/bin/bash
# Worker: for each YYYY-MM, take a lock, download, extract to Parquet, delete zip.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p data/processed/events logs locks
for ym in "$@"; do
  Y=${ym%-*}; M=${ym#*-}
  OUT="data/processed/events/permit_${Y}_${M}.parquet"
  if [ -f "$OUT" ] && [ -f "$OUT.json" ]; then echo "HAVE $ym"; continue; fi
  mkdir "locks/permit_$ym" 2>/dev/null || { echo "LOCKED $ym"; continue; }
  bash src/fetch_month.sh permit "$Y" "$M" "data/raw/streetmanager/permit/$Y" || { echo "FAIL download $ym"; rmdir "locks/permit_$ym"; continue; }
  python -P -E src/extract_events.py "data/raw/streetmanager/permit/$Y/$M.zip" "$OUT" src || { echo "FAIL extract $ym"; rmdir "locks/permit_$ym"; continue; }
  echo "$ym $(python -c "import json;print(json.load(open('$OUT.json'))['archive_sha256'])")" >> logs/archive_hashes.txt
  rm -f "data/raw/streetmanager/permit/$Y/$M.zip"
  echo "EXTRACTED $ym"
done
echo "ALLDONE"
