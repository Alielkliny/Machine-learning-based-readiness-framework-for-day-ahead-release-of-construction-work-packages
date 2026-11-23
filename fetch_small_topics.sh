#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
for T in activity section_58; do
  for ym in 2023-10 2023-11 2023-12 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06 2024-07 2024-08 2024-09 2024-10 2024-11 2024-12 2025-01; do
    Y=${ym%-*}; M=${ym#*-}; OUT="data/processed/events/${T}_${Y}_${M}.parquet"
    [ -f "$OUT.json" ] && continue
    P=4 bash src/fetch_month.sh $T $Y $M "data/raw/streetmanager/$T/$Y" || continue
    python -P -E src/extract_generic.py "data/raw/streetmanager/$T/$Y/$M.zip" "$OUT" src && echo "$T $ym $(python -c "import json;print(json.load(open('$OUT.json'))['archive_sha256'])")" >> logs/archive_hashes.txt && rm -f "data/raw/streetmanager/$T/$Y/$M.zip"
  done
done
echo SMALLDONE
