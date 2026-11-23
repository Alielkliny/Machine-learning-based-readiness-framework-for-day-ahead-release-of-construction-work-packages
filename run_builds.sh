#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
python -P src/build_ledger.py base 2024-05 2024-06 2024-07 2024-08
python -P src/build_ledger.py snap1 2023-12 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06 2024-07
echo BUILDS_DONE
