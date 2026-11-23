#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
python -P src/build_ledger.py base 2024-09 2024-10 2024-11 2024-12 2025-01
python -P src/build_ledger.py snap1 2024-08 2024-09 2024-10 2024-11 2024-12 2025-01
echo BUILDS_DONE
