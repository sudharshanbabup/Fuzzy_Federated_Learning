#!/usr/bin/env bash
# Generate tables and evaluation metrics from run records.
set -e
cd "$(dirname "$0")"
[ -d results_ledger ] || tar xzf results_ledger.tar.gz
python make_fodm.py
python analyze_ledger.py results_ledger generated tables_fodm
python fodm/fig_fuzzy.py
echo "Tables and evaluation metrics generated successfully."
