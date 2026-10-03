#!/usr/bin/env bash
# Regenerate every table, figure panel and quoted number from the run records.
set -e
cd "$(dirname "$0")"
python make_fodm.py            # main tables, panel JSONs, logs/fodm_numbers.txt
python make_aux_tables.py      # six auxiliary tables
python fodm/fig_fuzzy.py       # fuzzy membership functions and trust surfaces
