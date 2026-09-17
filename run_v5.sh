#!/usr/bin/env bash
# v5: extend the two sweeps to four (non-IID) and three (Byzantine fraction)
# seeds so that every quantitative claim in Section VI-C rests on more than one
# federation draw.  Runs are cached, so a restart resumes where it stopped.
cd "$(dirname "$0")"
for i in 1 2 3 4 5 6; do
  python run_experiments.py --suite noniid  --workers 2 && \
  python run_experiments.py --suite byzfrac --workers 2 && break
  echo "retry $i"; sleep 5
done
echo "v5 sweeps complete"
