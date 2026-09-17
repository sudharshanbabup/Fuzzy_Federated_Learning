#!/usr/bin/env bash
# Third stage: the alignment-budget sweep of the white-box adaptive attack.
cd "$(dirname "$0")"
for i in 1 2 3 4 5 6; do
  python run_experiments.py --suite adaptive_rho --workers 2 && break
  echo "retry adaptive_rho ($i)"; sleep 5
done
echo "adaptive_rho complete"
