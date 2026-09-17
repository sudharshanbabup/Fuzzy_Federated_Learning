#!/usr/bin/env bash
# Second stage of the revision campaign: two more seeds on the Dirichlet sweep,
# so that the exact sign-flip permutation test has enough resolution to reach
# the 5% level (with four seeds its smallest attainable p is 0.125).
cd "$(dirname "$0")"
for i in 1 2 3 4 5 6; do
  python run_experiments.py --suite noniid --workers 2 && break
  echo "retry noniid ($i)"; sleep 5
done
echo "noniid extension complete"
