#!/usr/bin/env bash
# Revision campaign, in priority order. Every run is cached by configuration, so
# this resumes whatever an earlier invocation completed.
cd "$(dirname "$0")"
run () {
  for i in 1 2 3 4 5 6 7 8; do
    python run_experiments.py --suite "$1" --workers 2 && return 0
    echo "retry $1 ($i)"; sleep 5
  done
}
run main_v6        # nine rules x eight attacks x five seeds, Fashion-MNIST
run noniid         # Dirichlet sweep extended to six seeds
run adaptive_rho   # alignment-budget sweep of the white-box attack
run cifar_v6       # the two non-fuzzy controls on CIFAR-10
echo "v6 campaign complete"
