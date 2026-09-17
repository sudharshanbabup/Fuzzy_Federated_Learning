#!/usr/bin/env bash
# Resilient driver: the cloud container may be recycled; every run is cached on
# disk, so simply re-entering the driver resumes where it stopped.
cd "$(dirname "$0")"
for i in 1 2 3 4 5 6 7 8; do
  python3 -u run_experiments.py --suite main_cifar --workers 2 && break
  echo "=== driver exited, retrying ($i) ==="
  sleep 5
done
