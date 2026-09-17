#!/usr/bin/env bash
# Revision campaign: the Fashion-MNIST study is rebuilt with nine rules, eight
# attacks (including the white-box adaptive one) and five seeds, then the
# CIFAR-10 additions. Runs are cached per configuration, so a restart resumes.
cd "$(dirname "$0")"
for i in 1 2 3 4 5 6 7 8; do
  python run_experiments.py --suite main_v6 --workers 2 && break
  echo "retry main_v6 ($i)"; sleep 5
done
for i in 1 2 3 4 5 6 7 8; do
  python run_experiments.py --suite cifar_v6 --workers 2 && break
  echo "retry cifar_v6 ($i)"; sleep 5
done
echo "v6 campaign complete"
