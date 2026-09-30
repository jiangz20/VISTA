#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for seed in 0 1 2; do
  bash "$ROOT_DIR/scripts/train.sh" "$@" \
    "seed=$seed" "trainer.experiment_name=vista-seed-$seed" \
    "trainer.default_local_dir=checkpoints/vista-seed-$seed"
done
