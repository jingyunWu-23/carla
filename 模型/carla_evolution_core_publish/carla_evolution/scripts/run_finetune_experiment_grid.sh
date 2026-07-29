#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EGO_SET_SIZES="${EGO_SET_SIZES:-20 30 50}"
POPULATION_SIZES="${POPULATION_SIZES:-10 30 50}"

for ego_size in ${EGO_SET_SIZES}; do
  for population_size in ${POPULATION_SIZES}; do
    echo "================================================================"
    echo "Running fine-tune experiment: frozen_ego_set_size=${ego_size}, population_size=${population_size}"
    echo "================================================================"
    python scripts/finetune_ego_enhanced_poet.py \
      --frozen-ego-set-size "${ego_size}" \
      --population-size "${population_size}" \
      "$@"
  done
done
