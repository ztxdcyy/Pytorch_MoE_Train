#!/usr/bin/env bash
set -euo pipefail

# Simple runner for MoE demos. Usage:
#   MODE=dp ./run_moe.sh            # data parallel (full experts per rank)
#   MODE=ep ./run_moe.sh            # expert parallel (all_gather-based)
# You can override ranks via torchrun flags below.

MODE="${MODE:-dp}"
NPROC="${NPROC:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "${MODE}" == "dp" ]]; then
  SCRIPT="main_dp.py"
elif [[ "${MODE}" == "ep" ]]; then
  SCRIPT="main_ep.py"
else
  echo "Unknown MODE=${MODE}, expected dp or ep" >&2
  exit 1
fi

echo "Running ${SCRIPT} with ${NPROC} processes..."
torchrun --nproc_per_node="${NPROC}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" "${SCRIPT}"
