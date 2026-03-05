#!/usr/bin/env bash
# Phase 2-B: Fine-tune Phase 1 checkpoint on MATH
set -euxo pipefail

BASE_MODEL="${BASE_MODEL:=checkpoints/phase1}" \
DATA_DIR=data/math \
OUTPUT_DIR=checkpoints/phase2_math \
LOG_NAME=phase2_math \
    bash "$(dirname "$0")/train_phase2.sh"
