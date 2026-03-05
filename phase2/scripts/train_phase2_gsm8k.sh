#!/usr/bin/env bash
# Phase 2-A: Fine-tune Phase 1 checkpoint on GSM8K
set -euxo pipefail

BASE_MODEL="${BASE_MODEL:=checkpoints/phase1}" \
DATA_DIR=data/gsm8k \
OUTPUT_DIR=checkpoints/phase2_gsm8k \
LOG_NAME=phase2_gsm8k \
    bash "$(dirname "$0")/train_phase2.sh"
