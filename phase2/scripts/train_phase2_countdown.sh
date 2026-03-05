#!/usr/bin/env bash
# Phase 2-C: Fine-tune Phase 1 checkpoint on Countdown-4
set -euxo pipefail

BASE_MODEL="${BASE_MODEL:=checkpoints/phase1}" \
DATA_DIR=data/countdown \
OUTPUT_DIR=checkpoints/phase2_countdown \
LOG_NAME=phase2_countdown \
    bash "$(dirname "$0")/train_phase2.sh"
