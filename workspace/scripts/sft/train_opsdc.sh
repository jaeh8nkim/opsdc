#!/bin/bash
# OPSDC base training for Qwen3-8B on 4x H100
#
# Based on paper config (qwen3-8b-opsd-length-prune.json), adjusted for 4 GPUs.
#
# Changes from paper's 8-GPU config:
#   N_GPUS:            8 -> 4    fewer GPUs available
#   MICRO_BATCH_SIZE:  2 -> 1    halved to fit 4-GPU memory budget
#   ULYSSES_SP_SIZE:   4 -> 2    must divide N_GPUS evenly
#   TRAIN_BATCH_SIZE:  32 = 32   kept identical; grad accum compensates
#
# Usage:
#   MODEL_PATH=Qwen/Qwen3-8B ./workspace/scripts/sft/train_opsdc.sh

MODEL_PATH=${MODEL_PATH:?MODEL_PATH environment variable is required} \
SD_PROMPTS_PATH=./workspace/data/length_prune_concise/self_distill_prompts.parquet \
SD_VAL_PROMPTS_PATH=./workspace/data/length_prune_concise/self_distill_prompts_val.parquet \
OPSD_BETA=0.5 \
OPSD_LOSS_TYPE=reverse_kl \
SD_TEMPERATURE=1.0 \
SD_TOP_P=1.0 \
SD_MAX_TOKENS=8192 \
SFT_MAX_LENGTH=10240 \
TOTAL_EPOCHS=1 \
TRAIN_MAX_SAMPLES=3200 \
TRAIN_BATCH_SIZE=32 \
MICRO_BATCH_SIZE=1 \
LEARNING_RATE=1e-6 \
SAVE_FREQ=100 \
TEST_FREQ=25 \
N_GPUS=4 \
TP_SIZE=2 \
GPU_MEM_UTIL=0.75 \
ULYSSES_SP_SIZE=2 \
MAX_PROMPT_LENGTH=2048 \
MAX_RESPONSE_LENGTH=30000 \
VAL_MAX_TOKENS=30000 \
CHECK_STRUCTURE=false \
USE_LIGER=true \
TEACHER_UPDATE_FREQ=9999 \
VAL_BEFORE_TRAIN=true \
EXPERIMENT_NAME=opsdc_base \
RL_VAL_FILES="['./workspace/data/processed/val_math500.parquet', './workspace/data/processed/val_aime24.parquet', './workspace/data/processed/val_aime25.parquet']" \
bash workspace/scripts/sft/train_opsd.sh
