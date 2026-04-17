#!/bin/bash
# OPSDC Epiphany training for Qwen3-8B on 4x H100
#
# Epiphany mode: after the student generates (Turn 1), it reflects on its
# attempt (Turn 2) to produce a self-reflection memo.  This memo is injected
# into the teacher's prompt, giving it a genuine informational advantage over
# the student for the reverse KL training step.
#
# Based on paper config (qwen3-8b-opsd-length-prune.json), adjusted for 4 GPUs.
#
# Changes from paper's 8-GPU config:
#   N_GPUS:            8 -> 4    fewer GPUs available
#   MICRO_BATCH_SIZE:  2 -> 1    halved to fit 4-GPU memory budget
#   ULYSSES_SP_SIZE:   4 -> 2    must divide N_GPUS evenly
#   TRAIN_BATCH_SIZE:  32 = 32   kept identical; grad accum compensates
#
# Token budget breakdown:
#   SD_MAX_TOKENS               Student Turn 1 output cap (configurable below)
#   TURN2_MAX_TOKENS            Student Turn 2 output cap (configurable below)
#   MAX_PROMPT_LENGTH           = SD_MAX_TOKENS + TURN2_MAX_TOKENS + 2048  (adaptive)
#   SFT_MAX_LENGTH              = SD_MAX_TOKENS + TURN2_MAX_TOKENS + 2048  (adaptive) max sequence length for the distillation training step
#   MAX_RESPONSE_LENGTH=30000   hard ceiling on any single generation (fixed)
#   VAL_MAX_TOKENS=30000        validation generation cap (fixed)
#
# Usage:
#   MODEL_PATH=Qwen/Qwen3-8B ./workspace/scripts/sft/train_epiphany.sh

SD_MAX_TOKENS=8192
TURN2_MAX_TOKENS=8192
TURN2_RESCUE_TOKENS=2048
TURN2_TEACHER_CTX_TOKENS=null
TEACHER_CTX_MODE=${TEACHER_CTX_MODE:-reflection_from_gt}
# TEACHER_CTX_MODE options:
#   sd_prompt          — use precomputed sd_prompt from dataset (no Turn 2)
#   reflection_from_gt — Turn 2 self-reflection memo as teacher context
#   gt_directly        — ground truth as teacher context (no Turn 2)
#   conciseness_instruction — conciseness instruction only (no Turn 2, matches train_opsdc.sh)
KL_GATING=${KL_GATING:-all}
# KL_GATING options:
#   all                    — KL on every sample
#   correct_only           — KL only on correct rollouts
#   correct_and_truncated  — KL on correct + truncated rollouts
#   incorrect_only         — KL only on incorrect rollouts
EXPERT_DEMO_PATH=${EXPERT_DEMO_PATH:-./workspace/data/expert_demonstrations/expert_demos_3200.parquet}

# ---- KL-by-position probe ----
KL_POS_LOGGING=${KL_POS_LOGGING:-true}
KL_ANALYZE_ONLY=${KL_ANALYZE_ONLY:-false}
PLOT_FREQ=${PLOT_FREQ:-50}

# ---- distance_weighted_kl ----
DISTANCE_WEIGHT_SCHEDULE=${DISTANCE_WEIGHT_SCHEDULE:-off}
DISTANCE_WEIGHT_LATE_MULT=${DISTANCE_WEIGHT_LATE_MULT:-2.0}
DISTANCE_WEIGHT_ALPHA=${DISTANCE_WEIGHT_ALPHA:-2.0}

# ---- teacher_ctx_reinjection ----
REINJECTION_ENABLED=${REINJECTION_ENABLED:-false}
REINJECTION_MODE=${REINJECTION_MODE:-multi_pass}   # multi_pass (default) | cumulative
REINJECTION_INTERVAL=${REINJECTION_INTERVAL:-2048}
REINJECTION_CONTENT=${REINJECTION_CONTENT:-specific_context}

MODEL_PATH=${MODEL_PATH:?MODEL_PATH environment variable is required} \
SD_PROMPTS_PATH=./workspace/data/length_prune_concise/self_distill_prompts.parquet \
SD_VAL_PROMPTS_PATH=./workspace/data/length_prune_concise/self_distill_prompts_val.parquet \
OPSD_BETA=0.5 \
OPSD_LOSS_TYPE=reverse_kl \
SD_TEMPERATURE=1.0 \
SD_TOP_P=1.0 \
SD_MAX_TOKENS=$SD_MAX_TOKENS \
SFT_MAX_LENGTH=$(( SD_MAX_TOKENS + TURN2_MAX_TOKENS + 2048 )) \
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
MAX_PROMPT_LENGTH=$(( SD_MAX_TOKENS + TURN2_MAX_TOKENS + 2048 )) \
MAX_RESPONSE_LENGTH=30000 \
VAL_MAX_TOKENS=30000 \
CHECK_STRUCTURE=false \
USE_LIGER=true \
TEACHER_UPDATE_FREQ=9999 \
VAL_BEFORE_TRAIN=false \
EXPERIMENT_NAME=opsd_epiphany \
RL_VAL_FILES="['./workspace/data/processed/val_math500.parquet', './workspace/data/processed/val_aime24.parquet', './workspace/data/processed/val_aime25.parquet']" \
bash workspace/scripts/sft/train_opsd.sh \
    opsd.teacher_ctx_mode=$TEACHER_CTX_MODE \
    opsd.turn2_max_tokens=$TURN2_MAX_TOKENS \
    opsd.turn2_rescue_tokens=$TURN2_RESCUE_TOKENS \
    opsd.turn2_teacher_ctx_tokens=$TURN2_TEACHER_CTX_TOKENS \
    opsd.expert_demo_path="$EXPERT_DEMO_PATH" \
    opsd.kl_gating=$KL_GATING \
    opsd.kl_by_pos.enabled=$KL_POS_LOGGING \
    opsd.kl_by_pos.plot_freq=$PLOT_FREQ \
    opsd.kl_analyze_only=$KL_ANALYZE_ONLY \
    opsd.distance_weighting.schedule=$DISTANCE_WEIGHT_SCHEDULE \
    opsd.distance_weighting.late_multiplier=$DISTANCE_WEIGHT_LATE_MULT \
    opsd.distance_weighting.alpha=$DISTANCE_WEIGHT_ALPHA \
    opsd.teacher_ctx_reinjection.enabled=$REINJECTION_ENABLED \
    opsd.teacher_ctx_reinjection.mode=$REINJECTION_MODE \
    opsd.teacher_ctx_reinjection.interval=$REINJECTION_INTERVAL \
    opsd.teacher_ctx_reinjection.content=$REINJECTION_CONTENT
