#!/bin/bash
# Vanilla GRPO training for Qwen3-8B on 4x H100
#
# Standard GRPO with KL regularization, symmetric clipping, seq-mean-token-mean loss.
# Uses verl's main_ppo entry point with sglang rollout.
#
# Usage:
#   MODEL_PATH=Qwen/Qwen3-8B ./workspace/scripts/rl/train_grpo.sh

set -x

ulimit -n 65535

# =============================================================================
# Environment setup
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-$(cd "${WORKSPACE_ROOT}/../verl" && pwd)}"
SD_SRC="${WORKSPACE_ROOT}/src"

export PYTHONPATH="${VERL_ROOT}:${SD_SRC}:${PYTHONPATH}"
echo "PYTHONPATH: $PYTHONPATH"

SITE_PKGS=$(python3 -c "import torch; import os; print(os.path.dirname(os.path.dirname(torch.__file__)))")
TORCH_CUDA_LIBS="${SITE_PKGS}/torch/lib"
for pkg in cuda_runtime cublas cudnn cuda_cupti cufft curand cusolver cusparse nccl nvtx; do
    d="${SITE_PKGS}/nvidia/${pkg}/lib"
    [ -d "$d" ] && TORCH_CUDA_LIBS="${TORCH_CUDA_LIBS}:${d}"
done
export LD_LIBRARY_PATH="${TORCH_CUDA_LIBS}:${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_COMPILE_THREADS=1

# =============================================================================
# Required
# =============================================================================
MODEL_PATH=${MODEL_PATH:?MODEL_PATH environment variable is required}

# =============================================================================
# Data — same dataset as OPSDC training
# =============================================================================
TRAIN_FILE=${TRAIN_FILE:-./workspace/data/processed/train.parquet}
VAL_FILES=${VAL_FILES:-"['./workspace/data/processed/val_math500.parquet', './workspace/data/processed/val_aime24.parquet', './workspace/data/processed/val_aime25.parquet']"}

# =============================================================================
# Training budget
# =============================================================================
TOTAL_STEPS=${TOTAL_STEPS:-100}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}

# =============================================================================
# Batch sizes
# =============================================================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
N=${N:-8}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}

# =============================================================================
# Optimizer
# =============================================================================
LEARNING_RATE=${LEARNING_RATE:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
WARMUP_STEPS=${WARMUP_STEPS:-10}
GRAD_CLIP=${GRAD_CLIP:-1.0}

# =============================================================================
# Generation
# =============================================================================
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
VAL_RESPONSE_LENGTH=${VAL_RESPONSE_LENGTH:-30000}

# =============================================================================
# GRPO-specific: symmetric clipping, KL loss, seq-mean-token-mean
# =============================================================================
CLIP_RATIO=${CLIP_RATIO:-0.2}
LOSS_AGG=${LOSS_AGG:-seq-mean-token-mean}
USE_KL_LOSS=${USE_KL_LOSS:-True}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}

# =============================================================================
# Infrastructure
# =============================================================================
N_GPUS=${N_GPUS:-4}
TP_SIZE=${TP_SIZE:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}
ULYSSES_SP_SIZE=${ULYSSES_SP_SIZE:-2}

# =============================================================================
# Logging & checkpointing
# =============================================================================
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-25}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-}

# =============================================================================
# Experiment tracking
# =============================================================================
PROJECT_NAME=${PROJECT_NAME:-grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo}

MODEL_NAME=$(basename "$(dirname "${MODEL_PATH}")" | tr '[:upper:]' '[:lower:]')
if [[ "${EXPERIMENT_NAME}" != *"${MODEL_NAME}"* ]]; then
    EXPERIMENT_NAME="${MODEL_NAME}_${EXPERIMENT_NAME}"
fi

NFS_OUTPUT_DIR="${OUTPUT_DIR:-./outputs/${EXPERIMENT_NAME}}"
CHECKPOINT_DIR="${NFS_OUTPUT_DIR}/checkpoints"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${NFS_OUTPUT_DIR}/val_generations}"
mkdir -p "${CHECKPOINT_DIR}"

echo "============================================="
echo "  GRPO Training (vanilla + KL)"
echo "============================================="
echo "  Model:           ${MODEL_PATH}"
echo "  Train data:      ${TRAIN_FILE}"
echo "  Steps:           ${TOTAL_STEPS}"
echo "  LR:              ${LEARNING_RATE}"
echo "  Batch:           ${TRAIN_BATCH_SIZE} prompts × ${N} responses"
echo "  Context (train): ${MAX_PROMPT_LENGTH} prompt + ${MAX_RESPONSE_LENGTH} response"
echo "  Context (val):   ${MAX_PROMPT_LENGTH} prompt + ${VAL_RESPONSE_LENGTH} response"
echo "  Clip:            ${CLIP_RATIO} (symmetric)"
echo "  Loss agg:        ${LOSS_AGG}"
echo "  KL loss:         ${USE_KL_LOSS} (coef=${KL_LOSS_COEF}, type=${KL_LOSS_TYPE})"
echo "  GPUs:            ${N_GPUS} (TP=${TP_SIZE}, SP=${ULYSSES_SP_SIZE})"
echo "  Checkpoint dir:  ${CHECKPOINT_DIR}"
echo "============================================="

# =============================================================================
# Process validation data
# =============================================================================
DATA_DIR="${WORKSPACE_ROOT}/data"
PROCESSED_DIR="${DATA_DIR}/processed"

if [ -d "${DATA_DIR}" ]; then
    echo "Processing validation data on-cluster..."
    python3 "${SD_SRC}/data/process_eval_data.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${PROCESSED_DIR}"
    echo "Data processing complete."
fi

# =============================================================================
# Launch training
# =============================================================================
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="${WARMUP_STEPS}" \
    actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY}" \
    actor_rollout_ref.actor.grad_clip="${GRAD_CLIP}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio="${CLIP_RATIO}" \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${ULYSSES_SP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n="${N}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${ULYSSES_SP_SIZE}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    \
    reward_model.reward_manager=dapo \
    \
    trainer.critic_warmup=0 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    'trainer.logger=["console"]' \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.resume_mode=auto \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
    +trainer.val_max_response_length="${VAL_RESPONSE_LENGTH}" \
    "$@"

echo ""
echo "============================================="
echo "  GRPO Training complete!"
echo "============================================="
