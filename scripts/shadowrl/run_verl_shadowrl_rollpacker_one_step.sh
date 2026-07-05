#!/usr/bin/env bash
set -euo pipefail

SYSTEM="${1:?system verl|shadowrl|rollpacker}"
MODEL_KEY="${2:?model key qwen3|qwen3_8b|llama31_8b|deepseek_r1_qwen7b}"
DATASET_KEY="${3:?dataset key deepmath|eurus|hh}"
PORT="${4:-30000}"

ROOT=/root/shadowrl_verl_sglang
MODEL_ROOT=/mnt/L202500425/hwh/model
DATA_ROOT=/mnt/L202500425/hwh/dataset/verl_shadowrl_bench
LOG_ROOT=${LOG_ROOT:-/root/shadowrl_verl_sglang/bench_logs/verl_full_rl}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.50}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
TP_SIZE=${TP_SIZE:-2}
DP_SIZE=${DP_SIZE:-2}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-512}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
NCCL_TIMEOUT=${NCCL_TIMEOUT:-600}
ROLLOUT_MODE=${ROLLOUT_MODE:-sync}
PDMUX_PREFILL_SM=${PDMUX_PREFILL_SM:-66}
PDMUX_DECODE_SM=${PDMUX_DECODE_SM:-66}

case "$MODEL_KEY" in
  qwen3) MODEL_PATH="$MODEL_ROOT/Qwen3-0.6B" ;;
  qwen3_8b) MODEL_PATH="$MODEL_ROOT/Qwen3-8B" ;;
  llama31_8b) MODEL_PATH="$MODEL_ROOT/Meta-Llama-3.1-8B-Instruct" ;;
  deepseek_r1_qwen7b) MODEL_PATH="$MODEL_ROOT/DeepSeek-R1-Distill-Qwen-7B" ;;
  *) echo "unknown model key: $MODEL_KEY" >&2; exit 2 ;;
esac

TRAIN_FILE="$DATA_ROOT/$DATASET_KEY/train.parquet"
VAL_FILE="$DATA_ROOT/$DATASET_KEY/val.parquet"
if [[ ! -f "$TRAIN_FILE" ]]; then
  source /root/envs/shadowrl/bin/activate
  cd "$ROOT"
  python scripts/shadowrl/prepare_verl_bench_data.py
fi

mkdir -p "$LOG_ROOT" "$LOG_ROOT/pdmux"
LOG_FILE="$LOG_ROOT/${SYSTEM}__${MODEL_KEY}__${DATASET_KEY}__${RUN_TAG}.log"
RESULT_JSON="$LOG_ROOT/results_${RUN_TAG}.jsonl"
PDMUX_CONFIG="$LOG_ROOT/pdmux/${MODEL_KEY}_${DATASET_KEY}_${RUN_TAG}.yaml"

cat > "$PDMUX_CONFIG" <<YAML
sm_group_num: 3
manual_divisions:
  - [$PDMUX_PREFILL_SM, $PDMUX_DECODE_SM, 0]
shadow_decode_cutoff_steps: 1000
shadow_decode_stream_idx: 1
shadow_decode_step_policy: min
YAML

PROMPT_BATCH=${SHADOWRL_PROMPT_BATCH:-128}
OVER_SAMPLE_RATE=0
ROLLPACKER_ENABLE=False
SGLANG_ENGINE_OVERRIDES=()
if [[ "$SYSTEM" == "shadowrl" ]]; then
  SGLANG_ENGINE_OVERRIDES=(
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.enable_pdmux=True"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.pdmux_config_path=$PDMUX_CONFIG"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=-1"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=True"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend=pytorch"
  )
elif [[ "$SYSTEM" == "rollpacker" ]]; then
  PROMPT_BATCH=${ROLLPACKER_PROMPT_BATCH:-160}
  OVER_SAMPLE_RATE=0.2
  ROLLPACKER_ENABLE=True
elif [[ "$SYSTEM" == "verl" ]]; then
  PROMPT_BATCH=${VERL_PROMPT_BATCH:-128}
  OVER_SAMPLE_RATE=0
  ROLLPACKER_ENABLE=False
else
  echo "unknown system: $SYSTEM" >&2
  exit 2
fi

ROLLOUT_N=4
if [[ "$SYSTEM" != "shadowrl" ]]; then
  CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-$(( (PROMPT_BATCH * ROLLOUT_N + DP_SIZE - 1) / DP_SIZE ))}
  SGLANG_ENGINE_OVERRIDES=(
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=False"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.disable_prefill_cuda_graph=True"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_backend_decode=full"
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs_decode=$CUDA_GRAPH_MAX_BS"
  )
fi

source /root/envs/shadowrl/bin/activate
cd "$ROOT"
VENV_SITE="/root/envs/shadowrl/lib/python3.10/site-packages"
CUDA13_LIBS=(
  "$VENV_SITE/nvidia/cu13/lib"
  "$VENV_SITE/nvidia/cuda_runtime/lib"
  "$VENV_SITE/nvidia/cublas/lib"
  "$VENV_SITE/nvidia/cudnn/lib"
  "$VENV_SITE/nvidia/cufft/lib"
  "$VENV_SITE/nvidia/curand/lib"
  "$VENV_SITE/nvidia/cusolver/lib"
  "$VENV_SITE/nvidia/cusparse/lib"
  "$VENV_SITE/nvidia/nccl/lib"
  "$VENV_SITE/nvidia/nvjitlink/lib"
  "$VENV_SITE/nvidia/nvtx/lib"
)
for lib_dir in "${CUDA13_LIBS[@]}"; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
  fi
done
export PYTHONPATH="/root/shadowrl_verl_sglang:/root/shadowrl_sglang:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_PORT="$PORT"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
unset SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK
unset SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false

ray stop --force >/dev/null 2>&1 || true

set -x
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$ROOT/scripts/shadowrl/zero_length_reward.py" \
  custom_reward_function.name=compute_score \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$PROMPT_BATCH" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=right \
  data.return_raw_chat=True \
  data.shuffle=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.nccl_timeout="$NCCL_TIMEOUT" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode="$ROLLOUT_MODE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
  actor_rollout_ref.rollout.data_parallel_size="$DP_SIZE" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.ignore_eos=True \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
  actor_rollout_ref.rollout.over_sample_rate="$OVER_SAMPLE_RATE" \
  actor_rollout_ref.rollout.rollpacker_enable="$ROLLPACKER_ENABLE" \
  actor_rollout_ref.rollout.rollpacker_filter_aborted=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  critic.enable=False \
  reward_model.enable=False \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.total_training_steps=1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=False \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.logger='["console"]' \
  trainer.project_name=shadowrl_full_rl_bench \
  trainer.experiment_name="${SYSTEM}_${MODEL_KEY}_${DATASET_KEY}_${RUN_TAG}" \
  "${SGLANG_ENGINE_OVERRIDES[@]}" \
  2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
ray stop --force >/dev/null 2>&1 || true
python scripts/shadowrl/summarize_verl_results.py --log-dir "$LOG_ROOT" --output "$RESULT_JSON" || true
exit "$status"
