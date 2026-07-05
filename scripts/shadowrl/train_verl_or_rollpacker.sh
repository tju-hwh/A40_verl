#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/shadowrl/train_verl_or_rollpacker.sh --system verl|rollpacker --model MODEL --dataset DATASET [options]

Models:
  qwen3                 /mnt/L202500425/hwh/model/Qwen3-0.6B
  qwen3_8b              /mnt/L202500425/hwh/model/Qwen3-8B
  llama31_8b            /mnt/L202500425/hwh/model/Meta-Llama-3.1-8B-Instruct
  deepseek_r1_qwen7b    /mnt/L202500425/hwh/model/DeepSeek-R1-Distill-Qwen-7B

Datasets:
  deepmath
  eurus
  hh

Options:
  --port PORT                 SGLang base port, default 30000
  --run-tag TAG               log/result tag, default timestamp
  --max-prompt-length N       default 1024
  --max-response-length N     default 1024
  --log-root PATH             default /root/shadowrl_verl_sglang/bench_logs/verl_full_rl
USAGE
}

ROOT=/root/shadowrl_verl_sglang
SYSTEM=""
MODEL=""
DATASET=""
PORT=30000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)
      SYSTEM="${2:?missing --system value}"
      shift 2
      ;;
    --model)
      MODEL="${2:?missing --model value}"
      shift 2
      ;;
    --dataset)
      DATASET="${2:?missing --dataset value}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing --port value}"
      shift 2
      ;;
    --run-tag)
      export RUN_TAG="${2:?missing --run-tag value}"
      shift 2
      ;;
    --max-prompt-length)
      export MAX_PROMPT_LENGTH="${2:?missing --max-prompt-length value}"
      shift 2
      ;;
    --max-response-length)
      export MAX_RESPONSE_LENGTH="${2:?missing --max-response-length value}"
      shift 2
      ;;
    --log-root)
      export LOG_ROOT="${2:?missing --log-root value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SYSTEM" || -z "$MODEL" || -z "$DATASET" ]]; then
  usage >&2
  exit 2
fi

case "$SYSTEM" in
  verl|rollpacker) ;;
  *) echo "--system must be verl or rollpacker, got: $SYSTEM" >&2; exit 2 ;;
esac

case "$MODEL" in
  qwen3|qwen3_8b|llama31_8b|deepseek_r1_qwen7b) ;;
  *) echo "unknown --model: $MODEL" >&2; exit 2 ;;
esac

case "$DATASET" in
  deepmath|eurus|hh) ;;
  *) echo "unknown --dataset: $DATASET" >&2; exit 2 ;;
esac

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot auto-select GPUs" >&2
  exit 1
fi

CUDA_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | head -n 4 | paste -sd, -)"
GPU_COUNT="$(awk -F, 'NF {print NF}' <<<"$CUDA_DEVICES")"
if [[ "$GPU_COUNT" -ne 4 ]]; then
  echo "need 4 visible GPUs from nvidia-smi, got $GPU_COUNT: ${CUDA_DEVICES:-none}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export N_GPUS_PER_NODE=4
export TP_SIZE=2
export DP_SIZE=2
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-512}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
if [[ -z "${PPO_MICRO_BATCH_SIZE_PER_GPU+x}" ]]; then
  if [[ "$MAX_RESPONSE_LENGTH" -ge 4096 ]]; then
    export PPO_MICRO_BATCH_SIZE_PER_GPU=1
  else
    export PPO_MICRO_BATCH_SIZE_PER_GPU=8
  fi
fi
if [[ -z "${GPU_MEMORY_UTILIZATION+x}" ]]; then
  if [[ "$MAX_RESPONSE_LENGTH" -ge 4096 ]]; then
    export GPU_MEMORY_UTILIZATION=0.35
  else
    export GPU_MEMORY_UTILIZATION=0.50
  fi
fi
if [[ -z "${NCCL_TIMEOUT+x}" ]]; then
  if [[ "$MAX_RESPONSE_LENGTH" -ge 4096 ]]; then
    export NCCL_TIMEOUT=7200
  else
    export NCCL_TIMEOUT=600
  fi
fi
export ROLLOUT_MODE=${ROLLOUT_MODE:-sync}
if [[ -z "${ROLLOUT_MAX_NUM_SEQS+x}" ]]; then
  if [[ "$MAX_RESPONSE_LENGTH" -ge 4096 ]]; then
    export ROLLOUT_MAX_NUM_SEQS=32
  else
    export ROLLOUT_MAX_NUM_SEQS=64
  fi
fi
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}

# VERL sees prompt batch before rollout.n=4. These defaults produce 512
# baseline rollout samples, and 640 RollPacker issued samples with 512 kept.
export VERL_PROMPT_BATCH=${VERL_PROMPT_BATCH:-128}
export ROLLPACKER_PROMPT_BATCH=${ROLLPACKER_PROMPT_BATCH:-160}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "N_GPUS_PER_NODE=$N_GPUS_PER_NODE TP_SIZE=$TP_SIZE DP_SIZE=$DP_SIZE"
echo "NCCL_TIMEOUT=$NCCL_TIMEOUT ROLLOUT_MAX_NUM_SEQS=$ROLLOUT_MAX_NUM_SEQS ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
echo "system=$SYSTEM model=$MODEL dataset=$DATASET port=$PORT"

cd "$ROOT"
bash scripts/shadowrl/run_verl_shadowrl_rollpacker_one_step.sh "$SYSTEM" "$MODEL" "$DATASET" "$PORT"
