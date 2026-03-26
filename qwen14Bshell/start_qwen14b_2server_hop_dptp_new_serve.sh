#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate verl

NODE_IP="${NODE_IP:?set NODE_IP to this machine IP}"
CUDA_VISIBLE_DEVICES_LOCAL="${CUDA_VISIBLE_DEVICES_LOCAL:-0,1,2,3,4,5,6,7}"
MODEL_PATH="${MODEL_PATH:-/root/model/Qwen-14B}"
PYTHON_BIN="${PYTHON_BIN:-/root/anaconda3/envs/verl/bin/python}"

DP_SIZE="${DP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-4}"
ROUTER_PORT="${ROUTER_PORT:-8200}"
OWNER_STATE_PORT="${OWNER_STATE_PORT:-8300}"
SERVER1_BASE_PORT="${SERVER1_BASE_PORT:-8101}"
SERVER2_BASE_PORT="${SERVER2_BASE_PORT:-8102}"
SERVER_KV_BASE_PORT="${SERVER_KV_BASE_PORT:-18101}"

MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-5120}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
OWNER_GPU_MEM_UTIL="${OWNER_GPU_MEM_UTIL:-0.24}"
CONSUMER_GPU_MEM_UTIL="${CONSUMER_GPU_MEM_UTIL:-0.24}"
SEND_PUBLISH_TOKEN_STRIDE="${SEND_PUBLISH_TOKEN_STRIDE:-1536}"
DECODE_CUTOVER_TOKENS="${DECODE_CUTOVER_TOKENS:-1024}"
SEND_ACTIVATION_MARGIN_TOKENS="${SEND_ACTIVATION_MARGIN_TOKENS:-0}"
DP_ROUTING_STRATEGY="${DP_ROUTING_STRATEGY:-request_id_hash}"

ENABLE_CUDA_MPS="${ENABLE_CUDA_MPS:-true}"
MPS_OWNER_PERCENTAGE="${MPS_OWNER_PERCENTAGE:-100}"
MPS_CONSUMER_PERCENTAGE="${MPS_CONSUMER_PERCENTAGE:-40}"

CONSUMER_COMPILATION_CONFIG="${CONSUMER_COMPILATION_CONFIG:-{\"level\":3,\"use_inductor\":true,\"use_cudagraph\":true}}"
RUNTIME_DIR="${RUNTIME_DIR:-/tmp/qwen14b_2server_hop_dptp_${NODE_IP//./_}}"

cd /root

cmd=(
  python /root/A40_verl/qwen14Bshell/test_qwen14b_2server_hop_dptp_new.py
  --serve-only
  --model-path "${MODEL_PATH}"
  --python-bin "${PYTHON_BIN}"
  --data-parallel-size "${DP_SIZE}"
  --tensor-parallel-size "${TP_SIZE}"
  --bind-host "0.0.0.0"
  --advertise-host "${NODE_IP}"
  --router-port "${ROUTER_PORT}"
  --owner-state-port "${OWNER_STATE_PORT}"
  --server1-base-port "${SERVER1_BASE_PORT}"
  --server2-base-port "${SERVER2_BASE_PORT}"
  --server-kv-base-port "${SERVER_KV_BASE_PORT}"
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES_LOCAL}"
  --max-tokens "${MAX_TOKENS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --owner-gpu-memory-utilization "${OWNER_GPU_MEM_UTIL}"
  --consumer-gpu-memory-utilization "${CONSUMER_GPU_MEM_UTIL}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --send-activation-margin-tokens "${SEND_ACTIVATION_MARGIN_TOKENS}"
  --send-publish-token-stride "${SEND_PUBLISH_TOKEN_STRIDE}"
  --decode-cutover-tokens "${DECODE_CUTOVER_TOKENS}"
  --dp-routing-strategy "${DP_ROUTING_STRATEGY}"
  --consumer-compilation-config "${CONSUMER_COMPILATION_CONFIG}"
  --runtime-dir "${RUNTIME_DIR}"
)

if [[ "${ENABLE_CUDA_MPS}" == "true" || "${ENABLE_CUDA_MPS}" == "1" ]]; then
  cmd+=(
    --enable-cuda-mps
    --mps-owner-percentage "${MPS_OWNER_PERCENTAGE}"
    --mps-consumer-percentage "${MPS_CONSUMER_PERCENTAGE}"
  )
fi

exec "${cmd[@]}"
