#!/usr/bin/env bash
set -euo pipefail

cd /root/vllm

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0,eth1,eth2,eth3}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0,eth1,eth2,eth3}"

NODE_IP="${NODE_IP:?set NODE_IP to this machine's routable IP}"
MODEL_PATH="${MODEL_PATH:-/root/model/Qwen3-8B}"
LOCAL_CUDA_VISIBLE_DEVICES="${LOCAL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
LOCAL_DP_SIZE="${LOCAL_DP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-4}"
OWNER_STATE_URL="${OWNER_STATE_URL:?set OWNER_STATE_URL, e.g. http://node0:8300}"

SERVER1_BASE_PORT="${SERVER1_BASE_PORT:-8101}"
SERVER2_BASE_PORT="${SERVER2_BASE_PORT:-8102}"
SERVER_KV_BASE_PORT="${SERVER_KV_BASE_PORT:-18101}"

OWNER_GPU_MEM_UTIL="${OWNER_GPU_MEM_UTIL:-0.22}"
CONSUMER_GPU_MEM_UTIL="${CONSUMER_GPU_MEM_UTIL:-0.22}"
OWNER_MAX_NUM_SEQS="${OWNER_MAX_NUM_SEQS:-256}"
CONSUMER_MAX_NUM_SEQS="${CONSUMER_MAX_NUM_SEQS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
SHARED_KV_POOL_META_PATH="${SHARED_KV_POOL_META_PATH:-/dev/shm/vllm_shared_kv_pool_dp4tp4}"
SEND_ACTIVATION_MARGIN_TOKENS="${SEND_ACTIVATION_MARGIN_TOKENS:-0}"
SEND_PUBLISH_TOKEN_STRIDE="${SEND_PUBLISH_TOKEN_STRIDE:-2048}"
OWNER_COMPILATION_CONFIG="${OWNER_COMPILATION_CONFIG:-'{\"level\":3,\"use_inductor\":true,\"use_cudagraph\":true}'}"
CONSUMER_COMPILATION_CONFIG="${CONSUMER_COMPILATION_CONFIG:-'{\"level\":3,\"use_inductor\":true,\"use_cudagraph\":true}'}"
CONSUMER_ATTENTION_BACKEND="${CONSUMER_ATTENTION_BACKEND:-FLASH_ATTN}"

ENABLE_CUDA_MPS="${ENABLE_CUDA_MPS:-true}"
MPS_OWNER_PERCENTAGE="${MPS_OWNER_PERCENTAGE:-100}"
MPS_CONSUMER_PERCENTAGE="${MPS_CONSUMER_PERCENTAGE:-60}"

LOG_PATH="${LOG_PATH:-/tmp/hop_local_cluster_${NODE_IP//./_}.log}"

export VLLM_HOST_IP="${NODE_IP}"

CMD=(
  python -m vllm.proxy_cluster.launch_dp_tp_cluster
  --model "${MODEL_PATH}"
  --host "${NODE_IP}"
  --data-parallel-size "${LOCAL_DP_SIZE}"
  --tensor-parallel-size "${TP_SIZE}"
  --server1-base-port "${SERVER1_BASE_PORT}"
  --server2-base-port "${SERVER2_BASE_PORT}"
  --server-kv-base-port "${SERVER_KV_BASE_PORT}"
  --cuda-visible-devices "${LOCAL_CUDA_VISIBLE_DEVICES}"
  --owner-gpu-memory-utilization "${OWNER_GPU_MEM_UTIL}"
  --consumer-gpu-memory-utilization "${CONSUMER_GPU_MEM_UTIL}"
  --owner-max-num-seqs "${OWNER_MAX_NUM_SEQS}"
  --consumer-max-num-seqs "${CONSUMER_MAX_NUM_SEQS}"
  --owner-max-model-len "${MAX_MODEL_LEN}"
  --consumer-max-model-len "${MAX_MODEL_LEN}"
  --owner-compilation-config "${OWNER_COMPILATION_CONFIG}"
  --consumer-compilation-config "${CONSUMER_COMPILATION_CONFIG}"
  --consumer-attention-backend "${CONSUMER_ATTENTION_BACKEND}"
  --kv-owner-state-url "${OWNER_STATE_URL}"
  --kv-transfer-config-template '{"kv_connector":"CudaIpcConnector","kv_role":"kv_both","kv_rank":0,"kv_parallel_size":1}'
  --shared-kv-pool-enable
  --shared-kv-pool-meta-path "${SHARED_KV_POOL_META_PATH}"
  --send-activation-margin-tokens "${SEND_ACTIVATION_MARGIN_TOKENS}"
  --send-publish-token-stride "${SEND_PUBLISH_TOKEN_STRIDE}"
  --no-consumer-enforce-eager
)

if [[ "${ENABLE_CUDA_MPS}" == "true" || "${ENABLE_CUDA_MPS}" == "1" ]]; then
  CMD+=(
    --enable-cuda-mps
    --mps-owner-percentage "${MPS_OWNER_PERCENTAGE}"
    --mps-consumer-percentage "${MPS_CONSUMER_PERCENTAGE}"
  )
fi

nohup "${CMD[@]}" >"${LOG_PATH}" 2>&1 &

echo $! > "/tmp/hop_local_cluster_${NODE_IP//./_}.pid"
echo "local hop cluster started pid=$(cat /tmp/hop_local_cluster_${NODE_IP//./_}.pid) log=${LOG_PATH}"
