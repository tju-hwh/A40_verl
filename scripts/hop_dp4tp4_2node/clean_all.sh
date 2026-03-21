#!/usr/bin/env bash
set -euo pipefail

# Cleans the local 2-node dp4tp4 VERL + hop-vLLM stack on this machine.
# If REMOTE_HOST is set, it will also ssh to the peer and run the same script there.

SCRIPT_PATH="/root/A40_verl/scripts/hop_dp4tp4_2node/clean_all.sh"
RUN_REMOTE="${RUN_REMOTE:-1}"
REMOTE_HOST="${REMOTE_HOST:-}"

echo "[clean] stopping training and local ray/hop processes on $(hostname)"

pkill -f 'recipe.one_step_off_policy.main_ppo|main_ppo.py|ray_trainer.py' || true
pkill -f 'launch_dp_tp_cluster|launch_dp_tp_router|launch_four_server_ipc_vllm|launch_sequential_decode_router' || true
pkill -f 'kv_owner_state_server:create_app|uvicorn vllm.proxy_cluster.kv_owner_state_server:create_app' || true
pkill -f 'vllm.entrypoints.openai.api_server|vllm.v1.engine.core|VllmWorker' || true
pkill -f 'vLLMHttpServer|AgentLoopWorker|OneStepTaskRunner' || true

ray stop --force >/dev/null 2>&1 || true

pkill -f 'nvidia-cuda-mps-control|nvidia-cuda-mps-server' || true

rm -f /tmp/hop_owner_state_dp4tp4.pid \
      /tmp/hop_global_router_dp4tp4.pid \
      /tmp/hop_local_cluster_*.pid || true

rm -f /tmp/hop_owner_state_dp4tp4.log \
      /tmp/hop_global_router_dp4tp4.log \
      /tmp/hop_local_cluster_*.log \
      /root/1.log || true

rm -f /tmp/vllm_ipc_meta_dp* \
      /tmp/vllm_ipc_meta* || true

rm -rf /tmp/vllm_kv_ipc \
       /tmp/vllm_shared_kv_pool* || true

rm -f /dev/shm/vllm_shared_kv_pool_dp4tp4* \
      /dev/shm/vllm_shared_kv_pool* || true

echo "[clean] local cleanup done on $(hostname)"

if [[ "${RUN_REMOTE}" == "1" && -n "${REMOTE_HOST}" ]]; then
  echo "[clean] running peer cleanup on ${REMOTE_HOST}"
  ssh "${REMOTE_HOST}" "RUN_REMOTE=0 bash ${SCRIPT_PATH}"
fi

