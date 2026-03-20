#!/usr/bin/env bash
set -euo pipefail

cd /root/vllm

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0,eth1,eth2,eth3}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0,eth1,eth2,eth3}"

HOST="${HOST:-0.0.0.0}"
OWNER_STATE_PORT="${OWNER_STATE_PORT:-8300}"
LOG_PATH="${LOG_PATH:-/tmp/hop_owner_state_dp4tp4.log}"

nohup python -m uvicorn vllm.proxy_cluster.kv_owner_state_server:create_app \
  --factory \
  --host "${HOST}" \
  --port "${OWNER_STATE_PORT}" \
  >"${LOG_PATH}" 2>&1 &

echo $! > /tmp/hop_owner_state_dp4tp4.pid
echo "owner-state started pid=$(cat /tmp/hop_owner_state_dp4tp4.pid) log=${LOG_PATH}"
