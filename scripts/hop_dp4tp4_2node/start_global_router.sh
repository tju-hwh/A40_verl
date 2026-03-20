#!/usr/bin/env bash
set -euo pipefail

cd /root/vllm

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0,eth1,eth2,eth3}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0,eth1,eth2,eth3}"

ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-8200}"
NODE0_IP="${NODE0_IP:?set NODE0_IP}"
NODE1_IP="${NODE1_IP:?set NODE1_IP}"
OWNER_STATE_URL="${OWNER_STATE_URL:?set OWNER_STATE_URL, e.g. http://node0:8300}"

DECODE_CUTOVERS="${DECODE_CUTOVERS:-1536}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-3600}"
CONNECT_TIMEOUT_S="${CONNECT_TIMEOUT_S:-60}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
UPSTREAM_MAX_MODEL_LEN="${UPSTREAM_MAX_MODEL_LEN:-12288}"
DP_ROUTING_STRATEGY="${DP_ROUTING_STRATEGY:-request_id_hash}"
LOG_PATH="${LOG_PATH:-/tmp/hop_global_router_dp4tp4.log}"

SERVER1_URLS="${SERVER1_URLS:-http://${NODE0_IP}:8101,http://${NODE0_IP}:8103,http://${NODE1_IP}:8101,http://${NODE1_IP}:8103}"
SERVER2_URLS="${SERVER2_URLS:-http://${NODE0_IP}:8102,http://${NODE0_IP}:8104,http://${NODE1_IP}:8102,http://${NODE1_IP}:8104}"
SERVER_KV_PORT_GROUPS="${SERVER_KV_PORT_GROUPS:-18101,18102;18103,18104;18101,18102;18103,18104}"

nohup python -m vllm.proxy_cluster.launch_dp_tp_router \
  --host "${ROUTER_HOST}" \
  --port "${ROUTER_PORT}" \
  --server1-urls "${SERVER1_URLS}" \
  --server2-urls "${SERVER2_URLS}" \
  --server-kv-port-groups "${SERVER_KV_PORT_GROUPS}" \
  --routing-mode sequential_handoff \
  --dp-routing-strategy "${DP_ROUTING_STRATEGY}" \
  --decode-cutovers "${DECODE_CUTOVERS}" \
  --request-timeout-s "${REQUEST_TIMEOUT_S}" \
  --connect-timeout-s "${CONNECT_TIMEOUT_S}" \
  --max-response-length "${MAX_RESPONSE_LENGTH}" \
  --upstream-max-model-len "${UPSTREAM_MAX_MODEL_LEN}" \
  --kv-owner-state-url "${OWNER_STATE_URL}" \
  >"${LOG_PATH}" 2>&1 &

echo $! > /tmp/hop_global_router_dp4tp4.pid
echo "global router started pid=$(cat /tmp/hop_global_router_dp4tp4.pid) log=${LOG_PATH}"
