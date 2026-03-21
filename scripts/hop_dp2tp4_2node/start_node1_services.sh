#!/usr/bin/env bash
set -euo pipefail

NODE0_IP="${NODE0_IP:-172.24.79.15}"
NODE1_IP="${NODE1_IP:-172.24.79.13}"
CUDA_VISIBLE_DEVICES_LOCAL="${CUDA_VISIBLE_DEVICES_LOCAL:-0,1,2,3,4,5,6,7}"
HOP_LOG="${HOP_LOG:-/tmp/hop_dp2tp4_node1.log}"

env NODE_IP="${NODE1_IP}" \
  HEAD_IP="${NODE0_IP}" \
  bash /root/A40_verl/scripts/hop_dp4tp4_2node/start_ray_worker.sh

pkill -f 'test_qwen8b_2server_hop_dptp_new.py.*--serve-only' >/dev/null 2>&1 || true

nohup env \
  NODE_IP="${NODE1_IP}" \
  CUDA_VISIBLE_DEVICES_LOCAL="${CUDA_VISIBLE_DEVICES_LOCAL}" \
  bash /root/model/start_qwen8b_2server_hop_dptp_new_serve.sh \
  >"${HOP_LOG}" 2>&1 &

echo "node1 ray worker joined ${NODE0_IP}:6379"
echo "node1 local hop log: ${HOP_LOG}"
