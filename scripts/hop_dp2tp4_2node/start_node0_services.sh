#!/usr/bin/env bash
set -euo pipefail

NODE0_IP="${NODE0_IP:-172.24.79.15}"
CUDA_VISIBLE_DEVICES_LOCAL="${CUDA_VISIBLE_DEVICES_LOCAL:-0,1,2,3,4,5,6,7}"
HOP_LOG="${HOP_LOG:-/tmp/hop_dp2tp4_node0.log}"

env NODE_IP="${NODE0_IP}" \
  bash /root/A40_verl/scripts/hop_dp4tp4_2node/start_ray_head.sh

pkill -f 'test_qwen8b_2server_hop_dptp_new.py.*--serve-only' >/dev/null 2>&1 || true

nohup env \
  NODE_IP="${NODE0_IP}" \
  CUDA_VISIBLE_DEVICES_LOCAL="${CUDA_VISIBLE_DEVICES_LOCAL}" \
  bash /root/model/start_qwen8b_2server_hop_dptp_new_serve.sh \
  >"${HOP_LOG}" 2>&1 &

echo "node0 ray head started at ${NODE0_IP}:6379"
echo "node0 local hop log: ${HOP_LOG}"
