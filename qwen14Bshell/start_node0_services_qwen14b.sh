#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate verl

NODE0_IP="${NODE0_IP:-172.24.79.15}"
CUDA_VISIBLE_DEVICES_LOCAL="${CUDA_VISIBLE_DEVICES_LOCAL:-0,1,2,3,4,5,6,7}"
MODEL_PATH="${MODEL_PATH:-/root/model/Qwen-14B}"
HOP_LOG="${HOP_LOG:-/tmp/hop_dp2tp4_qwen14b_node0.log}"
RUNTIME_DIR="${RUNTIME_DIR:-/tmp/qwen14b_2server_hop_dptp_172_24_79_15}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
ETH_IFNAMES="${ETH_IFNAMES:-eth0,eth1,eth2,eth3}"
HF_MODULES_CACHE="${HF_MODULES_CACHE:-/root/.cache/huggingface/modules}"

export NCCL_SOCKET_IFNAME="${ETH_IFNAMES}"
export GLOO_SOCKET_IFNAME="${ETH_IFNAMES}"
export HF_MODULES_CACHE

python - <<'PY'
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_cached_module_file
p="/root/model/Qwen-14B"
AutoTokenizer.from_pretrained(p, trust_remote_code=True)
AutoConfig.from_pretrained(p, trust_remote_code=True)
get_cached_module_file(p, "configuration_qwen.py")
get_cached_module_file(p, "modeling_qwen.py")
print("qwen14b dynamic modules warmed on node0")
PY

ray stop --force >/dev/null 2>&1 || true
ray start \
  --head \
  --node-ip-address="${NODE0_IP}" \
  --port="${RAY_PORT}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${RAY_DASHBOARD_PORT}"

pkill -f 'qwen14Bshell/test_qwen14b_2server_hop_dptp_new.py.*--serve-only' >/dev/null 2>&1 || true

nohup env \
  NODE_IP="${NODE0_IP}" \
  MODEL_PATH="${MODEL_PATH}" \
  RUNTIME_DIR="${RUNTIME_DIR}" \
  CUDA_VISIBLE_DEVICES_LOCAL="${CUDA_VISIBLE_DEVICES_LOCAL}" \
  bash /root/A40_verl/qwen14Bshell/start_qwen14b_2server_hop_dptp_new_serve.sh \
  >"${HOP_LOG}" 2>&1 &

echo "node0 ray head started at ${NODE0_IP}:${RAY_PORT}"
echo "node0 local hop log: ${HOP_LOG}"
