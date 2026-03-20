#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
if [[ -z "${ROLE}" ]]; then
  echo "Usage: $0 <head|worker|stop|status>"
  exit 1
fi

source /root/anaconda3/etc/profile.d/conda.sh
conda activate verl

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

ETH_ENV_FILE="${ETH_ENV_FILE:-/root/A40_verl/recipe/eth_multinic.env}"
if [[ -f "${ETH_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ETH_ENV_FILE}"
fi

HEAD_NODE_IP="${HEAD_NODE_IP:-172.24.79.15}"
WORKER_NODE_IP="${WORKER_NODE_IP:-172.24.79.13}"
THIS_NODE_IP="${THIS_NODE_IP:-}"

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
NUM_CPUS_PER_NODE="${NUM_CPUS_PER_NODE:-$(nproc)}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

# Multi-NIC ethernet / NCCL socket defaults.
ETH_IFNAMES="${ETH_IFNAMES:-${ETH_MULTI_NIC_IFACES:-eth0,eth1,eth2,eth3}}"
PRIMARY_ETH_IFNAME="${PRIMARY_ETH_IFNAME:-${ETH_IFNAMES%%,*}}"

export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$PRIMARY_ETH_IFNAME}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$ETH_IFNAMES}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
unset NCCL_NET
unset NCCL_IB_HCA
unset NCCL_IB_GID_INDEX
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-${THIS_NODE_IP}}"

if [[ -z "${THIS_NODE_IP}" ]]; then
  if [[ "${ROLE}" == "head" ]]; then
    THIS_NODE_IP="${HEAD_NODE_IP}"
  elif [[ "${ROLE}" == "worker" ]]; then
    THIS_NODE_IP="${WORKER_NODE_IP}"
  fi
  export VLLM_HOST_IP="${THIS_NODE_IP:-}"
fi

echo "ROLE=${ROLE}"
echo "HEAD_NODE_IP=${HEAD_NODE_IP}"
echo "WORKER_NODE_IP=${WORKER_NODE_IP}"
echo "THIS_NODE_IP=${THIS_NODE_IP:-unset}"
echo "RAY_PORT=${RAY_PORT}"
echo "RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT}"
echo "ETH_IFNAMES=${ETH_IFNAMES}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"

case "${ROLE}" in
  head)
    ray stop --force >/dev/null 2>&1 || true
    ray start \
      --head \
      --node-ip-address="${THIS_NODE_IP}" \
      --port="${RAY_PORT}" \
      --dashboard-host=0.0.0.0 \
      --dashboard-port="${RAY_DASHBOARD_PORT}" \
      --num-gpus="${NUM_GPUS_PER_NODE}" \
      --num-cpus="${NUM_CPUS_PER_NODE}" \
      --disable-usage-stats
    ;;
  worker)
    ray stop --force >/dev/null 2>&1 || true
    ray start \
      --address="${HEAD_NODE_IP}:${RAY_PORT}" \
      --node-ip-address="${THIS_NODE_IP}" \
      --num-gpus="${NUM_GPUS_PER_NODE}" \
      --num-cpus="${NUM_CPUS_PER_NODE}" \
      --disable-usage-stats
    ;;
  stop)
    ray stop --force
    ;;
  status)
    ray status
    ;;
  *)
    echo "Unknown role: ${ROLE}"
    exit 1
    ;;
esac
