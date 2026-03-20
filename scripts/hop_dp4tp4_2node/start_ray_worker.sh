#!/usr/bin/env bash
set -euo pipefail

NODE_IP="${NODE_IP:?set NODE_IP to this worker-node IP}"
HEAD_IP="${HEAD_IP:?set HEAD_IP to head-node IP}"
RAY_PORT="${RAY_PORT:-6379}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0,eth1,eth2,eth3}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0,eth1,eth2,eth3}"

ray stop --force >/dev/null 2>&1 || true
ray start \
  --address="${HEAD_IP}:${RAY_PORT}" \
  --node-ip-address="${NODE_IP}"
