#!/usr/bin/env bash
set -euo pipefail

export HOP_OWNER_TP_SIZE=2
export HOP_OWNER_DP_SIZE=2
export HOP_CONSUMER_TP_SIZE=2
export HOP_CONSUMER_DP_SIZE=2

export HOP_OWNER_CUDA_VISIBLE_DEVICES="0,1,2,3"
export HOP_CONSUMER_CUDA_VISIBLE_DEVICES_ALL="0,1,2,3"

cd /root/A40_verl
exec /root/A40_verl/rlos_test_deepmath_4gpu.sh "$@"
