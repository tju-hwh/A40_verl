#!/usr/bin/env bash
set -euo pipefail

export micro_batch_size=8
export rollout_max_num_seqs=256
export ppo_mini_batch_size=128

export HOP_OWNER_MAX_NUM_SEQS=256
export HOP_CONSUMER_MAX_NUM_SEQS=256

export HOP_MPS_ACTIVE_THREAD_PERCENTAGES='[100,60]'
export ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE=70
export REF_MPS_ACTIVE_THREAD_PERCENTAGE=70
export CRITIC_MPS_ACTIVE_THREAD_PERCENTAGE=70

export HOP_DECODE_CUTOVERS='[1536]'
export HOP_SEND_PUBLISH_TOKEN_STRIDE=2048

export TRAIN_GROUP_PLAN='[12,12,12,12,12,12,12,12,8,8,8,8]'

exec /root/A40_verl/rlos_test_deepmath_4gpu.sh "$@"
