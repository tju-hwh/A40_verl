#!/usr/bin/env bash
set -euo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate verl

cd /root/A40_verl

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

ETH_ENV_FILE="${ETH_ENV_FILE:-/root/A40_verl/recipe/eth_multinic.env}"
if [[ -f "${ETH_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ETH_ENV_FILE}"
fi

project_name="${PROJECT_NAME:-GRPO}"
exp_name="${EXP_NAME:-deepmath-qwen3-14b-mainline-sync-grpo-2node16gpu-rdma}"

MODEL_PATH="${MODEL_PATH:-/root/model/Qwen-14B}"
TRAIN_FILE="${TRAIN_FILE:-/root/data/deepmath/train.parquet}"
TEST_FILE="${TEST_FILE:-/root/data/deepmath/test.parquet}"
CKPTS_DIR="${CKPTS_DIR:-/root/A40_verl/ckpts/${project_name}/${exp_name}}"

HEAD_NODE_IP="${HEAD_NODE_IP:-172.24.79.15}"
WORKER_NODE_IP="${WORKER_NODE_IP:-172.24.79.13}"
NNODES="${NNODES:-2}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
EXPECTED_NODES="${EXPECTED_NODES:-2}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

# Multi-NIC ethernet / NCCL socket defaults.
ETH_IFNAMES="${ETH_IFNAMES:-${ETH_MULTI_NIC_IFACES:-eth0,eth1,eth2,eth3}}"
PRIMARY_ETH_IFNAME="${PRIMARY_ETH_IFNAME:-${ETH_IFNAMES%%,*}}"

# Keep the major rollout settings aligned with the 4-GPU reference script unless
# the user explicitly asked for a change.
max_prompt_length="${MAX_PROMPT_LENGTH:-1024}"
max_response_length="${MAX_RESPONSE_LENGTH:-4096}"
rollout_max_model_len="${ROLLOUT_MAX_MODEL_LEN:-5120}"
rollout_tp="${ROLLOUT_TP:-4}"
rollout_gpu_mem_util="${ROLLOUT_GPU_MEM_UTIL:-0.6}"
rollout_temperature="${ROLLOUT_TEMPERATURE:-0.6}"
rollout_top_p="${ROLLOUT_TOP_P:-0.95}"
rollout_top_k="${ROLLOUT_TOP_K:-20}"
rollout_max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-256}"

train_prompt_bsz="${TRAIN_PROMPT_BSZ:-128}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-2}"
micro_batch_size="${MICRO_BATCH_SIZE:-8}"
mini_batch_size="${MINI_BATCH_SIZE:-128}"
actor_lr="${ACTOR_LR:-1e-6}"
total_training_steps="${TOTAL_TRAINING_STEPS:-23}"
total_epochs="${TOTAL_EPOCHS:-1}"

RUNTIME_ENV_JSON="$(python3 - <<PY
import json
import os
env = {
    "TOKENIZERS_PARALLELISM": "false",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_DEBUG": "WARN",
    "TORCH_NCCL_HIGH_PRIORITY": "1",
    "NCCL_SOCKET_IFNAME": "${ETH_IFNAMES}",
    "GLOO_SOCKET_IFNAME": "${PRIMARY_ETH_IFNAME}",
    "NCCL_IB_DISABLE": "1",
    "NCCL_CROSS_NIC": "1",
    "HF_MODULES_CACHE": "/root/.cache/huggingface/modules",
    "PYTHONPATH": "/root/.cache/huggingface/modules" + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
}
print(json.dumps({"working_dir": "/root/A40_verl", "env_vars": env}))
PY
)"

echo "Waiting for Ray cluster to have ${EXPECTED_NODES} nodes..."
for _ in $(seq 1 60); do
  node_count="$(python3 - <<'PY'
import contextlib
import ray

with contextlib.suppress(Exception):
    ray.init(address="auto", logging_level="ERROR")
    print(sum(1 for node in ray.nodes() if node.get("Alive")))
    raise SystemExit(0)

print(0)
PY
)"
  if [[ "${node_count}" -ge "${EXPECTED_NODES}" ]]; then
    break
  fi
  sleep 5
done

ray status

ray job submit \
  --address="http://${HEAD_NODE_IP}:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  --no-wait \
  -- \
  python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=prompt \
  data.reward_fn_key=data_source \
  data.return_raw_chat=True \
  data.trust_remote_code=True \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.train_batch_size=${train_prompt_bsz} \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  +critic.model.override_config.attn_implementation=eager \
  actor_rollout_ref.actor.strategy=fsdp2 \
  critic.strategy=fsdp2 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=${actor_lr} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.temperature=${rollout_temperature} \
  actor_rollout_ref.rollout.top_p=${rollout_top_p} \
  actor_rollout_ref.rollout.top_k=${rollout_top_k} \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
  actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
  actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs} \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_model_len} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size} \
  actor_rollout_ref.rollout.val_kwargs.temperature=${rollout_temperature} \
  actor_rollout_ref.rollout.val_kwargs.top_p=${rollout_top_p} \
  actor_rollout_ref.rollout.val_kwargs.top_k=${rollout_top_k} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  algorithm.use_kl_in_reward=False \
  reward_model.reward_manager=naive \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.save_freq=0 \
  trainer.test_freq=0 \
  trainer.total_epochs=${total_epochs} \
  trainer.total_training_steps=${total_training_steps} \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.resume_mode=disable \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  "$@"
