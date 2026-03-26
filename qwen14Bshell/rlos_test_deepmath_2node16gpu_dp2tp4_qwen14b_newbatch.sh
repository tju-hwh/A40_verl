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
exp_name="${EXP_NAME:-deepmath-qwen14b-hop-dp2tp4-newbatch-2node16gpu}"

MODEL_PATH="${MODEL_PATH:-/root/model/Qwen-14B}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-/root/A40_verl/qwen14Bshell/qwen_chat_template.jinja}"
HF_MODULES_CACHE="${HF_MODULES_CACHE:-/root/.cache/huggingface/modules}"
TRAIN_FILE="${TRAIN_FILE:-/root/data/deepmath/train.parquet}"
TEST_FILE="${TEST_FILE:-/root/data/deepmath/test.parquet}"
CKPTS_DIR="${CKPTS_DIR:-/root/A40_verl/ckpts/${project_name}/${exp_name}}"

HEAD_NODE_IP="${HEAD_NODE_IP:-172.24.79.15}"
WORKER_NODE_IP="${WORKER_NODE_IP:-172.24.79.13}"
NNODES="${NNODES:-2}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
EXPECTED_NODES="${EXPECTED_NODES:-2}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
ETH_IFNAMES="${ETH_IFNAMES:-${ETH_MULTI_NIC_IFACES:-eth0,eth1,eth2,eth3}}"
PRIMARY_ETH_IFNAME="${PRIMARY_ETH_IFNAME:-${ETH_IFNAMES%%,*}}"

max_prompt_length="${MAX_PROMPT_LENGTH:-1024}"
max_response_length="${MAX_RESPONSE_LENGTH:-4096}"
rollout_max_model_len="${ROLLOUT_MAX_MODEL_LEN:-5120}"
rollout_tp="${ROLLOUT_TP:-4}"
rollout_gpu_mem_util="${ROLLOUT_GPU_MEM_UTIL:-0.24}"
rollout_temperature="${ROLLOUT_TEMPERATURE:-0.25}"
rollout_top_p="${ROLLOUT_TOP_P:-0.95}"
rollout_top_k="${ROLLOUT_TOP_K:-20}"
rollout_max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-1024}"
TRAIN_ATTN_IMPLEMENTATION="${TRAIN_ATTN_IMPLEMENTATION:-eager}"

train_prompt_bsz="${TRAIN_PROMPT_BSZ:-128}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-2}"
micro_batch_size="${MICRO_BATCH_SIZE:-8}"
mini_batch_size="${MINI_BATCH_SIZE:-128}"
actor_lr="${ACTOR_LR:-1e-6}"
total_training_steps="${TOTAL_TRAINING_STEPS:-5}"
total_epochs="${TOTAL_EPOCHS:-1}"

TRAIN_GROUP_PLAN="${TRAIN_GROUP_PLAN:-[16,16,16,16,16,16,16,16]}"
TRAIN_GROUP_PLAN_HEX="$(printf '%s' "${TRAIN_GROUP_PLAN}" | xxd -p -c 256)"
CHAT_TEMPLATE_JSON="$(python3 - <<PY
import json
from pathlib import Path
print(json.dumps(Path(r"${CHAT_TEMPLATE_FILE}").read_text()))
PY
)"

HOP_ROUTER_URLS="${HOP_ROUTER_URLS:-['http://${HEAD_NODE_IP}:8200','http://${WORKER_NODE_IP}:8200']}"
HOP_OWNER_STATE_URLS="${HOP_OWNER_STATE_URLS:-['http://${HEAD_NODE_IP}:8300','http://${WORKER_NODE_IP}:8300']}"
HOP_SERVER1_URLS="${HOP_SERVER1_URLS:-['http://${HEAD_NODE_IP}:8101','http://${HEAD_NODE_IP}:8103','http://${WORKER_NODE_IP}:8101','http://${WORKER_NODE_IP}:8103']}"
HOP_SERVER2_URLS="${HOP_SERVER2_URLS:-['http://${HEAD_NODE_IP}:8102','http://${HEAD_NODE_IP}:8104','http://${WORKER_NODE_IP}:8102','http://${WORKER_NODE_IP}:8104']}"
HOP_SERVER_URLS="${HOP_SERVER_URLS:-['http://${HEAD_NODE_IP}:8101','http://${HEAD_NODE_IP}:8102','http://${HEAD_NODE_IP}:8103','http://${HEAD_NODE_IP}:8104','http://${WORKER_NODE_IP}:8101','http://${WORKER_NODE_IP}:8102','http://${WORKER_NODE_IP}:8103','http://${WORKER_NODE_IP}:8104']}"

HOP_REQUEST_TIMEOUT_S="${HOP_REQUEST_TIMEOUT_S:-3600.0}"
HOP_CONNECT_TIMEOUT_S="${HOP_CONNECT_TIMEOUT_S:-60.0}"
HOP_STARTUP_TIMEOUT_S="${HOP_STARTUP_TIMEOUT_S:-900.0}"
HOP_HTTP_MAX_CONNECTIONS="${HOP_HTTP_MAX_CONNECTIONS:-256}"
HOP_HTTP_MAX_KEEPALIVE_CONNECTIONS="${HOP_HTTP_MAX_KEEPALIVE_CONNECTIONS:-256}"
HOP_SHARED_KV_POOL_META_PATH="${HOP_SHARED_KV_POOL_META_PATH:-/dev/shm/vllm_shared_kv_pool_qwen14b_newbatch}"
HOP_SEND_ACTIVATION_MARGIN_TOKENS="${HOP_SEND_ACTIVATION_MARGIN_TOKENS:-0}"
HOP_SEND_PUBLISH_TOKEN_STRIDE="${HOP_SEND_PUBLISH_TOKEN_STRIDE:-1536}"
HOP_DECODE_CUTOVERS="${HOP_DECODE_CUTOVERS:-[1024]}"
HOP_OWNER_FLUSH_EACH_LAYER="${HOP_OWNER_FLUSH_EACH_LAYER:-false}"
HOP_OWNER_GPU_MEM_UTIL="${HOP_OWNER_GPU_MEM_UTIL:-0.24}"
HOP_CONSUMER_GPU_MEM_UTIL="${HOP_CONSUMER_GPU_MEM_UTIL:-0.24}"
HOP_OWNER_MAX_NUM_SEQS="${HOP_OWNER_MAX_NUM_SEQS:-1024}"
HOP_CONSUMER_MAX_NUM_SEQS="${HOP_CONSUMER_MAX_NUM_SEQS:-1024}"
HOP_OWNER_TP_SIZE="${HOP_OWNER_TP_SIZE:-4}"
HOP_CONSUMER_TP_SIZE="${HOP_CONSUMER_TP_SIZE:-4}"
HOP_OWNER_DP_SIZE="${HOP_OWNER_DP_SIZE:-2}"
HOP_CONSUMER_DP_SIZE="${HOP_CONSUMER_DP_SIZE:-2}"
HOP_ENABLE_CUDA_MPS="${HOP_ENABLE_CUDA_MPS:-true}"
HOP_MPS_ACTIVE_THREAD_PERCENTAGES="${HOP_MPS_ACTIVE_THREAD_PERCENTAGES:-[100,40]}"
HOP_CONSUMER_ATTENTION_BACKEND="${HOP_CONSUMER_ATTENTION_BACKEND:-FLASH_ATTN}"
HOP_MACHINE_ROUTING_STRATEGY="${HOP_MACHINE_ROUTING_STRATEGY:-round_robin}"

ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE="${ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE:-100}"
REF_MPS_ACTIVE_THREAD_PERCENTAGE="${REF_MPS_ACTIVE_THREAD_PERCENTAGE:-100}"
CRITIC_MPS_ACTIVE_THREAD_PERCENTAGE="${CRITIC_MPS_ACTIVE_THREAD_PERCENTAGE:-100}"

VERL_HOP_CONFIG="{enabled:true,external_managed:true,router_urls:${HOP_ROUTER_URLS},owner_state_urls:${HOP_OWNER_STATE_URLS},server1_urls:${HOP_SERVER1_URLS},server2_urls:${HOP_SERVER2_URLS},server_urls:${HOP_SERVER_URLS},decode_cutovers:${HOP_DECODE_CUTOVERS},max_response_length:${max_response_length},request_timeout_s:${HOP_REQUEST_TIMEOUT_S},connect_timeout_s:${HOP_CONNECT_TIMEOUT_S},startup_timeout_s:${HOP_STARTUP_TIMEOUT_S},http_max_connections:${HOP_HTTP_MAX_CONNECTIONS},http_max_keepalive_connections:${HOP_HTTP_MAX_KEEPALIVE_CONNECTIONS},shared_kv_pool_meta_path:'${HOP_SHARED_KV_POOL_META_PATH}',send_activation_margin_tokens:${HOP_SEND_ACTIVATION_MARGIN_TOKENS},send_publish_token_stride:${HOP_SEND_PUBLISH_TOKEN_STRIDE},owner_flush_each_layer:${HOP_OWNER_FLUSH_EACH_LAYER},owner_gpu_memory_utilization:${HOP_OWNER_GPU_MEM_UTIL},consumer_gpu_memory_utilization:${HOP_CONSUMER_GPU_MEM_UTIL},owner_max_num_seqs:${HOP_OWNER_MAX_NUM_SEQS},consumer_max_num_seqs:${HOP_CONSUMER_MAX_NUM_SEQS},owner_tensor_parallel_size:${HOP_OWNER_TP_SIZE},consumer_tensor_parallel_size:${HOP_CONSUMER_TP_SIZE},owner_data_parallel_size:${HOP_OWNER_DP_SIZE},consumer_data_parallel_size:${HOP_CONSUMER_DP_SIZE},consumer_attention_backend:'${HOP_CONSUMER_ATTENTION_BACKEND}',enable_cuda_mps:${HOP_ENABLE_CUDA_MPS},mps_active_thread_percentages:${HOP_MPS_ACTIVE_THREAD_PERCENTAGES},machine_routing_strategy:'${HOP_MACHINE_ROUTING_STRATEGY}'}"

export NCCL_SOCKET_IFNAME="${ETH_IFNAMES}"
export GLOO_SOCKET_IFNAME="${PRIMARY_ETH_IFNAME}"
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_HIGH_PRIORITY=1
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HF_MODULES_CACHE

python3 - <<'PY'
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_cached_module_file
p="/root/model/Qwen-14B"
AutoTokenizer.from_pretrained(p, trust_remote_code=True)
AutoConfig.from_pretrained(p, trust_remote_code=True)
get_cached_module_file(p, "configuration_qwen.py")
get_cached_module_file(p, "modeling_qwen.py")
print("qwen14b dynamic modules warmed on trainer head")
PY

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

python3 -m recipe.one_step_off_policy.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=prompt \
  data.reward_fn_key=data_source \
  data.return_raw_chat=True \
  data.trust_remote_code="${TRUST_REMOTE_CODE}" \
  "+data.apply_chat_template_kwargs.chat_template=${CHAT_TEMPLATE_JSON}" \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.train_batch_size="${train_prompt_bsz}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code="${TRUST_REMOTE_CODE}" \
  "+actor_rollout_ref.model.custom_chat_template=${CHAT_TEMPLATE_JSON}" \
  "+actor_rollout_ref.model.override_config.attn_implementation=${TRAIN_ATTN_IMPLEMENTATION}" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  critic.strategy=fsdp2 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${actor_lr}" \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.actor.ppo_mini_batch_size="${mini_batch_size}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${micro_batch_size}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${micro_batch_size}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.temperature="${rollout_temperature}" \
  actor_rollout_ref.rollout.top_p="${rollout_top_p}" \
  actor_rollout_ref.rollout.top_k="${rollout_top_k}" \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${rollout_tp}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_mem_util}" \
  actor_rollout_ref.rollout.max_model_len="${rollout_max_model_len}" \
  actor_rollout_ref.rollout.max_num_seqs="${rollout_max_num_seqs}" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens="${rollout_max_model_len}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${micro_batch_size}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${rollout_temperature}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${rollout_top_p}" \
  actor_rollout_ref.rollout.val_kwargs.top_k="${rollout_top_k}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.verl_hop=${VERL_HOP_CONFIG}" \
  algorithm.use_kl_in_reward=False \
  reward_model.reward_manager=naive \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.save_freq=0 \
  trainer.test_freq=0 \
  trainer.total_epochs="${total_epochs}" \
  trainer.total_training_steps="${total_training_steps}" \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.resume_mode=disable \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.stream_train=True \
  "+trainer.stream_train_pipe=False" \
  "+ray_kwargs.ray_init.address=${RAY_ADDRESS}" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.TRAIN_GROUP_PLAN_HEX=${TRAIN_GROUP_PLAN_HEX}" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.HF_MODULES_CACHE=${HF_MODULES_CACHE}" \
  rollout.nnodes="${NNODES}" \
  rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
  "$@"
