#!/usr/bin/env bash
set -euo pipefail

cd /root/A40_verl

export mylog="${mylog:-1}"
rm -rf /root/1.log

project_name="${project_name:-GRPO}"
exp_name="${exp_name:-GRPO-Qwen3-8B-deepmath-hop-dp4tp4-2node16gpu}"

MODEL_PATH="${MODEL_PATH:-/root/model/Qwen3-8B}"
TRAIN_FILE="${TRAIN_FILE:-/root/data/deepmath/train.parquet}"
TEST_FILE="${TEST_FILE:-/root/data/deepmath/test.parquet}"

NNODES="${NNODES:-2}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
n_gpus_rollout="${n_gpus_rollout:-8}"
n_gpus_training="${n_gpus_training:-8}"

micro_batch_size="${micro_batch_size:-8}"
rollout_max_num_seqs="${rollout_max_num_seqs:-256}"
ppo_mini_batch_size="${ppo_mini_batch_size:-128}"
temperature="${temperature:-0.6}"
top_p="${top_p:-0.95}"
top_k="${top_k:-20}"

export TRAIN_GROUP_PLAN="${TRAIN_GROUP_PLAN:-[12,12,12,12,12,12,12,12,8,8,8,8]}"
export HOP_MPS_ACTIVE_THREAD_PERCENTAGES="${HOP_MPS_ACTIVE_THREAD_PERCENTAGES:-[100,60]}"
export ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE="${ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE:-70}"
export REF_MPS_ACTIVE_THREAD_PERCENTAGE="${REF_MPS_ACTIVE_THREAD_PERCENTAGE:-70}"
export CRITIC_MPS_ACTIVE_THREAD_PERCENTAGE="${CRITIC_MPS_ACTIVE_THREAD_PERCENTAGE:-70}"
export HOP_DECODE_CUTOVERS="${HOP_DECODE_CUTOVERS:-[1536]}"
export HOP_SEND_PUBLISH_TOKEN_STRIDE="${HOP_SEND_PUBLISH_TOKEN_STRIDE:-2048}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0,eth1,eth2,eth3}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0,eth1,eth2,eth3}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"

HEAD_NODE_IP="${HEAD_NODE_IP:?set HEAD_NODE_IP to machine-0 IP}"
WORKER_NODE_IP="${WORKER_NODE_IP:?set WORKER_NODE_IP to machine-1 IP}"
HOP_ROUTER_URL="${HOP_ROUTER_URL:-http://${HEAD_NODE_IP}:8200}"
HOP_OWNER_STATE_URL="${HOP_OWNER_STATE_URL:-http://${HEAD_NODE_IP}:8300}"

HOP_SERVER1_URLS="${HOP_SERVER1_URLS:-['http://${HEAD_NODE_IP}:8101','http://${HEAD_NODE_IP}:8103','http://${WORKER_NODE_IP}:8101','http://${WORKER_NODE_IP}:8103']}"
HOP_SERVER2_URLS="${HOP_SERVER2_URLS:-['http://${HEAD_NODE_IP}:8102','http://${HEAD_NODE_IP}:8104','http://${WORKER_NODE_IP}:8102','http://${WORKER_NODE_IP}:8104']}"
HOP_SERVER_URLS="${HOP_SERVER_URLS:-['http://${HEAD_NODE_IP}:8101','http://${HEAD_NODE_IP}:8102','http://${HEAD_NODE_IP}:8103','http://${HEAD_NODE_IP}:8104','http://${WORKER_NODE_IP}:8101','http://${WORKER_NODE_IP}:8102','http://${WORKER_NODE_IP}:8103','http://${WORKER_NODE_IP}:8104']}"

HOP_OWNER_GPU_MEM_UTIL="${HOP_OWNER_GPU_MEM_UTIL:-0.22}"
HOP_CONSUMER_GPU_MEM_UTIL="${HOP_CONSUMER_GPU_MEM_UTIL:-0.22}"
HOP_OWNER_MAX_NUM_SEQS="${HOP_OWNER_MAX_NUM_SEQS:-256}"
HOP_CONSUMER_MAX_NUM_SEQS="${HOP_CONSUMER_MAX_NUM_SEQS:-256}"
HOP_OWNER_TP_SIZE="${HOP_OWNER_TP_SIZE:-4}"
HOP_CONSUMER_TP_SIZE="${HOP_CONSUMER_TP_SIZE:-4}"
HOP_OWNER_DP_SIZE="${HOP_OWNER_DP_SIZE:-4}"
HOP_CONSUMER_DP_SIZE="${HOP_CONSUMER_DP_SIZE:-4}"
HOP_REQUEST_TIMEOUT_S="${HOP_REQUEST_TIMEOUT_S:-3600.0}"
HOP_CONNECT_TIMEOUT_S="${HOP_CONNECT_TIMEOUT_S:-60.0}"
HOP_STARTUP_TIMEOUT_S="${HOP_STARTUP_TIMEOUT_S:-900.0}"
HOP_HTTP_MAX_CONNECTIONS="${HOP_HTTP_MAX_CONNECTIONS:-512}"
HOP_HTTP_MAX_KEEPALIVE_CONNECTIONS="${HOP_HTTP_MAX_KEEPALIVE_CONNECTIONS:-512}"
HOP_SHARED_KV_POOL_META_PATH="${HOP_SHARED_KV_POOL_META_PATH:-/dev/shm/vllm_shared_kv_pool_dp4tp4}"
HOP_SEND_ACTIVATION_MARGIN_TOKENS="${HOP_SEND_ACTIVATION_MARGIN_TOKENS:-0}"
HOP_OWNER_FLUSH_EACH_LAYER="${HOP_OWNER_FLUSH_EACH_LAYER:-false}"
HOP_CONSUMER_ATTENTION_BACKEND="${HOP_CONSUMER_ATTENTION_BACKEND:-FLASH_ATTN}"
HOP_ENABLE_CUDA_MPS="${HOP_ENABLE_CUDA_MPS:-true}"
HOP_DP_ROUTING_STRATEGY="${HOP_DP_ROUTING_STRATEGY:-request_id_hash}"

OWNER_COMPILATION_CONFIG="${OWNER_COMPILATION_CONFIG:-{level:3,use_inductor:true,use_cudagraph:true}}"
SERVER2_COMPILATION_CONFIG="${SERVER2_COMPILATION_CONFIG:-{level:3,use_inductor:true,use_cudagraph:true}}"

if [[ "${ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE}" != "" ]]; then
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE}"
fi

VERL_HOP_CONFIG="{enabled:true,external_managed:true,router_url:'${HOP_ROUTER_URL}',router_host:'${HEAD_NODE_IP}',router_port:8200,owner_state_url:'${HOP_OWNER_STATE_URL}',server1_urls:${HOP_SERVER1_URLS},server2_urls:${HOP_SERVER2_URLS},server_urls:${HOP_SERVER_URLS},decode_cutovers:${HOP_DECODE_CUTOVERS},max_response_length:8192,request_timeout_s:${HOP_REQUEST_TIMEOUT_S},connect_timeout_s:${HOP_CONNECT_TIMEOUT_S},startup_timeout_s:${HOP_STARTUP_TIMEOUT_S},http_max_connections:${HOP_HTTP_MAX_CONNECTIONS},http_max_keepalive_connections:${HOP_HTTP_MAX_KEEPALIVE_CONNECTIONS},shared_kv_pool_meta_path:'${HOP_SHARED_KV_POOL_META_PATH}',send_activation_margin_tokens:${HOP_SEND_ACTIVATION_MARGIN_TOKENS},send_publish_token_stride:${HOP_SEND_PUBLISH_TOKEN_STRIDE},owner_flush_each_layer:${HOP_OWNER_FLUSH_EACH_LAYER},owner_gpu_memory_utilization:${HOP_OWNER_GPU_MEM_UTIL},consumer_gpu_memory_utilization:${HOP_CONSUMER_GPU_MEM_UTIL},owner_max_num_seqs:${HOP_OWNER_MAX_NUM_SEQS},consumer_max_num_seqs:${HOP_CONSUMER_MAX_NUM_SEQS},owner_tensor_parallel_size:${HOP_OWNER_TP_SIZE},consumer_tensor_parallel_size:${HOP_CONSUMER_TP_SIZE},owner_data_parallel_size:${HOP_OWNER_DP_SIZE},consumer_data_parallel_size:${HOP_CONSUMER_DP_SIZE},consumer_attention_backend:'${HOP_CONSUMER_ATTENTION_BACKEND}',enable_cuda_mps:${HOP_ENABLE_CUDA_MPS},mps_active_thread_percentages:${HOP_MPS_ACTIVE_THREAD_PERCENTAGES},owner_compilation_config:${OWNER_COMPILATION_CONFIG},server2_compilation_config:${SERVER2_COMPILATION_CONFIG},dp_routing_strategy:'${HOP_DP_ROUTING_STRATEGY}'}"

python3 -m recipe.one_step_off_policy.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.train_batch_size=128 \
  data.max_prompt_length=1024 \
  data.max_response_length=8192 \
  "+data.apply_chat_template_kwargs.enable_thinking=False" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.actor.strategy=fsdp2 \
  critic.strategy=fsdp2 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size} \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.temperature=${temperature} \
  actor_rollout_ref.rollout.top_p=${top_p} \
  actor_rollout_ref.rollout.top_k=${top_k} \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs} \
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.verl_hop=${VERL_HOP_CONFIG}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size} \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger="['console','tensorboard']" \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.save_freq=0 \
  trainer.test_freq=0 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=5 \
  trainer.nnodes="${NNODES}" \
  trainer.stream_train=True \
  "+trainer.stream_train_pipe=False" \
  ray_kwargs.ray_init.address="${RAY_ADDRESS}" \
  trainer.n_gpus_per_node="${n_gpus_training}" \
  rollout.nnodes="${NNODES}" \
  rollout.n_gpus_per_node="${n_gpus_rollout}" \
  "$@"
