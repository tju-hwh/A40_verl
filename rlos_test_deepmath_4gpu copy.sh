export mylog="1"

set -x

project_name='GRPO'
exp_name='GRPO-Qwen3-8B-deepmath-2server-hop-tp4'

RAY_DATA_HOME="/root/A40_verl"
MODEL_PATH=${MODEL_PATH:-"/root/model/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/root/data/deepmath/train.parquet"}
TEST_FILE=${TEST_FILE:-"/root/data/deepmath/test.parquet"}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
n_gpus_rollout=${n_gpus_rollout:-4}
n_gpus_training=${n_gpus_training:-4}

micro_batch_size=${micro_batch_size:-8}
temperature=${temperature:-0.6}
top_p=${top_p:-0.95}
top_k=${top_k:-20}

ENABLE_VERL_HOP=${ENABLE_VERL_HOP:-"true"}
HOP_HOST=${HOP_HOST:-"127.0.0.1"}
HOP_ROUTER_PORT=${HOP_ROUTER_PORT:-8200}
HOP_OWNER_STATE_PORT=${HOP_OWNER_STATE_PORT:-8300}
HOP_SERVER_PORTS=${HOP_SERVER_PORTS:-"[8101,8102]"}
HOP_SERVER_KV_PORTS=${HOP_SERVER_KV_PORTS:-"[18101,18102]"}
HOP_DECODE_CUTOVERS=${HOP_DECODE_CUTOVERS:-"[1024]"}
HOP_MAX_RESPONSE_LENGTH=${HOP_MAX_RESPONSE_LENGTH:-4096}

HOP_REQUEST_TIMEOUT_S=${HOP_REQUEST_TIMEOUT_S:-3600.0}
HOP_CONNECT_TIMEOUT_S=${HOP_CONNECT_TIMEOUT_S:-60.0}
HOP_STARTUP_TIMEOUT_S=${HOP_STARTUP_TIMEOUT_S:-900.0}

HOP_HTTP_MAX_CONNECTIONS=${HOP_HTTP_MAX_CONNECTIONS:-256}
HOP_HTTP_MAX_KEEPALIVE_CONNECTIONS=${HOP_HTTP_MAX_KEEPALIVE_CONNECTIONS:-256}
HOP_SHARED_KV_POOL_META_PATH=${HOP_SHARED_KV_POOL_META_PATH:-"/tmp/vllm_shared_kv_pool_from_verl.pkl"}
HOP_SEND_ACTIVATION_MARGIN_TOKENS=${HOP_SEND_ACTIVATION_MARGIN_TOKENS:-0}
HOP_SEND_PUBLISH_TOKEN_STRIDE=${HOP_SEND_PUBLISH_TOKEN_STRIDE:-1024}
HOP_OWNER_FLUSH_EACH_LAYER=${HOP_OWNER_FLUSH_EACH_LAYER:-false}
KV_HANDOFF_WAIT_TIMEOUT_S=${KV_HANDOFF_WAIT_TIMEOUT_S:-120}
KV_HANDOFF_STABLE_POLLS=${KV_HANDOFF_STABLE_POLLS:-1}

HOP_OWNER_GPU_MEM_UTIL=${HOP_OWNER_GPU_MEM_UTIL:-0.22}
HOP_CONSUMER_GPU_MEM_UTIL=${HOP_CONSUMER_GPU_MEM_UTIL:-0.22}
HOP_OWNER_MAX_NUM_SEQS=${HOP_OWNER_MAX_NUM_SEQS:-256}
HOP_CONSUMER_MAX_NUM_SEQS=${HOP_CONSUMER_MAX_NUM_SEQS:-256}
HOP_OWNER_TP_SIZE=${HOP_OWNER_TP_SIZE:-4}
HOP_CONSUMER_TP_SIZE=${HOP_CONSUMER_TP_SIZE:-4}
HOP_OWNER_CUDA_VISIBLE_DEVICES=${HOP_OWNER_CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
HOP_CONSUMER_CUDA_VISIBLE_DEVICES_ALL=${HOP_CONSUMER_CUDA_VISIBLE_DEVICES_ALL:-"0,1,2,3"}
HOP_NO_CONSUMER_ENFORCE_EAGER=${HOP_NO_CONSUMER_ENFORCE_EAGER:-true}
HOP_CONSUMER_ATTENTION_BACKEND=${HOP_CONSUMER_ATTENTION_BACKEND:-"FLASH_ATTN"}

HOP_ENABLE_CUDA_MPS=${HOP_ENABLE_CUDA_MPS:-true}
HOP_MPS_ACTIVE_THREAD_PERCENTAGES=${HOP_MPS_ACTIVE_THREAD_PERCENTAGES:-"[100,100]"}

TRAINING_ENABLE_CUDA_MPS=${TRAINING_ENABLE_CUDA_MPS:-true}
ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE=${ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE:-100}
REF_MPS_ACTIVE_THREAD_PERCENTAGE=${REF_MPS_ACTIVE_THREAD_PERCENTAGE:-100}
CRITIC_MPS_ACTIVE_THREAD_PERCENTAGE=${CRITIC_MPS_ACTIVE_THREAD_PERCENTAGE:-100}

# Training-side Ray workers inherit a single process-level MPS percentage.
# Keep role-specific knobs visible in the script, but they currently collapse
# to one shared value unless the worker launch path is split by role.
if [ "${TRAINING_ENABLE_CUDA_MPS}" = "true" ] || [ "${TRAINING_ENABLE_CUDA_MPS}" = "1" ]; then
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE}"
fi

# [1,2,4,8,16,24,32,40,48,56,64,72,64,72,80,88,96,104,112,120,128]
# [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256]
# OWNER_COMPILATION_CONFIG=${OWNER_COMPILATION_CONFIG:-'{level:3,use_inductor:true,use_cudagraph:true,cudagraph_capture_sizes:[120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256]}'}
# SERVER2_COMPILATION_CONFIG=${SERVER2_COMPILATION_CONFIG:-'{level:3,use_inductor:true,use_cudagraph:true,cudagraph_capture_sizes:[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160]}'}
OWNER_COMPILATION_CONFIG=${OWNER_COMPILATION_CONFIG:-'{level:3,use_inductor:true,use_cudagraph:true}'}
SERVER2_COMPILATION_CONFIG=${SERVER2_COMPILATION_CONFIG:-'{level:3,use_inductor:true,use_cudagraph:true}'}
SERVER3_COMPILATION_CONFIG=${SERVER3_COMPILATION_CONFIG:-'{level:0,use_inductor:false,use_cudagraph:false}'}
SERVER4_COMPILATION_CONFIG=${SERVER4_COMPILATION_CONFIG:-'{level:0,use_inductor:false,use_cudagraph:false}'}


VERL_HOP_CONFIG="{enabled:${ENABLE_VERL_HOP},host:'${HOP_HOST}',router_port:${HOP_ROUTER_PORT},owner_state_port:${HOP_OWNER_STATE_PORT},server_ports:${HOP_SERVER_PORTS},server_kv_ports:${HOP_SERVER_KV_PORTS},decode_cutovers:${HOP_DECODE_CUTOVERS},max_response_length:${HOP_MAX_RESPONSE_LENGTH},request_timeout_s:${HOP_REQUEST_TIMEOUT_S},connect_timeout_s:${HOP_CONNECT_TIMEOUT_S},startup_timeout_s:${HOP_STARTUP_TIMEOUT_S},http_max_connections:${HOP_HTTP_MAX_CONNECTIONS},http_max_keepalive_connections:${HOP_HTTP_MAX_KEEPALIVE_CONNECTIONS},shared_kv_pool_meta_path:'${HOP_SHARED_KV_POOL_META_PATH}',send_activation_margin_tokens:${HOP_SEND_ACTIVATION_MARGIN_TOKENS},send_publish_token_stride:${HOP_SEND_PUBLISH_TOKEN_STRIDE},owner_flush_each_layer:${HOP_OWNER_FLUSH_EACH_LAYER},owner_gpu_memory_utilization:${HOP_OWNER_GPU_MEM_UTIL},consumer_gpu_memory_utilization:${HOP_CONSUMER_GPU_MEM_UTIL},owner_max_num_seqs:${HOP_OWNER_MAX_NUM_SEQS},consumer_max_num_seqs:${HOP_CONSUMER_MAX_NUM_SEQS},owner_tensor_parallel_size:${HOP_OWNER_TP_SIZE},consumer_tensor_parallel_size:${HOP_CONSUMER_TP_SIZE},owner_cuda_visible_devices:'${HOP_OWNER_CUDA_VISIBLE_DEVICES}',consumer_cuda_visible_devices_all:'${HOP_CONSUMER_CUDA_VISIBLE_DEVICES_ALL}',no_consumer_enforce_eager:${HOP_NO_CONSUMER_ENFORCE_EAGER},consumer_attention_backend:'${HOP_CONSUMER_ATTENTION_BACKEND}',enable_cuda_mps:${HOP_ENABLE_CUDA_MPS},mps_active_thread_percentages:${HOP_MPS_ACTIVE_THREAD_PERCENTAGES},owner_compilation_config:${OWNER_COMPILATION_CONFIG},server2_compilation_config:${SERVER2_COMPILATION_CONFIG},server3_compilation_config:${SERVER3_COMPILATION_CONFIG},server4_compilation_config:${SERVER4_COMPILATION_CONFIG}}"

KV_HANDOFF_WAIT_TIMEOUT_S="${KV_HANDOFF_WAIT_TIMEOUT_S}" \
KV_HANDOFF_STABLE_POLLS="${KV_HANDOFF_STABLE_POLLS}" \
python3 -m recipe.one_step_off_policy.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    "+data.apply_chat_template_kwargs.enable_thinking=False" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
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
    actor_rollout_ref.rollout.max_num_seqs=256 \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.verl_hop=${VERL_HOP_CONFIG}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=6 \
    trainer.nnodes="${NNODES}" \
    trainer.stream_train=True \
    "+trainer.stream_train_pipe=False" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" $@ 
