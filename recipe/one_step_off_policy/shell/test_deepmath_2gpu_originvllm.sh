export mylog="1"

set -x

project_name='GRPO'
exp_name='GRPO-Qwen3-0.6b-deepmath-origin-vllm-4gpu'

PYTHON_BIN=${PYTHON_BIN:-/root/anaconda3/envs/verl/bin/python}

RAY_DATA_HOME="/root/A40_verl"
MODEL_PATH=${MODEL_PATH:-"/root/model/Qwen2-7B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"/root/data/deepmath/train.parquet"}
TEST_FILE=${TEST_FILE:-"/root/data/deepmath/test.parquet"}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
n_gpus_rollout=${n_gpus_rollout:-2}
n_gpus_training=${n_gpus_training:-2}

micro_batch_size=${micro_batch_size:-8}
ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.25}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-512}
ROLLOUT_N=${ROLLOUT_N:-4}
ACTOR_READONLY_IPC_ENABLED=${ACTOR_READONLY_IPC_ENABLED:-true}

TRAINING_ENABLE_CUDA_MPS=${TRAINING_ENABLE_CUDA_MPS:-true}
ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE=${ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE:-80}
REF_MPS_ACTIVE_THREAD_PERCENTAGE=${REF_MPS_ACTIVE_THREAD_PERCENTAGE:-80}
CRITIC_MPS_ACTIVE_THREAD_PERCENTAGE=${CRITIC_MPS_ACTIVE_THREAD_PERCENTAGE:-80}

# Training-side Ray workers inherit one process-level MPS setting.
if [ "${TRAINING_ENABLE_CUDA_MPS}" = "true" ] || [ "${TRAINING_ENABLE_CUDA_MPS}" = "1" ]; then
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE}"
fi

"${PYTHON_BIN}" -m recipe.one_step_off_policy.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=3 \
    trainer.nnodes="${NNODES}" \
    trainer.stream_train=True \
    "+trainer.actor_readonly_ipc_enabled=${ACTOR_READONLY_IPC_ENABLED}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" $@
