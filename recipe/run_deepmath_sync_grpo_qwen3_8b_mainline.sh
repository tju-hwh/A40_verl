#!/usr/bin/env bash
set -xeuo pipefail

source /root/anaconda3/etc/profile.d/conda.sh
conda activate verl

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

project_name=${PROJECT_NAME:-"GRPO"}
exp_name=${EXP_NAME:-"deepmath-qwen3-8b-mainline-sync-grpo"}

# Paths
MODEL_PATH=${MODEL_PATH:-"/root/model/Qwen3-8B"}
TRAIN_FILE=${TRAIN_FILE:-"/root/data/deepmath/train.parquet"}
TEST_FILE=${TEST_FILE:-"/root/data/deepmath/test.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"/root/A40_verl/ckpts/${project_name}/${exp_name}"}

# Cluster
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

# Match /root/model/test_qwen8b.py rollout behavior as closely as possible.
# test_qwen8b.py effective settings:
# - model=/root/model/Qwen3-8B
# - tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)
# - max_tokens=8192
# - temperature=0.6
# - top_p=0.95
# - top_k comes from generation_config.json => 20
# - tensor_parallel_size=4
# - gpu_memory_utilization=0.6
# - max_model_len=max(12288, prompt_len + 8192 + 256), which is 12288 for DeepMath
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-12288}
rollout_tp=${ROLLOUT_TP:-4}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_temperature=${ROLLOUT_TEMPERATURE:-0.6}
rollout_top_p=${ROLLOUT_TOP_P:-0.95}
rollout_top_k=${ROLLOUT_TOP_K:-20}

# Keep train_batch_size * rollout.n = 128 to match test_qwen8b.py max_num_seqs=128.
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-32}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-4}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-128}

# Actor / PPO
micro_batch_size=${MICRO_BATCH_SIZE:-2}
mini_batch_size=${MINI_BATCH_SIZE:-32}
actor_lr=${ACTOR_LR:-1e-6}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.reward_fn_key=data_source \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size} \
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
    trainer.total_epochs=1 \
    trainer.total_training_steps=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    "$@"
