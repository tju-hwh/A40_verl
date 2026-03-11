import asyncio
import json
import os
import socket
import time
import uuid
from pprint import pprint

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from verl import DataProto
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.fs import copy_to_local
from verl.utils.torch_functional import pad_2d_list_to_length
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


def _get_gen_batch(batch: DataProto) -> DataProto:
    reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
    gen_batch = batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
    )
    gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
    return gen_batch


def _strip_pad(token_ids, pad_token_id):
    ids = np.asarray(token_ids).tolist()
    if pad_token_id is None:
        return ids
    return [int(tok) for tok in ids if int(tok) != int(pad_token_id)]


def _dump_rollout_samples_from_ids(prompt_ids_list, response_ids_list, tokenizer, dump_path: str) -> None:
    pad_token_id = tokenizer.pad_token_id
    records = []
    for idx, (prompt_ids_raw, response_ids_raw) in enumerate(zip(prompt_ids_list, response_ids_list, strict=True)):
        prompt_ids = _strip_pad(prompt_ids_raw, pad_token_id)
        response_ids = _strip_pad(response_ids_raw, pad_token_id)
        records.append(
            {
                "sample_index": idx,
                "decode_tokens": len(response_ids),
                "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                "response": tokenizer.decode(response_ids, skip_special_tokens=True),
            }
        )
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


async def _generate_one(server, prompt_ids, sampling_params, request_id):
    t0 = time.perf_counter()
    output = await server.generate.remote(
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        request_id=request_id,
        image_data=None,
    )
    elapsed = time.perf_counter() - t0
    return output, elapsed


async def _run_rollout(config) -> dict:
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    train_dataset = create_rl_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        processor,
        max_samples=config.data.get("train_max_samples", -1),
    )
    train_sampler = create_rl_sampler(config.data, train_dataset)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.data.train_batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    batch_dict = next(iter(train_dataloader))
    batch = DataProto.from_single_dict(batch_dict)
    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

    gen_batch = _get_gen_batch(batch)
    gen_batch.meta_info["global_steps"] = 1
    gen_batch = gen_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)

    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model
    server = vLLMHttpServer.options(name=f"rollout_only_server_{uuid.uuid4().hex}").remote(
        config=rollout_config,
        model_config=model_config,
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=config.trainer.n_gpus_per_node,
        nnodes=1,
    )
    await server.launch_server.remote()
    await server.clear_kv_cache.remote()

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_ids_list = [
        _strip_pad(prompt_ids, pad_token_id)
        for prompt_ids in gen_batch.batch["input_ids"].cpu().numpy()
    ]
    sampling_params = {
        "max_tokens": int(config.data.max_response_length),
        "temperature": float(rollout_config.temperature),
        "top_p": float(rollout_config.top_p),
        "top_k": int(rollout_config.top_k),
        "stop": [],
        "logprobs": False,
    }

    request_ids = [str(uuid.uuid4()) for _ in prompt_ids_list]
    t0 = time.perf_counter()
    outputs = await asyncio.gather(
        *[
            _generate_one(server, prompt_ids, sampling_params, request_id)
            for prompt_ids, request_id in zip(prompt_ids_list, request_ids, strict=True)
        ]
    )
    elapsed = time.perf_counter() - t0

    token_outputs = [x[0] for x in outputs]
    latencies = [float(x[1]) for x in outputs]
    response_ids = [out.token_ids for out in token_outputs]
    responses = pad_2d_list_to_length(
        response_ids,
        pad_token_id,
        max_length=int(config.data.max_response_length),
    ).to(dtype=torch.long)

    batch = batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
    batch.batch["responses"] = responses
    response_len = batch.batch["responses"].shape[-1] if "responses" in batch.batch else -1
    dump_path = f"/tmp/rollout_only_samples_{int(time.time())}.json"
    _dump_rollout_samples_from_ids(prompt_ids_list, response_ids, tokenizer, dump_path)

    await server.clear_kv_cache.remote()
    await server.sleep.remote()
    ray.kill(server, no_restart=True)

    slowest_idx = int(np.argmax(latencies)) if latencies else -1
    metrics = {
        "agent_loop/generate_sequences/min": float(min(latencies)) if latencies else 0.0,
        "agent_loop/generate_sequences/max": float(max(latencies)) if latencies else 0.0,
        "agent_loop/generate_sequences/mean": float(np.mean(latencies)) if latencies else 0.0,
        "agent_loop/tool_calls/min": 0.0,
        "agent_loop/tool_calls/max": 0.0,
        "agent_loop/tool_calls/mean": 0.0,
        "agent_loop/slowest/generate_sequences": float(max(latencies)) if latencies else 0.0,
        "agent_loop/slowest/tool_calls": 0.0,
        "agent_loop/slowest/prompt_length": int(len(prompt_ids_list[slowest_idx])) if slowest_idx >= 0 else 0,
        "agent_loop/slowest/response_length": int(len(response_ids[slowest_idx])) if slowest_idx >= 0 else 0,
    }

    return {
        "epoch": 0,
        "wall_time_s": float(elapsed),
        "prompt_count": int(len(prompt_ids_list) // config.actor_rollout_ref.rollout.n),
        "sample_count": int(len(prompt_ids_list)),
        "rollout_n": int(config.actor_rollout_ref.rollout.n),
        "response_tensor_len": int(response_len),
        "timing_raw": {"generate_async": float(elapsed)},
        "metrics": metrics,
        "samples_dump_path": dump_path,
    }


@hydra.main(config_path="config", config_name="one_step_off_ppo_trainer", version_base=None)
def main(config):
    print(f"rollout-only hostname: {socket.gethostname()}, pid: {os.getpid()}", flush=True)
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    ray_kwargs = {
        "num_cpus": None,
        "runtime_env": {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_CUMEM_ENABLE": "0",
            },
            "working_dir": None,
        },
    }
    print(f"ray init kwargs: {ray_kwargs}", flush=True)
    if not ray.is_initialized():
        ray.init(**ray_kwargs)

    result = asyncio.run(_run_rollout(config))
    print("[rollout_only_result]", result, flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
