#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify vLLM inference with CUDA green context setup.")
    parser.add_argument("--model", type=str, default="/root/model/Qwen2-7B-Instruct")
    parser.add_argument("--prompt", type=str, default="Please reply with exactly: OK")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--sm-limits", type=str, default="100,80,60,40")
    parser.add_argument("--disable-green-context", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Keep parity with rollout server defaults where possible.
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
    # Avoid CUDA re-init crash in EngineCore subprocess.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method may already be set by the launcher.
        pass

    green_mgr = None
    try:
        if not args.disable_green_context:
            from verl.workers.rollout.vllm_rollout.vllm_async_server import CudaGreenContextManager

            sm_limits = [int(x.strip()) for x in args.sm_limits.split(",") if x.strip()]
            green_mgr = CudaGreenContextManager(sm_limits_pct=sm_limits, cuda_device=0)
            metas = green_mgr.create_all()
            print("[green_contexts]")
            print(json.dumps(metas, indent=2))

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tp_size,
            trust_remote_code=True,
            dtype=args.dtype,
            enforce_eager=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            disable_custom_all_reduce=True,
            enable_prefix_caching=True,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_tokens,
        )
        outputs = llm.generate([args.prompt], sampling_params=sampling_params, use_tqdm=False)
        out = outputs[0]
        text = out.outputs[0].text
        token_ids = out.outputs[0].token_ids
        print("[inference]")
        print(f"prompt={args.prompt!r}")
        print(f"text={text!r}")
        print(f"token_ids={token_ids}")
        return 0
    except Exception as exc:
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if green_mgr is not None:
            try:
                green_mgr.destroy_all()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
