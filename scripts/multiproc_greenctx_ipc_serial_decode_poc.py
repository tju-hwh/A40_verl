#!/usr/bin/env python3
"""Multi-process PoC for CUDA IPC + green context.

Design:
- Parent process allocates owner parameter W on GPU and exports CUDA IPC handle.
- Spawn 4 worker processes.
- Each worker creates one green context (100/80/60/40), opens IPC handle,
  and runs one decode-like request.
- Parent dispatches workers serially (request1->worker1->...->worker4).
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

import cupy as cp
import numpy as np
import torch

from verl.workers.rollout.vllm_rollout.vllm_async_server import CudaGreenContextManager


class CudaIpcDriver:
    _CUDA_SUCCESS = 0
    _CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1

    class CUipcMemHandle(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_char * 64)]

    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._setup_signatures()
        self._check(self.lib.cuInit(0), "cuInit")

    def _setup_signatures(self):
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuInit.restype = ctypes.c_int
        self.lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.lib.cuCtxSetCurrent.restype = ctypes.c_int
        self.lib.cuIpcGetMemHandle.argtypes = [ctypes.POINTER(self.CUipcMemHandle), ctypes.c_uint64]
        self.lib.cuIpcGetMemHandle.restype = ctypes.c_int
        self.lib.cuIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            self.CUipcMemHandle,
            ctypes.c_uint,
        ]
        self.lib.cuIpcOpenMemHandle.restype = ctypes.c_int
        self.lib.cuIpcCloseMemHandle.argtypes = [ctypes.c_uint64]
        self.lib.cuIpcCloseMemHandle.restype = ctypes.c_int
        self.lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.cuGetErrorName.restype = ctypes.c_int
        self.lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.cuGetErrorString.restype = ctypes.c_int

    def _check(self, code: int, where: str):
        if code == self._CUDA_SUCCESS:
            return
        name_ptr = ctypes.c_char_p()
        msg_ptr = ctypes.c_char_p()
        self.lib.cuGetErrorName(code, ctypes.byref(name_ptr))
        self.lib.cuGetErrorString(code, ctypes.byref(msg_ptr))
        name = name_ptr.value.decode() if name_ptr.value else f"CUDA_ERROR_{code}"
        msg = msg_ptr.value.decode() if msg_ptr.value else "Unknown CUDA error"
        raise RuntimeError(f"{where} failed: {name} ({msg})")

    def set_current(self, cu_ctx: int):
        self._check(self.lib.cuCtxSetCurrent(ctypes.c_void_p(cu_ctx)), "cuCtxSetCurrent")

    def export_handle_bytes(self, ptr: int) -> bytes:
        h = self.CUipcMemHandle()
        self._check(self.lib.cuIpcGetMemHandle(ctypes.byref(h), ctypes.c_uint64(ptr)), "cuIpcGetMemHandle")
        return bytes(h.reserved)

    def open_handle_bytes(self, handle_bytes: bytes) -> int:
        h = self.CUipcMemHandle()
        if len(handle_bytes) != 64:
            raise ValueError("Invalid CUipcMemHandle bytes")
        ctypes.memmove(h.reserved, handle_bytes, 64)
        opened_ptr = ctypes.c_uint64()
        self._check(
            self.lib.cuIpcOpenMemHandle(
                ctypes.byref(opened_ptr),
                h,
                ctypes.c_uint(self._CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
            ),
            "cuIpcOpenMemHandle",
        )
        return int(opened_ptr.value)

    def close_opened(self, ptr: int):
        self._check(self.lib.cuIpcCloseMemHandle(ctypes.c_uint64(ptr)), "cuIpcCloseMemHandle")


@dataclass
class WorkerConfig:
    rank: int
    sm_limit: int
    hidden_size: int
    decode_steps: int
    dtype: str
    nbytes: int


def _cupy_dtype(dtype: str):
    return cp.float16 if dtype == "float16" else cp.float32


def worker_main(cfg: WorkerConfig, handle_bytes: bytes, task_q: mp.Queue, result_q: mp.Queue):
    try:
        ipc = CudaIpcDriver()
        g = CudaGreenContextManager([cfg.sm_limit], cuda_device=0)
        meta = g.create_all()[0]
        ipc.set_current(int(meta["cu_ctx"]))

        opened_ptr = ipc.open_handle_bytes(handle_bytes)
        unowned = cp.cuda.UnownedMemory(opened_ptr, cfg.nbytes, owner=None)
        memptr = cp.cuda.MemoryPointer(unowned, 0)
        w = cp.ndarray((cfg.hidden_size, cfg.hidden_size), dtype=_cupy_dtype(cfg.dtype), memptr=memptr)

        while True:
            item = task_q.get()
            if item is None:
                break
            req_id = int(item)
            x = cp.random.standard_normal((cfg.hidden_size,), dtype=_cupy_dtype(cfg.dtype))
            t0 = time.time()
            for _ in range(cfg.decode_steps):
                x = cp.tanh(w.dot(x))
            cp.cuda.runtime.deviceSynchronize()
            elapsed_ms = (time.time() - t0) * 1000.0
            checksum = float(cp.sum(x).get())
            result_q.put(
                {
                    "worker_rank": cfg.rank,
                    "req_id": req_id,
                    "sm_limit": cfg.sm_limit,
                    "sm_count": int(meta["sm_count"]),
                    "elapsed_ms": elapsed_ms,
                    "checksum": checksum,
                }
            )
        ipc.close_opened(opened_ptr)
    except Exception as e:
        result_q.put({"worker_rank": cfg.rank, "error": str(e)})


def parse_args():
    p = argparse.ArgumentParser(description="Multiprocess green context + IPC serial decode PoC")
    p.add_argument("--sm-limits", type=str, default="100,80,60,40")
    p.add_argument("--hidden-size", type=int, default=1024)
    p.add_argument("--decode-steps", type=int, default=8)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    mp.set_start_method("spawn", force=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sm_limits = [int(x.strip()) for x in args.sm_limits.split(",") if x.strip()]
    if len(sm_limits) != 4:
        raise ValueError("sm-limits must contain exactly 4 values")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    # Owner param in parent process.
    torch.cuda.set_device(0)
    dtype_torch = torch.float16 if args.dtype == "float16" else torch.float32
    w_owner = torch.randn(args.hidden_size, args.hidden_size, device="cuda", dtype=dtype_torch)
    owner_ptr = int(w_owner.data_ptr())
    nbytes = int(w_owner.numel() * w_owner.element_size())
    ipc_parent = CudaIpcDriver()
    handle_bytes = ipc_parent.export_handle_bytes(owner_ptr)

    task_queues: list[mp.Queue] = []
    result_q: mp.Queue = mp.Queue()
    procs: list[mp.Process] = []

    for i, sm in enumerate(sm_limits):
        q: mp.Queue = mp.Queue()
        cfg = WorkerConfig(
            rank=i + 1,
            sm_limit=sm,
            hidden_size=args.hidden_size,
            decode_steps=args.decode_steps,
            dtype=args.dtype,
            nbytes=nbytes,
        )
        p = mp.Process(target=worker_main, args=(cfg, handle_bytes, q, result_q), daemon=False)
        p.start()
        task_queues.append(q)
        procs.append(p)

    print("[serial_decode_requests]")
    for req_id in range(1, 5):
        worker_idx = req_id - 1
        task_queues[worker_idx].put(req_id)
        out = result_q.get(timeout=600)
        if "error" in out:
            raise RuntimeError(f"worker-{out['worker_rank']} failed: {out['error']}")
        print(
            f"req={out['req_id']} worker={out['worker_rank']} sm_limit={out['sm_limit']} "
            f"sm_count={out['sm_count']} elapsed_ms={out['elapsed_ms']:.2f} checksum={out['checksum']:.6f}"
        )

    for q in task_queues:
        q.put(None)
    for p in procs:
        p.join(timeout=30)
        if p.exitcode != 0:
            raise RuntimeError(f"worker pid={p.pid} exited with code {p.exitcode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
