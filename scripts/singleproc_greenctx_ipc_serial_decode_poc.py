#!/usr/bin/env python3
"""Single-process PoC:
1) Create 4 CUDA green contexts (100/80/60/40 by default).
2) Export one owner parameter tensor via CUDA IPC handle.
3) Open that handle in each green context.
4) Run 4 serial "decode-like" requests, each request on one context.

This validates green-context round-robin + IPC shared parameter memory path.
"""

import argparse
import ctypes
import json
import os
import time

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

        self.lib.cuIpcGetMemHandle.argtypes = [
            ctypes.POINTER(self.CUipcMemHandle),
            ctypes.c_uint64,
        ]
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

    def export_mem_handle(self, ptr: int) -> "CudaIpcDriver.CUipcMemHandle":
        h = self.CUipcMemHandle()
        self._check(self.lib.cuIpcGetMemHandle(ctypes.byref(h), ctypes.c_uint64(ptr)), "cuIpcGetMemHandle")
        return h

    def open_mem_handle(self, handle: "CudaIpcDriver.CUipcMemHandle") -> int:
        opened_ptr = ctypes.c_uint64()
        self._check(
            self.lib.cuIpcOpenMemHandle(
                ctypes.byref(opened_ptr),
                handle,
                ctypes.c_uint(self._CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
            ),
            "cuIpcOpenMemHandle",
        )
        return int(opened_ptr.value)

    def close_mem_handle(self, ptr: int):
        self._check(self.lib.cuIpcCloseMemHandle(ctypes.c_uint64(ptr)), "cuIpcCloseMemHandle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-process green-context + IPC serial decode PoC")
    p.add_argument("--sm-limits", type=str, default="100,80,60,40")
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--decode-steps", type=int, default=16)
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _torch_dtype(dtype: str):
    return torch.float16 if dtype == "float16" else torch.float32


def _cupy_dtype(dtype: str):
    return cp.float16 if dtype == "float16" else cp.float32


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sm_limits = [int(x.strip()) for x in args.sm_limits.split(",") if x.strip()]
    if len(sm_limits) != 4:
        raise ValueError("Expected exactly 4 SM limits, e.g. 100,80,60,40")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    # Owner allocation in primary context.
    torch.cuda.set_device(0)
    w_torch = torch.randn(
        args.hidden_size,
        args.hidden_size,
        device="cuda",
        dtype=_torch_dtype(args.dtype),
    )
    owner_ptr = int(w_torch.data_ptr())
    nbytes = int(w_torch.numel() * w_torch.element_size())

    green_mgr = CudaGreenContextManager(sm_limits_pct=sm_limits, cuda_device=0)
    green_meta = green_mgr.create_all()
    print("[green_contexts]")
    print(json.dumps(green_meta, indent=2))

    ipc = CudaIpcDriver()
    handle = ipc.export_mem_handle(owner_ptr)

    # Open the same owner parameter memory in each green context.
    opened_ptrs: list[int] = []
    w_views: list[cp.ndarray] = []
    for meta in green_meta:
        cu_ctx = int(meta["cu_ctx"])
        ipc.set_current(cu_ctx)
        opened_ptr = ipc.open_mem_handle(handle)
        opened_ptrs.append(opened_ptr)

        unowned = cp.cuda.UnownedMemory(opened_ptr, nbytes, owner=None)
        memptr = cp.cuda.MemoryPointer(unowned, 0)
        arr = cp.ndarray(
            (args.hidden_size, args.hidden_size),
            dtype=_cupy_dtype(args.dtype),
            memptr=memptr,
        )
        w_views.append(arr)

    # 4 serial requests, one request per green context in order.
    print("[serial_decode_requests]")
    for req_id in range(4):
        meta = green_meta[req_id]
        cu_ctx = int(meta["cu_ctx"])
        ipc.set_current(cu_ctx)

        x = cp.random.standard_normal((args.hidden_size,), dtype=_cupy_dtype(args.dtype))
        t0 = time.time()
        for _ in range(args.decode_steps):
            # decode-like step: y = W*x, then nonlinearity
            x = cp.tanh(w_views[req_id].dot(x))
        cp.cuda.runtime.deviceSynchronize()
        ms = (time.time() - t0) * 1000.0
        checksum = float(cp.sum(x).get())
        print(
            f"req={req_id+1} sm_limit={meta['sm_limit_pct']} sm_count={meta['sm_count']} "
            f"elapsed_ms={ms:.2f} checksum={checksum:.6f}"
        )

    # Cleanup mapped pointers.
    for meta, opened_ptr in zip(green_meta, opened_ptrs):
        ipc.set_current(int(meta["cu_ctx"]))
        ipc.close_mem_handle(opened_ptr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
