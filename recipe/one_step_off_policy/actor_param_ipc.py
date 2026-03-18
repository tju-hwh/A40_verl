import cloudpickle
import torch
from torch.multiprocessing.reductions import reduce_tensor


def export_cuda_tensor_ipc_blob(tensor: torch.Tensor) -> bytes:
    if not tensor.is_cuda:
        raise ValueError("Only CUDA tensors support IPC export")
    tensor = tensor.detach()
    reduced = reduce_tensor(tensor)
    return cloudpickle.dumps(reduced)


def import_cuda_tensor_ipc_blob(blob: bytes) -> torch.Tensor:
    rebuild_fn, rebuild_args = cloudpickle.loads(blob)
    return rebuild_fn(*rebuild_args)

