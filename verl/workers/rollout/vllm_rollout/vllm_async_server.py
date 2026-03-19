# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pprint import pprint
from typing import Any, Callable, Optional

import cloudpickle as pickle
import httpx
import numpy as np
import ray
import vllm.entrypoints.cli.serve
import zmq
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_app,
    init_app_state,
)
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, get_tcp_uri
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngineProcManager
from vllm.v1.executor.abstract import Executor

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches
from verl.workers.config import HFModelConfig, RewardModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address, run_unvicorn
from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class ExternalZeroMQDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        dp_rank_local = self.vllm_config.parallel_config.data_parallel_rank_local
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size

        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        addresses = addresses[dp_rank_local * tp_size : (dp_rank_local + 1) * tp_size]
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            if address.startswith("tcp://["):
                socket.setsockopt(zmq.IPV6, 1)
            socket.connect(address)
            self.sockets.append(socket)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
        **kwargs_extra: Any,
    ) -> list[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        outputs = []
        for socket in self.sockets:
            outputs.append(pickle.loads(socket.recv()))

        for output in outputs:
            if isinstance(output, Exception):
                raise output
        return outputs

    def check_health(self):
        return


class vLLMHttpServerBase:
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        """
        Args:
            config (RolloutConfig): full config.
            model_config (HFModelConfig): model config.
            rollout_mode (RolloutMode): rollout mode.
            replica_rank (int): replica rank, a replica may contain multiple nodes.
            node_rank (int): node rank.
            gpus_per_node (int): number of gpus per node.
            nnodes (int): number of nodes.
        """
        super().__init__()

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.config.max_model_len = self.config.prompt_length + self.config.response_length
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address)
            logger.info(
                f"vLLMHttpServer, replica_rank: {self.replica_rank}, master address: {self._master_address}, "
                f"master port: {self._master_port}, data parallel master port: {self._dp_master_port}"
            )
        else:
            self._master_address = None
            self._master_port = None
        self._hop_cfg: dict[str, Any] = {}
        self._hop_enabled: bool = False
        self._hop_router_url: str = ""
        self._hop_http_client: Optional[httpx.AsyncClient] = None
        self._hop_processes: list[subprocess.Popen] = []

    def get_master_address(self):
        """Get master address and port for data parallel."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        hop_cfg = dict(engine_kwargs.pop("verl_hop", {}) or {})
        self._hop_cfg = hop_cfg
        self._hop_enabled = bool(hop_cfg.get("enabled", False))
        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes
        if self._hop_enabled:
            await self._launch_hop_cluster()
            return

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        override_generation_config = dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self.config.response_length,
        )
        logger.info(f"override_generation_config: {override_generation_config}")
        quantization = self.config.quantization
        if quantization is not None:
            if quantization == "fp8":
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
                # Apply vllm fp8 patches
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
                apply_vllm_fp8_patches()
            else:
                raise ValueError(f"Currently only support fp8 quantization, got: {quantization}")
        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": True,
            "disable_custom_all_reduce": True,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": {"quantization_config": fp8_block_quant_kwargs} if quantization == "fp8" else None,
            **engine_kwargs,
        }

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

        if self.config.expert_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size, (
                f"num workers ({len(self.workers)}) should be equal to dp_size_local "
            )
            f"({data_parallel_size_local}) * tp_size ({self.config.tensor_model_parallel_size})"

            args.update(
                {
                    "enable_expert_parallel": self.config.expert_parallel_size > 1,
                    "data_parallel_size": self.config.data_parallel_size,
                    "data_parallel_size_local": data_parallel_size_local,
                    "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._master_port,
                }
            )

        # update lora-related args
        if self.model_config.lora_rank > 0:
            args.update(
                {
                    "enable_lora": True,
                    "max_loras": 1,
                    "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank),
                }
            )

        server_args = ["serve", self.model_config.local_path]
        for k, v in args.items():
            if isinstance(v, bool):
                if v:
                    server_args.append(f"--{k}")
            elif v is not None:
                server_args.append(f"--{k}")
                # Use json.dumps for dict to ensure valid JSON format
                server_args.append(json.dumps(v) if isinstance(v, dict) else str(v))

        if self.replica_rank == 0:
            pprint(server_args)

        CMD_MODULES = [vllm.entrypoints.cli.serve]
        parser = FlexibleArgumentParser(description="vLLM CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # 2. setup distributed executor backend
        distributed_executor_backend = ExternalZeroMQDistributedExecutor if len(self.workers) > 0 else None
        server_args.distributed_executor_backend = distributed_executor_backend

        zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in self.workers])
        logger.info(
            f"replica_rank={self.replica_rank}, node_rank={self.node_rank}, nnodes={self.nnodes}, "
            f"get worker zmq addresses: {zmq_addresses}"
        )
        os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        # 3. launch server
        if self.node_rank == 0:
            await self.run_server(server_args)
        else:
            await self.run_headless(server_args)

    async def _launch_hop_cluster(self) -> None:
        if self.node_rank != 0:
            return

        host = str(self._hop_cfg.get("host", "127.0.0.1"))
        router_port = int(self._hop_cfg.get("router_port", 8200))
        owner_state_port = int(self._hop_cfg.get("owner_state_port", 8300))
        server_ports = self._hop_cfg.get("server_ports", [8101, 8102])
        kv_ports = self._hop_cfg.get("server_kv_ports", [18101, 18102])
        decode_cutovers = self._hop_cfg.get("decode_cutovers", [1024])
        request_timeout_s = float(self._hop_cfg.get("request_timeout_s", 600.0))
        connect_timeout_s = float(self._hop_cfg.get("connect_timeout_s", 60.0))
        max_response_length = int(self._hop_cfg.get("max_response_length", self.config.response_length))
        hop_http_max_connections = int(self._hop_cfg.get("http_max_connections", 1024))
        hop_http_max_keepalive = int(self._hop_cfg.get("http_max_keepalive_connections", 512))

        if len(server_ports) != len(kv_ports):
            raise RuntimeError("verl_hop.server_ports and verl_hop.server_kv_ports must have the same size")
        if len(server_ports) not in (2, 4):
            raise RuntimeError("verl_hop currently supports exactly 2 or 4 servers")

        py = sys.executable
        env = dict(os.environ)
        py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"/root/vllm:{py_path}" if py_path else "/root/vllm"

        def _cleanup_stale_hop_processes() -> None:
            patterns = [
                "launch_four_server_ipc_vllm",
                "launch_sequential_decode_router",
                "kv_owner_state_server:create_app",
                "vllm.entrypoints.openai.api_server",
                "vllm.v1.engine.core",
            ]
            for pattern in patterns:
                try:
                    subprocess.run(
                        ["pkill", "-9", "-f", pattern],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    logger.exception("failed to cleanup stale hop process pattern=%s", pattern)
            time.sleep(1.0)

        def _j(obj: Any) -> str:
            return json.dumps(obj) if isinstance(obj, (dict, list)) else str(obj)

        log_prefix = f"/tmp/verl_hop_replica_{self.replica_rank}_{self.node_rank}"
        owner_log_path = f"{log_prefix}_owner_state.log"
        launch_log_path = f"{log_prefix}_launch4.log"
        router_log_path = f"{log_prefix}_router.log"
        owner_gpu_mem = float(self._hop_cfg.get("owner_gpu_memory_utilization", self.config.gpu_memory_utilization))
        consumer_gpu_mem = float(
            self._hop_cfg.get("consumer_gpu_memory_utilization", self.config.gpu_memory_utilization)
        )
        owner_max_num_seqs = int(self._hop_cfg.get("owner_max_num_seqs", self.config.max_num_seqs))
        consumer_max_num_seqs = int(self._hop_cfg.get("consumer_max_num_seqs", self.config.max_num_seqs))
        owner_tp = int(self._hop_cfg.get("owner_tensor_parallel_size", self.config.tensor_model_parallel_size))
        consumer_tp = int(self._hop_cfg.get("consumer_tensor_parallel_size", owner_tp))
        owner_cvd = str(self._hop_cfg.get("owner_cuda_visible_devices", "0,1"))
        consumer_cvd = str(self._hop_cfg.get("consumer_cuda_visible_devices_all", owner_cvd))
        owner_comp_cfg = self._hop_cfg.get("owner_compilation_config")
        server2_comp_cfg = self._hop_cfg.get("server2_compilation_config")
        server3_comp_cfg = self._hop_cfg.get("server3_compilation_config")
        server4_comp_cfg = self._hop_cfg.get("server4_compilation_config")
        send_activation_margin_tokens = int(self._hop_cfg.get("send_activation_margin_tokens", 512))
        send_publish_token_stride = int(self._hop_cfg.get("send_publish_token_stride", 64))
        owner_flush_each_layer = bool(self._hop_cfg.get("owner_flush_each_layer", False))
        num_servers = len(server_ports)

        if self._hop_cfg.get("enable_cuda_mps", False):
            default_mps = [100, 60] if len(server_ports) == 2 else [100, 60, 40, 20]
            percentages = self._hop_cfg.get("mps_active_thread_percentages", default_mps)
            if len(percentages) >= 1:
                env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percentages[0])

        owner_cmd = [
            py,
            "-m",
            "uvicorn",
            "vllm.proxy_cluster.kv_owner_state_server:create_app",
            "--factory",
            "--host",
            host,
            "--port",
            str(owner_state_port),
        ]

        async def _wait_http_ready(url: str, timeout_s: float) -> bool:
            deadline = time.time() + max(0.1, timeout_s)
            timeout = httpx.Timeout(min(5.0, max(1.0, timeout_s)), connect=min(2.0, max(0.5, timeout_s)))
            async with httpx.AsyncClient(timeout=timeout) as client:
                while time.time() < deadline:
                    try:
                        resp = await client.get(url)
                        if 200 <= resp.status_code < 300:
                            return True
                    except Exception:
                        pass
                    await asyncio.sleep(0.25)
            return False

        def _tail_log(path: str, lines: int = 80) -> str:
            if not os.path.exists(path):
                return f"<missing log {path}>"
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    data = f.readlines()
                return "".join(data[-lines:])
            except Exception as e:
                return f"<failed to read {path}: {e}>"

        _cleanup_stale_hop_processes()

        self._hop_processes.append(
            subprocess.Popen(
                owner_cmd,
                env=env,
                start_new_session=True,
                stdout=open(owner_log_path, "w"),
                stderr=subprocess.STDOUT,
            )
        )

        owner_ready = await _wait_http_ready(f"http://{host}:{owner_state_port}/healthz", 60.0)
        if not owner_ready:
            raise RuntimeError(
                "verl hop owner-state failed to become ready\n" f"owner_state_tail:\n{_tail_log(owner_log_path)}"
            )

        launch_cmd = [
            py,
            "-m",
            "vllm.proxy_cluster.launch_four_server_ipc_vllm",
            "--num-servers",
            str(num_servers),
            "--model",
            self.model_config.local_path,
            "--host",
            host,
            "--server1-port",
            str(server_ports[0]),
            "--server2-port",
            str(server_ports[1]),
            "--owner-gpu-memory-utilization",
            str(owner_gpu_mem),
            "--consumer-gpu-memory-utilization",
            str(consumer_gpu_mem),
            "--owner-max-num-seqs",
            str(owner_max_num_seqs),
            "--consumer-max-num-seqs",
            str(consumer_max_num_seqs),
            "--owner-max-model-len",
            str(self.config.max_model_len),
            "--consumer-max-model-len",
            str(self.config.max_model_len),
            "--owner-cuda-visible-devices",
            owner_cvd,
            "--consumer-cuda-visible-devices-all",
            consumer_cvd,
            "--owner-tensor-parallel-size",
            str(owner_tp),
            "--consumer-tensor-parallel-size",
            str(consumer_tp),
            "--consumer-attention-backend",
            str(self._hop_cfg.get("consumer_attention_backend", "TORCH_SDPA")),
            "--kv-owner-state-url",
            f"http://{host}:{owner_state_port}",
            "--kv-transfer-config-template",
            '{"kv_connector":"CudaIpcConnector","kv_role":"kv_both","kv_rank":0,"kv_parallel_size":1}',
            "--shared-kv-pool-enable",
            "--shared-kv-pool-meta-path",
            str(self._hop_cfg.get("shared_kv_pool_meta_path", "/tmp/vllm_shared_kv_pool_from_verl.pkl")),
            "--send-activation-margin-tokens",
            str(send_activation_margin_tokens),
            "--send-publish-token-stride",
            str(send_publish_token_stride),
        ]
        if owner_flush_each_layer:
            launch_cmd.append("--owner-flush-each-layer")
        if num_servers >= 3:
            launch_cmd += ["--server3-port", str(server_ports[2])]
        if num_servers >= 4:
            launch_cmd += ["--server4-port", str(server_ports[3])]
        if self._hop_cfg.get("no_consumer_enforce_eager", True):
            launch_cmd.append("--no-consumer-enforce-eager")
        if owner_comp_cfg:
            launch_cmd += ["--owner-compilation-config", _j(owner_comp_cfg)]
        if server2_comp_cfg:
            launch_cmd += ["--server2-compilation-config", _j(server2_comp_cfg)]
        if num_servers >= 3 and server3_comp_cfg:
            launch_cmd += ["--server3-compilation-config", _j(server3_comp_cfg)]
        if num_servers >= 4 and server4_comp_cfg:
            launch_cmd += ["--server4-compilation-config", _j(server4_comp_cfg)]
        if self._hop_cfg.get("enable_cuda_mps", False):
            default_mps = [100, 60] if num_servers == 2 else [100, 60, 40, 20]
            launch_cmd += [
                "--enable-cuda-mps",
                "--mps-active-thread-percentages",
                ",".join(str(x) for x in self._hop_cfg.get("mps_active_thread_percentages", default_mps)),
            ]

        self._hop_processes.append(
            subprocess.Popen(
                launch_cmd,
                env=env,
                start_new_session=True,
                stdout=open(launch_log_path, "w"),
                stderr=subprocess.STDOUT,
            )
        )

        startup_timeout_s = float(self._hop_cfg.get("startup_timeout_s", max(300.0, connect_timeout_s)))
        upstream_timeout_s = max(startup_timeout_s, connect_timeout_s)
        for port in server_ports:
            ready = await _wait_http_ready(f"http://{host}:{port}/v1/models", upstream_timeout_s)
            if not ready:
                raise RuntimeError(
                    f"verl hop upstream server at {host}:{port} failed to become ready\n"
                    f"owner_state_tail:\n{_tail_log(owner_log_path)}\n"
                    f"launch4_tail:\n{_tail_log(launch_log_path)}"
                )

        router_cmd = [
            py,
            "-m",
            "vllm.proxy_cluster.launch_sequential_decode_router",
            "--num-servers",
            str(num_servers),
            "--host",
            host,
            "--port",
            str(router_port),
            "--server1-url",
            f"http://{host}:{server_ports[0]}",
            "--server2-url",
            f"http://{host}:{server_ports[1]}",
            "--server-kv-ports",
            ",".join(str(x) for x in kv_ports),
            "--routing-mode",
            "sequential_handoff",
            "--decode-cutovers",
            ",".join(str(x) for x in decode_cutovers),
            "--upstream-max-model-len",
            str(self.config.max_model_len),
            "--max-response-length",
            str(max_response_length),
            "--request-timeout-s",
            str(request_timeout_s),
            "--connect-timeout-s",
            str(connect_timeout_s),
            "--kv-owner-state-url",
            f"http://{host}:{owner_state_port}",
        ]
        if num_servers >= 3:
            router_cmd += ["--server3-url", f"http://{host}:{server_ports[2]}"]
        if num_servers >= 4:
            router_cmd += ["--server4-url", f"http://{host}:{server_ports[3]}"]
        router_proc = subprocess.Popen(
            router_cmd,
            env=env,
            start_new_session=True,
            stdout=open(router_log_path, "w"),
            stderr=subprocess.STDOUT,
        )
        self._hop_processes.append(router_proc)

        self._hop_router_url = str(self._hop_cfg.get("router_url", f"http://{host}:{router_port}")).rstrip("/")
        self._hop_http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout_s, connect=connect_timeout_s, pool=request_timeout_s),
            limits=httpx.Limits(
                max_connections=max(32, hop_http_max_connections),
                max_keepalive_connections=max(16, hop_http_max_keepalive),
            ),
        )
        await self._wait_hop_router_ready()
        self._server_address = host
        self._server_port = router_port
        router_ready = await _wait_http_ready(self._hop_router_url + "/healthz", 60.0)
        if not router_ready:
            raise RuntimeError(
                "verl hop router failed to become ready\n"
                f"launch4_tail:\n{_tail_log(launch_log_path)}\n"
                f"router_tail:\n{_tail_log(router_log_path)}"
            )
        if router_proc.poll() is not None:
            raise RuntimeError(
                "verl hop router exited unexpectedly after startup\n"
                f"returncode={router_proc.returncode}\n"
                f"launch4_tail:\n{_tail_log(launch_log_path)}\n"
                f"router_tail:\n{_tail_log(router_log_path)}"
            )

        logger.info("verl hop router ready at %s", self._hop_router_url)

    async def _wait_hop_router_ready(self, timeout_s: float = 300.0) -> None:
        assert self._hop_http_client is not None
        deadline = time.time() + timeout_s
        url = self._hop_router_url + "/healthz"
        last_err = ""
        while time.time() < deadline:
            try:
                resp = await self._hop_http_client.get(url)
                if 200 <= resp.status_code < 300:
                    return
                last_err = f"status={resp.status_code}"
            except Exception as exc:
                last_err = str(exc)
            await asyncio.sleep(1.0)
        raise RuntimeError(f"hop router not ready within {timeout_s}s ({last_err})")

    async def _generate_via_hop_router(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
    ) -> TokenOutput:
        if self._hop_http_client is None:
            raise RuntimeError("hop router client is not initialized")
        if self._hop_cfg.get("router_mode", "sequential_handoff") != "sequential_handoff":
            raise NotImplementedError("Only sequential_handoff router_mode is supported in verl hop rollout")

        req_max_tokens = sampling_params.get("max_tokens")
        if req_max_tokens is None:
            req_max_tokens = self.config.response_length
        req_max_tokens = int(req_max_tokens)
        max_tokens = min(
            req_max_tokens,
            int(self._hop_cfg.get("max_response_length", self.config.response_length)),
            self.config.max_model_len - len(prompt_ids),
        )
        payload = {
            "model": self.model_config.local_path,
            "request_id": request_id,
            "prompt": prompt_ids,
            "max_tokens": max(1, int(max_tokens)),
            "temperature": sampling_params.get("temperature", self.config.temperature),
            "top_p": sampling_params.get("top_p", self.config.top_p),
            "top_k": sampling_params.get("top_k", self.config.top_k),
            "stop": sampling_params.get("stop", []),
            "return_token_ids": True,
        }
        if sampling_params.get("logprobs", False):
            payload["logprobs"] = 1
        if "repetition_penalty" in sampling_params:
            payload["repetition_penalty"] = sampling_params["repetition_penalty"]
        if sampling_params.get("ignore_eos", False):
            payload["ignore_eos"] = True
        if sampling_params.get("presence_penalty") is not None:
            payload["presence_penalty"] = sampling_params["presence_penalty"]
        if sampling_params.get("frequency_penalty") is not None:
            payload["frequency_penalty"] = sampling_params["frequency_penalty"]

        resp = await self._hop_http_client.post(self._hop_router_url + "/v1/completions", json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"hop router error status={resp.status_code} body={resp.text}")
        obj = resp.json()
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("hop router response missing choices[0]")
        token_ids = choices[0].get("token_ids")
        if not isinstance(token_ids, list) or not all(isinstance(x, int) for x in token_ids):
            raise RuntimeError("hop router response missing choices[0].token_ids")
        log_probs = None
        choice_logprobs = choices[0].get("logprobs")
        if isinstance(choice_logprobs, dict):
            token_logprobs = choice_logprobs.get("token_logprobs")
            if isinstance(token_logprobs, list):
                parsed: list[float] = []
                ok = True
                for x in token_logprobs[: len(token_ids)]:
                    if isinstance(x, (int, float)):
                        parsed.append(float(x))
                    else:
                        ok = False
                        break
                if ok and len(parsed) == len(token_ids):
                    log_probs = parsed
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        engine_client = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            disable_log_requests=engine_args.disable_log_requests,
            disable_log_stats=engine_args.disable_log_stats,
        )

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()

        app = build_app(args)
        await init_app_state(engine_client, vllm_config, app.state, args)
        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_unvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        # Create the EngineConfig.
        engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context, headless=True)

        parallel_config = vllm_config.parallel_config
        local_engine_count = parallel_config.data_parallel_size_local

        host = parallel_config.data_parallel_master_ip
        port = engine_args.data_parallel_rpc_port  # add to config too
        handshake_address = get_tcp_uri(host, port)

        # Create the engines.
        self.engine_manager = CoreEngineProcManager(
            target_fn=EngineCoreProc.run_engine_core,
            local_engine_count=local_engine_count,
            start_index=vllm_config.parallel_config.data_parallel_rank,
            local_start_index=0,
            vllm_config=vllm_config,
            local_client=False,
            handshake_address=handshake_address,
            executor_class=Executor.get_class(vllm_config),
            log_stats=not engine_args.disable_log_stats,
        )

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        if self._hop_enabled:
            if image_data:
                raise NotImplementedError("verl hop rollout does not support image_data yet")
            return await self._generate_via_hop_router(prompt_ids, sampling_params, request_id)
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        max_tokens = self.config.max_model_len - len(prompt_ids)
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        prompt = TokensPrompt(
            prompt_token_ids=prompt_ids, multi_modal_data={"image": image_data} if image_data else None
        )

        # Add lora request
        lora_request = None
        if self.model_config.lora_rank > 0:
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        generator = self.engine.generate(
            prompt=prompt, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request
        )

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)

    async def wake_up(self):
        if self._hop_enabled:
            logger.info("skip wake_up in verl hop mode")
            return
        if self.rollout_mode == RolloutMode.HYBRID:
            # Call all workers to switch between trainer mode and rollout mode.
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            if self.node_rank == 0:
                await self.engine.wake_up(tags=["kv_cache", "weights"])
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self._hop_enabled:
            logger.info("skip sleep in verl hop mode")
            return
        if self.rollout_mode == RolloutMode.HYBRID:
            if self.node_rank == 0:
                await self.engine.reset_prefix_cache()
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            if self.node_rank == 0:
                await self.engine.reset_prefix_cache()
                await self.engine.sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def clear_kv_cache(self):
        if self._hop_enabled:
            host = str(self._hop_cfg.get("host", "127.0.0.1"))
            server_ports = list(self._hop_cfg.get("server_ports", [8101, 8102]))
            owner_state_port = int(self._hop_cfg.get("owner_state_port", 8300))
            handoff_pin_s = float(self._hop_cfg.get("handoff_pin_s", 1.5))
            # Let deferred producer-side frees mature before resetting cache state.
            await asyncio.sleep(max(0.0, handoff_pin_s) + 0.2)
            timeout = httpx.Timeout(30.0, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                for port in server_ports:
                    try:
                        resp = await client.post(f"http://{host}:{port}/reset_hop_state")
                        resp.raise_for_status()
                    except Exception as exc:
                        logger.warning(
                            "hop clear_kv_cache reset_hop_state failed upstream=%s:%s err=%r",
                            host,
                            port,
                            exc,
                        )
                try:
                    resp = await client.post(
                        f"http://{host}:{owner_state_port}/reset_all",
                        json={"confirm": True},
                    )
                    resp.raise_for_status()
                except Exception as exc:
                    logger.warning(
                        "hop clear_kv_cache owner_state reset_all failed host=%s port=%s err=%r",
                        host,
                        owner_state_port,
                        exc,
                    )
            return
        if self.node_rank == 0:
            await self.engine.reset_prefix_cache()

    async def wait_for_requests_to_drain(self):
        if self._hop_enabled:
            return
        await self.engine.wait_for_requests_to_drain()


@ray.remote(num_cpus=1)
class vLLMHttpServer(vLLMHttpServerBase):
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        super().__init__(config, model_config, rollout_mode, workers, replica_rank, node_rank, gpus_per_node, nnodes)


_rollout_worker_actor_cls = ray.remote(vLLMAsyncRollout)


class vLLMReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig | RewardModelConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = vLLMHttpServer

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get node_id of all workers
        worker_node_ids = await asyncio.gather(
            *[
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.workers
            ]
        )

        # For non-data parallel case, there's only one server whether it's single or multi nodes.
        nnodes, gpus_per_node = self.nnodes, self.gpus_per_node
        if self.config.data_parallel_size == 1:
            nnodes = 1
            gpus_per_node = self.world_size

        # create server actor in each node with node affinity
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_node : (node_rank + 1) * gpus_per_node]
            node_id = worker_node_ids[node_rank * gpus_per_node]
            name = (
                f"vllm_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"vllm_server_reward_{self.replica_rank}_{node_rank}"
            )
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_node,
                nnodes=nnodes,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def sleep(self):
        """Sleep each rollout server."""
        # Drain DP engines for safe sleep.
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])


def _qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
    <|image_pad|> token by image_data.

    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        prompt_ids = np.array(prompt_ids)

        # Create a mask where True indicates elements to keep
        mask = np.ones(len(prompt_ids), dtype=bool)

        # Find where the array equals the value
        is_value = prompt_ids == processor.image_token_id

        # Find consecutive duplicates by checking if previous element is also the value
        mask[1:] &= ~(is_value[1:] & is_value[:-1])

        return prompt_ids[mask].tolist()
    else:
        return prompt_ids
