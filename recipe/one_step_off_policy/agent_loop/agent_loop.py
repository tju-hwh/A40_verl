# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import asyncio
import logging
import os

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OneStepOffAgentLoopManager(AgentLoopManager):
    async def generate_sequences_async(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers (async version).

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        # Use asyncio.gather with ray.get wrapped in asyncio.to_thread to avoid blocking
        import asyncio

        outputs = await asyncio.gather(
            *[
                asyncio.to_thread(ray.get, worker.generate_sequences.remote(chunk))
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    async def generate_sequences_async_stream(
        self,
        prompts: DataProto,
        stream_queue,
        stream_group_size: int,
        stream_end_token=None,
    ) -> DataProto:
        """Split input batch and dispatch to agent loop workers (async version) with streaming output."""
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                asyncio.to_thread(
                    ray.get,
                    worker.generate_sequences.remote(
                        chunk,
                        stream_queue=stream_queue,
                        stream_group_size=stream_group_size,
                        stream_end_token=None,  # 结束符只由 manager 发出
                    ),
                )
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        keep_indices = []
        running_offset = 0
        saw_early_stop = False
        for worker_output in outputs:
            worker_keep = worker_output.meta_info.pop("early_stop_indices", None)
            if worker_keep is None:
                worker_keep = list(range(len(worker_output)))
            else:
                saw_early_stop = True
            for idx in worker_keep:
                if 0 <= idx < len(worker_output):
                    keep_indices.append(running_offset + idx)
            running_offset += len(worker_output)
        output = DataProto.concat(outputs)
        metrics = [output.meta_info.pop("metrics") for output in outputs]
        timing = self._performance_metrics(metrics, output)
        if saw_early_stop and keep_indices:
            output.meta_info["early_stop_indices"] = keep_indices
        output.meta_info = {"timing": timing, **output.meta_info}
        if stream_queue is not None and stream_end_token is not None:
            # 只发送一个结束符，避免多 worker 重复
            await asyncio.to_thread(stream_queue.put, stream_end_token)
        return output

    async def wake_up(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    async def sleep(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])

    async def clear_kv_cache(self):
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])

    async def wait_for_requests_to_drain(self):
        await asyncio.gather(*[replica.wait_for_requests_to_drain() for replica in self.rollout_replicas])
