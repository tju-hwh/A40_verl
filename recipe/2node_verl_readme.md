- 多网卡以太网 + Ray 集群启动脚本：
  [start_ray_cluster_rdma.sh](/root/A40_verl/recipe/start_ray_cluster_rdma.sh)
- 两机 16 卡训练提交脚本：
  [run_deepmath_sync_grpo_qwen3_8b_2node16gpu_rdma.sh](/root/A40_verl/recipe/run_deepmath_sync_grpo_qwen3_8b_2node16gpu_rdma.sh)
- 本机 `eth0` 到 `eth3` 自动采集脚本：
  [generate_rdma_eth_env.sh](/root/A40_verl/recipe/generate_rdma_eth_env.sh)

当前训练脚本已经对齐到你要求的主配置：

- `data.train_batch_size=512`
- `data.max_prompt_length=1024`
- `data.max_response_length=4096`
- `+data.apply_chat_template_kwargs.enable_thinking=False`
- `actor_rollout_ref.actor.ppo_mini_batch_size=512`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2`
- `actor_rollout_ref.rollout.n=2`

保留的主 rollout 参数：

- `rollout.temperature=0.6`
- `rollout.top_p=0.95`
- `rollout.top_k=20`
- `rollout.tensor_model_parallel_size=4`
- `rollout.mode=async`
- `rollout.name=vllm`








• 已经改成非 RDMA 版本了，通信只用 eth0 到 eth3。

现在脚本行为是：

- NCCL_SOCKET_IFNAME=eth0,eth1,eth2,eth3
- GLOO_SOCKET_IFNAME=eth0
- NCCL_IB_DISABLE=1
- 不再设置 NCCL_NET=IB
- 不再依赖 mlx5 / ib0 / UCX

改动文件：

- A40_verl/recipe/generate_rdma_eth_env.sh
- A40_verl/recipe/start_ray_cluster_rdma.sh
- A40_verl/recipe/run_deepmath_sync_grpo_qwen3_8b_2node16gpu_rdma.sh
- A40_verl/recipe/2node_verl_readme.md

文件名我暂时没改，还是带 rdma，但内容已经是普通多网卡以太网版。

执行流程

1. 在 node2 先生成本机网卡 env，拿到它的 eth0 IP：

bash /root/A40_verl/recipe/generate_rdma_eth_env.sh OUTPUT_FILE=/tmp/node2_eth.env
cat /tmp/node2_eth.env

2. 在 node1 生成正式 env：

bash /root/A40_verl/recipe/generate_rdma_eth_env.sh \
ROLE=head \
PEER_ETH0_IP=172.24.79.13 \
OUTPUT_FILE=/root/A40_verl/recipe/eth_multinic.env

3. 在 node2 生成正式 env：

bash /root/A40_verl/recipe/generate_rdma_eth_env.sh \
ROLE=worker \
PEER_ETH0_IP=172.24.79.15 \
OUTPUT_FILE=/root/A40_verl/recipe/eth_multinic.env

4. 启动 Ray：

bash /root/A40_verl/recipe/start_ray_cluster_rdma.sh head

bash /root/A40_verl/recipe/start_ray_cluster_rdma.sh worker

5. 在 node1 提交训练：

bash /root/A40_verl/recipe/run_deepmath_sync_grpo_qwen3_8b_2node16gpu_rdma.sh

bash /root/A40_verl/recipe/run_deepmath_sync_grpo_qwen3_14b_2node16gpu_rdma.sh

6. 看日志：

ray job list --address="http://172.24.79.15:8265"
ray job logs raysubmit_JNQhe9uPNFE2M5Yx --address="http://172.24.79.15:8265" --follow