# Hop-vLLM DP4 TP4 on 2 Nodes / 16 GPUs

This setup uses the latest external-DP-shard hop-vLLM design:

- global DP = 4
- TP per DP rank = 4
- total GPUs = 16
- node0 hosts global `dp0, dp1`
- node1 hosts global `dp2, dp3`
- each DP rank is a local mirrored hop pair:
  - `server1(owner) -> server2(consumer)`

The training process does not launch vLLM locally. Instead, VERL connects to an externally managed global hop router.

## Fixed parameters

These scripts default to the requested tuning:

- `HOP_MPS_ACTIVE_THREAD_PERCENTAGES='[100,60]'`
- `ACTOR_MPS_ACTIVE_THREAD_PERCENTAGE=70`
- `REF_MPS_ACTIVE_THREAD_PERCENTAGE=70`
- `CRITIC_MPS_ACTIVE_THREAD_PERCENTAGE=70`
- `HOP_DECODE_CUTOVERS='[1536]'`
- `HOP_SEND_PUBLISH_TOKEN_STRIDE=2048`
- `TRAIN_GROUP_PLAN='[12,12,12,12,12,12,12,12,8,8,8,8]'`

## Network

All scripts pin communication to the regular Ethernet interfaces:

- `NCCL_SOCKET_IFNAME=eth0,eth1,eth2,eth3`
- `GLOO_SOCKET_IFNAME=eth0,eth1,eth2,eth3`

## Files

- owner-state:
  - [start_owner_state.sh](/root/A40_verl/scripts/hop_dp4tp4_2node/start_owner_state.sh)
- local per-node hop cluster:
  - [start_local_cluster.sh](/root/A40_verl/scripts/hop_dp4tp4_2node/start_local_cluster.sh)
- global router:
  - [start_global_router.sh](/root/A40_verl/scripts/hop_dp4tp4_2node/start_global_router.sh)
- Ray head:
  - [start_ray_head.sh](/root/A40_verl/scripts/hop_dp4tp4_2node/start_ray_head.sh)
- Ray worker:
  - [start_ray_worker.sh](/root/A40_verl/scripts/hop_dp4tp4_2node/start_ray_worker.sh)
- training:
  - [rlos_test_deepmath_2node16gpu_dp4tp4.sh](/root/A40_verl/recipe/one_step_off_policy/shell/rlos_test_deepmath_2node16gpu_dp4tp4.sh)

## Topology

Node 0, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`:

- global `dp0`: GPUs `0,1,2,3`
  - owner `http://NODE0_IP:8101`
  - consumer `http://NODE0_IP:8102`
- global `dp1`: GPUs `4,5,6,7`
  - owner `http://NODE0_IP:8103`
  - consumer `http://NODE0_IP:8104`

Node 1, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`:

- global `dp2`: GPUs `0,1,2,3`
  - owner `http://NODE1_IP:8101`
  - consumer `http://NODE1_IP:8102`
- global `dp3`: GPUs `4,5,6,7`
  - owner `http://NODE1_IP:8103`
  - consumer `http://NODE1_IP:8104`

## Startup order

Assume:

- `NODE0_IP=<head machine IP>`
- `NODE1_IP=<worker machine IP>`


  本机清理：

  bash /root/A40_verl/scripts/hop_dp4tp4_2node/clean_all.sh

  如果你想在 node0 上一键把 node1 也一起清掉：

  REMOTE_HOST=172.24.79.13 bash /root/A40_verl/scripts/hop_dp4tp4_2node/clean_all.sh

### 1. Start Ray

On node0:

```bash
NODE_IP=172.24.79.15 bash /root/A40_verl/scripts/hop_dp4tp4_2node/start_ray_head.sh
```

On node1:

```bash
NODE_IP=172.24.79.13 HEAD_IP=172.24.79.15 bash /root/A40_verl/scripts/hop_dp4tp4_2node/start_ray_worker.sh
```

### 2. Start owner-state on node0

On node0:

```bash
OWNER_STATE_PORT=8300 bash /root/A40_verl/scripts/hop_dp4tp4_2node/start_owner_state.sh
```

### 3. Start local hop clusters

On node0:

```bash
export NODE0_IP=172.24.79.15
NODE_IP=172.24.79.15 \
OWNER_STATE_URL=http://${NODE0_IP}:8300 \
LOCAL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash /root/A40_verl/scripts/hop_dp4tp4_2node/start_local_cluster.sh
```

On node1:

```bash
export NODE0_IP=172.24.79.15
NODE_IP=172.24.79.13 \
OWNER_STATE_URL=http://${NODE0_IP}:8300 \
LOCAL_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash /root/A40_verl/scripts/hop_dp4tp4_2node/start_local_cluster.sh
```

### 4. Start global router on node0

On node0:

```bash
export NODE0_IP=172.24.79.15
export NODE1_IP=172.24.79.13
NODE0_IP=${NODE0_IP} \
NODE1_IP=${NODE1_IP} \
OWNER_STATE_URL=http://${NODE0_IP}:8300 \
bash /root/A40_verl/scripts/hop_dp4tp4_2node/start_global_router.sh
```

### 5. Start training on node0

On node0:

```bash
HEAD_NODE_IP=${NODE0_IP} \
WORKER_NODE_IP=${NODE1_IP} \
bash /root/A40_verl/recipe/one_step_off_policy/shell/rlos_test_deepmath_2node16gpu_dp4tp4.sh
```

## Logs

- owner-state:
  - `/tmp/hop_owner_state_dp4tp4.log`
- node-local clusters:
  - `/tmp/hop_local_cluster_${NODE_IP//./_}.log`
- global router:
  - `/tmp/hop_global_router_dp4tp4.log`
- VERL:
  - `/root/1.log`

## VERL adaptation

The VERL bridge has been adapted so that `verl_hop` can use:

- `external_managed: true`
- `router_url`
- `owner_state_url`
- `server1_urls`
- `server2_urls`
- `server_urls`

In this mode, VERL:

- does not launch local vLLM hop servers
- only waits for the external router
- resets all external upstreams via `/reset_hop_state`
- resets owner-state via `/reset_all`
