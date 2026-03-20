#!/usr/bin/env bash
set -euo pipefail

OUTPUT_FILE="${OUTPUT_FILE:-/root/A40_verl/recipe/eth_multinic.env}"
ROLE="${ROLE:-}"
PEER_ETH0_IP="${PEER_ETH0_IP:-}"
IFACES_CSV="${IFACES_CSV:-eth0,eth1,eth2,eth3}"

IFS=',' read -r -a IFACES <<< "${IFACES_CSV}"

declare -a LOCAL_IPS=()
declare -a FOUND_IFACES=()
for IFACE in "${IFACES[@]}"; do
  CIDR="$(ip -4 -o addr show dev "${IFACE}" scope global | awk 'NR==1 {print $4}')"
  if [[ -z "${CIDR}" ]]; then
    echo "Missing IPv4 on ${IFACE}" >&2
    exit 1
  fi
  LOCAL_IPS+=("${CIDR%/*}")
  FOUND_IFACES+=("${IFACE}")
done

THIS_NODE_IP="${LOCAL_IPS[0]}"
GLOO_IFACE="${FOUND_IFACES[0]}"
NCCL_IFACES="$(IFS=,; echo "${FOUND_IFACES[*]}")"
LOCAL_ETH_IPS="$(IFS=,; echo "${LOCAL_IPS[*]}")"

HEAD_NODE_IP_LINE=""
WORKER_NODE_IP_LINE=""
if [[ -n "${ROLE}" ]]; then
  if [[ -z "${PEER_ETH0_IP}" ]]; then
    echo "PEER_ETH0_IP is required when ROLE is set" >&2
    exit 1
  fi
  case "${ROLE}" in
    head)
      HEAD_NODE_IP_LINE="export HEAD_NODE_IP=\"${THIS_NODE_IP}\""
      WORKER_NODE_IP_LINE="export WORKER_NODE_IP=\"${PEER_ETH0_IP}\""
      ;;
    worker)
      HEAD_NODE_IP_LINE="export HEAD_NODE_IP=\"${PEER_ETH0_IP}\""
      WORKER_NODE_IP_LINE="export WORKER_NODE_IP=\"${THIS_NODE_IP}\""
      ;;
    *)
      echo "ROLE must be head or worker, got ${ROLE}" >&2
      exit 1
      ;;
  esac
fi

cat > "${OUTPUT_FILE}" <<EOF
#!/usr/bin/env bash
export ETH_MULTI_NIC_IFACES="${NCCL_IFACES}"
export ETH_MULTI_NIC_IPS="${LOCAL_ETH_IPS}"
export THIS_NODE_IP="${THIS_NODE_IP}"
export VLLM_HOST_IP="${THIS_NODE_IP}"
export GLOO_SOCKET_IFNAME="${GLOO_IFACE}"
export NCCL_SOCKET_IFNAME="${NCCL_IFACES}"
export NCCL_CROSS_NIC="\${NCCL_CROSS_NIC:-1}"
export NCCL_IB_DISABLE="\${NCCL_IB_DISABLE:-1}"
${HEAD_NODE_IP_LINE}
${WORKER_NODE_IP_LINE}
EOF

chmod +x "${OUTPUT_FILE}"

echo "Wrote ${OUTPUT_FILE}"
echo "ETH_MULTI_NIC_IFACES=${NCCL_IFACES}"
echo "ETH_MULTI_NIC_IPS=${LOCAL_ETH_IPS}"
echo "THIS_NODE_IP=${THIS_NODE_IP}"
if [[ -n "${HEAD_NODE_IP_LINE}" ]]; then
  echo "${HEAD_NODE_IP_LINE#export }"
  echo "${WORKER_NODE_IP_LINE#export }"
fi
