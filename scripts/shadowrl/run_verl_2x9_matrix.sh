#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/shadowrl_verl_sglang
source /root/envs/shadowrl/bin/activate
cd "$ROOT"

python scripts/shadowrl/prepare_verl_bench_data.py

export RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
export LOG_ROOT=${LOG_ROOT:-$ROOT/bench_logs/verl_full_rl}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export TP_SIZE=${TP_SIZE:-2}
export DP_SIZE=${DP_SIZE:-2}

python - <<'PY'
import os
import sys
import torch

required = int(os.environ.get("N_GPUS_PER_NODE", "4"))
count = torch.cuda.device_count()
print(f"CUDA visible device_count={count}, required={required}")
for i in range(count):
    prop = torch.cuda.get_device_properties(i)
    print(
        f"  gpu{i}: {torch.cuda.get_device_name(i)}, "
        f"mem_gb={prop.total_memory / 1024**3:.2f}, sms={prop.multi_processor_count}"
    )
if count < required:
    sys.exit(
        f"Insufficient visible GPUs: need {required} for the requested TP/DP matrix, got {count}. "
        "Fix CUDA_VISIBLE_DEVICES/MIG allocation before launching."
    )
PY

models=(qwen3 llama31_8b deepseek_r1_qwen7b)
datasets=(deepmath eurus hh)
systems=(shadowrl rollpacker)
port=30000

for system in "${systems[@]}"; do
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      echo "===== ${system} ${model} ${dataset} ====="
      bash scripts/shadowrl/run_verl_shadowrl_rollpacker_one_step.sh "$system" "$model" "$dataset" "$port"
      port=$((port + 20))
      if (( port > 30180 )); then
        port=30000
      fi
    done
  done
done

python scripts/shadowrl/summarize_verl_results.py \
  --log-dir "$LOG_ROOT" \
  --output "$LOG_ROOT/results_${RUN_TAG}.jsonl"

echo "logs: $LOG_ROOT"
echo "results: $LOG_ROOT/results_${RUN_TAG}.jsonl"
