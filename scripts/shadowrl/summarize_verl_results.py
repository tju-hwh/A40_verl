#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


METRIC_RE = re.compile(r"([A-Za-z0-9_./-]+)[:=] ?(-?[0-9]+(?:\.[0-9]+)?)")


def parse_log(path):
    text = path.read_text(errors="ignore")
    metrics = {}
    for key, value in METRIC_RE.findall(text):
        if key.startswith(("timing_s/", "perf/", "training/", "rollpacker/")):
            try:
                metrics[key] = float(value)
            except ValueError:
                pass
    if "timing_s/step" not in metrics:
        m = re.findall(r"'timing_s/step': ([0-9.]+)|\"timing_s/step\": ([0-9.]+)", text)
        if m:
            metrics["timing_s/step"] = float(next(x or y for x, y in m[::-1]))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    rows = []
    for log_path in sorted(log_dir.glob("*.log")):
        stem = log_path.stem
        parts = stem.split("__")
        if len(parts) < 3:
            continue
        system, model, dataset = parts[:3]
        row = {
            "system": system,
            "model": model,
            "dataset": dataset,
            "log": str(log_path),
        }
        row.update(parse_log(log_path))
        rows.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
