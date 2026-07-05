#!/usr/bin/env python3
import argparse
import gzip
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset


DATASETS = {
    "deepmath": {
        "path": "DeepMath-103K",
        "files": ["data/*.parquet"],
        "kind": "parquet",
    },
    "eurus": {
        "path": "Eurus-2-RL-Data",
        "files": ["train.parquet", "validation.parquet"],
        "kind": "parquet",
    },
    "hh": {
        "path": "HH-RLHF",
        "files": ["harmless-base/*.jsonl.gz", "helpful-*/*.jsonl.gz"],
        "kind": "json",
    },
}


def stringify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role") or item.get("from") or "user"
                content = item.get("content") or item.get("value") or ""
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        for key in ("content", "text", "prompt", "question", "instruction", "chosen"):
            if key in value:
                return stringify(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_prompt(row):
    for key in (
        "question",
        "problem",
        "prompt",
        "instruction",
        "query",
        "input",
        "chosen",
        "messages",
        "conversation",
        "conversations",
    ):
        if key in row and row[key] is not None:
            text = stringify(row[key]).strip()
            if text:
                return text
    return stringify(row).strip()


def build_chat(prompt):
    return [{"role": "user", "content": prompt}]


def load_local_dataset(root, spec):
    base = root / spec["path"]
    files = []
    for pattern in spec["files"]:
        files.extend(sorted(str(path) for path in base.glob(pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched for {base}")
    if spec["kind"] == "json":
        rows = []
        for file_name in files:
            with gzip.open(file_name, "rt") as f:
                for line in f:
                    rows.append(json.loads(line))
        return Dataset.from_list(rows)
    ds = load_dataset(spec["kind"], data_files=files, split="train")
    return ds


def convert_one(name, root, out_dir, train_size, val_size):
    ds = load_local_dataset(root, DATASETS[name])
    rows = []
    for i, row in enumerate(ds):
        prompt = extract_prompt(row)
        if not prompt:
            continue
        rows.append(
            {
                "data_source": name,
                "prompt": build_chat(prompt),
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"index": i, "source": name},
            }
        )
        if len(rows) >= train_size + val_size:
            break
    if len(rows) < train_size:
        raise RuntimeError(f"{name} only yielded {len(rows)} usable prompts")

    ds_dir = out_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(rows[:train_size])
    val = pd.DataFrame(rows[train_size : train_size + val_size] or rows[: min(val_size, len(rows))])
    train.to_parquet(ds_dir / "train.parquet", index=False)
    val.to_parquet(ds_dir / "val.parquet", index=False)
    print(f"{name}: train={len(train)} val={len(val)} -> {ds_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/mnt/L202500425/hwh/dataset")
    parser.add_argument("--output-dir", default="/mnt/L202500425/hwh/dataset/verl_shadowrl_bench")
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--datasets", nargs="*", default=sorted(DATASETS))
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    for name in args.datasets:
        convert_one(name, root, out_dir, args.train_size, args.val_size)


if __name__ == "__main__":
    main()
