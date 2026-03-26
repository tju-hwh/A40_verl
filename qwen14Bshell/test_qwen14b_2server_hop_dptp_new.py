#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BASE_SCRIPT = Path("/root/model/test_qwen8b_2server_hop_dptp_new.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("qwen14b_hop_new", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load base script: {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _inject_defaults(argv: list[str]) -> list[str]:
    out = list(argv)

    def _has(flag: str) -> bool:
        return flag in out

    def _add_default(flag: str, value: str) -> None:
        if not _has(flag):
            out.extend([flag, value])

    _add_default("--model-path", "/root/model/Qwen-14B")
    _add_default("--runtime-dir",
                 "/tmp/qwen14b_2server_hop_dptp_serve"
                 if "--serve-only" in out else
                 "/tmp/qwen14b_2server_hop_dptp_run")
    return out


def main() -> None:
    mod = _load_module()
    sys.argv = [__file__, *_inject_defaults(sys.argv[1:])]
    mod.main()


if __name__ == "__main__":
    main()
