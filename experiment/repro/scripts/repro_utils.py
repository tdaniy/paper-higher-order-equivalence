#!/usr/bin/env python3
"""Utilities for reproducible runs (run IDs, manifests, environment logging)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional


THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def compute_config_hash(config_path: str) -> str:
    data = read_file_bytes(config_path)
    return _sha256_bytes(data)


def compute_run_id(config_path: str, base_seed: int, prefix: str = "run") -> str:
    stem = os.path.splitext(os.path.basename(config_path))[0]
    h = compute_config_hash(config_path)[:8]
    return f"{prefix}_{stem}_seed{base_seed}_hash{h}"


def _cmd_output(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def collect_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {
        "python": platform.python_version(),
    }
    try:
        import numpy as np  # type: ignore

        versions["numpy"] = np.__version__
    except Exception:
        versions["numpy"] = None
    try:
        import matplotlib  # type: ignore

        versions["matplotlib"] = matplotlib.__version__
    except Exception:
        versions["matplotlib"] = None

    versions["uv"] = _cmd_output(["uv", "--version"])
    return versions


def collect_git_info(repo_root: str) -> Dict[str, Any]:
    if not os.path.isdir(os.path.join(repo_root, ".git")):
        return {"commit": None, "dirty": None}
    commit = _cmd_output(["git", "-C", repo_root, "rev-parse", "HEAD"])
    status = _cmd_output(["git", "-C", repo_root, "status", "--porcelain"])
    dirty = bool(status) if status is not None else None
    return {"commit": commit, "dirty": dirty}


def collect_platform_info() -> Dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def collect_thread_env() -> Dict[str, Optional[str]]:
    return {key: os.environ.get(key) for key in THREAD_ENV_VARS}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


@dataclass
class RunPaths:
    run_id: str
    outputs_dir: str
    plots_dir: str
    logs_dir: str


def build_run_paths(root: str, experiment: str, run_id: str) -> RunPaths:
    outputs_dir = os.path.join(root, "outputs", experiment, run_id, "tables")
    plots_dir = os.path.join(root, "plots", experiment, run_id, "figs")
    logs_dir = os.path.join(root, "logs", run_id)
    return RunPaths(run_id=run_id, outputs_dir=outputs_dir, plots_dir=plots_dir, logs_dir=logs_dir)


def write_manifest(
    manifest_path: str,
    *,
    run_id: str,
    config_path: str,
    base_seed: int,
    spawn_keys: Optional[list[Any]],
    command: list[str],
    repo_root: str,
) -> None:
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "config_path": config_path,
        "config_hash": compute_config_hash(config_path),
        "base_seed": base_seed,
        "spawn_keys": spawn_keys,
        "command": command,
        "versions": collect_versions(),
        "git": collect_git_info(repo_root),
        "platform": collect_platform_info(),
        "thread_env": collect_thread_env(),
    }
    write_json(manifest_path, payload)
