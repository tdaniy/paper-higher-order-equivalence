#!/usr/bin/env python3
"""Run only the parity module."""
from __future__ import annotations

import argparse
import os

import tomllib

from repro_utils import compute_run_id
from run_sim import run_parity, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parity module")
    parser.add_argument("--config", required=True, help="TOML config path")
    parser.add_argument("--seed", type=int, help="override base seed")
    parser.add_argument("--run-id", help="override run ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = os.path.abspath(args.config)
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    seed = args.seed if args.seed is not None else cfg["meta"]["base_seed"]
    run_id = args.run_id or compute_run_id(config_path, seed)

    base_dir = os.path.dirname(__file__)
    repro_root = os.path.abspath(os.path.join(base_dir, ".."))

    master_rows = []
    alpha = cfg["meta"]["alpha"]
    run_parity(cfg["parity"], seed, run_id, repro_root, master_rows, config_path, alpha)

    master_path = os.path.join(repro_root, "outputs", "master", run_id, "tables", "master_table.csv")
    write_csv(master_path, master_rows)


if __name__ == "__main__":
    main()
