#!/usr/bin/env python3
"""Entry point scaffold for reproducible simulations."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List

from repro_utils import build_run_paths, compute_run_id, ensure_dir, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible simulation runner (scaffold)")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--experiment", required=True, help="experiment name (e.g., rate_grid)")
    parser.add_argument("--seed", type=int, required=True, help="base RNG seed")
    parser.add_argument("--run-id", help="override run ID (optional)")
    parser.add_argument("--dry-run", action="store_true", help="only create run folders + manifest")
    return parser.parse_args()


def append_summary(summary_path: str, row: List[str]) -> None:
    ensure_dir(os.path.dirname(summary_path))
    exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                ["run_id", "config", "base_seed", "outputs_path", "plots_path", "logs_path", "notes"]
            )
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        raise SystemExit(f"Config not found: {config_path}")

    base_dir = os.path.dirname(__file__)
    repro_root = os.path.abspath(os.path.join(base_dir, ".."))

    run_id = args.run_id or compute_run_id(config_path, args.seed)
    paths = build_run_paths(repro_root, args.experiment, run_id)

    ensure_dir(paths.outputs_dir)
    ensure_dir(paths.plots_dir)
    ensure_dir(paths.logs_dir)

    manifest_path = os.path.join(paths.logs_dir, "manifest.json")
    write_manifest(
        manifest_path,
        run_id=run_id,
        config_path=config_path,
        base_seed=args.seed,
        spawn_keys=None,
        command=sys.argv,
        repo_root=os.path.abspath(os.path.join(repro_root, "..")),
    )
    # Mirror manifest into outputs/plots directories (policy requirement)
    for target_dir in (os.path.dirname(paths.outputs_dir), os.path.dirname(paths.plots_dir)):
        write_manifest(
            os.path.join(target_dir, "manifest.json"),
            run_id=run_id,
            config_path=config_path,
            base_seed=args.seed,
            spawn_keys=None,
            command=sys.argv,
            repo_root=os.path.abspath(os.path.join(repro_root, "..")),
        )

    summary_path = os.path.join(repro_root, "outputs", args.experiment, "summary.csv")
    append_summary(
        summary_path,
        [
            run_id,
            os.path.relpath(config_path, repro_root),
            str(args.seed),
            os.path.relpath(os.path.dirname(paths.outputs_dir), repro_root),
            os.path.relpath(os.path.dirname(paths.plots_dir), repro_root),
            os.path.relpath(paths.logs_dir, repro_root),
            "dry-run" if args.dry_run else "pending simulation",
        ],
    )

    if args.dry_run:
        print(f"Dry run complete: {run_id}")
        return

    raise SystemExit("Simulation logic not implemented yet. Add experiment code here.")


if __name__ == "__main__":
    main()
