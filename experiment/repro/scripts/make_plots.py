#!/usr/bin/env python3
"""Entry point scaffold for reproducible plot generation."""
from __future__ import annotations

import argparse
import os
import sys

from repro_utils import build_run_paths, compute_run_id, ensure_dir, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible plot generator (scaffold)")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--experiment", required=True, help="experiment name (e.g., rate_grid)")
    parser.add_argument("--seed", type=int, required=True, help="base RNG seed")
    parser.add_argument("--run-id", help="override run ID (optional)")
    parser.add_argument("--dry-run", action="store_true", help="only create run folders + manifest")
    return parser.parse_args()


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

    if args.dry_run:
        print(f"Dry run complete: {run_id}")
        return

    raise SystemExit("Plot generation logic not implemented yet. Add plotting code here.")


if __name__ == "__main__":
    main()
