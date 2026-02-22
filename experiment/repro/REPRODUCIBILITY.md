# Reproducibility Guide

This document provides exact steps to recreate the numerical results and figures in the paper.

## Environment Setup
```bash
uv sync --frozen
```

If cache initialization fails due to permissions:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --frozen
```

## Canonical Runs
Replace `<config>` with the desired config file (e.g., `configs/base.toml`).

```bash
uv run python scripts/run_sim.py --config <config>
uv run python scripts/make_plots.py --config <config>
```

## Figure Regeneration
The file `figure_map.json` maps each paper figure to the exact run ID and plot files.
Use it to locate the plot artifacts under `plots/<experiment>/<run_id>/figs/`.

## Determinism Check
To compare two run directories for byte-for-byte equality:
```bash
uv run python scripts/determinism_check.py --dir-a <pathA> --dir-b <pathB>
```

## Notes
Exact reproducibility requires identical Python and NumPy versions and the same seed configuration.
