# Reproducible Environment

This folder is a self-contained, minimal reproduction bundle for the paper’s simulations and plots.
It is designed to be archived (GitHub/Zenodo) and re-run deterministically.

## Contents
- `scripts/`: runnable entry points for simulations and plot generation
- `configs/`: immutable run configurations (TOML/YAML/JSON)
- `data/`: immutable input data (if any)
- `outputs/`: generated numeric outputs (CSV/NPZ)
- `plots/`: generated figures
- `logs/`: run metadata and provenance logs
- `figure_map.json`: map from paper figure IDs to run IDs and plot files
- `REPRODUCIBILITY.md`: exact steps to regenerate figures
- `CHANGELOG.md`: experiment versioning notes
- `Makefile`: canonical run targets
- `.python-version`, `pyproject.toml`, `uv.lock`: pinned environment

## Requirements
- `uv` installed and on PATH
- Python version pinned in `.python-version` (e.g., 3.12.3)

## Setup
```bash
uv sync --frozen
```

If cache initialization fails due to permissions:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --frozen
```

If `uv.lock` is missing, generate it once with network access:
```bash
UV_CACHE_DIR=/tmp/uv-cache uv lock
```

## Reproduction Steps

### 1. Run simulations
```bash
uv run python scripts/run_sim.py --config configs/base.toml
```

### 2. Generate plots (optional)
```bash
uv run python scripts/make_plots.py --config configs/base.toml
```

Outputs will be written to `outputs/` and `plots/` (one subfolder per run).

## RNG and Determinism
- RNG uses `numpy.random.Generator(Philox)` and `SeedSequence.spawn`.
- The base seed and per-scenario spawn keys are logged to `logs/`.
- Determinism is guaranteed given:
  - identical Python + NumPy versions (as locked in `uv.lock`)
  - identical code + config
  - identical seed

## Provenance Logging
Each run records:
- OS + CPU
- Python, NumPy, uv versions
- Git commit (if applicable)
- Full command line
- Base seed and spawn keys

## Output Organization Policy
All generated artifacts must be written under `outputs/`, `plots/`, and `logs/` and never mixed.

Structure:
- `outputs/<experiment>/run_<config>_seed<seed>_hash<short>/tables/` for CSV/NPZ data
- `plots/<experiment>/run_<config>_seed<seed>_hash<short>/figs/` for figures
- `logs/run_<config>_seed<seed>_hash<short>/` for provenance metadata

Each run directory must include a `manifest.json` (or `run.json`) recording config content or checksum, seeds/spawn keys, versions, and command line.

## Notes on Reproducibility
Changing Python or NumPy versions may change random streams and numerical results.
This is why `.python-version` and `uv.lock` are required for exact reproduction.
