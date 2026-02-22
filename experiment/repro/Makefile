CONFIG ?= configs/base.toml

.PHONY: sync lock run plots determinism-check

sync:
	uv sync --frozen

lock:
	UV_CACHE_DIR=/tmp/uv-cache uv lock

run:
	uv run python scripts/run_sim.py --config $(CONFIG)

plots:
	uv run python scripts/make_plots.py --config $(CONFIG)

determinism-check:
	uv run python scripts/determinism_check.py --dir-a $(A) --dir-b $(B)
