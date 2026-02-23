# Experiments

This directory hosts the reproducible experiment bundle and protocol.

- `repro`: Self-contained reproducible environment (uv + pinned Python + NumPy/Philox) for archival runs.

Recent diagnostics: C2/C3 rate-slope estimates from the 128x MC runs are noise-limited in several regimes (errors at or below MCSE), so the simulation evidence for rate separation should be treated as inconclusive until errors clearly exceed MCSE.

Parity update: a 512x extended-N parity run (`run_parity_highmc512x_extB_seed20260222`) supports O(m_N^{-1/2}) under parity-fails but still shows shallow slopes under parity-holds; see `experiment/repro/REPRODUCIBILITY.md` for details.
Further parity-holds stress tests with heavier-tailed symmetric DGPs (t_df=3 and t_df=2.5, delta=1.0) remain inconclusive for O(m_N^{-1}); details are documented in `experiment/repro/REPRODUCIBILITY.md`.
