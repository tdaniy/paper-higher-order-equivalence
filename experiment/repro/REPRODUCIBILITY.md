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
Module runs print periodic progress lines (once per minute) showing global task progress and ETA.

## C5 Strong Violation Confirmation
The cluster leverage failure mode is supported when leverage is fixed at a large share and the largest cluster is always treated.

| run_id | violation_share | forced_big_cluster | design | outcome | mean error (calibrated) |
| --- | --- | --- | --- | --- | --- |
| run_cluster_violation_strongest_seed20260222 | 0.80 | true | violation | continuous | 0.00356 |
| run_cluster_violation_strongest_seed20260222 | 0.80 | true | regime_a | continuous | 0.00069 |
| run_cluster_violation_strongest_seed20260222 | 0.80 | true | regime_b | continuous | 0.00094 |

Plot: `plots/cluster/run_cluster_violation_strongest_seed20260222/figs/cluster_coverage.png`

## C6 Targeted Stability Checks (N=800)
One-sided coverage at N=800 for the high-MC run and the high-R (assignment-noise) run.

| run_id | method | coverage | error | mcse |
| --- | --- | --- | --- | --- |
| run_one_sided_highmc5x_seed20260222 | gaussian_one_sided | 0.950030 | 0.000030 | 0.000109 |
| run_one_sided_highmc5x_seed20260222 | cornish_fisher_one_sided | 0.949961 | 0.000039 | 0.000109 |
| run_one_sided_highmc5x_seed20260222 | calibrated_one_sided | 0.950707 | 0.000707 | 0.000108 |
| run_one_sided_n800_highR_seed20260222 | gaussian_one_sided | 0.949701 | 0.000300 | 0.000155 |
| run_one_sided_n800_highR_seed20260222 | cornish_fisher_one_sided | 0.949306 | 0.000694 | 0.000155 |
| run_one_sided_n800_highR_seed20260222 | calibrated_one_sided | 0.949832 | 0.000168 | 0.000154 |
| run_one_sided_n800_highR_highB600_seed20260222 | gaussian_one_sided | 0.949797 | 0.000203 | 0.000089 |
| run_one_sided_n800_highR_highB600_seed20260222 | cornish_fisher_one_sided | 0.949733 | 0.000267 | 0.000089 |
| run_one_sided_n800_highR_highB600_seed20260222 | calibrated_one_sided | 0.949790 | 0.000210 | 0.000089 |

## C2/C3 Rate-Slope Diagnostics (128x MC)
Runs:
- `run_parity_highmc128x_seed20260222`
- `run_cra_sampling_highmc128x_seed20260222`

Rate-slope estimates (from `outputs/master/<run_id>/tables/rate_slopes.csv`, largest 4 N points):

| module | design | outcome | method | scale | slope | slope_low | slope_high | n_points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parity | parity_holds | continuous | calibrated | error | -0.2746 | -0.5540 | -0.1840 | 4 |
| parity | parity_fails | continuous | calibrated | error | 0.7516 | 0.4176 | 0.6445 | 4 |
| cra_sampling | f=0.5 | continuous | calibrated | error | -1.0353 | -0.3770 | 0.2212 | 4 |
| cra_sampling | near-census c=1 | continuous | calibrated | error | -0.7564 | -1.2062 | -0.5132 | 4 |
| cra_sampling | near-census c=2 | continuous | calibrated | error | 1.5660 | 0.5422 | 0.7686 | 4 |

Noise-floor check (calibrated coverage error vs MCSE, from `master_table.csv`):
- Parity, parity_holds: error range 2.38e-4 to 4.08e-4; MCSE ~1.52e-4; err/MCSE ~1.6 to 2.7.
- Parity, parity_fails: error range 5.6e-5 to 4.52e-4; MCSE ~1.52e-4; err/MCSE ~0.37 to 2.96 (median ~0.47).
- CRA, f=0.5: error range 2.85e-5 to 2.02e-4; MCSE ~1.36e-4; err/MCSE ~0.21 to 1.49 (median ~0.78).
- CRA, near-census c=1: error range 6.48e-5 to 5.52e-4; MCSE ~1.36e-4; err/MCSE ~0.48 to 4.03 (median ~2.13).
- CRA, near-census c=2: error range 4.26e-5 to 7.07e-4; MCSE ~1.36e-4; err/MCSE ~0.31 to 5.23 (median ~1.75).

Implication for paper: The theoretical claims are unaffected, but the C2/C3 simulation evidence is currently noise-limited; rate-slope diagnostics should be treated as inconclusive until errors are clearly above MCSE (e.g., higher MC, larger N, or a harder DGP).

## C2 Parity 512x Extended N (High-MC)
Run:
- `run_parity_highmc512x_extB_seed20260222`

Config summary:
- `parity_highmc512x_extB.toml`
- N grid: 200, 400, 800, 1200, 1600, 2000, 2400
- B=1280, r_skew=12800, r_cov=25600
- Runtime: ~3h 34m (8 workers)

Summary (calibrated method):
- Log–log slope of coverage error vs N using all 7 points:
  - parity_holds: −0.22
  - parity_fails: −0.44
- Parity-fails is broadly consistent with O(m_N^{-1/2}).
- Parity-holds remains shallow and close to the MCSE in multiple points; evidence for O(m_N^{-1}) is still weak.
- Error/MCSE ranges:
  - parity_holds: ~0.07–1.98
  - parity_fails: ~0.53–2.82

## C2 Parity-Holds Stress Tests (Harder Symmetric DGPs)
Purpose: increase the second-order signal under parity-holds while preserving symmetry.

### Run A: heavy-tailed symmetric t (df=3)
Run:
- `run_parity_holds_hard_256x_extB_seed20260222`

Config:
- `parity_holds_hard_256x_extB.toml`
- symmetric_dgp = `t`, t_df = 3.0, delta = 0.5
- N grid: 200, 400, 800, 1200, 1600, 2000, 2400
- B=640, r_skew=6400, r_cov=12800
- Runtime: ~59 min (6 workers)

Summary (calibrated parity_holds):
- Log–log slope of coverage error vs N: **−0.30**
- Error/MCSE range: ~0.41–3.27

### Run B: heavier-tailed symmetric t (df=2.5) + larger delta
Run:
- `run_parity_holds_harder_256x_extB_seed20260222`

Config:
- `parity_holds_harder_256x_extB.toml`
- symmetric_dgp = `t`, t_df = 2.5, delta = 1.0
- N grid: 200, 400, 800, 1200, 1600, 2000, 2400
- B=640, r_skew=6400, r_cov=12800
- Runtime: ~1h 7m (6 workers)

Summary (calibrated parity_holds):
- Log–log slope of coverage error vs N: **+0.05**
- Error/MCSE range: ~0.47–3.53

Conclusion: even with heavier tails and larger delta, parity-holds does not exhibit a clear O(m_N^{-1}) decay; errors remain close to MCSE for multiple N values, so slope estimates remain unstable.

### Run C: spike-symmetric DGP (high-signal)
Run:
- `run_parity_holds_spike_256x_extB_seed20260222`

Config:
- `parity_holds_spike_256x_extB.toml`
- symmetric_dgp = `spike` (small-probability large-magnitude symmetric spikes), delta = 1.0
- N grid: 200, 400, 800, 1200, 1600, 2000, 2400
- B=640, r_skew=6400, r_cov=12800
- Runtime: ~1h 8m (6 workers)

Summary (calibrated parity_holds):
- Log–log slope of coverage error vs N (tail 4 points): **−0.63** (MCSE band: −0.71 to −0.57)
- Last-segment slope (N=2000→2400): **−0.99**
- Error/MCSE range: ~7.24–97.32 (signal is well above MCSE across N)

Interpretation: the spike DGP produces a stronger signal, and the last segment is consistent with O(m_N^{-1}), but the tail slope over multiple points remains shallower than −1.

### Next Steps (Parity-Holds Rate)
To credibly claim O(m_N^{-1}), extend the N grid upward and re-estimate slopes over a wider tail window (at least 5 tail points). The goal is to confirm whether the near −1 *local* slope observed at N=2000→2400 persists across higher N values (trend confirmation).

Note: the parity scaling plot (`parity_scaling.png`, sqrt(N)·|err|) would especially benefit from an extended N-grid to make the tail trend clearer and more diagnostic.

Additional note: a few more large-N points (e.g., 3–4 additional grid values) would improve trend diagnostics across the parity plots, especially for tail-slope confirmation.

Pending item: the CRA 512× extended run (`run_cra_sampling_highmc512x_extB_seed20260222`) remains on the agenda and should be executed after the parity tail extension to complete the high-MC rate diagnostics.

## C7 Objective Bayes (Faithful Missing-Mass Posterior)
Run:
- `run_objective_bayes_stage2_B30_R200_S2000_seed20260222`

Faithful posterior construction:
- Missing-mass finite-population draws via Pólya-urn / Dirichlet-multinomial for the unobserved units in each arm.
- This matches the paper’s benchmark of Neyman-conservative variance with the missing-mass factor.

Key results (posterior variance ratio and endpoint alignment vs pivot):

| design | N | variance_ratio | endpoint_diff |
| --- | --- | --- | --- |
| CRA | 200 | 0.9797 | 0.5362 |
| CRA | 400 | 0.9895 | 0.3896 |
| CRA | 800 | 0.9952 | 0.2778 |
| stratified | 200 | 0.9799 | 0.3643 |
| stratified | 400 | 0.9907 | 0.2542 |
| stratified | 800 | 0.9944 | 0.1670 |

Runtime note: the faithful missing-mass posterior is substantially more expensive; the Stage-2 run above took ~1h 17m on this machine.
