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
For long runs, the runner now checkpoints after each module and skips missing config sections instead of failing.
Module-only runners (e.g., `scripts/run_cra_sampling_only.py`) are available to avoid unrelated-module crashes.

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

## CRA 512× Tail-Only Extension (2026-02-24)
Goal: reduce continuous heat exposure while extending the CRA tail window (N=3600–4800).

Config:
- `configs/cra_sampling_highmc512x_extB_tail.toml`

Run attempt:
- `run_cra_sampling_highmc512x_extB_tail_seed20260222_hash680b2fef`
- **CRA tasks completed (32/32), but the process crashed afterward** due to a missing `parity` section in the config (`KeyError: 'parity'`).
- Result: **no `outputs/master/<run_id>/tables/master_table.csv`** was written, so plots could not be generated.

### Safeguards Added (Post‑mortem)
To prevent long-run data loss:
1. **Module guard + checkpointing** in `scripts/run_sim.py`:
   - Missing config sections are skipped (no hard failure).
   - The master table is written after each module and on exceptions.
2. **CRA-only runner**: `scripts/run_cra_sampling_only.py`
   - Runs only the CRA module and writes the master table directly.

Recommended rerun (safe):
```
REPRO_WORKERS=8 UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl-cache \
  uv run python scripts/run_cra_sampling_only.py \
  --config configs/cra_sampling_highmc512x_extB_tail.toml
```

## Deterministic Parity Experiment (Option 1)
Decision: reformulate the parity experiment to eliminate outer-loop population noise by fixing a deterministic finite population per N and pushing Monte Carlo effort into assignment randomness. This targets the asymptotic rate claims directly and avoids MCSE-limited tails.

Implementation notes:
- New deterministic generator (symmetric spike for parity-holds; lognormal quantiles for parity-fails).
- Optional antithetic assignment pairing at f=0.5 to reduce variance.
- New runner: `scripts/run_parity_deterministic.py`.
- New plots support: `make_plots.py` now handles `module=parity_det`.

Proposed config:
- `configs/parity_deterministic_highsignal_tailN.toml`
- Tail-focused N grid: 800–6400
- r_skew=10000, r_cov=50000
- parity_holds_f=0.5, parity_fails_f=0.5
- antithetic_assignments=true
- spike_prob=0.005, spike_scale=80.0 (high-signal symmetric)
- parity_fails_lognormal_sigma=1.8 (high-skew)

Run command (example):
```
REPRO_WORKERS=12 MPLCONFIGDIR=/tmp/mpl-cache UV_CACHE_DIR=/tmp/uv-cache \
  uv run python scripts/run_parity_deterministic.py \
  --config configs/parity_deterministic_highsignal_tailN.toml \
  --run-id run_parity_deterministic_highsignal_tailN_seed20260223
```

## Deterministic Parity Periodicity Diagnostics (2026-02-23)
Goal: test whether the non‑monotone tail behavior is a periodic/rounding artifact of the deterministic spike population.

### A. Baseline periodicity check (rounding spikes)
Config:
- `configs/parity_deterministic_periodicity_diag_v3.toml`
- N grid: 3000..3800 by 50
- r_cov=200000, r_skew=20000
- spike_prob=0.002, spike_scale=200.0, spike_mode=rounding

Run:
- `run_parity_deterministic_periodicity_diag_v3_seed20260223_hashb7264c55`

Finding (gaussian method):
- Parity‑holds error exhibits a **sharp step at N=3500**, flipping from about +0.018–0.019 to about −0.015–0.017.
- Root cause: `spike_pairs = round((N/2)*spike_prob)` jumps from 3 to 4 at N=3500.

### B. Smooth spike mixture (remove rounding artifact)
Config:
- `configs/parity_deterministic_periodicity_diag_v3_smooth.toml`
- spike_mode = `"smooth"` (fractional spike mass)

Run:
- `run_parity_deterministic_periodicity_diag_v3_smooth_seed20260223_hash6cedad3c`

Finding (gaussian method):
- The step at N=3500 disappears.
- Parity‑holds error **drifts smoothly** and crosses zero around N≈3550; no mod‑100/mod‑200 periodicity.

### C. Randomized spike assignment (same spike_prob/scale)
Config:
- `configs/parity_deterministic_periodicity_diag_v3_random.toml`
- spike_mode = `"randomized"`

Runs (3 seeds):
- `run_parity_deterministic_periodicity_diag_v3_random_seed20260223_hasha157a417`
- `run_parity_deterministic_periodicity_diag_v3_random_seed20260224_hasha157a417`
- `run_parity_deterministic_periodicity_diag_v3_random_seed20260225_hasha157a417`

Finding (gaussian method):
- Parity‑holds still shows a **downward drift** vs N in every seed.
  - Per‑seed linear slopes: −4.94e‑05, −2.82e‑05, −2.47e‑05 per N.
  - Mean slope across seeds: −3.41e‑05 per N.
  - Errors span roughly +0.049 to −0.019 across the grid (crossing zero).
- Parity‑fails remains stable with near‑zero slope (~1e‑06 per N), error ≈ −0.039.

Conclusion: the **rounding artifact is real**, but even after removing it (smooth or randomized spikes), the parity‑holds drift persists. The irregularity appears **structural to the DGP**, not periodicity.

### Recommendations (next steps)
1. **Redesign DGP to a fully smooth, randomized population** (no deterministic quantiles/spikes), e.g.:
   - Draw y0 i.i.d. from a smooth distribution (e.g., standardized Student‑t or skew‑normal), then fix that population per N (randomized once per N, but not quantile‑based).
   - Use multiple seeds to verify stability and avoid deterministic lattice effects.
2. **Switch to a regime where theory approximations are expected to be accurate**:
   - Focus on parity‑fails or CRA sampling regimes where the observed slopes already align with O(N^{-1/2}) or O(N^{-1}).
   - Document parity‑holds as unsupported under current DGPs rather than forcing asymptotics.

## Option A: Smooth Randomized DGP (Symmetric t + Lognormal) (2026-02-23)
Goal: remove deterministic grid artifacts by drawing a **random smooth population per N**, then fixing it for assignment MC.

Config:
- `configs/parity_deterministic_periodicity_diag_optionA.toml`
- parity_holds_dgp = `symmetric_t`, t_df = 3.0 (standardized)
- parity_fails_dgp = `lognormal`, sigma = 1.8 (standardized)
- N grid: 3000..3800 by 50
- r_cov=200000, r_skew=20000, antithetic=true

Runs (3 seeds):
- `run_parity_deterministic_periodicity_diag_optionA_seed20260223_hash05f7fb49`
- `run_parity_deterministic_periodicity_diag_optionA_seed20260224_hash05f7fb49`
- `run_parity_deterministic_periodicity_diag_optionA_seed20260225_hash05f7fb49`

Findings (gaussian method):
- Parity‑holds errors are **near zero and flat** across N; per‑seed slopes are ~0.
- Parity‑fails remains negatively biased; no clean decay over this narrow window.

### Extended N test (3000–12000 by 1000)
Config:
- `configs/parity_deterministic_optionA_extN.toml`

Runs (3 seeds):
- `run_parity_deterministic_optionA_extN_seed20260223_hashde32ad7a`
- `run_parity_deterministic_optionA_extN_seed20260224_hashde32ad7a`
- `run_parity_deterministic_optionA_extN_seed20260225_hashde32ad7a`

Plots:
- `plots/parity_det/<run_id>/figs/parity_coverage.png`
- `plots/parity_det/<run_id>/figs/parity_rate_N.png`
- `plots/parity_det/<run_id>/figs/parity_rate_sqrtN.png`
- `plots/parity_det/<run_id>/figs/parity_scaling.png`

Summary:
- Parity‑holds (gaussian): mean error ≈ 0 with **no visible trend**; slope ≈ −4e‑08 per N.
- Parity‑fails (gaussian): negative bias persists, but slope ≈ 0 on this window.
- Calibrated method: errors are small and flat for both regimes (as expected after removing leading skew term).

### Conclusion (Decision)
At the current compute scale, **parity‑holds is not empirically resolvable** for these DGPs. The error signal is too small and non‑monotone to support an O(N^{-1}) rate claim.
We will **document this limitation** rather than push further brute‑force runs.

## Parity Tail Extension (2026-02-23)
Decision: extend the parity N grid with `2800, 3200, 3600, 4000, 4400, 4800` using a tail-only run, then merge with the prior 512× extB results to avoid recomputing existing N values. Tail slope diagnostics are computed with `k=5` largest-N points.

Run (tail only):
- config: `configs/parity_highmc512x_extB_tail.toml`
- run_id: `run_parity_highmc512x_extB_tail_seed20260223`
- N grid: 2800, 3200, 3600, 4000, 4400, 4800
- B=1280, r_skew=12800, r_cov=25600, delta=0.5
- workers: 12
- runtime: ~3h 05m 34s
- master table: `outputs/master/run_parity_highmc512x_extB_tail_seed20260223/tables/master_table.csv`

Merged (full grid for plots/slopes):
- combined run_id: `run_parity_highmc512x_extB_combo_seed20260223`
- master table: `outputs/master/run_parity_highmc512x_extB_combo_seed20260223/tables/master_table.csv`
- plots: `plots/parity/run_parity_highmc512x_extB_combo_seed20260223/figs/`
  - `parity_coverage.png`
  - `parity_rate_N.png`
  - `parity_rate_sqrtN.png`
  - `parity_scaling.png`
- tail slope (k=5): `outputs/master/run_parity_highmc512x_extB_combo_seed20260223/tables/parity_tail_slopes_k5.csv`

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
