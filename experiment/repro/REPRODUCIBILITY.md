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

## Symflip N^-1/2 Isolation (Completed 2026-02-26)
Goal: isolate the odd N^-1/2 term in the coverage error expansion by canceling even-order terms.

Design (completed):
- Symflip skew-baseline, constant-effect DGP with one-sided coverage and CRN pairing between + and - runs.
- Lognormal skew component, g_log_sigma = 0.5, s = 1.0, tau = 0.5, f = 0.25.
- N grid: 36 log-spaced points from 800 to 100000.

Run:
- `run_skew_symflip_log36_800_100000_seed20260225_hasha5043a19`

Plot:
- `plots/skew_symflip/run_skew_symflip_log36_800_100000_seed20260225_hasha5043a19/figs/symflip_delta_error.png`

Summary:
- |Delta e(N)| follows the N^-1/2 reference line across the full grid; the N^-1 guide is too steep.
- The trend remains stable through large N, with wider MCSE bands at the tail but no slope change.

Interpretation: the symflip difference cleanly exposes the odd-order term, providing direct empirical support for the N^-1/2 component predicted by the theory. This completes the symflip experiment block.

## CRA Symflip Diagnostics (2026-02-27)
Goal: apply symflip (odd-term isolation) to CRA and assess whether |Δe(N)| follows N^-1/2 for f=0.5.

### Starter sanity check (small N + near-census)
Config:
- `configs/cra_symflip_example.toml`
- N grid: 800, 1200, 1600, 2400
- f=0.5 plus near-census (c=1,2), B=200, R=5000

Run:
- `run_cra_symflip_example_seed20260227_hashab5fc012`

Plots:
- `plots/cra_symflip/run_cra_symflip_example_seed20260227_hashab5fc012/figs/cra_symflip_delta_error.png`
- `plots/cra_symflip/run_cra_symflip_example_seed20260227_hashab5fc012/figs/cra_symflip_delta_sqrtN.png`

Finding:
- f=0.5 curve is small and noisy at this grid; near-census points sit much higher and are not slope-diagnostic (f varies with N).

### Production grid with near-census (log-spaced N up to 50k)
Config:
- `configs/cra_symflip_prod.toml`
- N grid: 2000..50000 (9 points), f=0.5 plus near-census c=1
- B=200, R=8000, g_log_sigma=0.5

Run:
- `run_cra_symflip_prod_seed20260227_hashf52d7ff8`

Plots:
- `plots/cra_symflip/run_cra_symflip_prod_seed20260227_hashf52d7ff8/figs/cra_symflip_delta_error.png`
- `plots/cra_symflip/run_cra_symflip_prod_seed20260227_hashf52d7ff8/figs/cra_symflip_delta_sqrtN.png`

Finding:
- f=0.5 series shows decline but remains non‑monotone; sqrt(N) scaling is not flat.
- Near-census points are large but still not slope-diagnostic (f changes with N).

### Production f=0.5 only (extended tail + higher MC)
Config:
- `configs/cra_symflip_prod_f05.toml`
- N grid: 3000..110000 (10 points), f=0.5 only
- B=400, R=12000, g_log_sigma=0.6

Run:
- `run_cra_symflip_prod_f05_seed20260227_hash34bdd38e`

Plots:
- `plots/cra_symflip/run_cra_symflip_prod_f05_seed20260227_hash34bdd38e/figs/cra_symflip_delta_error.png`
- `plots/cra_symflip/run_cra_symflip_prod_f05_seed20260227_hash34bdd38e/figs/cra_symflip_delta_sqrtN.png`

Finding:
- |Δe| remains non‑monotone and does not track a clean N^-1/2 slope.
- sqrt(N)*|Δe| is not flat (noticeable bump near mid/upper tail), so asymptotics are not yet cleanly resolved.

Conclusion:
CRA symflip currently does **not** deliver a stable N^-1/2 diagnostic at f=0.5, even with extended N and higher MC. This contrasts with the successful non‑CRA symflip experiment.

## Parity-Holds Spike Tail Extension (Completed 2026-02-26)
Goal: extend the parity-holds spike DGP tail to confirm O(N^-1) decay with larger N.

Config:
- `configs/parity_holds_spike_256x_extB_tail4.toml`
- Added tail points: N = 3200, 4000, 5000, 6400 (base grid up to 2400).

Run:
- `run_parity_holds_spike_256x_extB_tail4_seed20260222_hashfcb11350`

Key plot:
- `plots/parity/run_parity_holds_spike_256x_extB_tail4_seed20260222_hashfcb11350/figs/parity_error_loglog.png`

Summary:
- Parity-holds series follows the O(N^-1) guide across the extended tail.
- Parity-fails remains MCSE-limited in this design; N^-1/2 is not recoverable here.

Conclusion: the parity-holds spike experiment now cleanly demonstrates O(N^-1). Task complete.

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

### Deterministic Parity (Calibrated) Summary (2026-02-25)
Computed from `outputs/master/<run_id>/tables/master_table.csv` for `module=parity_det`, `design=parity_holds`, `method=calibrated`.
No explicit `error` column exists in these tables, so we compute `error = |coverage - 0.95|`.

| run_id | N range (pts) | error range | err/mcse range | median err/mcse |
| --- | --- | --- | --- | --- |
| run_parity_deterministic_highsignal_tailN_seed20260223 | 800..6400 (8) | 0.00102–0.00610 | 1.06–6.66 | 2.84 |
| run_parity_deterministic_highsignal_tailN_v2_seed20260223 | 800..6400 (8) | 0.00074–0.00404 | 1.51–8.63 | 3.27 |
| run_parity_deterministic_highsignal_tailN_v3_seed20260223 | 800..6400 (8) | 0.00038–0.00815 | 0.783–18.2 | 3.11 |
| run_parity_deterministic_optionA_extN_seed20260223_hashde32ad7a | 3000..12000 (10) | 4.0e-05–0.00360 | 0.0821–7.65 | 2.63 |
| run_parity_deterministic_optionA_extN_seed20260224_hashde32ad7a | 3000..12000 (10) | 0–0.00421 | 0–8.31 | 3.28 |
| run_parity_deterministic_optionA_extN_seed20260225_hashde32ad7a | 3000..12000 (10) | 0.00071–0.00438 | 1.45–9.39 | 4.77 |
| run_parity_deterministic_periodicity_diag_optionA_seed20260223_hash05f7fb49 | 3000..3800 (17) | 0.00021–0.00486 | 0.43–9.54 | 2.76 |
| run_parity_deterministic_periodicity_diag_optionA_seed20260224_hash05f7fb49 | 3000..3800 (17) | 0.00041–0.00419 | 0.845–8.96 | 4.74 |
| run_parity_deterministic_periodicity_diag_optionA_seed20260225_hash05f7fb49 | 3000..3800 (17) | 0.00017–0.00565 | 0.349–11 | 3.63 |
| run_parity_deterministic_periodicity_diag_v3_random_seed20260223_hasha157a417 | 3000..3800 (17) | 0.00043–0.00694 | 0.879–15.3 | 5.16 |
| run_parity_deterministic_periodicity_diag_v3_random_seed20260224_hasha157a417 | 3000..3800 (17) | 0.00021–0.00449 | 0.432–9.62 | 5.72 |
| run_parity_deterministic_periodicity_diag_v3_random_seed20260225_hasha157a417 | 3000..3800 (17) | 0.00021–0.00627 | 0.43–13.7 | 3.70 |
| run_parity_deterministic_periodicity_diag_v3_seed20260223_hashb7264c55 | 3000..3800 (17) | 1.0e-05–0.00368 | 0.0205–7.82 | 3.11 |
| run_parity_deterministic_periodicity_diag_v3_smooth_seed20260223_hash6cedad3c | 3000..3800 (17) | 0.00021–0.00369 | 0.432–7.84 | 2.27 |

Rate-slope estimates from `tables/rate_slopes.csv` (scale=error, last 4 N points):
- `run_parity_deterministic_highsignal_tailN_seed20260223`: slope −1.02 (low −5.19, high −0.535; n=4)
- `run_parity_deterministic_highsignal_tailN_v2_seed20260223`: slope +2.13 (low +1.54, high +3.75; n=4)
- `run_parity_deterministic_highsignal_tailN_v3_seed20260223`: slope −2.48 (low −3.79, high −1.96; n=4)
- `run_parity_deterministic_optionA_extN_seed20260223_hashde32ad7a`: slope +5.23 (low −8.68, high +0.196; n=4)
- `run_parity_deterministic_optionA_extN_seed20260224_hashde32ad7a`: slope −3.99 (low −4.77, high −2.12; n=4)
- `run_parity_deterministic_optionA_extN_seed20260225_hashde32ad7a`: slope −5.61 (low −8.86, high −4.31; n=4)

Interpretation: calibrated parity‑holds errors remain small but non‑monotone, with err/mcse often near 1 and slopes that flip sign across runs, so deterministic runs still do not yield a stable O(N^{-1}) rate signal. This is consistent with the observation that deterministic runs did not help demonstrate an N^{-1/2} decay (or any stable rate) for parity‑holds.

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

## Next Steps
- Re-run the CRA tail-only high-MC job with the **CRA-only runner** so results are saved:  
  `REPRO_WORKERS=8 uv run python scripts/run_cra_sampling_only.py --config configs/cra_sampling_highmc512x_extB_tail.toml`
- Generate plots and tail diagnostics from the CRA tail-only master table (once the rerun completes).
- Decide whether to run the **full** CRA 512× extended grid (`configs/cra_sampling_highmc512x_extB.toml`, now includes 3600–4800) or keep a tail-only extension; document the decision.
- Update the C3 near-census summary once CRA tail diagnostics are available (rate slopes + err/MCSE in the tail).
- Add explicit **C1 forced-pivot / Gaussian-shift proxy diagnostics** (QQ/distance metrics) and include their rate plots in the reproducibility report.
- Add explicit **C4 lattice jitter** before/after periodicity summary and plots to the reproducibility report.
- Expand `figure_map.json` so every paper figure maps to a run ID and artifacts (not just stress tests).
- Reconsider parity-holds using **variance-reduced MLMC/CRN** instead of brute-force N:
  - Couple N and 2N runs with shared RNG streams for population + assignments (common random numbers).
  - Estimate `Δerr = err(2N) − err(N)` directly to amplify the O(N^{-1}) term.
  - Reuse existing RNG tagging and add a small wrapper that draws paired assignments so both N and 2N share randomness where possible.
  - Report slopes from `Δerr` vs N (paired differences) rather than raw errors.
 - CRA symflip next steps (if we continue):
   - Drop near‑census permanently and focus on f=0.5 only.
   - Add 2–3 larger N points above 110k and/or run multiple seeds to average out non‑monotone tails.
   - Consider MLMC‑style paired N/2N differences for CRA to reduce variance instead of further increasing B/R.
