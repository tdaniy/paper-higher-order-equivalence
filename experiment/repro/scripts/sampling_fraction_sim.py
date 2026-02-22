#!/usr/bin/env python3
"""Sampling-fraction simulation: coverage error vs sampling fraction p.

Pure-Python (no numpy) so it runs in minimal environments. Produces a CSV and
optionally a plot if matplotlib is available.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
import time
from typing import List, Tuple

Z_LOW = -1.959963984540054
Z_HIGH = 1.959963984540054


@dataclass
class CoverageStats:
    count: int = 0
    covered: int = 0

    def add(self, ok: bool) -> None:
        self.count += 1
        if ok:
            self.covered += 1

    def rate(self) -> float:
        return self.covered / self.count if self.count else float("nan")

    def mcse(self) -> float:
        if self.count == 0:
            return float("nan")
        p = self.rate()
        return math.sqrt(max(p * (1 - p), 0.0) / self.count)


def skewness(xs: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return float("nan")
    mean = sum(xs) / n
    m2 = sum((x - mean) ** 2 for x in xs) / n
    if m2 <= 0:
        return 0.0
    m3 = sum((x - mean) ** 3 for x in xs) / n
    return m3 / (m2 ** 1.5)


def simulate_one_population(
    y0: List[float],
    y1: List[float],
    n1: int,
    n0: int,
    tau: float,
    r_skew: int,
    r_cov: int,
    alpha: float,
    rng: random.Random,
) -> Tuple[float, CoverageStats, CoverageStats, CoverageStats]:
    n = len(y0)
    idx = list(range(n))

    total_y0 = sum(y0)
    total_y0_sq = sum(v * v for v in y0)

    def draw_stat() -> Tuple[float, float]:
        treated = rng.sample(idx, n1)
        sum_y1_t = 0.0
        sumsq_y1_t = 0.0
        sum_y0_t = 0.0
        sumsq_y0_t = 0.0
        for i in treated:
            v1 = y1[i]
            v0 = y0[i]
            sum_y1_t += v1
            sumsq_y1_t += v1 * v1
            sum_y0_t += v0
            sumsq_y0_t += v0 * v0

        sum_y0_c = total_y0 - sum_y0_t
        sumsq_y0_c = total_y0_sq - sumsq_y0_t

        mean1 = sum_y1_t / n1
        mean0 = sum_y0_c / n0

        var1 = (sumsq_y1_t - n1 * mean1 * mean1) / max(n1 - 1, 1)
        var0 = (sumsq_y0_c - n0 * mean0 * mean0) / max(n0 - 1, 1)
        vhat = var1 / n1 + var0 / n0
        if vhat <= 0:
            vhat = 1e-12
        tau_hat = mean1 - mean0
        t_stat = (tau_hat - tau) / math.sqrt(vhat)
        return tau_hat, vhat, t_stat

    # Estimate skewness and quantiles from a separate randomization batch
    t_vals = []
    for _ in range(r_skew):
        _, _, t_stat = draw_stat()
        t_vals.append(t_stat)
    gamma_hat = skewness(t_vals)
    t_vals.sort()
    if not t_vals:
        q_low = Z_LOW
        q_high = Z_HIGH
    else:
        lo_idx = max(0, int((alpha / 2) * len(t_vals)) - 1)
        hi_idx = min(len(t_vals) - 1, int((1 - alpha / 2) * len(t_vals)))
        q_low = t_vals[lo_idx]
        q_high = t_vals[hi_idx]

    # Coverage evaluation
    gauss = CoverageStats()
    cf = CoverageStats()
    calib = CoverageStats()

    z_low_cf = Z_LOW + (Z_LOW * Z_LOW - 1.0) * gamma_hat / 6.0
    z_high_cf = Z_HIGH + (Z_HIGH * Z_HIGH - 1.0) * gamma_hat / 6.0

    for _ in range(r_cov):
        tau_hat, vhat, _ = draw_stat()
        se = math.sqrt(vhat)
        lo = tau_hat + Z_LOW * se
        hi = tau_hat + Z_HIGH * se
        gauss.add(lo <= tau <= hi)

        lo_cf = tau_hat + z_low_cf * se
        hi_cf = tau_hat + z_high_cf * se
        cf.add(lo_cf <= tau <= hi_cf)

        lo_cal = tau_hat - q_high * se
        hi_cal = tau_hat - q_low * se
        calib.add(lo_cal <= tau <= hi_cal)

    return gamma_hat, gauss, cf, calib


def run_simulation(
    n: int,
    b: int,
    r_skew: int,
    r_cov: int,
    p_grid: List[float],
    alpha: float,
    seed: int,
    delta: float,
    mu: float,
    sigma: float,
) -> List[dict]:
    rng = random.Random(seed)
    target = 1.0 - alpha
    results = []

    for p in p_grid:
        n1 = max(5, int(round(p * n)))
        n0 = n - n1
        if n0 < 5:
            n0 = 5
            n1 = n - n0
        p = n1 / n

        gauss_all = CoverageStats()
        cf_all = CoverageStats()
        calib_all = CoverageStats()
        gamma_vals = []

        for _ in range(b):
            y0 = [math.exp(rng.gauss(mu, sigma)) for _ in range(n)]
            y1 = [v + delta for v in y0]
            tau = delta

            gamma_hat, gauss, cf, calib = simulate_one_population(
                y0, y1, n1, n0, tau, r_skew, r_cov, alpha, rng
            )
            gamma_vals.append(gamma_hat)
            gauss_all.covered += gauss.covered
            gauss_all.count += gauss.count
            cf_all.covered += cf.covered
            cf_all.count += cf.count
            calib_all.covered += calib.covered
            calib_all.count += calib.count

        gauss_rate = gauss_all.rate()
        cf_rate = cf_all.rate()
        calib_rate = calib_all.rate()
        gauss_mcse = gauss_all.mcse()
        cf_mcse = cf_all.mcse()
        calib_mcse = calib_all.mcse()
        results.append(
            {
                "p": p,
                "method": "gaussian",
                "coverage": gauss_rate,
                "coverage_error": abs(gauss_rate - target),
                "mcse": gauss_mcse,
                "gamma_mean": sum(gamma_vals) / len(gamma_vals) if gamma_vals else float("nan"),
            }
        )
        results.append(
            {
                "p": p,
                "method": "cornish_fisher",
                "coverage": cf_rate,
                "coverage_error": abs(cf_rate - target),
                "mcse": cf_mcse,
                "gamma_mean": sum(gamma_vals) / len(gamma_vals) if gamma_vals else float("nan"),
            }
        )
        results.append(
            {
                "p": p,
                "method": "calibrated",
                "coverage": calib_rate,
                "coverage_error": abs(calib_rate - target),
                "mcse": calib_mcse,
                "gamma_mean": sum(gamma_vals) / len(gamma_vals) if gamma_vals else float("nan"),
            }
        )

    return results


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def try_plot(rows: List[dict], path: str, log_scale: bool = False) -> None:
    try:
        import os
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".mpl_cache"))
        os.environ.setdefault("MPLCONFIGDIR", cache_dir)
        import matplotlib.pyplot as plt
    except Exception:
        return

    ps = sorted({row["p"] for row in rows})
    for method in sorted({row["method"] for row in rows}):
        vals = [row["coverage_error"] for row in rows if row["method"] == method]
        errs = [row.get("mcse", 0.0) for row in rows if row["method"] == method]
        plt.plot(ps, vals, marker="o", label=method)
        if any(e > 0 for e in errs):
            lower = [max(v - e, 1e-6) for v, e in zip(vals, errs)]
            upper = [v + e for v, e in zip(vals, errs)]
            plt.fill_between(ps, lower, upper, alpha=0.15)

    plt.xlabel("Sampling fraction p")
    if log_scale:
        plt.yscale("log")
        plt.ylabel("|coverage - (1-α)| (log scale)")
    else:
        plt.ylabel("|coverage - (1-α)|")
    plt.title("Coverage error vs sampling fraction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sampling-fraction coverage simulation")
    parser.add_argument("--n", type=int, default=200, help="population size")
    parser.add_argument("--b", type=int, default=30, help="number of populations")
    parser.add_argument("--r-skew", type=int, default=400, help="assignments for skewness estimation")
    parser.add_argument("--r-cov", type=int, default=1000, help="assignments for coverage evaluation")
    parser.add_argument("--alpha", type=float, default=0.05, help="nominal alpha")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--delta", type=float, default=0.5, help="constant treatment effect")
    parser.add_argument("--lognormal-mu", type=float, default=0.0, help="lognormal mean (on log scale)")
    parser.add_argument("--lognormal-sigma", type=float, default=1.2, help="lognormal sigma (on log scale)")
    parser.add_argument(
        "--p-grid",
        type=float,
        nargs="+",
        default=[0.2, 0.35, 0.5, 0.65, 0.8, 0.9],
        help="sampling fractions",
    )
    parser.add_argument("--run-id", type=str, help="optional run ID for output organization")
    base_dir = os.path.dirname(__file__)
    repro_root = os.path.abspath(os.path.join(base_dir, ".."))
    outputs_root = os.path.join(repro_root, "outputs", "sampling_fraction")
    plots_root = os.path.join(repro_root, "plots", "sampling_fraction")
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(outputs_root, "sampling_fraction_results.csv"),
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=os.path.join(plots_root, "sampling_fraction_results.png"),
    )
    parser.add_argument("--logy", action="store_true", help="log-scale y-axis for coverage error")
    args = parser.parse_args()
    if args.run_id:
        outputs_root = os.path.join(outputs_root, args.run_id, "tables")
        plots_root = os.path.join(plots_root, args.run_id, "figs")
        if args.csv.endswith("sampling_fraction_results.csv"):
            args.csv = os.path.join(outputs_root, "sampling_fraction_results.csv")
        if args.plot.endswith("sampling_fraction_results.png"):
            args.plot = os.path.join(plots_root, "sampling_fraction_results.png")
    return args


def main() -> None:
    args = parse_args()
    start = time.time()
    rows = run_simulation(
        n=args.n,
        b=args.b,
        r_skew=args.r_skew,
        r_cov=args.r_cov,
        p_grid=args.p_grid,
        alpha=args.alpha,
        seed=args.seed,
        delta=args.delta,
        mu=args.lognormal_mu,
        sigma=args.lognormal_sigma,
    )
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot), exist_ok=True)
    write_csv(args.csv, rows)
    try_plot(rows, args.plot, log_scale=args.logy)
    # Print a concise summary
    print("p\tmethod\tcoverage\t|err|\tmcse\tgamma_mean")
    for row in rows:
        print(
            f"{row['p']:.2f}\t{row['method']}\t{row['coverage']:.4f}\t"
            f"{row['coverage_error']:.4f}\t{row.get('mcse', float('nan')):.4f}\t"
            f"{row['gamma_mean']:.4f}"
        )
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
