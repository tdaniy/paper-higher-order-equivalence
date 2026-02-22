#!/usr/bin/env python3
"""Sampling-fraction simulation: coverage error vs sampling fraction p.

Uses NumPy Generator(Philox) with SeedSequence.spawn for deterministic,
order-invariant RNG streams. Produces a CSV and optionally a plot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
from dataclasses import dataclass
import time
from typing import List, Tuple

import numpy as np
from numpy.random import Generator, Philox, SeedSequence

from repro_utils import build_run_paths, compute_run_id, ensure_dir, write_manifest

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


def skewness(xs: np.ndarray) -> float:
    if xs.size == 0:
        return float("nan")
    mean = float(xs.mean())
    m2 = float(np.mean((xs - mean) ** 2))
    if m2 <= 0:
        return 0.0
    m3 = float(np.mean((xs - mean) ** 3))
    return m3 / (m2 ** 1.5)


def stable_hash_int(tag: str) -> int:
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def scenario_tag(
    *,
    n: int,
    p: float,
    b: int,
    r_skew: int,
    r_cov: int,
    alpha: float,
    delta: float,
    mu: float,
    sigma: float,
) -> str:
    return (
        f"sampling_fraction|n={n}|p={p:.8f}|b={b}|r_skew={r_skew}|r_cov={r_cov}|"
        f"alpha={alpha:.6f}|delta={delta:.6f}|mu={mu:.6f}|sigma={sigma:.6f}"
    )


def simulate_one_population(
    y0: np.ndarray,
    y1: np.ndarray,
    n1: int,
    n0: int,
    tau: float,
    r_skew: int,
    r_cov: int,
    alpha: float,
    rng_skew: Generator,
    rng_cov: Generator,
) -> Tuple[float, CoverageStats, CoverageStats, CoverageStats]:
    n = y0.size

    total_y0 = float(y0.sum())
    total_y0_sq = float(np.square(y0).sum())

    def draw_stat(rng: Generator) -> Tuple[float, float, float]:
        treated = rng.choice(n, size=n1, replace=False)
        y1_t = y1[treated]
        y0_t = y0[treated]
        sum_y1_t = float(y1_t.sum())
        sumsq_y1_t = float(np.square(y1_t).sum())
        sum_y0_t = float(y0_t.sum())
        sumsq_y0_t = float(np.square(y0_t).sum())

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
    if r_skew > 0:
        t_vals = np.empty(r_skew, dtype=float)
        for i in range(r_skew):
            _, _, t_stat = draw_stat(rng_skew)
            t_vals[i] = t_stat
        gamma_hat = skewness(t_vals)
        t_vals.sort()
    else:
        t_vals = np.array([], dtype=float)
        gamma_hat = float("nan")
    if t_vals.size == 0:
        q_low = Z_LOW
        q_high = Z_HIGH
    else:
        lo_idx = max(0, int((alpha / 2) * t_vals.size) - 1)
        hi_idx = min(t_vals.size - 1, int((1 - alpha / 2) * t_vals.size))
        q_low = t_vals[lo_idx]
        q_high = t_vals[hi_idx]

    # Coverage evaluation
    gauss = CoverageStats()
    cf = CoverageStats()
    calib = CoverageStats()

    z_low_cf = Z_LOW + (Z_LOW * Z_LOW - 1.0) * gamma_hat / 6.0
    z_high_cf = Z_HIGH + (Z_HIGH * Z_HIGH - 1.0) * gamma_hat / 6.0

    for _ in range(r_cov):
        tau_hat, vhat, _ = draw_stat(rng_cov)
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

        tag = scenario_tag(
            n=n,
            p=p,
            b=b,
            r_skew=r_skew,
            r_cov=r_cov,
            alpha=alpha,
            delta=delta,
            mu=mu,
            sigma=sigma,
        )
        scenario_ss = SeedSequence(entropy=[seed, stable_hash_int(tag)])
        pop_seqs = scenario_ss.spawn(b)

        for pop_ss in pop_seqs:
            ss_pop, ss_skew, ss_cov = pop_ss.spawn(3)
            rng_pop = Generator(Philox(ss_pop))
            rng_skew = Generator(Philox(ss_skew))
            rng_cov = Generator(Philox(ss_cov))

            y0 = rng_pop.lognormal(mean=mu, sigma=sigma, size=n).astype(float, copy=False)
            y1 = y0 + delta
            tau = delta

            gamma_hat, gauss, cf, calib = simulate_one_population(
                y0, y1, n1, n0, tau, r_skew, r_cov, alpha, rng_skew, rng_cov
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
    ensure_dir(os.path.dirname(path))
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
        import matplotlib
        matplotlib.use("Agg")
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
    ensure_dir(os.path.dirname(path))
    plt.savefig(path, dpi=200)


def append_summary(path: str, row: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                ["run_id", "config", "base_seed", "outputs_path", "plots_path", "logs_path", "notes"]
            )
        writer.writerow(row)


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
    parser.add_argument("--config", type=str, help="optional config file for run ID hashing")
    parser.add_argument(
        "--p-grid",
        type=float,
        nargs="+",
        default=[0.2, 0.35, 0.5, 0.65, 0.8, 0.9],
        help="sampling fractions",
    )
    parser.add_argument("--run-id", type=str, help="optional run ID (overrides config-based hash)")
    base_dir = os.path.dirname(__file__)
    repro_root = os.path.abspath(os.path.join(base_dir, ".."))
    outputs_root = os.path.join(repro_root, "outputs", "sampling_fraction")
    plots_root = os.path.join(repro_root, "plots", "sampling_fraction")
    parser.add_argument("--csv", type=str, help="output CSV path")
    parser.add_argument("--plot", type=str, help="output plot path (PNG)")
    parser.add_argument("--logy", action="store_true", help="log-scale y-axis for coverage error")
    args = parser.parse_args()
    run_id = args.run_id
    if run_id is None and args.config:
        run_id = compute_run_id(args.config, args.seed)
    if run_id is None:
        run_id = f"run_manual_seed{args.seed}"
    args.run_id = run_id
    outputs_root = os.path.join(outputs_root, run_id, "tables")
    plots_root = os.path.join(plots_root, run_id, "figs")
    if not args.csv:
        args.csv = os.path.join(outputs_root, "sampling_fraction_results.csv")
    if not args.plot:
        args.plot = os.path.join(plots_root, "sampling_fraction_results.png")
    return args


def main() -> None:
    args = parse_args()
    base_dir = os.path.dirname(__file__)
    repro_root = os.path.abspath(os.path.join(base_dir, ".."))
    paths = build_run_paths(repro_root, "sampling_fraction", args.run_id)
    ensure_dir(paths.outputs_dir)
    ensure_dir(paths.plots_dir)
    ensure_dir(paths.logs_dir)
    spawn_keys = [
        {
            "p": float(p),
            "hash": stable_hash_int(
                scenario_tag(
                    n=args.n,
                    p=float(p),
                    b=args.b,
                    r_skew=args.r_skew,
                    r_cov=args.r_cov,
                    alpha=args.alpha,
                    delta=args.delta,
                    mu=args.lognormal_mu,
                    sigma=args.lognormal_sigma,
                )
            ),
        }
        for p in args.p_grid
    ]
    write_manifest(
        os.path.join(paths.logs_dir, "manifest.json"),
        run_id=args.run_id,
        config_path=args.config,
        base_seed=args.seed,
        spawn_keys=spawn_keys,
        command=sys.argv,
        repo_root=os.path.abspath(os.path.join(repro_root, "..")),
    )
    for target_dir in (os.path.dirname(paths.outputs_dir), os.path.dirname(paths.plots_dir)):
        write_manifest(
            os.path.join(target_dir, "manifest.json"),
            run_id=args.run_id,
            config_path=args.config,
            base_seed=args.seed,
            spawn_keys=spawn_keys,
            command=sys.argv,
            repo_root=os.path.abspath(os.path.join(repro_root, "..")),
        )
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
    write_csv(args.csv, rows)
    try_plot(rows, args.plot, log_scale=args.logy)
    summary_path = os.path.join(repro_root, "outputs", "sampling_fraction", "summary.csv")
    append_summary(
        summary_path,
        [
            args.run_id,
            os.path.relpath(args.config, repro_root) if args.config else "",
            str(args.seed),
            os.path.relpath(os.path.dirname(paths.outputs_dir), repro_root),
            os.path.relpath(os.path.dirname(paths.plots_dir), repro_root),
            os.path.relpath(paths.logs_dir, repro_root),
            "completed",
        ],
    )
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
