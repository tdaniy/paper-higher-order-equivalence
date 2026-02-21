#!/usr/bin/env python3
"""Rate-grid simulation: scaled coverage error vs N for selected p values.

Uses kill_shot_sim.run_simulation to compute coverage errors, then rescales
by sqrt(N) and N for rate diagnostics. Produces CSV and plots.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from typing import List

import sys

sys.path.insert(0, os.path.dirname(__file__))
import kill_shot_sim as ks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rate-grid simulation")
    parser.add_argument(
        "--n-grid",
        type=int,
        nargs="+",
        default=list(range(120, 961, 30)),
        help="list of N values",
    )
    parser.add_argument(
        "--p-grid",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 0.9],
        help="sampling fractions",
    )
    parser.add_argument("--b", type=int, default=20, help="number of populations")
    parser.add_argument("--r-skew", type=int, default=300, help="assignments for skewness estimation")
    parser.add_argument("--r-cov", type=int, default=800, help="assignments for coverage evaluation")
    parser.add_argument("--alpha", type=float, default=0.05, help="nominal alpha")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--delta", type=float, default=0.5, help="constant treatment effect")
    parser.add_argument("--lognormal-mu", type=float, default=0.0, help="lognormal mean (on log scale)")
    parser.add_argument("--lognormal-sigma", type=float, default=1.2, help="lognormal sigma (on log scale)")
    parser.add_argument("--csv", type=str, default="experiment/kill_shot_rate_grid.csv")
    parser.add_argument("--plot-sqrt", type=str, default="experiment/kill_shot_rate_grid_sqrt.png")
    parser.add_argument("--plot-linear", type=str, default="experiment/kill_shot_rate_grid_linear.png")
    parser.add_argument("--logy", action="store_true", help="log-scale y-axis for plots")
    parser.add_argument("--ylim", type=float, nargs=2, help="y-axis limits, e.g. --ylim -2 2")
    parser.add_argument("--slope-k", type=int, default=6, help="number of largest-N points for slope fit")
    parser.add_argument("--slope-csv", type=str, default="experiment/kill_shot_rate_grid_slopes.csv")
    parser.add_argument("--plot-only", action="store_true", help="skip simulation and replot from CSV")
    parser.add_argument("--input-csv", type=str, help="CSV to read when --plot-only is set")
    return parser.parse_args()


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ("p", "coverage", "coverage_error", "mcse", "gamma_mean", "err_sqrt", "err_linear"):
                if key in row and row[key] not in ("", None):
                    row[key] = float(row[key])
            if "N" in row and row["N"] not in ("", None):
                row["N"] = int(float(row["N"]))
            rows.append(row)
    return rows


def plot_scaled(rows: List[dict], path: str, scale: str, logy: bool, ylim: List[float] | None) -> None:
    try:
        import os
        os.environ.setdefault("MPLCONFIGDIR", "experiment/.mpl_cache")
        import matplotlib.pyplot as plt
    except Exception:
        return

    methods = sorted({r["method"] for r in rows})
    p_vals = sorted({float(r["p"]) for r in rows})

    plt.figure(figsize=(7, 4))
    for method in methods:
        for p in p_vals:
            subset = [r for r in rows if r["method"] == method and float(r["p"]) == p]
            subset = sorted(subset, key=lambda r: int(r["N"]))
            xs = [int(r["N"]) for r in subset]
            if scale == "sqrt":
                ys = [float(r["err_sqrt"]) for r in subset]
                errs = [
                    math.sqrt(int(r["N"])) * float(r.get("mcse", 0.0)) for r in subset
                ]
                label = f"{method}, p={p:.2f}"
            else:
                ys = [float(r["err_linear"]) for r in subset]
                errs = [int(r["N"]) * float(r.get("mcse", 0.0)) for r in subset]
                label = f"{method}, p={p:.2f}"
            plt.plot(xs, ys, marker="o", label=label)
            if any(e > 0 for e in errs):
                lower = [max(v - e, 1e-8) for v, e in zip(ys, errs)]
                upper = [v + e for v, e in zip(ys, errs)]
                plt.fill_between(xs, lower, upper, alpha=0.15)

    plt.xlabel("N")
    plt.ylabel("scaled |coverage error|")
    if logy:
        plt.yscale("log")
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    title = "Scaled error vs N (sqrt scaling)" if scale == "sqrt" else "Scaled error vs N (linear scaling)"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)


def _weighted_log_slope(xs: List[float], ys: List[float], sigmas: List[float]) -> Tuple[float, float]:
    vals = []
    for x, y, s in zip(xs, ys, sigmas):
        if x <= 0 or y <= 0 or s <= 0:
            continue
        vals.append((math.log(x), math.log(y), (y / s) ** 2))
    if len(vals) < 2:
        return float("nan"), float("nan")
    sum_w = sum(w for _, _, w in vals)
    x_bar = sum(w * x for x, _, w in vals) / sum_w
    y_bar = sum(w * y for _, y, w in vals) / sum_w
    s_xx = sum(w * (x - x_bar) ** 2 for x, _, w in vals)
    s_xy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in vals)
    if s_xx <= 0:
        return float("nan"), float("nan")
    slope = s_xy / s_xx
    se = math.sqrt(1.0 / s_xx)
    return slope, se


def estimate_slopes(rows: List[dict], scale: str, k: int) -> List[dict]:
    methods = sorted({r["method"] for r in rows})
    p_vals = sorted({float(r["p"]) for r in rows})
    out = []
    for method in methods:
        for p in p_vals:
            subset = [r for r in rows if r["method"] == method and float(r["p"]) == p]
            subset = sorted(subset, key=lambda r: int(r["N"]))
            if k > 0:
                subset = subset[-k:]
            xs = [float(r["N"]) for r in subset]
            if scale == "sqrt":
                ys = [float(r["err_sqrt"]) for r in subset]
                sigmas = [math.sqrt(float(r["N"])) * float(r.get("mcse", 0.0)) for r in subset]
            else:
                ys = [float(r["err_linear"]) for r in subset]
                sigmas = [float(r["N"]) * float(r.get("mcse", 0.0)) for r in subset]
            slope, se = _weighted_log_slope(xs, ys, sigmas)
            out.append(
                {
                    "method": method,
                    "p": p,
                    "scale": scale,
                    "k": len(subset),
                    "slope": slope,
                    "slope_se": se,
                }
            )
    return out


def main() -> None:
    args = parse_args()
    start = time.time()
    rows: List[dict] = []

    if args.plot_only:
        source = args.input_csv or args.csv
        rows = read_csv(source)
        for r in rows:
            if "err_sqrt" not in r or "err_linear" not in r:
                n = int(r["N"])
                err = float(r["coverage_error"])
                r["err_sqrt"] = math.sqrt(n) * err
                r["err_linear"] = n * err
    else:
        total = len(args.n_grid)
        for idx, n in enumerate(args.n_grid, 1):
            sim_rows = ks.run_simulation(
                n=n,
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
            for r in sim_rows:
                err = float(r["coverage_error"])
                r["N"] = n
                r["err_sqrt"] = math.sqrt(n) * err
                r["err_linear"] = n * err
                rows.append(r)
            elapsed = time.time() - start
            avg = elapsed / idx
            eta = avg * (total - idx)
            print(f"[{idx}/{total}] N={n} done. Elapsed={elapsed:.1f}s ETA={eta:.1f}s")

        write_csv(args.csv, rows)
    plot_scaled(rows, args.plot_sqrt, scale="sqrt", logy=args.logy, ylim=args.ylim)
    plot_scaled(rows, args.plot_linear, scale="linear", logy=args.logy, ylim=args.ylim)
    slope_rows = estimate_slopes(rows, scale="sqrt", k=args.slope_k) + estimate_slopes(
        rows, scale="linear", k=args.slope_k
    )
    write_csv(args.slope_csv, slope_rows)
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
