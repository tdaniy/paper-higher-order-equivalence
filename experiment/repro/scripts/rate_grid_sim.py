#!/usr/bin/env python3
"""Rate-grid simulation: scaled coverage error vs N for selected p values.

Uses sampling_fraction_sim.run_simulation to compute coverage errors, then rescales
by sqrt(N) and N for rate diagnostics. Produces CSV and plots.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from typing import List, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed

import sys

base_dir = os.path.dirname(__file__)
sys.path.insert(0, base_dir)
import sampling_fraction_sim as sim
from repro_utils import build_run_paths, compute_run_id, ensure_dir, write_manifest


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
    parser.add_argument("--config", type=str, help="optional config file for run ID hashing")
    parser.add_argument("--run-id", type=str, help="optional run ID (overrides config-based hash)")
    outputs_dir = os.path.abspath(os.path.join(base_dir, "..", "outputs", "rate_grid"))
    plots_dir = os.path.abspath(os.path.join(base_dir, "..", "plots", "rate_grid"))
    parser.add_argument("--csv", type=str, help="output CSV path")
    parser.add_argument("--plot-sqrt", type=str, help="output path for sqrt-scaled plot")
    parser.add_argument("--plot-linear", type=str, help="output path for linear-scaled plot")
    parser.add_argument("--logy", action="store_true", help="log-scale y-axis for plots")
    parser.add_argument("--ylim", type=float, nargs=2, help="y-axis limits, e.g. --ylim -2 2")
    parser.add_argument("--slope-k", type=int, default=6, help="number of largest-N points for slope fit")
    parser.add_argument("--slope-csv", type=str, help="output path for slopes CSV")
    parser.add_argument("--plot-only", action="store_true", help="skip simulation and replot from CSV")
    parser.add_argument("--input-csv", type=str, help="CSV to read when --plot-only is set")
    parser.add_argument("--plot-raw-p05", type=str, help="output path for raw error p=0.5 plot")
    parser.add_argument("--jobs", type=int, default=0, help="number of parallel workers (0=auto)")
    args = parser.parse_args()
    run_id = args.run_id
    if run_id is None and args.config:
        run_id = compute_run_id(args.config, args.seed)
    if run_id is None:
        run_id = f"run_manual_seed{args.seed}"
    args.run_id = run_id
    outputs_dir = os.path.join(outputs_dir, run_id, "tables")
    plots_dir = os.path.join(plots_dir, run_id, "figs")
    if not args.csv:
        args.csv = os.path.join(outputs_dir, "rate_grid.csv")
    if not args.plot_sqrt:
        args.plot_sqrt = os.path.join(plots_dir, "rate_grid_sqrt.png")
    if not args.plot_linear:
        args.plot_linear = os.path.join(plots_dir, "rate_grid_linear.png")
    if not args.slope_csv:
        args.slope_csv = os.path.join(outputs_dir, "rate_grid_slopes.csv")
    if args.plot_raw_p05 is None:
        args.plot_raw_p05 = os.path.join(plots_dir, "rate_grid_raw_p05.png")
    return args


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def plot_scaled(rows: List[dict], path: str, scale: str, logy: bool, ylim: List[float] | None) -> None:
    try:
        import os
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".mpl_cache"))
        os.environ.setdefault("MPLCONFIGDIR", cache_dir)
        import matplotlib
        matplotlib.use("Agg")
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200)


def plot_raw_p05(rows: List[dict], path: str) -> None:
    try:
        import os
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".mpl_cache"))
        os.environ.setdefault("MPLCONFIGDIR", cache_dir)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    p_target = 0.5
    methods = sorted({r["method"] for r in rows})
    plt.figure(figsize=(6, 3.5))
    for method in methods:
        subset = [r for r in rows if r["method"] == method and abs(float(r["p"]) - p_target) < 1e-9]
        subset = sorted(subset, key=lambda r: int(r["N"]))
        if not subset:
            continue
        xs = [int(r["N"]) for r in subset]
        ys = [float(r["coverage_error"]) for r in subset]
        errs = [float(r.get("mcse", 0.0)) for r in subset]
        plt.plot(xs, ys, marker="o", label=method)
        if any(e > 0 for e in errs):
            lower = [max(v - e, 0.0) for v, e in zip(ys, errs)]
            upper = [v + e for v, e in zip(ys, errs)]
            plt.fill_between(xs, lower, upper, alpha=0.15)

    plt.xlabel("N")
    plt.ylabel("|coverage - (1-α)|")
    plt.title("Raw coverage error vs N (p=0.5)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def _simulate_one_n(args: Tuple[int, int, int, int, List[float], float, int, float, float, float]) -> Tuple[int, List[dict]]:
    n, b, r_skew, r_cov, p_grid, alpha, seed, delta, mu, sigma = args
    sim_rows = sim.run_simulation(
        n=n,
        b=b,
        r_skew=r_skew,
        r_cov=r_cov,
        p_grid=p_grid,
        alpha=alpha,
        seed=seed,
        delta=delta,
        mu=mu,
        sigma=sigma,
    )
    return n, sim_rows


def main() -> None:
    args = parse_args()
    base_dir = os.path.dirname(__file__)
    repro_root = os.path.abspath(os.path.join(base_dir, ".."))
    paths = build_run_paths(repro_root, "rate_grid", args.run_id)
    ensure_dir(paths.outputs_dir)
    ensure_dir(paths.plots_dir)
    ensure_dir(paths.logs_dir)
    spawn_keys = []
    for n in args.n_grid:
        for p in args.p_grid:
            tag = sim.scenario_tag(
                n=n,
                p=float(p),
                b=args.b,
                r_skew=args.r_skew,
                r_cov=args.r_cov,
                alpha=args.alpha,
                delta=args.delta,
                mu=args.lognormal_mu,
                sigma=args.lognormal_sigma,
            )
            spawn_keys.append({"N": int(n), "p": float(p), "hash": sim.stable_hash_int(tag)})
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
        jobs = args.jobs
        if jobs <= 0:
            try:
                jobs = min(total, os.cpu_count() or 1)
            except Exception:
                jobs = 1

        if jobs <= 1 or total == 1:
            for idx, n in enumerate(args.n_grid, 1):
                sim_rows = sim.run_simulation(
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
        else:
            tasks = []
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                for n in args.n_grid:
                    task_args = (
                        n,
                        args.b,
                        args.r_skew,
                        args.r_cov,
                        list(args.p_grid),
                        args.alpha,
                        args.seed,
                        args.delta,
                        args.lognormal_mu,
                        args.lognormal_sigma,
                    )
                    tasks.append(ex.submit(_simulate_one_n, task_args))

                completed = 0
                for fut in as_completed(tasks):
                    n, sim_rows = fut.result()
                    for r in sim_rows:
                        err = float(r["coverage_error"])
                        r["N"] = n
                        r["err_sqrt"] = math.sqrt(n) * err
                        r["err_linear"] = n * err
                        rows.append(r)
                    completed += 1
                    elapsed = time.time() - start
                    avg = elapsed / completed
                    eta = avg * (total - completed)
                    print(f"[{completed}/{total}] N={n} done. Elapsed={elapsed:.1f}s ETA={eta:.1f}s")

        write_csv(args.csv, rows)
    plot_scaled(rows, args.plot_sqrt, scale="sqrt", logy=args.logy, ylim=args.ylim)
    plot_scaled(rows, args.plot_linear, scale="linear", logy=args.logy, ylim=args.ylim)
    if args.plot_raw_p05:
        plot_raw_p05(rows, args.plot_raw_p05)
    slope_rows = estimate_slopes(rows, scale="sqrt", k=args.slope_k) + estimate_slopes(
        rows, scale="linear", k=args.slope_k
    )
    write_csv(args.slope_csv, slope_rows)
    summary_path = os.path.join(repro_root, "outputs", "rate_grid", "summary.csv")
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
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
