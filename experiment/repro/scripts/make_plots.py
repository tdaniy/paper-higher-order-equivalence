#!/usr/bin/env python3
"""Reproducible plot generation from the master table."""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repro_utils import build_run_paths, compute_run_id, ensure_dir, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible plot generator")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--seed", type=int, required=True, help="base RNG seed")
    parser.add_argument("--run-id", help="override run ID (optional)")
    parser.add_argument("--compare-run-id", help="optional run ID to overlay (one-sided only)")
    parser.add_argument("--dry-run", action="store_true", help="only create run folders + manifest")
    return parser.parse_args()


def _to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def read_master_table(path: str) -> List[Dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    numeric_fields = {
        "N",
        "m_N",
        "f",
        "B",
        "R",
        "S",
        "coverage",
        "mcse",
        "avg_length",
        "skew",
        "kurtosis",
        "periodicity",
        "lambda_N",
        "variance_ratio",
        "endpoint_diff",
    }
    for row in rows:
        for key in numeric_fields:
            if key in row and row[key] not in ("", None):
                row[key] = _to_float(row[key])
            elif key in row:
                row[key] = math.nan
    return rows


def _round_key(value: float, digits: int = 6) -> float:
    if value != value:
        return float("nan")
    return round(value, digits)


def _coverage_error(row: Dict, alpha: float) -> float:
    cov = row.get("coverage", math.nan)
    if cov != cov:
        return math.nan
    return abs(cov - (1 - alpha))


def _coverage_error_band(row: Dict, alpha: float) -> Tuple[float, float]:
    cov = row.get("coverage", math.nan)
    mcse = row.get("mcse", math.nan)
    if cov != cov or mcse != mcse:
        err = _coverage_error(row, alpha)
        return err, err
    low = abs((cov - mcse) - (1 - alpha))
    high = abs((cov + mcse) - (1 - alpha))
    return min(low, high), max(low, high)


def _loglog_slope(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return math.nan
    xs = [x for x in xs if x > 0]
    ys = [y for y in ys if y > 0]
    if len(xs) < 2 or len(ys) < 2:
        return math.nan
    logx = [math.log(x) for x in xs]
    logy = [math.log(y) for y in ys]
    mean_x = sum(logx) / len(logx)
    mean_y = sum(logy) / len(logy)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(logx, logy))
    den = sum((x - mean_x) ** 2 for x in logx)
    if den == 0:
        return math.nan
    return num / den


def _slope_with_band(xs: List[float], ys: List[float], ys_low: List[float], ys_high: List[float]) -> Tuple[float, float, float]:
    slope = _loglog_slope(xs, ys)
    slope_low = _loglog_slope(xs, ys_low)
    slope_high = _loglog_slope(xs, ys_high)
    lo = min(slope_low, slope_high)
    hi = max(slope_low, slope_high)
    return slope, lo, hi


def _select_tail(xs: List[float], ys: List[float], ys_low: List[float], ys_high: List[float], k: int = 4) -> Tuple[List[float], List[float], List[float], List[float]]:
    if not xs:
        return [], [], [], []
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    order = order[-min(k, len(order)) :]
    sel_x = [xs[i] for i in order]
    sel_y = [ys[i] for i in order]
    sel_low = [ys_low[i] for i in order]
    sel_high = [ys_high[i] for i in order]
    return sel_x, sel_y, sel_low, sel_high


def _write_rate_slopes(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_rate_scaled(
    ax: plt.Axes,
    label: str,
    xs: List[float],
    err: List[float],
    err_low: List[float],
    err_high: List[float],
    scale_power: float,
    color: Optional[str] = None,
) -> None:
    ys = [e * (x ** scale_power) for e, x in zip(err, xs)]
    ys_low = [e * (x ** scale_power) for e, x in zip(err_low, xs)]
    ys_high = [e * (x ** scale_power) for e, x in zip(err_high, xs)]
    ax.plot(xs, ys, marker="o", label=label, color=color)
    ax.fill_between(xs, ys_low, ys_high, color=color, alpha=0.2)


def _plot_theory_line(ax: plt.Axes, xs: List[float], y0: float, power: float, label: str) -> None:
    if not xs:
        return
    x0 = xs[0]
    ys = [y0 * (x / x0) ** power for x in xs]
    ax.plot(xs, ys, linestyle="--", color="black", linewidth=1, label=label)


def plot_parity_rates(rows: List[Dict], run_id: str, repro_root: str, alpha: float, slope_rows: List[Dict]) -> None:
    data = [r for r in rows if r["module"] == "parity" and r["method"] == "calibrated"]
    if not data:
        return
    series = {}
    for regime in ["parity_holds", "parity_fails"]:
        pts = sorted([r for r in data if r["design"] == regime], key=lambda x: x["m_N"])
        xs = [p["m_N"] for p in pts]
        err = [_coverage_error(p, alpha) for p in pts]
        bands = [_coverage_error_band(p, alpha) for p in pts]
        err_low = [b[0] for b in bands]
        err_high = [b[1] for b in bands]
        series[regime] = (xs, err, err_low, err_high)
        sel_x, sel_y, sel_low, sel_high = _select_tail(xs, err, err_low, err_high)
        slope, slope_low, slope_high = _slope_with_band(sel_x, sel_y, sel_low, sel_high)
        slope_rows.append(
            {
                "module": "parity",
                "design": regime,
                "outcome": "continuous",
                "method": "calibrated",
                "scale": "error",
                "slope": slope,
                "slope_low": slope_low,
                "slope_high": slope_high,
                "n_points": len(sel_x),
            }
        )
    for scale_power, suffix, ylabel in [
        (0.5, "sqrtN", "sqrt(N) * |coverage error|"),
        (1.0, "N", "N * |coverage error|"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for regime, (xs, err, err_low, err_high) in series.items():
            _plot_rate_scaled(ax, regime, xs, err, err_low, err_high, scale_power)
        # Theory guide lines (anchored at first point of each regime)
        if "parity_holds" in series:
            xs_h, err_h, _, _ = series["parity_holds"]
            y0_h = err_h[0] * (xs_h[0] ** scale_power)
            _plot_theory_line(ax, xs_h, y0_h, scale_power - 1.0, "theory O(N^-1)")
        if "parity_fails" in series:
            xs_f, err_f, _, _ = series["parity_fails"]
            y0_f = err_f[0] * (xs_f[0] ** scale_power)
            _plot_theory_line(ax, xs_f, y0_f, scale_power - 0.5, "theory O(N^-1/2)")
        ax.set_xlabel("m_N")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Parity rate scaling ({suffix})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(repro_root, "plots", "parity", run_id, "figs", f"parity_rate_{suffix}.png")
        ensure_dir(os.path.dirname(plot_path))
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)


def _cra_near_census_sequence(rows: List[Dict], c_val: float, tol: float = 5e-3) -> List[Dict]:
    seq = []
    n_vals = sorted({r["N"] for r in rows})
    for n in n_vals:
        target = 1.0 - c_val / math.sqrt(n)
        candidates = [r for r in rows if r["N"] == n]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: abs(r["f"] - target))
        if abs(best["f"] - target) <= tol:
            seq.append(best)
    return seq


def plot_cra_sampling_rates(rows: List[Dict], run_id: str, repro_root: str, alpha: float, slope_rows: List[Dict]) -> None:
    data = [r for r in rows if r["module"] == "cra_sampling" and r["method"] == "calibrated"]
    if not data:
        return
    sequences = {
        "f=0.5": sorted([r for r in data if abs(r["f"] - 0.5) < 1e-6], key=lambda x: x["m_N"]),
        "near-census c=1": _cra_near_census_sequence(data, 1.0),
        "near-census c=2": _cra_near_census_sequence(data, 2.0),
    }
    series = {}
    for label, seq in sequences.items():
        if not seq:
            continue
        pts = sorted(seq, key=lambda x: x["m_N"])
        xs = [p["m_N"] for p in pts]
        err = [_coverage_error(p, alpha) for p in pts]
        bands = [_coverage_error_band(p, alpha) for p in pts]
        err_low = [b[0] for b in bands]
        err_high = [b[1] for b in bands]
        series[label] = (xs, err, err_low, err_high)
        sel_x, sel_y, sel_low, sel_high = _select_tail(xs, err, err_low, err_high)
        slope, slope_low, slope_high = _slope_with_band(sel_x, sel_y, sel_low, sel_high)
        slope_rows.append(
            {
                "module": "cra_sampling",
                "design": label,
                "outcome": "continuous",
                "method": "calibrated",
                "scale": "error",
                "slope": slope,
                "slope_low": slope_low,
                "slope_high": slope_high,
                "n_points": len(sel_x),
            }
        )
    for scale_power, suffix, ylabel in [
        (0.5, "sqrtN", "sqrt(N) * |coverage error|"),
        (1.0, "N", "N * |coverage error|"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for label, (xs, err, err_low, err_high) in series.items():
            _plot_rate_scaled(ax, label, xs, err, err_low, err_high, scale_power)
        ax.set_xlabel("m_N")
        ax.set_ylabel(ylabel)
        ax.set_title(f"CRA sampling rate scaling ({suffix})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(repro_root, "plots", "cra_sampling", run_id, "figs", f"cra_sampling_rate_{suffix}.png")
        ensure_dir(os.path.dirname(plot_path))
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)


def plot_parity(rows: List[Dict], run_id: str, repro_root: str, alpha: float) -> None:
    data = [r for r in rows if r["module"] == "parity" and r["method"] == "calibrated"]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for regime in ["parity_holds", "parity_fails"]:
        pts = sorted([r for r in data if r["design"] == regime], key=lambda x: x["N"])
        xs = [p["N"] for p in pts]
        ys = [p["coverage"] for p in pts]
        yerr = [p["mcse"] for p in pts]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", label=regime)
    ax.axhline(1 - alpha, color="black", linestyle="--", linewidth=1, label="theory (nominal)")
    ax.set_xlabel("N")
    ax.set_ylabel("coverage")
    ax.set_title("Parity modules (calibrated)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(repro_root, "plots", "parity", run_id, "figs", "parity_coverage.png")
    ensure_dir(os.path.dirname(plot_path))
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)


def plot_parity_scaling(rows: List[Dict], run_id: str, repro_root: str, alpha: float) -> None:
    data = [r for r in rows if r["module"] == "parity" and r["method"] == "gaussian"]
    if not data:
        return
    series = {}
    for regime in ["parity_holds", "parity_fails"]:
        pts = sorted([r for r in data if r["design"] == regime], key=lambda x: x["N"])
        xs = [p["N"] for p in pts]
        err = [_coverage_error(p, alpha) for p in pts]
        series[regime] = (xs, err)

    fig, ax = plt.subplots(figsize=(7, 4))
    for regime, (xs, err) in series.items():
        ys = [e * math.sqrt(x) for e, x in zip(err, xs)]
        ax.plot(xs, ys, marker="o", label=regime)

    # Theory guide lines (anchored at first point of each regime)
    if "parity_holds" in series:
        xs_h, err_h = series["parity_holds"]
        y0_h = err_h[0] * math.sqrt(xs_h[0])
        _plot_theory_line(ax, xs_h, y0_h, -0.5, "theory O(N^-1)")
    if "parity_fails" in series:
        xs_f, err_f = series["parity_fails"]
        y0_f = err_f[0] * math.sqrt(xs_f[0])
        _plot_theory_line(ax, xs_f, y0_f, 0.0, "theory O(N^-1/2)")

    ax.set_xlabel("N")
    ax.set_ylabel("sqrt(N)*|coverage error|")
    ax.set_title("Parity regimes: gaussian scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(repro_root, "plots", "parity", run_id, "figs", "parity_scaling.png")
    ensure_dir(os.path.dirname(plot_path))
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)


def plot_stratified(rows: List[Dict], run_id: str, repro_root: str, alpha: float) -> None:
    for outcome in ["continuous", "binary"]:
        data = [r for r in rows if r["module"] == "stratified" and r["method"] == "calibrated" and r["outcome"] == outcome]
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4))
        f_vals = sorted({_round_key(r["f"]) for r in data})
        for f in f_vals:
            pts = sorted([r for r in data if _round_key(r["f"]) == f], key=lambda x: x["N"])
            xs = [p["N"] for p in pts]
            ys = [p["coverage"] for p in pts]
            yerr = [p["mcse"] for p in pts]
            ax.errorbar(xs, ys, yerr=yerr, marker="o", label=f"f={f}")
        ax.axhline(1 - alpha, color="black", linestyle="--", linewidth=1, label="nominal")
        ax.set_xlabel("N")
        ax.set_ylabel("coverage")
        ax.set_title(f"Stratified ({outcome}, calibrated)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(repro_root, "plots", "stratified", run_id, "figs", f"stratified_{outcome}_coverage.png")
        ensure_dir(os.path.dirname(plot_path))
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)


def plot_cluster(rows: List[Dict], run_id: str, repro_root: str, alpha: float) -> None:
    data = [r for r in rows if r["module"] == "cluster" and r["method"] == "calibrated" and r["outcome"] == "continuous"]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for regime in ["regime_a", "regime_b", "violation"]:
        pts = sorted([r for r in data if r["design"] == regime], key=lambda x: x["N"])
        xs = [p["N"] for p in pts]
        ys = [p["coverage"] for p in pts]
        yerr = [p["mcse"] for p in pts]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", label=regime)
    ax.axhline(1 - alpha, color="black", linestyle="--", linewidth=1, label="nominal")
    ax.set_xlabel("N")
    ax.set_ylabel("coverage")
    ax.set_title("Cluster leverage regimes (calibrated)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(repro_root, "plots", "cluster", run_id, "figs", "cluster_coverage.png")
    ensure_dir(os.path.dirname(plot_path))
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)


def plot_one_sided(
    rows: List[Dict],
    run_id: str,
    repro_root: str,
    alpha: float,
    compare_rows: List[Dict] | None = None,
) -> None:
    data = [r for r in rows if r["module"] == "one_sided" and r["method"] in ("gaussian_one_sided", "calibrated_one_sided")]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method in ["gaussian_one_sided", "calibrated_one_sided"]:
        pts = sorted([r for r in data if r["method"] == method], key=lambda x: x["N"])
        xs = [p["N"] for p in pts]
        ys = [p["coverage"] for p in pts]
        yerr = [p["mcse"] for p in pts]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", label=method.replace("_", " "))
    if compare_rows:
        comp = [r for r in compare_rows if r["module"] == "one_sided" and r["method"] in ("gaussian_one_sided", "calibrated_one_sided")]
        if comp:
            for method, marker in [("gaussian_one_sided", "x"), ("calibrated_one_sided", "^")]:
                pts = sorted([r for r in comp if r["method"] == method], key=lambda x: x["N"])
                if not pts:
                    continue
                xs = [p["N"] for p in pts]
                ys = [p["coverage"] for p in pts]
                ax.scatter(xs, ys, marker=marker, s=50, label=f"{method.replace('_', ' ')} (highR)")
    ax.axhline(1 - alpha, color="black", linestyle="--", linewidth=1, label="nominal")
    ax.set_xlabel("N")
    ax.set_ylabel("coverage")
    ax.set_title("One-sided coverage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(repro_root, "plots", "one_sided", run_id, "figs", "one_sided_coverage.png")
    ensure_dir(os.path.dirname(plot_path))
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)


def plot_objective_bayes(rows: List[Dict], run_id: str, repro_root: str) -> None:
    data = [r for r in rows if r["module"] == "objective_bayes"]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for design in ["CRA", "stratified"]:
        pts = sorted([r for r in data if r["design"] == design], key=lambda x: x["N"])
        xs = [p["N"] for p in pts]
        ys = [p["variance_ratio"] for p in pts]
        ax.plot(xs, ys, marker="o", label=design)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="ratio=1")
    ax.set_xlabel("N")
    ax.set_ylabel("posterior variance ratio")
    ax.set_title("Objective Bayes variance ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(repro_root, "plots", "objective_bayes", run_id, "figs", "objective_bayes_variance_ratio.png")
    ensure_dir(os.path.dirname(plot_path))
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for design in ["CRA", "stratified"]:
        pts = sorted([r for r in data if r["design"] == design], key=lambda x: x["N"])
        xs = [p["N"] for p in pts]
        ys = [p["endpoint_diff"] for p in pts]
        ax.plot(xs, ys, marker="o", label=design)
    ax.set_xlabel("N")
    ax.set_ylabel("endpoint diff")
    ax.set_title("Objective Bayes endpoint alignment")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(repro_root, "plots", "objective_bayes", run_id, "figs", "objective_bayes_endpoint_diff.png")
    ensure_dir(os.path.dirname(plot_path))
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)


def plot_fpc(rows: List[Dict], run_id: str, repro_root: str) -> None:
    data = [r for r in rows if r["module"] == "fpc"]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    pts = sorted(data, key=lambda x: x["f"])
    xs = [p["f"] for p in pts]
    ys = [p["variance_ratio"] for p in pts]
    yerr = [p["mcse"] for p in pts]
    ax.errorbar(xs, ys, yerr=yerr, marker="o")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="ratio=1")
    ax.set_xlabel("f")
    ax.set_ylabel("variance ratio")
    ax.set_title("FPC sanity check")
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(repro_root, "plots", "fpc", run_id, "figs", "fpc_variance_ratio.png")
    ensure_dir(os.path.dirname(plot_path))
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)


def main() -> None:
    args = parse_args()
    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        raise SystemExit(f"Config not found: {config_path}")

    base_dir = os.path.dirname(__file__)
    repro_root = os.path.abspath(os.path.join(base_dir, ".."))

    run_id = args.run_id or compute_run_id(config_path, args.seed)
    paths = build_run_paths(repro_root, "master", run_id)
    ensure_dir(paths.outputs_dir)
    ensure_dir(paths.plots_dir)
    ensure_dir(paths.logs_dir)

    manifest_path = os.path.join(paths.logs_dir, "manifest.json")
    write_manifest(
        manifest_path,
        run_id=run_id,
        config_path=config_path,
        base_seed=args.seed,
        spawn_keys=None,
        command=sys.argv,
        repo_root=os.path.abspath(os.path.join(repro_root, "..")),
    )
    for target_dir in (os.path.dirname(paths.outputs_dir), os.path.dirname(paths.plots_dir)):
        write_manifest(
            os.path.join(target_dir, "manifest.json"),
            run_id=run_id,
            config_path=config_path,
            base_seed=args.seed,
            spawn_keys=None,
            command=sys.argv,
            repo_root=os.path.abspath(os.path.join(repro_root, "..")),
        )

    if args.dry_run:
        print(f"Dry run complete: {run_id}")
        return

    master_path = os.path.join(repro_root, "outputs", "master", run_id, "tables", "master_table.csv")
    if not os.path.isfile(master_path):
        raise SystemExit(f"Master table not found: {master_path}")

    rows = read_master_table(master_path)
    compare_rows = None
    if args.compare_run_id:
        compare_path = os.path.join(repro_root, "outputs", "master", args.compare_run_id, "tables", "master_table.csv")
        if os.path.isfile(compare_path):
            compare_rows = read_master_table(compare_path)

    alpha = 0.05
    slope_rows: List[Dict] = []
    plot_parity(rows, run_id, repro_root, alpha)
    plot_parity_rates(rows, run_id, repro_root, alpha, slope_rows)
    plot_parity_scaling(rows, run_id, repro_root, alpha)
    plot_stratified(rows, run_id, repro_root, alpha)
    plot_cluster(rows, run_id, repro_root, alpha)
    plot_one_sided(rows, run_id, repro_root, alpha, compare_rows=compare_rows)
    plot_objective_bayes(rows, run_id, repro_root)
    plot_fpc(rows, run_id, repro_root)
    plot_cra_sampling_rates(rows, run_id, repro_root, alpha, slope_rows)
    if slope_rows:
        slopes_path = os.path.join(repro_root, "outputs", "master", run_id, "tables", "rate_slopes.csv")
        _write_rate_slopes(slopes_path, slope_rows)


if __name__ == "__main__":
    main()
