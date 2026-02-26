#!/usr/bin/env python3
"""Run the full reproduction suite from a TOML config."""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import tomllib
import numpy as np
from numpy.random import Generator, Philox, SeedSequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repro_utils import build_run_paths, compute_run_id, ensure_dir, write_manifest
from sim_core import (
    CoverageStats,
    compute_vhat,
    draw_cra,
    draw_stratified,
    draw_clustered,
    draw_clustered_force_big,
    interval_gaussian,
    interval_cf,
    interval_calibrated,
    one_sided_interval_gaussian,
    one_sided_interval_cf,
    kurtosis_excess,
    skewness,
    periodicity_metric,
    lattice_span,
    norm_ppf,
)


def _cra_worker(args: Tuple[SeedSequence, int, int, int, int, float, float, float, float]):
    pop_ss, n, n1, r_skew, r_cov, alpha, mu, sigma, delta = args
    ss_pop, ss_sk, ss_cv = pop_ss.spawn(3)
    rng_pop = Generator(Philox(ss_pop))
    rng_skew = Generator(Philox(ss_sk))
    rng_cov = Generator(Philox(ss_cv))
    y0, y1, tau = gen_lognormal(rng_pop, n, mu, sigma, delta)
    assign_fn = lambda rng, n=n, n1=n1: draw_cra(rng, n, n1)
    gamma_hat, kurt, _ql, _qh, ga, cf, cal, _q1, _p, _pj = simulate_population(
        y0,
        y1,
        tau,
        assign_fn,
        r_skew,
        r_cov,
        alpha,
        rng_skew,
        rng_cov,
    )
    return (
        ga.covered,
        ga.count,
        ga.length_sum,
        cf.covered,
        cf.count,
        cf.length_sum,
        cal.covered,
        cal.count,
        cal.length_sum,
        gamma_hat,
        kurt,
    )


def _parity_worker(
    args: Tuple[
        SeedSequence,
        int,
        int,
        int,
        int,
        float,
        float,
        str,
        str,
        float,
        float,
        float,
        float,
        float,
        bool,
        bool,
    ]
):
    (
        pop_ss,
        n,
        n1,
        r_skew,
        r_cov,
        alpha,
        delta,
        regime,
        symmetric_dgp,
        t_df,
        mix_weight,
        mix_scale,
        spike_prob,
        spike_scale,
        spike_standardize,
        antithetic,
    ) = args
    ss_pop, ss_sk, ss_cv = pop_ss.spawn(3)
    rng_pop = Generator(Philox(ss_pop))
    rng_skew = Generator(Philox(ss_sk))
    rng_cov = Generator(Philox(ss_cv))
    if regime == "parity_holds":
        if symmetric_dgp == "normal":
            y0, y1, tau = gen_symmetric(rng_pop, n, delta)
        elif symmetric_dgp == "t":
            y0, y1, tau = gen_symmetric_t(rng_pop, n, delta, t_df)
        elif symmetric_dgp == "mixture":
            y0, y1, tau = gen_symmetric_mixture(rng_pop, n, delta, mix_weight, mix_scale)
        elif symmetric_dgp == "spike":
            y0, y1, tau = gen_symmetric_spike(
                rng_pop, n, delta, spike_prob, spike_scale, spike_standardize
            )
        else:
            raise ValueError(f"Unknown symmetric_dgp: {symmetric_dgp}")
    else:
        y0, y1, tau = gen_lognormal(rng_pop, n, 0.0, 1.2, delta)
    assign_fn = lambda rng, n=n, n1=n1: draw_cra(rng, n, n1)
    gamma_hat, _k, _ql, _qh, ga, cf, cal, _q1, _p, _pj = simulate_population(
        y0,
        y1,
        tau,
        assign_fn,
        r_skew,
        r_cov,
        alpha,
        rng_skew,
        rng_cov,
        antithetic=antithetic,
    )
    return (
        ga.covered,
        ga.count,
        ga.length_sum,
        cf.covered,
        cf.count,
        cf.length_sum,
        cal.covered,
        cal.count,
        cal.length_sum,
        gamma_hat,
    )


def _cluster_worker(args: Tuple[SeedSequence, int, List[np.ndarray], int, int, int, float, float, float, float, float, str, bool]):
    pop_ss, n, clusters, n_treated_clusters, r_skew, r_cov, alpha, mu, sigma, delta, p0, p1, outcome, force_big = args
    ss_pop, ss_sk, ss_cv = pop_ss.spawn(3)
    rng_pop = Generator(Philox(ss_pop))
    rng_skew = Generator(Philox(ss_sk))
    rng_cov = Generator(Philox(ss_cv))
    if outcome == "continuous":
        y0, y1, tau = gen_lognormal(rng_pop, n, mu, sigma, delta)
    else:
        y0, y1, tau = gen_binary(rng_pop, n, p0, p1)
    if force_big:
        assign_fn = lambda rng, clusters=clusters, k=n_treated_clusters: draw_clustered_force_big(rng, clusters, k)
    else:
        assign_fn = lambda rng, clusters=clusters, k=n_treated_clusters: draw_clustered(rng, clusters, k)
    _g, _k, _ql, _qh, ga, cf, cal, _q1, _p, _pj = simulate_population(
        y0, y1, tau, assign_fn, r_skew, r_cov, alpha, rng_skew, rng_cov
    )
    return (
        ga.covered,
        ga.count,
        ga.length_sum,
        cf.covered,
        cf.count,
        cf.length_sum,
        cal.covered,
        cal.count,
        cal.length_sum,
    )

def _skew_rate_worker(
    args: Tuple[
        SeedSequence,
        int,
        int,
        int,
        int,
        float,
        float,
        str,
        float,
        bool,
        str,
        float,
        float,
        float,
    ]
):
    (
        pop_ss,
        n,
        n1,
        r_skew,
        r_cov,
        alpha,
        delta,
        base_dist,
        t_df,
        base_standardize,
        skew_dist,
        gamma_shape,
        log_sigma,
        skew_scale,
    ) = args
    ss_pop, ss_sk, ss_cv = pop_ss.spawn(3)
    rng_pop = Generator(Philox(ss_pop))
    rng_skew = Generator(Philox(ss_sk))
    rng_cov = Generator(Philox(ss_cv))

    if base_dist == "normal":
        y0 = rng_pop.standard_normal(size=n)
    elif base_dist == "t":
        y0 = rng_pop.standard_t(t_df, size=n)
    else:
        raise ValueError(f"Unknown base_dist: {base_dist}")

    if base_standardize:
        y0 = standardize_array(y0)

    skew_component = gen_skew_component(rng_pop, n, skew_dist, gamma_shape, log_sigma)
    y1 = y0 + delta + skew_scale * skew_component
    tau = float(delta)

    assign_fn = lambda rng, n=n, n1=n1: draw_cra(rng, n, n1)
    gamma_hat, kurt, _ql, _qh, ga, cf, cal, _q1, _p, _pj = simulate_population(
        y0,
        y1,
        tau,
        assign_fn,
        r_skew,
        r_cov,
        alpha,
        rng_skew,
        rng_cov,
    )
    return (
        ga.covered,
        ga.count,
        ga.length_sum,
        cf.covered,
        cf.count,
        cf.length_sum,
        cal.covered,
        cal.count,
        cal.length_sum,
        gamma_hat,
        kurt,
    )


def _lattice_worker(args: Tuple[SeedSequence, int, int, int, int, float, float, float]):
    pop_ss, n, n1, r_skew, r_cov, alpha, p0, p1 = args
    ss_pop, ss_sk, ss_cv = pop_ss.spawn(3)
    rng_pop = Generator(Philox(ss_pop))
    rng_skew = Generator(Philox(ss_sk))
    rng_cov = Generator(Philox(ss_cv))
    y0, y1, tau = gen_binary(rng_pop, n, p0, p1)
    assign_fn = lambda rng, n=n, n1=n1: draw_cra(rng, n, n1)

    _g, _k, _ql, _qh, ga, cf, cal, _q1, per, _per_j = simulate_population(
        y0,
        y1,
        tau,
        assign_fn,
        r_skew,
        r_cov,
        alpha,
        rng_skew,
        rng_cov,
        lattice=True,
        jitter=False,
    )
    _, _k2, _ql2, _qh2, _ga2, _cf2, cal_j, _q1b, _per2, perj2 = simulate_population(
        y0,
        y1,
        tau,
        assign_fn,
        r_skew,
        r_cov,
        alpha,
        rng_skew,
        rng_cov,
        lattice=True,
        jitter=True,
    )
    return (
        ga.covered,
        ga.count,
        ga.length_sum,
        cf.covered,
        cf.count,
        cf.length_sum,
        cal.covered,
        cal.count,
        cal.length_sum,
        cal_j.covered,
        cal_j.count,
        cal_j.length_sum,
        per,
        perj2,
    )


def _stratified_worker(args: Tuple[SeedSequence, int, List[np.ndarray], List[int], int, int, float, float, float, float, float, str, List[float]]):
    pop_ss, n, strata, n1_per, r_skew, r_cov, alpha, mu, delta, p0, p1, outcome, sigmas = args
    ss_pop, ss_sk, ss_cv = pop_ss.spawn(3)
    rng_pop = Generator(Philox(ss_pop))
    rng_skew = Generator(Philox(ss_sk))
    rng_cov = Generator(Philox(ss_cv))
    if outcome == "continuous":
        ys = []
        for sidx, sigma in zip(strata, sigmas):
            ys.append(rng_pop.lognormal(mean=mu, sigma=sigma, size=len(sidx)))
        y0 = np.concatenate(ys)
        y1 = y0 + delta
        tau = delta
    else:
        y0, y1, tau = gen_binary(rng_pop, n, p0, p1)
    assign_fn = lambda rng, strata=strata, n1_per=n1_per: draw_stratified(rng, strata, n1_per)
    _g, _k, _ql, _qh, ga, cf, cal, _q1, _p, _pj = simulate_population(
        y0, y1, tau, assign_fn, r_skew, r_cov, alpha, rng_skew, rng_cov
    )
    return (
        ga.covered,
        ga.count,
        ga.length_sum,
        cf.covered,
        cf.count,
        cf.length_sum,
        cal.covered,
        cal.count,
        cal.length_sum,
    )


def _one_sided_worker(args: Tuple[SeedSequence, int, int, int, int, float, float]):
    pop_ss, n, n1, r_skew, r_cov, alpha, delta = args
    ss_pop, ss_sk, ss_cv = pop_ss.spawn(3)
    rng_pop = Generator(Philox(ss_pop))
    rng_skew = Generator(Philox(ss_sk))
    rng_cov = Generator(Philox(ss_cv))
    y0, y1, tau = gen_symmetric(rng_pop, n, delta)
    assign_fn = lambda rng, n=n, n1=n1: draw_cra(rng, n, n1)
    _g, _k, _ql, _qh, ga, cf, cal, _q1, _p, _pj = simulate_population(
        y0, y1, tau, assign_fn, r_skew, r_cov, alpha, rng_skew, rng_cov, one_sided=True
    )
    return (
        ga.covered,
        ga.count,
        ga.length_sum,
        cf.covered,
        cf.count,
        cf.length_sum,
        cal.covered,
        cal.count,
        cal.length_sum,
    )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full reproduction suite")
    parser.add_argument("--config", required=True, help="TOML config path")
    parser.add_argument("--seed", type=int, help="override base seed")
    parser.add_argument("--run-id", help="override run ID")
    return parser.parse_args()


def _format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "inf"
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class ModuleProgress:
    def __init__(self, label: str, total_tasks: int, cadence_seconds: int = 60) -> None:
        self.label = label
        self.total_tasks = max(0, total_tasks)
        self.cadence_seconds = cadence_seconds
        self.start_ts = time.time()
        self.last_report = self.start_ts
        self.completed_tasks = 0
        self.completed_task_times: List[float] = []
        self.current_task_label: str | None = None
        self.current_task_total: int = 0
        self.current_task_done: int = 0
        self.current_task_start: float | None = None
        start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_ts))
        print(f"[{self.label}] START {start_str} | tasks={self.total_tasks}", flush=True)

    def start_task(self, label: str, total: int) -> None:
        self.current_task_label = label
        self.current_task_total = max(0, total)
        self.current_task_done = 0
        self.current_task_start = time.time()

    def update(self, inc: int = 1) -> None:
        self.current_task_done += inc
        now = time.time()
        if (now - self.last_report) >= self.cadence_seconds:
            self._report(now)
            self.last_report = now

    def finish_task(self) -> None:
        now = time.time()
        task_elapsed = (now - self.current_task_start) if self.current_task_start else 0.0
        self.completed_tasks += 1
        if task_elapsed > 0:
            self.completed_task_times.append(task_elapsed)
        label = self.current_task_label or "task"
        total = self.current_task_total
        done = self.current_task_done
        print(
            f"[{self.label}] done {label} | {done}/{total} | "
            f"task_time={_format_seconds(task_elapsed)} | tasks={self.completed_tasks}/{self.total_tasks}",
            flush=True,
        )
        self.current_task_label = None
        self.current_task_total = 0
        self.current_task_done = 0
        self.current_task_start = None

    def finish(self) -> None:
        now = time.time()
        total_elapsed = now - self.start_ts
        end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        print(f"[{self.label}] DONE {end_str} | total_time={_format_seconds(total_elapsed)}", flush=True)

    def _estimate_eta(self, now: float) -> float:
        elapsed = now - self.start_ts
        remaining_tasks = max(0, self.total_tasks - self.completed_tasks)
        avg_task = None
        if self.completed_task_times:
            avg_task = sum(self.completed_task_times) / len(self.completed_task_times)
        remaining_current = None
        if self.current_task_start and self.current_task_total > 0 and self.current_task_done > 0:
            frac = self.current_task_done / self.current_task_total
            est_task_total = (now - self.current_task_start) / frac
            remaining_current = max(0.0, est_task_total - (now - self.current_task_start))
        if avg_task is None:
            if remaining_current is None:
                return float("inf")
            return remaining_current * remaining_tasks
        if remaining_tasks == 0:
            return 0.0
        if remaining_current is None:
            return avg_task * remaining_tasks
        return remaining_current + avg_task * max(0, remaining_tasks - 1)

    def _report(self, now: float) -> None:
        elapsed = now - self.start_ts
        eta = self._estimate_eta(now)
        current = "none"
        if self.current_task_label:
            current = f"{self.current_task_label} ({self.current_task_done}/{self.current_task_total})"
        print(
            f"[{self.label}] progress {self.completed_tasks}/{self.total_tasks} | "
            f"current={current} | elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta)}",
            flush=True,
        )

def stable_hash_int(tag: str) -> int:
    import hashlib

    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def scenario_tag(prefix: str, **kwargs: float | int | str) -> str:
    parts = [prefix]
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.8f}")
        else:
            parts.append(f"{k}={v}")
    return "|".join(parts)


def rng_sequence(seed: int, tag: str) -> SeedSequence:
    return SeedSequence(entropy=[seed, stable_hash_int(tag)])


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def make_manifest(repro_root: str, experiment: str, run_id: str, config_path: str, seed: int, spawn_keys: List[Dict]) -> None:
    paths = build_run_paths(repro_root, experiment, run_id)
    ensure_dir(paths.outputs_dir)
    ensure_dir(paths.plots_dir)
    ensure_dir(paths.logs_dir)
    write_manifest(
        os.path.join(paths.logs_dir, "manifest.json"),
        run_id=run_id,
        config_path=config_path,
        base_seed=seed,
        spawn_keys=spawn_keys,
        command=sys.argv,
        repo_root=os.path.abspath(os.path.join(repro_root, "..")),
    )
    for target_dir in (os.path.dirname(paths.outputs_dir), os.path.dirname(paths.plots_dir)):
        write_manifest(
            os.path.join(target_dir, "manifest.json"),
            run_id=run_id,
            config_path=config_path,
            base_seed=seed,
            spawn_keys=spawn_keys,
            command=sys.argv,
            repo_root=os.path.abspath(os.path.join(repro_root, "..")),
        )
    append_summary(
        os.path.join(repro_root, "outputs", experiment, "summary.csv"),
        [
            run_id,
            os.path.relpath(config_path, repro_root),
            str(seed),
            os.path.relpath(os.path.dirname(paths.outputs_dir), repro_root),
            os.path.relpath(os.path.dirname(paths.plots_dir), repro_root),
            os.path.relpath(paths.logs_dir, repro_root),
            "completed",
        ],
    )


# ---------- DGP helpers ----------

def gen_lognormal(rng: Generator, n: int, mu: float, sigma: float, delta: float) -> Tuple[np.ndarray, np.ndarray, float]:
    y0 = rng.lognormal(mean=mu, sigma=sigma, size=n).astype(float, copy=False)
    y1 = y0 + delta
    return y0, y1, delta


def gen_symmetric(rng: Generator, n: int, delta: float) -> Tuple[np.ndarray, np.ndarray, float]:
    half = n // 2
    base = rng.standard_normal(size=half)
    if n % 2 == 1:
        extra = np.array([0.0])
        base_full = np.concatenate([base, -base, extra])
    else:
        base_full = np.concatenate([base, -base])
    y0 = base_full.astype(float)
    y1 = y0 + delta
    return y0, y1, delta


def gen_symmetric_t(rng: Generator, n: int, delta: float, df: float) -> Tuple[np.ndarray, np.ndarray, float]:
    half = n // 2
    base = rng.standard_t(df, size=half)
    if n % 2 == 1:
        extra = np.array([0.0])
        base_full = np.concatenate([base, -base, extra])
    else:
        base_full = np.concatenate([base, -base])
    y0 = base_full.astype(float)
    y1 = y0 + delta
    return y0, y1, delta


def gen_symmetric_mixture(
    rng: Generator, n: int, delta: float, mix_weight: float, mix_scale: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    half = n // 2
    base = rng.standard_normal(size=half)
    if mix_weight > 0.0:
        mask = rng.random(size=half) < mix_weight
        if np.any(mask):
            base[mask] = rng.standard_normal(size=int(np.sum(mask))) * mix_scale
    if n % 2 == 1:
        extra = np.array([0.0])
        base_full = np.concatenate([base, -base, extra])
    else:
        base_full = np.concatenate([base, -base])
    y0 = base_full.astype(float)
    y1 = y0 + delta
    return y0, y1, delta


def gen_symmetric_spike(
    rng: Generator,
    n: int,
    delta: float,
    spike_prob: float,
    spike_scale: float,
    standardize: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    half = n // 2
    draws = rng.random(size=half)
    mags = np.ones(half)
    mags[draws < spike_prob] = spike_scale
    if standardize:
        e_m2 = (1.0 - spike_prob) * 1.0 + spike_prob * (spike_scale ** 2)
        mags = mags / math.sqrt(e_m2)
    if n % 2 == 1:
        extra = np.array([0.0])
        base_full = np.concatenate([mags, -mags, extra])
    else:
        base_full = np.concatenate([mags, -mags])
    y0 = base_full.astype(float)
    y1 = y0 + delta
    return y0, y1, delta


def gen_binary(rng: Generator, n: int, p0: float, p1: float) -> Tuple[np.ndarray, np.ndarray, float]:
    y0 = (rng.random(size=n) < p0).astype(float)
    y1 = (rng.random(size=n) < p1).astype(float)
    return y0, y1, float(p1 - p0)


def gen_skew_component(
    rng: Generator,
    n: int,
    dist: str,
    gamma_shape: float,
    log_sigma: float,
) -> np.ndarray:
    if dist == "gamma":
        shape = max(gamma_shape, 1e-6)
        scale = 1.0
        x = rng.gamma(shape=shape, scale=scale, size=n)
        mean = shape * scale
        var = shape * (scale ** 2)
    elif dist == "lognormal":
        sigma = max(log_sigma, 1e-6)
        x = rng.lognormal(mean=0.0, sigma=sigma, size=n)
        mean = math.exp(0.5 * sigma ** 2)
        var = (math.exp(sigma ** 2) - 1.0) * math.exp(sigma ** 2)
    else:
        raise ValueError(f"Unknown skew_dist: {dist}")
    std = math.sqrt(var) if var > 0 else 1.0
    z = (x - mean) / std
    return z.astype(float, copy=False)


def gen_deterministic_symmetric_spike(
    n: int,
    spike_prob: float,
    spike_scale: float,
    base_scale: float = 1.0,
    standardize: bool = True,
    mode: str = "rounding",
    rng: Generator | None = None,
) -> np.ndarray:
    half = n // 2
    if half <= 0:
        return np.array([0.0], dtype=float)

    values = np.full(half, base_scale, dtype=float)
    if mode == "randomized":
        if rng is None:
            raise ValueError("rng is required for randomized spike mode")
        mask = rng.random(half) < spike_prob
        if spike_prob > 0 and not mask.any():
            mask[int(rng.integers(0, half))] = True
        values[mask] = spike_scale
    elif mode == "smooth":
        target = half * spike_prob
        spike_pairs = int(math.floor(target))
        frac = target - spike_pairs
        spike_pairs = min(max(spike_pairs, 0), half)
        if spike_pairs > 0:
            values[:spike_pairs] = spike_scale
        if spike_pairs < half and frac > 0:
            values[spike_pairs] = base_scale + frac * (spike_scale - base_scale)
    else:
        spike_pairs = int(round(half * spike_prob))
        if spike_prob > 0:
            spike_pairs = max(1, spike_pairs)
        spike_pairs = min(spike_pairs, half)
        if spike_pairs > 0:
            values[:spike_pairs] = spike_scale
    y = np.concatenate([values, -values])
    if n % 2 == 1:
        y = np.concatenate([y, np.array([0.0])])
    if standardize:
        mean = float(np.mean(y))
        std = float(np.std(y, ddof=0))
        if std > 0:
            y = (y - mean) / std
    return y.astype(float, copy=False)


def gen_deterministic_lognormal(
    n: int,
    mu: float,
    sigma: float,
    standardize: bool = True,
) -> np.ndarray:
    u = (np.arange(1, n + 1, dtype=float) - 0.5) / n
    z = np.array([norm_ppf(ui) for ui in u], dtype=float)
    y = np.exp(mu + sigma * z)
    if standardize:
        mean = float(np.mean(y))
        std = float(np.std(y, ddof=0))
        if std > 0:
            y = (y - mean) / std
    return y.astype(float, copy=False)


def standardize_array(y: np.ndarray) -> np.ndarray:
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=0))
    if std > 0:
        y = (y - mean) / std
    return y.astype(float, copy=False)


def build_strata(n: int, weights: List[float]) -> List[np.ndarray]:
    counts = [int(round(w * n)) for w in weights]
    diff = n - sum(counts)
    counts[-1] += diff
    idxs = []
    start = 0
    for c in counts:
        idxs.append(np.arange(start, start + c, dtype=int))
        start += c
    return idxs


def build_clusters_regime_a(n: int, size: int = 10) -> List[np.ndarray]:
    clusters = []
    start = 0
    while start < n:
        end = min(n, start + size)
        clusters.append(np.arange(start, end, dtype=int))
        start = end
    return clusters


def build_clusters_regime_b(n: int) -> List[np.ndarray]:
    size = int(n ** 0.4)
    size = max(2, size)
    clusters = []
    start = 0
    while start < n:
        end = min(n, start + size)
        clusters.append(np.arange(start, end, dtype=int))
        start = end
    return clusters


def build_clusters_violation(n: int, share: float = 0.30) -> List[np.ndarray]:
    share = max(0.05, min(share, 0.95))
    big = int(round(share * n))
    rest = n - big
    clusters = [np.arange(0, big, dtype=int)]
    size = max(5, rest // 10)
    start = big
    while start < n:
        end = min(n, start + size)
        clusters.append(np.arange(start, end, dtype=int))
        start = end
    return clusters


def simulate_population(
    y0: np.ndarray,
    y1: np.ndarray,
    tau: float,
    assign_fn,
    r_skew: int,
    r_cov: int,
    alpha: float,
    rng_skew: Generator,
    rng_cov: Generator,
    lattice: bool = False,
    jitter: bool = False,
    one_sided: bool = False,
    antithetic: bool = False,
) -> Tuple[float, float, float, float, CoverageStats, CoverageStats, CoverageStats, float, float, float]:
    n = y0.size

    def complement_assignment(treated: np.ndarray) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        mask[treated] = True
        return np.flatnonzero(~mask)

    def iter_assignments(rng: Generator, count: int):
        if count <= 0:
            return
        if not antithetic:
            for _ in range(count):
                yield assign_fn(rng)
            return

        first = assign_fn(rng)
        n1 = first.size
        n0 = n - n1
        if n1 != n0:
            yield first
            for _ in range(count - 1):
                yield assign_fn(rng)
            return

        yield first
        yield complement_assignment(first)
        produced = 2
        pairs = (count - produced) // 2
        for _ in range(pairs):
            tr = assign_fn(rng)
            yield tr
            yield complement_assignment(tr)
        if (count - produced) % 2 == 1:
            yield assign_fn(rng)

    t_vals = np.empty(r_skew, dtype=float)
    deltas = []
    for i, tr in enumerate(iter_assignments(rng_skew, r_skew)):
        tau_hat, vhat, _ = compute_vhat(y0, y1, tr)
        t = (tau_hat - tau) / math.sqrt(vhat)
        t_vals[i] = t
        if lattice:
            deltas.append(lattice_span(tr.size, y0.size - tr.size, vhat))
    gamma_hat = skewness(t_vals)
    kurt = kurtosis_excess(t_vals)
    periodicity = periodicity_metric(t_vals) if lattice else float("nan")

    if lattice and jitter:
        delta = float(np.mean(deltas)) if deltas else 0.0
        t_for_quant = np.sort(t_vals + rng_skew.uniform(-delta / 2.0, delta / 2.0, size=t_vals.size))
        periodicity_j = periodicity_metric(t_for_quant)
    else:
        t_for_quant = np.sort(t_vals)
        periodicity_j = float("nan")

    if t_for_quant.size == 0:
        q_low = -1.96
        q_high = 1.96
        q_one = norm_ppf(1 - alpha)
    else:
        lo_idx = max(0, int((alpha / 2) * t_for_quant.size) - 1)
        hi_idx = min(t_for_quant.size - 1, int((1 - alpha / 2) * t_for_quant.size))
        q_low = float(t_for_quant[lo_idx])
        q_high = float(t_for_quant[hi_idx])
        one_idx = min(t_for_quant.size - 1, int((1 - alpha) * t_for_quant.size))
        q_one = float(t_for_quant[one_idx])

    gauss = CoverageStats()
    cf = CoverageStats()
    calib = CoverageStats()

    for tr in iter_assignments(rng_cov, r_cov):
        tau_hat, vhat, _ = compute_vhat(y0, y1, tr)
        se = math.sqrt(vhat)
        if one_sided:
            lo_g, hi_g = one_sided_interval_gaussian(tau_hat, se, alpha, upper=True)
            lo_cf, hi_cf = one_sided_interval_cf(tau_hat, se, alpha, gamma_hat, upper=True)
            lo_cal = tau_hat - q_one * se
            hi_cal = float("inf")
        else:
            lo_g, hi_g = interval_gaussian(tau_hat, se)
            lo_cf, hi_cf = interval_cf(tau_hat, se, gamma_hat)
            lo_cal, hi_cal = interval_calibrated(tau_hat, se, q_low, q_high)

        gauss.add(lo_g <= tau <= hi_g, hi_g - lo_g if math.isfinite(hi_g) else float("inf"))
        cf.add(lo_cf <= tau <= hi_cf, hi_cf - lo_cf if math.isfinite(hi_cf) else float("inf"))
        calib.add(lo_cal <= tau <= hi_cal, hi_cal - lo_cal if math.isfinite(hi_cal) else float("inf"))

    return gamma_hat, kurt, q_low, q_high, gauss, cf, calib, q_one, periodicity, periodicity_j


# ---------- Module implementations ----------

def run_cra_sampling(cfg: Dict, seed: int, run_id: str, repro_root: str, master: List[Dict], config_path: str, alpha: float) -> None:
    n_grid = cfg["n_grid"]
    f_grid = cfg["f_grid"]
    near_c = cfg["near_census_c"]
    B = cfg["B"]
    r_skew = cfg["r_skew"]
    r_cov = cfg["r_cov"]
    mu = cfg["mu"]
    sigma = cfg["sigma"]
    delta = cfg["delta"]

    experiment = "cra_sampling"
    spawn_keys = []
    f_vals_by_n: Dict[int, List[float]] = {}
    total_tasks = 0

    for n in n_grid:
        f_vals = list(f_grid)
        for c in near_c:
            f_nc = 1.0 - c / math.sqrt(n)
            if 0.0 < f_nc < 1.0:
                f_vals.append(round(f_nc, 6))
        f_vals = sorted(set(f_vals))
        f_vals_by_n[n] = f_vals
        total_tasks += len(f_vals)

    progress = ModuleProgress("cra_sampling", total_tasks)

    for n in n_grid:
        f_vals = f_vals_by_n[n]
        for f in f_vals:
            n1 = max(5, int(round(f * n)))
            n0 = n - n1
            if n0 < 5:
                n0 = 5
                n1 = n - n0
            f_eff = n1 / n

            tag = scenario_tag("cra", n=n, f=f_eff, B=B, r_skew=r_skew, r_cov=r_cov, mu=mu, sigma=sigma, delta=delta)
            spawn_keys.append({"N": n, "f": f_eff, "hash": stable_hash_int(tag)})
            ss = rng_sequence(seed, tag)
            pop_seqs = ss.spawn(B)

            gauss_all = CoverageStats()
            cf_all = CoverageStats()
            cal_all = CoverageStats()
            gamma_sum = 0.0
            kurt_sum = 0.0
            pop_count = 0

            progress.start_task(f"N={n} f={f_eff:.3f}", B)
            workers = int(os.environ.get("REPRO_WORKERS", "1"))
            if workers > 1 and B > 1:
                from multiprocessing import get_context

                ctx = get_context("spawn")
                args = [
                    (pop_ss, n, n1, r_skew, r_cov, alpha, mu, sigma, delta)
                    for pop_ss in pop_seqs
                ]
                with ctx.Pool(processes=workers) as pool:
                    for result in pool.imap(_cra_worker, args, chunksize=1):
                        (
                            ga_cov,
                            ga_cnt,
                            ga_len,
                            cf_cov,
                            cf_cnt,
                            cf_len,
                            cal_cov,
                            cal_cnt,
                            cal_len,
                            gamma_hat,
                            kurt,
                        ) = result
                        gauss_all.covered += ga_cov
                        gauss_all.count += ga_cnt
                        gauss_all.length_sum += ga_len
                        cf_all.covered += cf_cov
                        cf_all.count += cf_cnt
                        cf_all.length_sum += cf_len
                        cal_all.covered += cal_cov
                        cal_all.count += cal_cnt
                        cal_all.length_sum += cal_len
                        gamma_sum += gamma_hat
                        kurt_sum += kurt
                        pop_count += 1
                        progress.update()
            else:
                for pop_ss in pop_seqs:
                    (
                        ga_cov,
                        ga_cnt,
                        ga_len,
                        cf_cov,
                        cf_cnt,
                        cf_len,
                        cal_cov,
                        cal_cnt,
                        cal_len,
                        gamma_hat,
                        kurt,
                    ) = _cra_worker((pop_ss, n, n1, r_skew, r_cov, alpha, mu, sigma, delta))
                    gauss_all.covered += ga_cov
                    gauss_all.count += ga_cnt
                    gauss_all.length_sum += ga_len
                    cf_all.covered += cf_cov
                    cf_all.count += cf_cnt
                    cf_all.length_sum += cf_len
                    cal_all.covered += cal_cov
                    cal_all.count += cal_cnt
                    cal_all.length_sum += cal_len
                    gamma_sum += gamma_hat
                    kurt_sum += kurt
                    pop_count += 1
                    progress.update()

            gamma_mean = gamma_sum / pop_count if pop_count else float("nan")
            kurt_mean = kurt_sum / pop_count if pop_count else float("nan")
            progress.finish_task()

            for method, stats in [("gaussian", gauss_all), ("cornish_fisher", cf_all), ("calibrated", cal_all)]:
                master.append(
                    {
                        "module": experiment,
                        "design": "CRA",
                        "outcome": "continuous",
                        "method": method,
                        "N": n,
                        "m_N": n,
                        "f": f_eff,
                        "B": B,
                        "R": r_cov,
                        "S": "",
                        "coverage": stats.rate(),
                        "mcse": stats.mcse(),
                        "avg_length": stats.mean_length(),
                        "skew": gamma_mean,
                        "kurtosis": kurt_mean,
                        "periodicity": "",
                        "lambda_N": "",
                        "variance_ratio": "",
                        "endpoint_diff": "",
                    }
                )

    make_manifest(repro_root, experiment, run_id, config_path, seed, spawn_keys)
    progress.finish()

    # Plot: coverage error vs f for N=max
    rows = [r for r in master if r["module"] == experiment and r["N"] == max(n_grid)]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in ["gaussian", "cornish_fisher", "calibrated"]:
            pts = sorted([r for r in rows if r["method"] == method], key=lambda x: x["f"])
            xs = [p["f"] for p in pts]
            ys = [abs(p["coverage"] - (1 - alpha)) for p in pts]
            ax.plot(xs, ys, marker="o", label=method)
        ax.set_xlabel("f")
        ax.set_ylabel("|coverage - (1-α)|")
        ax.set_title(f"CRA sampling-fraction errors (N={max(n_grid)})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(repro_root, "plots", experiment, run_id, "figs", "cra_sampling_errors.png")
        ensure_dir(os.path.dirname(plot_path))
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)


def run_parity(cfg: Dict, seed: int, run_id: str, repro_root: str, master: List[Dict], config_path: str, alpha: float) -> None:
    n_grid = cfg["n_grid"]
    B = cfg["B"]
    r_skew = cfg["r_skew"]
    r_cov = cfg["r_cov"]
    delta = cfg["delta"]
    symmetric_dgp = cfg.get("symmetric_dgp", "normal")
    t_df = float(cfg.get("t_df", 4.0))
    mix_weight = float(cfg.get("mix_weight", 0.1))
    mix_scale = float(cfg.get("mix_scale", 4.0))
    spike_prob = float(cfg.get("spike_prob", 0.02))
    spike_scale = float(cfg.get("spike_scale", 20.0))
    spike_standardize = bool(cfg.get("spike_standardize", True))
    antithetic = bool(cfg.get("antithetic_assignments", False))
    parity_holds_f = float(cfg.get("parity_holds_f", 0.5))
    parity_fails_f = float(cfg.get("parity_fails_f", 0.7))

    experiment = "parity"
    spawn_keys = []
    total_tasks = len(n_grid) * 2
    progress = ModuleProgress("parity", total_tasks)

    for regime in ["parity_holds", "parity_fails"]:
        for n in n_grid:
            f_eff = parity_holds_f if regime == "parity_holds" else parity_fails_f
            n1 = max(5, int(round(f_eff * n)))
            n0 = n - n1
            if n0 < 5:
                n0 = 5
                n1 = n - n0
            f_eff = n1 / n
            tag = scenario_tag(
                "parity",
                regime=regime,
                n=n,
                f=f_eff,
                B=B,
                r_skew=r_skew,
                r_cov=r_cov,
                delta=delta,
                symmetric_dgp=symmetric_dgp,
                t_df=t_df,
                mix_weight=mix_weight,
                mix_scale=mix_scale,
                spike_prob=spike_prob,
                spike_scale=spike_scale,
                spike_standardize=spike_standardize,
                antithetic=antithetic,
            )
            spawn_keys.append(
                {
                    "N": n,
                    "f": f_eff,
                    "regime": regime,
                    "symmetric_dgp": symmetric_dgp,
                    "t_df": t_df,
                    "mix_weight": mix_weight,
                    "mix_scale": mix_scale,
                    "spike_prob": spike_prob,
                    "spike_scale": spike_scale,
                    "spike_standardize": spike_standardize,
                    "antithetic": antithetic,
                    "hash": stable_hash_int(tag),
                }
            )
            ss = rng_sequence(seed, tag)
            pop_seqs = ss.spawn(B)

            gauss_all = CoverageStats()
            cf_all = CoverageStats()
            cal_all = CoverageStats()
            gamma_sum = 0.0
            pop_count = 0

            progress.start_task(f"{regime} N={n}", B)
            workers = int(os.environ.get("REPRO_WORKERS", "1"))
            if workers > 1 and B > 1:
                from multiprocessing import get_context

                ctx = get_context("spawn")
                args = [
                    (
                        pop_ss,
                        n,
                        n1,
                        r_skew,
                        r_cov,
                        alpha,
                        delta,
                        regime,
                        symmetric_dgp,
                        t_df,
                        mix_weight,
                        mix_scale,
                        spike_prob,
                        spike_scale,
                        spike_standardize,
                        antithetic,
                    )
                    for pop_ss in pop_seqs
                ]
                with ctx.Pool(processes=workers) as pool:
                    for result in pool.imap(_parity_worker, args, chunksize=1):
                        (
                            ga_cov,
                            ga_cnt,
                            ga_len,
                            cf_cov,
                            cf_cnt,
                            cf_len,
                            cal_cov,
                            cal_cnt,
                            cal_len,
                            gamma_hat,
                        ) = result
                        gauss_all.covered += ga_cov
                        gauss_all.count += ga_cnt
                        gauss_all.length_sum += ga_len
                        cf_all.covered += cf_cov
                        cf_all.count += cf_cnt
                        cf_all.length_sum += cf_len
                        cal_all.covered += cal_cov
                        cal_all.count += cal_cnt
                        cal_all.length_sum += cal_len
                        gamma_sum += gamma_hat
                        pop_count += 1
                        progress.update()
            else:
                for pop_ss in pop_seqs:
                    (
                        ga_cov,
                        ga_cnt,
                        ga_len,
                        cf_cov,
                        cf_cnt,
                        cf_len,
                        cal_cov,
                        cal_cnt,
                        cal_len,
                        gamma_hat,
                    ) = _parity_worker(
                        (
                            pop_ss,
                            n,
                            n1,
                            r_skew,
                            r_cov,
                            alpha,
                            delta,
                            regime,
                            symmetric_dgp,
                            t_df,
                            mix_weight,
                            mix_scale,
                            spike_prob,
                            spike_scale,
                            spike_standardize,
                            antithetic,
                        )
                    )
                    gauss_all.covered += ga_cov
                    gauss_all.count += ga_cnt
                    gauss_all.length_sum += ga_len
                    cf_all.covered += cf_cov
                    cf_all.count += cf_cnt
                    cf_all.length_sum += cf_len
                    cal_all.covered += cal_cov
                    cal_all.count += cal_cnt
                    cal_all.length_sum += cal_len
                    gamma_sum += gamma_hat
                    pop_count += 1
                    progress.update()

            gamma_mean = gamma_sum / pop_count if pop_count else float("nan")
            progress.finish_task()

            for method, stats in [("gaussian", gauss_all), ("cornish_fisher", cf_all), ("calibrated", cal_all)]:
                master.append(
                    {
                        "module": experiment,
                        "design": regime,
                        "outcome": "continuous",
                        "method": method,
                        "N": n,
                        "m_N": n,
                        "f": f_eff,
                        "B": B,
                        "R": r_cov,
                        "S": "",
                        "coverage": stats.rate(),
                        "mcse": stats.mcse(),
                        "avg_length": stats.mean_length(),
                        "skew": gamma_mean,
                        "kurtosis": "",
                        "periodicity": "",
                        "lambda_N": "",
                        "variance_ratio": "",
                        "endpoint_diff": "",
                    }
                )

    make_manifest(repro_root, experiment, run_id, config_path, seed, spawn_keys)
    progress.finish()

    # Plot: rate scaling for parity holds vs fails (gaussian)
    rows = [r for r in master if r["module"] == experiment and r["method"] == "gaussian"]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for regime in ["parity_holds", "parity_fails"]:
            pts = sorted([r for r in rows if r["design"] == regime], key=lambda x: x["N"])
            xs = [p["N"] for p in pts]
            ys = [abs(p["coverage"] - (1 - alpha)) * math.sqrt(p["N"]) for p in pts]
            ax.plot(xs, ys, marker="o", label=regime)
        ax.set_xlabel("N")
        ax.set_ylabel("sqrt(N)*|coverage error|")
        ax.set_title("Parity regimes: gaussian scaling")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_path = os.path.join(repro_root, "plots", experiment, run_id, "figs", "parity_scaling.png")
        ensure_dir(os.path.dirname(plot_path))
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)


def run_parity_deterministic(
    cfg: Dict,
    seed: int,
    run_id: str,
    repro_root: str,
    master: List[Dict],
    config_path: str,
    alpha: float,
) -> None:
    n_grid = cfg["n_grid"]
    r_skew = cfg["r_skew"]
    r_cov = cfg["r_cov"]
    delta = float(cfg.get("delta", 0.5))
    spike_prob = float(cfg.get("spike_prob", 0.02))
    spike_scale = float(cfg.get("spike_scale", 20.0))
    spike_standardize = bool(cfg.get("spike_standardize", True))
    spike_mode = str(cfg.get("spike_mode", "rounding"))
    base_scale = float(cfg.get("base_scale", 1.0))
    parity_holds_dgp = str(cfg.get("parity_holds_dgp", "deterministic_spike"))
    parity_holds_standardize = bool(cfg.get("parity_holds_standardize", True))
    t_df = float(cfg.get("t_df", 3.0))
    mix_weight = float(cfg.get("mix_weight", 0.05))
    mix_scale = float(cfg.get("mix_scale", 10.0))
    parity_holds_f = float(cfg.get("parity_holds_f", 0.5))
    parity_fails_f = float(cfg.get("parity_fails_f", 0.7))
    parity_fails_dgp = str(cfg.get("parity_fails_dgp", "deterministic_lognormal"))
    log_mu = float(cfg.get("parity_fails_lognormal_mu", 0.0))
    log_sigma = float(cfg.get("parity_fails_lognormal_sigma", 1.2))
    log_standardize = bool(cfg.get("parity_fails_standardize", True))
    antithetic = bool(cfg.get("antithetic_assignments", False))

    experiment = "parity_det"
    spawn_keys = []
    total_tasks = len(n_grid) * 2
    progress = ModuleProgress("parity_det", total_tasks)

    for regime in ["parity_holds", "parity_fails"]:
        for n in n_grid:
            f_eff = parity_holds_f if regime == "parity_holds" else parity_fails_f
            n1 = max(5, int(round(f_eff * n)))
            n0 = n - n1
            if n0 < 5:
                n0 = 5
                n1 = n - n0
            f_eff = n1 / n

            tag = scenario_tag(
                "parity_det",
                regime=regime,
                n=n,
                f=f_eff,
                r_skew=r_skew,
                r_cov=r_cov,
                delta=delta,
                spike_prob=spike_prob,
                spike_scale=spike_scale,
                spike_standardize=spike_standardize,
                spike_mode=spike_mode,
                base_scale=base_scale,
                parity_holds_dgp=parity_holds_dgp,
                parity_holds_standardize=parity_holds_standardize,
                t_df=t_df,
                mix_weight=mix_weight,
                mix_scale=mix_scale,
                parity_fails_dgp=parity_fails_dgp,
                log_mu=log_mu,
                log_sigma=log_sigma,
                log_standardize=log_standardize,
                antithetic=antithetic,
            )
            spawn_keys.append(
                {
                    "N": n,
                    "f": f_eff,
                    "regime": regime,
                    "spike_prob": spike_prob,
                    "spike_scale": spike_scale,
                    "spike_standardize": spike_standardize,
                    "spike_mode": spike_mode,
                    "base_scale": base_scale,
                    "parity_holds_dgp": parity_holds_dgp,
                    "parity_holds_standardize": parity_holds_standardize,
                    "t_df": t_df,
                    "mix_weight": mix_weight,
                    "mix_scale": mix_scale,
                    "parity_fails_dgp": parity_fails_dgp,
                    "log_mu": log_mu,
                    "log_sigma": log_sigma,
                    "log_standardize": log_standardize,
                    "antithetic": antithetic,
                    "hash": stable_hash_int(tag),
                }
            )

            ss = rng_sequence(seed, tag)
            ss_pop, ss_sk, ss_cv = ss.spawn(3)
            rng_pop = Generator(Philox(ss_pop))
            rng_skew = Generator(Philox(ss_sk))
            rng_cov = Generator(Philox(ss_cv))

            if regime == "parity_holds":
                if parity_holds_dgp == "deterministic_spike":
                    y0 = gen_deterministic_symmetric_spike(
                        n,
                        spike_prob=spike_prob,
                        spike_scale=spike_scale,
                        base_scale=base_scale,
                        standardize=spike_standardize,
                        mode=spike_mode,
                        rng=rng_pop,
                    )
                elif parity_holds_dgp == "symmetric_normal":
                    y0, _y1, _tau = gen_symmetric(rng_pop, n, delta)
                    if parity_holds_standardize:
                        y0 = standardize_array(y0)
                elif parity_holds_dgp == "symmetric_t":
                    y0, _y1, _tau = gen_symmetric_t(rng_pop, n, delta, t_df)
                    if parity_holds_standardize:
                        y0 = standardize_array(y0)
                elif parity_holds_dgp == "symmetric_mixture":
                    y0, _y1, _tau = gen_symmetric_mixture(rng_pop, n, delta, mix_weight, mix_scale)
                    if parity_holds_standardize:
                        y0 = standardize_array(y0)
                elif parity_holds_dgp == "symmetric_spike":
                    y0, _y1, _tau = gen_symmetric_spike(
                        rng_pop, n, delta, spike_prob, spike_scale, spike_standardize
                    )
                    if parity_holds_standardize and not spike_standardize:
                        y0 = standardize_array(y0)
                else:
                    raise ValueError(f"Unknown parity_holds_dgp: {parity_holds_dgp}")
            else:
                if parity_fails_dgp == "deterministic_lognormal":
                    y0 = gen_deterministic_lognormal(
                        n,
                        mu=log_mu,
                        sigma=log_sigma,
                        standardize=log_standardize,
                    )
                elif parity_fails_dgp == "lognormal":
                    y0, _y1, _tau = gen_lognormal(rng_pop, n, log_mu, log_sigma, delta)
                    if log_standardize:
                        y0 = standardize_array(y0)
                else:
                    raise ValueError(f"Unknown parity_fails_dgp: {parity_fails_dgp}")
            y1 = y0 + delta
            tau = float(delta)

            assign_fn = lambda rng, n=n, n1=n1: draw_cra(rng, n, n1)

            progress.start_task(f"{regime} N={n}", r_cov)
            gamma_hat, kurt, _ql, _qh, ga, cf, cal, _q1, _p, _pj = simulate_population(
                y0,
                y1,
                tau,
                assign_fn,
                r_skew,
                r_cov,
                alpha,
                rng_skew,
                rng_cov,
                antithetic=antithetic,
            )
            progress.finish_task()

            for method, stats in [("gaussian", ga), ("cornish_fisher", cf), ("calibrated", cal)]:
                master.append(
                    {
                        "module": experiment,
                        "design": regime,
                        "outcome": "continuous",
                        "method": method,
                        "N": n,
                        "m_N": n,
                        "f": f_eff,
                        "B": 1,
                        "R": r_cov,
                        "S": "",
                        "coverage": stats.rate(),
                        "mcse": stats.mcse(),
                        "avg_length": stats.mean_length(),
                        "skew": gamma_hat,
                        "kurtosis": kurt,
                        "periodicity": "",
                        "lambda_N": "",
                        "variance_ratio": "",
                        "endpoint_diff": "",
                    }
                )

    make_manifest(repro_root, experiment, run_id, config_path, seed, spawn_keys)
    progress.finish()


def run_skew_rate_diag(
    cfg: Dict,
    seed: int,
    run_id: str,
    repro_root: str,
    master: List[Dict],
    config_path: str,
    alpha: float,
) -> None:
    n_grid = cfg["n_grid"]
    f_grid = cfg["f_grid"]
    B = cfg.get("B", 1)
    r_skew = cfg["r_skew"]
    r_cov = cfg["r_cov"]
    delta = float(cfg.get("delta", 0.5))
    base_dist = str(cfg.get("base_dist", "normal"))
    t_df = float(cfg.get("t_df", 8.0))
    base_standardize = bool(cfg.get("base_standardize", True))
    skew_dist = str(cfg.get("skew_dist", "gamma"))
    gamma_shape = float(cfg.get("gamma_shape", 4.0))
    log_sigma = float(cfg.get("log_sigma", 0.4))
    skew_scale = float(cfg.get("skew_scale", 1.0))

    experiment = "skew_rate_diag"
    spawn_keys = []
    total_tasks = len(n_grid) * len(f_grid)
    progress = ModuleProgress("skew_rate_diag", total_tasks)

    for n in n_grid:
        for f in f_grid:
            n1 = max(5, int(round(f * n)))
            n0 = n - n1
            if n0 < 5:
                n0 = 5
                n1 = n - n0
            f_eff = n1 / n

            tag = scenario_tag(
                "skew_rate_diag",
                n=n,
                f=f_eff,
                B=B,
                r_skew=r_skew,
                r_cov=r_cov,
                delta=delta,
                base_dist=base_dist,
                t_df=t_df,
                base_standardize=base_standardize,
                skew_dist=skew_dist,
                gamma_shape=gamma_shape,
                log_sigma=log_sigma,
                skew_scale=skew_scale,
            )
            spawn_keys.append(
                {
                    "N": n,
                    "f": f_eff,
                    "base_dist": base_dist,
                    "t_df": t_df,
                    "base_standardize": base_standardize,
                    "skew_dist": skew_dist,
                    "gamma_shape": gamma_shape,
                    "log_sigma": log_sigma,
                    "skew_scale": skew_scale,
                    "hash": stable_hash_int(tag),
                }
            )

            ss = rng_sequence(seed, tag)
            pop_seqs = ss.spawn(B)

            gauss_all = CoverageStats()
            cf_all = CoverageStats()
            cal_all = CoverageStats()
            gamma_sum = 0.0
            kurt_sum = 0.0
            pop_count = 0

            progress.start_task(f"N={n} f={f_eff:.2f}", B)
            workers = int(os.environ.get("REPRO_WORKERS", "1"))
            if workers > 1 and B > 1:
                from multiprocessing import get_context

                ctx = get_context("spawn")
                args = [
                    (
                        pop_ss,
                        n,
                        n1,
                        r_skew,
                        r_cov,
                        alpha,
                        delta,
                        base_dist,
                        t_df,
                        base_standardize,
                        skew_dist,
                        gamma_shape,
                        log_sigma,
                        skew_scale,
                    )
                    for pop_ss in pop_seqs
                ]
                with ctx.Pool(processes=workers) as pool:
                    for result in pool.imap(_skew_rate_worker, args, chunksize=1):
                        (
                            ga_cov,
                            ga_cnt,
                            ga_len,
                            cf_cov,
                            cf_cnt,
                            cf_len,
                            cal_cov,
                            cal_cnt,
                            cal_len,
                            gamma_hat,
                            kurt,
                        ) = result
                        gauss_all.covered += ga_cov
                        gauss_all.count += ga_cnt
                        gauss_all.length_sum += ga_len
                        cf_all.covered += cf_cov
                        cf_all.count += cf_cnt
                        cf_all.length_sum += cf_len
                        cal_all.covered += cal_cov
                        cal_all.count += cal_cnt
                        cal_all.length_sum += cal_len
                        gamma_sum += gamma_hat
                        kurt_sum += kurt
                        pop_count += 1
                        progress.update()
            else:
                for pop_ss in pop_seqs:
                    (
                        ga_cov,
                        ga_cnt,
                        ga_len,
                        cf_cov,
                        cf_cnt,
                        cf_len,
                        cal_cov,
                        cal_cnt,
                        cal_len,
                        gamma_hat,
                        kurt,
                    ) = _skew_rate_worker(
                        (
                            pop_ss,
                            n,
                            n1,
                            r_skew,
                            r_cov,
                            alpha,
                            delta,
                            base_dist,
                            t_df,
                            base_standardize,
                            skew_dist,
                            gamma_shape,
                            log_sigma,
                            skew_scale,
                        )
                    )
                    gauss_all.covered += ga_cov
                    gauss_all.count += ga_cnt
                    gauss_all.length_sum += ga_len
                    cf_all.covered += cf_cov
                    cf_all.count += cf_cnt
                    cf_all.length_sum += cf_len
                    cal_all.covered += cal_cov
                    cal_all.count += cal_cnt
                    cal_all.length_sum += cal_len
                    gamma_sum += gamma_hat
                    kurt_sum += kurt
                    pop_count += 1
                    progress.update()

            gamma_mean = gamma_sum / pop_count if pop_count else float("nan")
            kurt_mean = kurt_sum / pop_count if pop_count else float("nan")
            progress.finish_task()

            design = f"f={f_eff:.2f}"
            for method, stats in [("gaussian", gauss_all), ("cornish_fisher", cf_all), ("calibrated", cal_all)]:
                master.append(
                    {
                        "module": experiment,
                        "design": design,
                        "outcome": "continuous",
                        "method": method,
                        "N": n,
                        "m_N": n,
                        "f": f_eff,
                        "B": B,
                        "R": r_cov,
                        "S": "",
                        "coverage": stats.rate(),
                        "mcse": stats.mcse(),
                        "avg_length": stats.mean_length(),
                        "skew": gamma_mean,
                        "kurtosis": kurt_mean,
                        "periodicity": "",
                        "lambda_N": "",
                        "variance_ratio": "",
                        "endpoint_diff": "",
                    }
                )

    make_manifest(repro_root, experiment, run_id, config_path, seed, spawn_keys)
    progress.finish()


def run_lattice(cfg: Dict, seed: int, run_id: str, repro_root: str, master: List[Dict], config_path: str, alpha: float) -> None:
    n_grid = cfg["n_grid"]
    B = cfg["B"]
    r_skew = cfg["r_skew"]
    r_cov = cfg["r_cov"]
    p0 = cfg["p0"]
    p1 = cfg["p1"]

    experiment = "lattice"
    spawn_keys = []
    progress = ModuleProgress("lattice", len(n_grid))

    for n in n_grid:
        f_eff = 0.5
        n1 = max(5, int(round(f_eff * n)))
        n0 = n - n1
        if n0 < 5:
            n0 = 5
            n1 = n - n0
        f_eff = n1 / n

        tag = scenario_tag("lattice", n=n, f=f_eff, B=B, r_skew=r_skew, r_cov=r_cov, p0=p0, p1=p1)
        spawn_keys.append({"N": n, "f": f_eff, "hash": stable_hash_int(tag)})
        ss = rng_sequence(seed, tag)
        pop_seqs = ss.spawn(B)

        gauss_all = CoverageStats()
        cf_all = CoverageStats()
        cal_all = CoverageStats()
        cal_jit_all = CoverageStats()
        periodicity_sum = 0.0
        periodicity_j_sum = 0.0
        pop_count = 0

        progress.start_task(f"N={n}", B)
        workers = int(os.environ.get("REPRO_WORKERS", "1"))
        if workers > 1 and B > 1:
            from multiprocessing import get_context

            ctx = get_context("spawn")
            args = [
                (pop_ss, n, n1, r_skew, r_cov, alpha, p0, p1)
                for pop_ss in pop_seqs
            ]
            with ctx.Pool(processes=workers) as pool:
                for result in pool.imap(_lattice_worker, args, chunksize=1):
                    (
                        ga_cov,
                        ga_cnt,
                        ga_len,
                        cf_cov,
                        cf_cnt,
                        cf_len,
                        cal_cov,
                        cal_cnt,
                        cal_len,
                        calj_cov,
                        calj_cnt,
                        calj_len,
                        per,
                        perj2,
                    ) = result
                    gauss_all.covered += ga_cov
                    gauss_all.count += ga_cnt
                    gauss_all.length_sum += ga_len
                    cf_all.covered += cf_cov
                    cf_all.count += cf_cnt
                    cf_all.length_sum += cf_len
                    cal_all.covered += cal_cov
                    cal_all.count += cal_cnt
                    cal_all.length_sum += cal_len
                    cal_jit_all.covered += calj_cov
                    cal_jit_all.count += calj_cnt
                    cal_jit_all.length_sum += calj_len
                    periodicity_sum += per
                    periodicity_j_sum += perj2
                    pop_count += 1
                    progress.update()
        else:
            for pop_ss in pop_seqs:
                (
                    ga_cov,
                    ga_cnt,
                    ga_len,
                    cf_cov,
                    cf_cnt,
                    cf_len,
                    cal_cov,
                    cal_cnt,
                    cal_len,
                    calj_cov,
                    calj_cnt,
                    calj_len,
                    per,
                    perj2,
                ) = _lattice_worker((pop_ss, n, n1, r_skew, r_cov, alpha, p0, p1))
                gauss_all.covered += ga_cov
                gauss_all.count += ga_cnt
                gauss_all.length_sum += ga_len
                cf_all.covered += cf_cov
                cf_all.count += cf_cnt
                cf_all.length_sum += cf_len
                cal_all.covered += cal_cov
                cal_all.count += cal_cnt
                cal_all.length_sum += cal_len
                cal_jit_all.covered += calj_cov
                cal_jit_all.count += calj_cnt
                cal_jit_all.length_sum += calj_len
                periodicity_sum += per
                periodicity_j_sum += perj2
                pop_count += 1
                progress.update()

        periodicity_mean = periodicity_sum / pop_count if pop_count else float("nan")
        periodicity_j_mean = periodicity_j_sum / pop_count if pop_count else float("nan")
        progress.finish_task()

        for method, stats in [
            ("gaussian", gauss_all),
            ("cornish_fisher", cf_all),
            ("calibrated", cal_all),
            ("calibrated_jitter", cal_jit_all),
        ]:
            master.append(
                {
                    "module": experiment,
                    "design": "CRA",
                    "outcome": "binary",
                    "method": method,
                    "N": n,
                    "m_N": n,
                    "f": f_eff,
                    "B": B,
                    "R": r_cov,
                    "S": "",
                    "coverage": stats.rate(),
                    "mcse": stats.mcse(),
                    "avg_length": stats.mean_length(),
                    "skew": "",
                    "kurtosis": "",
                    "periodicity": periodicity_j_mean if method == "calibrated_jitter" else periodicity_mean,
                    "lambda_N": "",
                    "variance_ratio": "",
                    "endpoint_diff": "",
                }
            )

    make_manifest(repro_root, experiment, run_id, config_path, seed, spawn_keys)
    progress.finish()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_grid, [float(periodicity_mean) for _ in n_grid], label="raw")
    ax.plot(n_grid, [float(periodicity_j_mean) for _ in n_grid], label="jittered")
    ax.set_xlabel("N")
    ax.set_ylabel("periodicity (max jump)")
    ax.set_title("Lattice periodicity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(repro_root, "plots", experiment, run_id, "figs", "lattice_periodicity.png")
    ensure_dir(os.path.dirname(plot_path))
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)


def run_stratified(cfg: Dict, seed: int, run_id: str, repro_root: str, master: List[Dict], config_path: str, alpha: float) -> None:
    n_grid = cfg["n_grid"]
    f_grid = cfg["f_grid"]
    B = cfg["B"]
    r_skew = cfg["r_skew"]
    r_cov = cfg["r_cov"]
    weights = cfg["strata_weights"]
    sigmas = cfg["strata_sigma"]
    mu = cfg["mu"]
    delta = cfg["delta"]
    p0 = cfg["p0"]
    p1 = cfg["p1"]

    experiment = "stratified"
    spawn_keys = []
    total_tasks = len(n_grid) * len(f_grid) * 2
    progress = ModuleProgress("stratified", total_tasks)

    for n in n_grid:
        strata = build_strata(n, weights)
        for f in f_grid:
            n1_per = [max(5, int(round(f * len(s)))) for s in strata]
            tag = scenario_tag("strat", n=n, f=f, B=B, r_skew=r_skew, r_cov=r_cov)
            spawn_keys.append({"N": n, "f": f, "hash": stable_hash_int(tag)})
            ss = rng_sequence(seed, tag)
            pop_seqs = ss.spawn(B)

            for outcome in ["continuous", "binary"]:
                gauss_all = CoverageStats()
                cf_all = CoverageStats()
                cal_all = CoverageStats()

                progress.start_task(f"{outcome} N={n} f={f:.2f}", B)
                workers = int(os.environ.get("REPRO_WORKERS", "1"))
                if workers > 1 and B > 1:
                    from multiprocessing import get_context

                    ctx = get_context("spawn")
                    args = [
                        (pop_ss, n, strata, n1_per, r_skew, r_cov, alpha, mu, delta, p0, p1, outcome, sigmas)
                        for pop_ss in pop_seqs
                    ]
                    with ctx.Pool(processes=workers) as pool:
                        for result in pool.imap(_stratified_worker, args, chunksize=1):
                            (
                                ga_cov,
                                ga_cnt,
                                ga_len,
                                cf_cov,
                                cf_cnt,
                                cf_len,
                                cal_cov,
                                cal_cnt,
                                cal_len,
                            ) = result
                            gauss_all.covered += ga_cov
                            gauss_all.count += ga_cnt
                            gauss_all.length_sum += ga_len
                            cf_all.covered += cf_cov
                            cf_all.count += cf_cnt
                            cf_all.length_sum += cf_len
                            cal_all.covered += cal_cov
                            cal_all.count += cal_cnt
                            cal_all.length_sum += cal_len
                            progress.update()
                else:
                    for pop_ss in pop_seqs:
                        (
                            ga_cov,
                            ga_cnt,
                            ga_len,
                            cf_cov,
                            cf_cnt,
                            cf_len,
                            cal_cov,
                            cal_cnt,
                            cal_len,
                        ) = _stratified_worker((pop_ss, n, strata, n1_per, r_skew, r_cov, alpha, mu, delta, p0, p1, outcome, sigmas))
                        gauss_all.covered += ga_cov
                        gauss_all.count += ga_cnt
                        gauss_all.length_sum += ga_len
                        cf_all.covered += cf_cov
                        cf_all.count += cf_cnt
                        cf_all.length_sum += cf_len
                        cal_all.covered += cal_cov
                        cal_all.count += cal_cnt
                        cal_all.length_sum += cal_len
                        progress.update()
                progress.finish_task()

                for method, stats in [("gaussian", gauss_all), ("cornish_fisher", cf_all), ("calibrated", cal_all)]:
                    master.append(
                        {
                            "module": experiment,
                            "design": "stratified",
                            "outcome": outcome,
                            "method": method,
                            "N": n,
                            "m_N": n,
                            "f": f,
                            "B": B,
                            "R": r_cov,
                            "S": "",
                            "coverage": stats.rate(),
                            "mcse": stats.mcse(),
                            "avg_length": stats.mean_length(),
                            "skew": "",
                            "kurtosis": "",
                            "periodicity": "",
                            "lambda_N": "",
                            "variance_ratio": "",
                            "endpoint_diff": "",
                        }
                    )

    make_manifest(repro_root, experiment, run_id, config_path, seed, spawn_keys)
    progress.finish()


def run_cluster(cfg: Dict, seed: int, run_id: str, repro_root: str, master: List[Dict], config_path: str, alpha: float) -> None:
    n_grid = cfg["n_grid"]
    B = cfg["B"]
    r_skew = cfg["r_skew"]
    r_cov = cfg["r_cov"]
    f = cfg["f"]
    mu = cfg["mu"]
    sigma = cfg["sigma"]
    delta = cfg["delta"]
    p0 = cfg["p0"]
    p1 = cfg["p1"]
    violation_share = cfg.get("violation_share", 0.30)
    violation_force_big = bool(cfg.get("violation_force_big_cluster", False))

    experiment = "cluster"
    spawn_keys = []
    total_tasks = len(n_grid) * 3 * 2
    progress = ModuleProgress("cluster", total_tasks)

    for n in n_grid:
        for regime, build_fn in [
            ("regime_a", build_clusters_regime_a),
            ("regime_b", build_clusters_regime_b),
            ("violation", lambda n_val: build_clusters_violation(n_val, violation_share)),
        ]:
            clusters = build_fn(n)
            G = len(clusters)
            n_treated_clusters = max(1, int(round(f * G)))
            lambda_n = max(len(c) for c in clusters) / n

            tag = scenario_tag("cluster", n=n, regime=regime, f=f, B=B, r_skew=r_skew, r_cov=r_cov)
            spawn_keys.append({"N": n, "regime": regime, "hash": stable_hash_int(tag)})
            ss = rng_sequence(seed, tag)
            pop_seqs = ss.spawn(B)

            for outcome in ["continuous", "binary"]:
                gauss_all = CoverageStats()
                cf_all = CoverageStats()
                cal_all = CoverageStats()

                progress.start_task(f"{regime} {outcome} N={n}", B)
                workers = int(os.environ.get("REPRO_WORKERS", "1"))
                force_big = regime == "violation" and violation_force_big
                if workers > 1 and B > 1:
                    from multiprocessing import get_context

                    ctx = get_context("spawn")
                    args = [
                        (pop_ss, n, clusters, n_treated_clusters, r_skew, r_cov, alpha, mu, sigma, delta, p0, p1, outcome, force_big)
                        for pop_ss in pop_seqs
                    ]
                    with ctx.Pool(processes=workers) as pool:
                        for result in pool.imap(_cluster_worker, args, chunksize=1):
                            (
                                ga_cov,
                                ga_cnt,
                                ga_len,
                                cf_cov,
                                cf_cnt,
                                cf_len,
                                cal_cov,
                                cal_cnt,
                                cal_len,
                            ) = result
                            gauss_all.covered += ga_cov
                            gauss_all.count += ga_cnt
                            gauss_all.length_sum += ga_len
                            cf_all.covered += cf_cov
                            cf_all.count += cf_cnt
                            cf_all.length_sum += cf_len
                            cal_all.covered += cal_cov
                            cal_all.count += cal_cnt
                            cal_all.length_sum += cal_len
                            progress.update()
                else:
                    for pop_ss in pop_seqs:
                        (
                            ga_cov,
                            ga_cnt,
                            ga_len,
                            cf_cov,
                            cf_cnt,
                            cf_len,
                            cal_cov,
                            cal_cnt,
                            cal_len,
                        ) = _cluster_worker((pop_ss, n, clusters, n_treated_clusters, r_skew, r_cov, alpha, mu, sigma, delta, p0, p1, outcome, force_big))
                        gauss_all.covered += ga_cov
                        gauss_all.count += ga_cnt
                        gauss_all.length_sum += ga_len
                        cf_all.covered += cf_cov
                        cf_all.count += cf_cnt
                        cf_all.length_sum += cf_len
                        cal_all.covered += cal_cov
                        cal_all.count += cal_cnt
                        cal_all.length_sum += cal_len
                        progress.update()
                progress.finish_task()

                for method, stats in [("gaussian", gauss_all), ("cornish_fisher", cf_all), ("calibrated", cal_all)]:
                    master.append(
                        {
                            "module": experiment,
                            "design": regime,
                            "outcome": outcome,
                            "method": method,
                            "N": n,
                            "m_N": G,
                            "f": f,
                            "B": B,
                            "R": r_cov,
                            "S": "",
                            "coverage": stats.rate(),
                            "mcse": stats.mcse(),
                            "avg_length": stats.mean_length(),
                            "skew": "",
                            "kurtosis": "",
                            "periodicity": "",
                            "lambda_N": lambda_n,
                            "variance_ratio": "",
                            "endpoint_diff": "",
                        }
                    )

    make_manifest(repro_root, experiment, run_id, config_path, seed, spawn_keys)
    progress.finish()


def run_one_sided(cfg: Dict, seed: int, run_id: str, repro_root: str, master: List[Dict], config_path: str, alpha: float) -> None:
    n_grid = cfg["n_grid"]
    B = cfg["B"]
    r_skew = cfg["r_skew"]
    r_cov = cfg["r_cov"]
    delta = cfg["delta"]

    experiment = "one_sided"
    spawn_keys = []
    progress = ModuleProgress("one_sided", len(n_grid))

    for n in n_grid:
        f_eff = 0.5
        n1 = max(5, int(round(f_eff * n)))
        n0 = n - n1
        if n0 < 5:
            n0 = 5
            n1 = n - n0
        f_eff = n1 / n
        tag = scenario_tag("one_sided", n=n, f=f_eff, B=B, r_skew=r_skew, r_cov=r_cov)
        spawn_keys.append({"N": n, "f": f_eff, "hash": stable_hash_int(tag)})
        ss = rng_sequence(seed, tag)
        pop_seqs = ss.spawn(B)

        gauss_all = CoverageStats()
        cf_all = CoverageStats()
        cal_all = CoverageStats()

        progress.start_task(f"N={n}", B)
        workers = int(os.environ.get("REPRO_WORKERS", "1"))
        if workers > 1 and B > 1:
            from multiprocessing import get_context

            ctx = get_context("spawn")
            args = [
                (pop_ss, n, n1, r_skew, r_cov, alpha, delta)
                for pop_ss in pop_seqs
            ]
            with ctx.Pool(processes=workers) as pool:
                for result in pool.imap(_one_sided_worker, args, chunksize=1):
                    (
                        ga_cov,
                        ga_cnt,
                        ga_len,
                        cf_cov,
                        cf_cnt,
                        cf_len,
                        cal_cov,
                        cal_cnt,
                        cal_len,
                    ) = result
                    gauss_all.covered += ga_cov
                    gauss_all.count += ga_cnt
                    gauss_all.length_sum += ga_len
                    cf_all.covered += cf_cov
                    cf_all.count += cf_cnt
                    cf_all.length_sum += cf_len
                    cal_all.covered += cal_cov
                    cal_all.count += cal_cnt
                    cal_all.length_sum += cal_len
                    progress.update()
        else:
            for pop_ss in pop_seqs:
                (
                    ga_cov,
                    ga_cnt,
                    ga_len,
                    cf_cov,
                    cf_cnt,
                    cf_len,
                    cal_cov,
                    cal_cnt,
                    cal_len,
                ) = _one_sided_worker((pop_ss, n, n1, r_skew, r_cov, alpha, delta))
                gauss_all.covered += ga_cov
                gauss_all.count += ga_cnt
                gauss_all.length_sum += ga_len
                cf_all.covered += cf_cov
                cf_all.count += cf_cnt
                cf_all.length_sum += cf_len
                cal_all.covered += cal_cov
                cal_all.count += cal_cnt
                cal_all.length_sum += cal_len
                progress.update()
        progress.finish_task()

        for method, stats in [("gaussian_one_sided", gauss_all), ("cornish_fisher_one_sided", cf_all), ("calibrated_one_sided", cal_all)]:
            master.append(
                {
                    "module": experiment,
                    "design": "CRA",
                    "outcome": "continuous",
                    "method": method,
                    "N": n,
                    "m_N": n,
                    "f": f_eff,
                    "B": B,
                    "R": r_cov,
                    "S": "",
                    "coverage": stats.rate(),
                    "mcse": stats.mcse(),
                    "avg_length": stats.mean_length(),
                    "skew": "",
                    "kurtosis": "",
                    "periodicity": "",
                    "lambda_N": "",
                    "variance_ratio": "",
                    "endpoint_diff": "",
                }
            )

    make_manifest(repro_root, experiment, run_id, config_path, seed, spawn_keys)
    progress.finish()


def run_objective_bayes(cfg: Dict, seed: int, run_id: str, repro_root: str, master: List[Dict], config_path: str, alpha: float) -> None:
    n_grid = cfg["n_grid"]
    B = cfg["B"]
    R = cfg["r_cov"]
    S = cfg["S"]
    mu = cfg["mu"]
    sigma = cfg["sigma"]
    delta = cfg["delta"]
    weights = cfg["strata_weights"]
    sigmas = cfg["strata_sigma"]

    experiment = "objective_bayes"
    spawn_keys = []
    progress = ModuleProgress("objective_bayes", len(n_grid) * 2)

    for design in ["CRA", "stratified"]:
        for n in n_grid:
            f_eff = 0.5
            n1 = max(5, int(round(f_eff * n)))
            n0 = n - n1
            if n0 < 5:
                n0 = 5
                n1 = n - n0
            f_eff = n1 / n
            tag = scenario_tag("obj_bayes", design=design, n=n, f=f_eff, B=B, R=R, S=S)
            spawn_keys.append({"N": n, "design": design, "hash": stable_hash_int(tag)})
            ss = rng_sequence(seed, tag)
            pop_seqs = ss.spawn(B)

            ratio_vals = []
            enddiff_vals = []
            progress.start_task(f"{design} N={n}", B)

            for pop_ss in pop_seqs:
                ss_pop, ss_asn, ss_post = pop_ss.spawn(3)
                rng_pop = Generator(Philox(ss_pop))
                rng_assign = Generator(Philox(ss_asn))
                rng_post = Generator(Philox(ss_post))

                if design == "CRA":
                    y0, y1, tau = gen_lognormal(rng_pop, n, mu, sigma, delta)
                    assign_fn = lambda rng, n=n, n1=n1: draw_cra(rng, n, n1)
                else:
                    strata = build_strata(n, weights)
                    ys = []
                    for sidx, s in zip(strata, sigmas):
                        ys.append(rng_pop.lognormal(mean=mu, sigma=s, size=len(sidx)))
                    y0 = np.concatenate(ys)
                    y1 = y0 + delta
                    tau = delta
                    n1_per = [max(5, int(round(f_eff * len(s)))) for s in strata]
                    assign_fn = lambda rng, strata=strata, n1_per=n1_per: draw_stratified(rng, strata, n1_per)

                for _ in range(R):
                    tr = assign_fn(rng_assign)
                    tau_hat, vhat, _ = compute_vhat(y0, y1, tr)
                    se = math.sqrt(vhat)
                    lo_g, hi_g = interval_gaussian(tau_hat, se)

                    y1_obs = y1[tr]
                    y0_obs = np.delete(y0, tr)
                    n1_obs = y1_obs.size
                    n0_obs = y0_obs.size
                    s1 = float(np.var(y1_obs, ddof=1))
                    s0 = float(np.var(y0_obs, ddof=1))

                    # Faithful Bayesian bootstrap posterior draws for finite-population tau_N
                    # Use a Pólya-urn / Dirichlet-multinomial draw for the missing units in each arm.
                    tau_draws = np.empty(S, dtype=float)
                    n_total = n
                    n_miss1 = n_total - n1_obs
                    n_miss0 = n_total - n0_obs
                    sum1_obs = float(np.sum(y1_obs))
                    sum0_obs = float(np.sum(y0_obs))

                    for s_idx in range(S):
                        w1 = rng_post.gamma(shape=1.0, scale=1.0, size=n1_obs)
                        w1 /= np.sum(w1)
                        if n_miss1 > 0:
                            counts1 = rng_post.multinomial(n_miss1, w1)
                            sum1_miss = float(np.dot(counts1, y1_obs))
                        else:
                            sum1_miss = 0.0
                        mean1 = (sum1_obs + sum1_miss) / n_total

                        w0 = rng_post.gamma(shape=1.0, scale=1.0, size=n0_obs)
                        w0 /= np.sum(w0)
                        if n_miss0 > 0:
                            counts0 = rng_post.multinomial(n_miss0, w0)
                            sum0_miss = float(np.dot(counts0, y0_obs))
                        else:
                            sum0_miss = 0.0
                        mean0 = (sum0_obs + sum0_miss) / n_total

                        tau_draws[s_idx] = mean1 - mean0

                    var_post = float(np.var(tau_draws, ddof=1))
                    q_low = float(np.quantile(tau_draws, alpha / 2))
                    q_high = float(np.quantile(tau_draws, 1 - alpha / 2))

                    # Neyman conservative variance
                    neyman = (1 - f_eff) * s1 / n1_obs + f_eff * s0 / n0_obs
                    ratio_vals.append(var_post / neyman if neyman > 0 else float("nan"))
                    enddiff_vals.append(abs(q_low - lo_g) + abs(q_high - hi_g))
                progress.update()
            progress.finish_task()

            master.append(
                {
                    "module": experiment,
                    "design": design,
                    "outcome": "continuous",
                    "method": "objective_bayes",
                    "N": n,
                    "m_N": n,
                    "f": f_eff,
                    "B": B,
                    "R": R,
                    "S": S,
                    "coverage": "",
                    "mcse": "",
                    "avg_length": "",
                    "skew": "",
                    "kurtosis": "",
                    "periodicity": "",
                    "lambda_N": "",
                    "variance_ratio": float(np.nanmean(ratio_vals)),
                    "endpoint_diff": float(np.nanmean(enddiff_vals)),
                }
            )

    make_manifest(repro_root, experiment, run_id, config_path, seed, spawn_keys)
    progress.finish()


def run_fpc(cfg: Dict, seed: int, run_id: str, repro_root: str, master: List[Dict], config_path: str) -> None:
    pairs = cfg["pairs"]
    B = cfg["B"]
    experiment = "fpc"
    spawn_keys = []
    progress = ModuleProgress("fpc", len(pairs))

    for N, n in pairs:
        tag = scenario_tag("fpc", N=N, n=n, B=B)
        spawn_keys.append({"N": N, "n": n, "hash": stable_hash_int(tag)})
        ss = rng_sequence(seed, tag)
        pop_seqs = ss.spawn(B)
        ratios = []
        progress.start_task(f"N={N} n={n}", B)
        for pop_ss in pop_seqs:
            rng = Generator(Philox(pop_ss))
            y = rng.standard_normal(size=N)
            # estimate var of sample mean via Monte Carlo
            R = 200
            means = []
            for _ in range(R):
                idx = rng.choice(N, size=n, replace=False)
                means.append(float(np.mean(y[idx])))
            var_hat = float(np.var(means, ddof=1))
            s2 = float(np.var(y, ddof=1))
            f = n / N
            ratio = var_hat / ((1 - f) * s2 / n)
            ratios.append(ratio)
            progress.update()
        progress.finish_task()
        master.append(
            {
                "module": experiment,
                "design": "SRSWOR",
                "outcome": "continuous",
                "method": "fpc_ratio",
                "N": N,
                "m_N": N,
                "f": n / N,
                "B": B,
                "R": 200,
                "S": "",
                "coverage": "",
                "mcse": float(np.std(ratios, ddof=1) / math.sqrt(len(ratios))),
                "avg_length": "",
                "skew": "",
                "kurtosis": "",
                "periodicity": "",
                "lambda_N": "",
                "variance_ratio": float(np.mean(ratios)),
                "endpoint_diff": "",
            }
        )

    make_manifest(repro_root, experiment, run_id, config_path, seed, spawn_keys)
    progress.finish()


def main() -> None:
    args = parse_args()
    config_path = os.path.abspath(args.config)
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    seed = args.seed if args.seed is not None else cfg["meta"]["base_seed"]
    run_id = args.run_id or compute_run_id(config_path, seed)

    base_dir = os.path.dirname(__file__)
    repro_root = os.path.abspath(os.path.join(base_dir, ".."))

    master_rows: List[Dict] = []
    alpha = cfg["meta"]["alpha"]
    master_path = os.path.join(repro_root, "outputs", "master", run_id, "tables", "master_table.csv")
    ensure_dir(os.path.dirname(master_path))

    def checkpoint() -> None:
        write_csv(master_path, master_rows)

    def run_if_present(key: str, fn, *fn_args) -> None:
        if key not in cfg:
            print(f"[{key}] SKIP: missing config section '{key}'")
            return
        try:
            fn(cfg[key], *fn_args)
            checkpoint()
        except Exception:
            checkpoint()
            raise

    # Run modules
    run_if_present("cra_sampling", run_cra_sampling, seed, run_id, repro_root, master_rows, config_path, alpha)
    run_if_present("parity", run_parity, seed, run_id, repro_root, master_rows, config_path, alpha)
    run_if_present("lattice", run_lattice, seed, run_id, repro_root, master_rows, config_path, alpha)
    run_if_present("stratified", run_stratified, seed, run_id, repro_root, master_rows, config_path, alpha)
    run_if_present("cluster", run_cluster, seed, run_id, repro_root, master_rows, config_path, alpha)
    run_if_present("one_sided", run_one_sided, seed, run_id, repro_root, master_rows, config_path, alpha)
    run_if_present("objective_bayes", run_objective_bayes, seed, run_id, repro_root, master_rows, config_path, alpha)
    run_if_present("fpc", run_fpc, seed, run_id, repro_root, master_rows, config_path)

    checkpoint()


if __name__ == "__main__":
    main()
