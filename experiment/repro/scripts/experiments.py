#!/usr/bin/env python3
"""Experiment modules for the reproduction protocol."""
from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.random import Generator, Philox, SeedSequence

from sim_core import (
    CoverageStats,
    compute_vhat,
    draw_cra,
    draw_stratified,
    draw_clustered,
    interval_gaussian,
    interval_cf,
    interval_calibrated,
    one_sided_interval_gaussian,
    one_sided_interval_cf,
    kurtosis_excess,
    skewness,
    periodicity_metric,
    lattice_span,
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


def rng_for(seed: int, tag: str) -> Generator:
    ss = SeedSequence(entropy=[seed, stable_hash_int(tag)])
    return Generator(Philox(ss))


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


def gen_binary(rng: Generator, n: int, p0: float, p1: float) -> Tuple[np.ndarray, np.ndarray, float]:
    y0 = (rng.random(size=n) < p0).astype(float)
    y1 = (rng.random(size=n) < p1).astype(float)
    tau = float(p1 - p0)
    return y0, y1, tau


def _evaluate_population(
    y0: np.ndarray,
    y1: np.ndarray,
    assign_fn,
    tau: float,
    r_skew: int,
    r_cov: int,
    alpha: float,
    rng_skew: Generator,
    rng_cov: Generator,
    lattice: bool = False,
    jitter: bool = False,
    one_sided: bool = False,
) -> Tuple[float, float, float, float, CoverageStats, CoverageStats, CoverageStats, float, float]:
    n = y0.size
    tr0 = assign_fn(rng_skew)
    n1 = tr0.size
    n0 = n - n1

    # Estimate skewness and quantiles
    t_vals = np.empty(r_skew, dtype=float)
    for i in range(r_skew):
        tr = assign_fn(rng_skew)
        tau_hat, vhat, _ = compute_vhat(y0, y1, tr)
        t_vals[i] = (tau_hat - tau) / math.sqrt(vhat)
    gamma_hat = skewness(t_vals)
    kurt = kurtosis_excess(t_vals)

    if lattice and jitter:
        # add jitter in standardized units
        # approximate lattice span using average vhat from t_vals
        vhat_mean = 1.0
        if t_vals.size > 0:
            # crude; use var of t if needed
            vhat_mean = 1.0
        delta = lattice_span(n1, n0, vhat_mean)
        jitter_vals = t_vals + rng_skew.uniform(-delta / 2.0, delta / 2.0, size=t_vals.size)
        t_for_quant = np.sort(jitter_vals)
    else:
        t_for_quant = np.sort(t_vals)

    if t_for_quant.size == 0:
        q_low = -1.96
        q_high = 1.96
        q_one = 1.645
    else:
        lo_idx = max(0, int((alpha / 2) * t_for_quant.size) - 1)
        hi_idx = min(t_for_quant.size - 1, int((1 - alpha / 2) * t_for_quant.size))
        q_low = float(t_for_quant[lo_idx])
        q_high = float(t_for_quant[hi_idx])
        one_idx = min(t_for_quant.size - 1, int((1 - alpha) * t_for_quant.size))
        q_one = float(t_for_quant[one_idx])

    # Coverage evaluation
    gauss = CoverageStats()
    cf = CoverageStats()
    calib = CoverageStats()
    for _ in range(r_cov):
        tr = assign_fn(rng_cov)
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

    return gamma_hat, kurt, q_low, q_high, gauss, cf, calib, q_one, periodicity_metric(t_vals)


# Placeholder: actual experiments will be orchestrated in run_sim.py
