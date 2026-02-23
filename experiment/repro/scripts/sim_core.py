#!/usr/bin/env python3
"""Core simulation utilities for reproducible experiments."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.random import Generator

Z_LOW = -1.959963984540054
Z_HIGH = 1.959963984540054


@dataclass
class CoverageStats:
    count: int = 0
    covered: int = 0
    length_sum: float = 0.0

    def add(self, ok: bool, length: float) -> None:
        self.count += 1
        if ok:
            self.covered += 1
        self.length_sum += length

    def rate(self) -> float:
        return self.covered / self.count if self.count else float("nan")

    def mcse(self) -> float:
        if self.count == 0:
            return float("nan")
        p = self.rate()
        return math.sqrt(max(p * (1 - p), 0.0) / self.count)

    def mean_length(self) -> float:
        return self.length_sum / self.count if self.count else float("nan")


def skewness(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    mean = float(x.mean())
    m2 = float(np.mean((x - mean) ** 2))
    if m2 <= 0:
        return 0.0
    m3 = float(np.mean((x - mean) ** 3))
    return m3 / (m2 ** 1.5)


def kurtosis_excess(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    mean = float(x.mean())
    m2 = float(np.mean((x - mean) ** 2))
    if m2 <= 0:
        return 0.0
    m4 = float(np.mean((x - mean) ** 4))
    return m4 / (m2 ** 2) - 3.0


def compute_vhat(y0: np.ndarray, y1: np.ndarray, treated: np.ndarray) -> Tuple[float, float, float]:
    n = y0.size
    n1 = treated.size
    n0 = n - n1
    y1_t = y1[treated]
    y0_t = y0[treated]

    sum_y1_t = float(y1_t.sum())
    sumsq_y1_t = float(np.square(y1_t).sum())
    sum_y0_t = float(y0_t.sum())
    sumsq_y0_t = float(np.square(y0_t).sum())

    total_y0 = float(y0.sum())
    total_y0_sq = float(np.square(y0).sum())

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
    return tau_hat, vhat, mean1 - mean0


def draw_cra(rng: Generator, n: int, n1: int) -> np.ndarray:
    return rng.choice(n, size=n1, replace=False)


def draw_stratified(rng: Generator, strata: List[np.ndarray], n1_per: List[int]) -> np.ndarray:
    treated = []
    for idxs, n1 in zip(strata, n1_per):
        treated.append(rng.choice(idxs, size=n1, replace=False))
    return np.concatenate(treated)


def draw_clustered(rng: Generator, clusters: List[np.ndarray], num_treated_clusters: int) -> np.ndarray:
    treated_clusters = rng.choice(len(clusters), size=num_treated_clusters, replace=False)
    treated = [clusters[i] for i in treated_clusters]
    if not treated:
        return np.array([], dtype=int)
    return np.concatenate(treated)


def draw_clustered_force_big(rng: Generator, clusters: List[np.ndarray], num_treated_clusters: int) -> np.ndarray:
    if num_treated_clusters <= 0 or not clusters:
        return np.array([], dtype=int)
    num_treated_clusters = min(num_treated_clusters, len(clusters))
    if num_treated_clusters == 1:
        treated_clusters = [0]
    else:
        remaining = rng.choice(len(clusters) - 1, size=num_treated_clusters - 1, replace=False)
        treated_clusters = [0] + [int(x) + 1 for x in remaining]
    treated = [clusters[i] for i in treated_clusters]
    return np.concatenate(treated)


def interval_gaussian(tau_hat: float, se: float) -> Tuple[float, float]:
    return tau_hat + Z_LOW * se, tau_hat + Z_HIGH * se


def interval_cf(tau_hat: float, se: float, gamma_hat: float) -> Tuple[float, float]:
    z_low_cf = Z_LOW + (Z_LOW * Z_LOW - 1.0) * gamma_hat / 6.0
    z_high_cf = Z_HIGH + (Z_HIGH * Z_HIGH - 1.0) * gamma_hat / 6.0
    return tau_hat + z_low_cf * se, tau_hat + z_high_cf * se


def interval_calibrated(tau_hat: float, se: float, q_low: float, q_high: float) -> Tuple[float, float]:
    lo = tau_hat - q_high * se
    hi = tau_hat - q_low * se
    return lo, hi


def one_sided_interval_gaussian(tau_hat: float, se: float, alpha: float, upper: bool = True) -> Tuple[float, float]:
    # upper one-sided: [L, inf), use z_{1-alpha}
    z = norm_ppf(1 - alpha)
    if upper:
        return tau_hat - z * se, float("inf")
    return float("-inf"), tau_hat + z * se


def one_sided_interval_cf(tau_hat: float, se: float, alpha: float, gamma_hat: float, upper: bool = True) -> Tuple[float, float]:
    z = norm_ppf(1 - alpha)
    z_cf = z + (z * z - 1.0) * gamma_hat / 6.0
    if upper:
        return tau_hat - z_cf * se, float("inf")
    return float("-inf"), tau_hat + z_cf * se


def periodicity_metric(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    # max frequency / n
    uniq, counts = np.unique(values, return_counts=True)
    return float(counts.max() / values.size)


def lattice_span(n1: int, n0: int, vhat: float) -> float:
    step = (1.0 / n1) + (1.0 / n0)
    return step / math.sqrt(vhat)


def norm_ppf(p: float) -> float:
    # Acklam's approximation
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
