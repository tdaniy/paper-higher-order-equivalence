#!/usr/bin/env python3
"""Run symflip one-sided diagnostic to isolate the N^{-1/2} term."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import time
from typing import Dict, List

import sys
import numpy as np
from numpy.random import Generator, Philox, SeedSequence

import tomllib

from repro_utils import build_run_paths, compute_run_id, ensure_dir, write_manifest
from sim_core import compute_vhat, draw_cra, kurtosis_excess, norm_ppf, skewness


def stable_hash_int(tag: str) -> int:
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def scenario_tag(prefix: str, **kwargs: float | int | str | bool) -> str:
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


def gen_skew_component(
    rng: Generator,
    n: int,
    dist: str,
    log_sigma: float,
    gamma_shape: float,
) -> np.ndarray:
    if dist == "lognormal":
        sigma = max(log_sigma, 1e-8)
        x = rng.lognormal(mean=0.0, sigma=sigma, size=n)
        mean = math.exp(0.5 * sigma * sigma)
        var = (math.exp(sigma * sigma) - 1.0) * math.exp(sigma * sigma)
    elif dist == "gamma":
        shape = max(gamma_shape, 1e-6)
        scale = 1.0
        x = rng.gamma(shape=shape, scale=scale, size=n)
        mean = shape * scale
        var = shape * (scale ** 2)
    else:
        raise ValueError(f"Unknown g_dist: {dist}")
    std = math.sqrt(var) if var > 0 else 1.0
    g = (x - mean) / std
    return g.astype(float, copy=False)


def _symflip_worker(
    args: tuple[
        SeedSequence,
        int,
        int,
        int,
        float,
        float,
        float,
        str,
        float,
        float,
        bool,
        bool,
        float,
    ]
) -> tuple[int, int, float, float, float, float]:
    (
        ss,
        n,
        n1,
        R,
        tau,
        mu,
        s,
        g_dist,
        g_log_sigma,
        g_gamma_shape,
        use_crn,
        upper,
        z,
    ) = args
    if use_crn:
        ss_pop, ss_assign = ss.spawn(2)
        rng_pop = Generator(Philox(ss_pop))
        rng_assign = Generator(Philox(ss_assign))
        rng_assign_minus = rng_assign
    else:
        ss_pop, ss_assign, ss_assign_minus = ss.spawn(3)
        rng_pop = Generator(Philox(ss_pop))
        rng_assign = Generator(Philox(ss_assign))
        rng_assign_minus = Generator(Philox(ss_assign_minus))

    g = gen_skew_component(rng_pop, n, g_dist, g_log_sigma, g_gamma_shape)
    g_skew = skewness(g)
    g_kurt = kurtosis_excess(g)

    y0_plus = mu + s * g
    y0_minus = mu - s * g
    y1_plus = y0_plus + tau
    y1_minus = y0_minus + tau

    plus_cov = 0
    minus_cov = 0
    delta_sum = 0.0
    delta_sum_sq = 0.0

    for _ in range(R):
        tr = draw_cra(rng_assign, n, n1)
        tau_hat_p, vhat_p, _ = compute_vhat(y0_plus, y1_plus, tr)
        se_p = math.sqrt(vhat_p)
        if upper:
            ok_p = tau <= (tau_hat_p + z * se_p)
        else:
            ok_p = tau >= (tau_hat_p - z * se_p)

        tr_minus = tr if use_crn else draw_cra(rng_assign_minus, n, n1)
        tau_hat_m, vhat_m, _ = compute_vhat(y0_minus, y1_minus, tr_minus)
        se_m = math.sqrt(vhat_m)
        if upper:
            ok_m = tau <= (tau_hat_m + z * se_m)
        else:
            ok_m = tau >= (tau_hat_m - z * se_m)

        plus_cov += int(ok_p)
        minus_cov += int(ok_m)
        delta = float(int(ok_p) - int(ok_m))
        delta_sum += delta
        delta_sum_sq += delta * delta

    return plus_cov, minus_cov, delta_sum, delta_sum_sq, g_skew, g_kurt


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


def run_symflip(
    cfg: Dict,
    seed: int,
    run_id: str,
    repro_root: str,
    config_path: str,
    alpha: float,
) -> List[Dict]:
    n_grid = cfg["n_grid"]
    f = float(cfg["f"])
    B = int(cfg["B"])
    R = int(cfg["R"])
    tau = float(cfg.get("tau", 0.5))
    mu = float(cfg.get("mu", 0.0))
    s = float(cfg.get("s", 1.0))
    g_dist = str(cfg.get("g_dist", "lognormal"))
    g_log_sigma = float(cfg.get("g_log_sigma", 0.5))
    g_gamma_shape = float(cfg.get("g_gamma_shape", 4.0))
    use_crn = bool(cfg.get("use_crn", True))
    one_sided = str(cfg.get("one_sided", "upper")).lower()
    upper = one_sided != "lower"

    z = norm_ppf(1 - float(alpha))

    master: List[Dict] = []

    total_steps = len(n_grid)
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))
    print(f"[skew_symflip] START {start_str} | steps={total_steps}", flush=True)

    spawn_keys = []
    for step_idx, n in enumerate(n_grid, start=1):
        print(f"[skew_symflip] step {step_idx}/{total_steps} | N={n}", flush=True)
        n1 = max(5, int(round(f * n)))
        n0 = n - n1
        if n0 < 5:
            n0 = 5
            n1 = n - n0
        f_eff = n1 / n

        total_reps = B * R
        plus_cov = 0
        minus_cov = 0
        delta_sum = 0.0
        delta_sum_sq = 0.0
        g_skew_sum = 0.0
        g_kurt_sum = 0.0

        worker_args = []
        for b in range(B):
            tag = scenario_tag(
                "symflip",
                n=n,
                f=f_eff,
                b=b,
                B=B,
                R=R,
                tau=tau,
                mu=mu,
                s=s,
                g_dist=g_dist,
                g_log_sigma=g_log_sigma,
                g_gamma_shape=g_gamma_shape,
                use_crn=use_crn,
            )
            spawn_keys.append(
                {
                    "N": n,
                    "f": f_eff,
                    "b": b,
                    "g_dist": g_dist,
                    "g_log_sigma": g_log_sigma,
                    "g_gamma_shape": g_gamma_shape,
                    "s": s,
                    "use_crn": use_crn,
                    "hash": stable_hash_int(tag),
                }
            )
            ss = rng_sequence(seed, tag)
            worker_args.append(
                (
                    ss,
                    n,
                    n1,
                    R,
                    tau,
                    mu,
                    s,
                    g_dist,
                    g_log_sigma,
                    g_gamma_shape,
                    use_crn,
                    upper,
                    z,
                )
            )

        workers = int(os.environ.get("REPRO_WORKERS", "1"))
        if workers > 1 and B > 1:
            from multiprocessing import get_context

            ctx = get_context("spawn")
            with ctx.Pool(processes=workers) as pool:
                for result in pool.imap(_symflip_worker, worker_args, chunksize=1):
                    plus_c, minus_c, delta_s, delta_s2, g_skew, g_kurt = result
                    plus_cov += plus_c
                    minus_cov += minus_c
                    delta_sum += delta_s
                    delta_sum_sq += delta_s2
                    g_skew_sum += g_skew
                    g_kurt_sum += g_kurt
        else:
            for args in worker_args:
                plus_c, minus_c, delta_s, delta_s2, g_skew, g_kurt = _symflip_worker(args)
                plus_cov += plus_c
                minus_cov += minus_c
                delta_sum += delta_s
                delta_sum_sq += delta_s2
                g_skew_sum += g_skew
                g_kurt_sum += g_kurt

        p_plus = plus_cov / total_reps if total_reps else float("nan")
        p_minus = minus_cov / total_reps if total_reps else float("nan")
        mcse_plus = math.sqrt(max(p_plus * (1 - p_plus), 0.0) / total_reps) if total_reps else float("nan")
        mcse_minus = math.sqrt(max(p_minus * (1 - p_minus), 0.0) / total_reps) if total_reps else float("nan")

        delta_mean = delta_sum / total_reps if total_reps else float("nan")
        if total_reps > 1:
            var_delta = (delta_sum_sq - total_reps * (delta_mean ** 2)) / (total_reps - 1)
            mcse_delta = math.sqrt(max(var_delta, 0.0) / total_reps)
        else:
            mcse_delta = float("nan")

        g_skew = g_skew_sum / B if B else float("nan")
        g_kurt = g_kurt_sum / B if B else float("nan")

        for design, cov, mcse in [
            ("plus", p_plus, mcse_plus),
            ("minus", p_minus, mcse_minus),
        ]:
            master.append(
                {
                    "module": "skew_symflip",
                    "design": design,
                    "outcome": "continuous",
                    "method": "gaussian_one_sided",
                    "N": n,
                    "m_N": n,
                    "f": f_eff,
                    "B": B,
                    "R": R,
                    "S": "",
                    "coverage": cov,
                    "mcse": mcse,
                    "avg_length": float("inf"),
                    "skew": g_skew,
                    "kurtosis": g_kurt,
                    "periodicity": "",
                    "lambda_N": "",
                    "variance_ratio": "",
                    "endpoint_diff": "",
                }
            )

        master.append(
            {
                "module": "skew_symflip",
                "design": "delta",
                "outcome": "continuous",
                "method": "symflip_delta",
                "N": n,
                "m_N": n,
                "f": f_eff,
                "B": B,
                "R": R,
                "S": "",
                "coverage": delta_mean,
                "mcse": mcse_delta,
                "avg_length": "",
                "skew": g_skew,
                "kurtosis": g_kurt,
                "periodicity": "",
                "lambda_N": "",
                "variance_ratio": "",
                "endpoint_diff": "",
            }
        )

    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    total_sec = end_ts - start_ts
    print(f"[skew_symflip] DONE {end_str} | total_time={total_sec:.1f}s", flush=True)

    paths = build_run_paths(repro_root, "skew_symflip", run_id)
    ensure_dir(paths.outputs_dir)
    ensure_dir(paths.plots_dir)
    ensure_dir(paths.logs_dir)
    write_manifest(
        os.path.join(paths.logs_dir, "manifest.json"),
        run_id=run_id,
        config_path=config_path,
        base_seed=seed,
        spawn_keys=spawn_keys,
        command=list(sys.argv),
        repo_root=os.path.abspath(os.path.join(repro_root, "..")),
    )
    for target_dir in (os.path.dirname(paths.outputs_dir), os.path.dirname(paths.plots_dir)):
        write_manifest(
            os.path.join(target_dir, "manifest.json"),
            run_id=run_id,
            config_path=config_path,
            base_seed=seed,
            spawn_keys=spawn_keys,
            command=list(sys.argv),
            repo_root=os.path.abspath(os.path.join(repro_root, "..")),
        )
    append_summary(
        os.path.join(repro_root, "outputs", "skew_symflip", "summary.csv"),
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

    return master


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run symflip diagnostic module")
    parser.add_argument("--config", required=True, help="TOML config path")
    parser.add_argument("--seed", type=int, help="override base seed")
    parser.add_argument("--run-id", help="override run ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = os.path.abspath(args.config)
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    seed = args.seed if args.seed is not None else cfg["meta"]["base_seed"]
    run_id = args.run_id or compute_run_id(config_path, seed)

    base_dir = os.path.dirname(__file__)
    repro_root = os.path.abspath(os.path.join(base_dir, ".."))

    alpha = cfg["meta"]["alpha"]
    master_rows = run_symflip(cfg["skew_symflip"], seed, run_id, repro_root, config_path, alpha)
    master_path = os.path.join(repro_root, "outputs", "master", run_id, "tables", "master_table.csv")
    write_csv(master_path, master_rows)


if __name__ == "__main__":
    main()
