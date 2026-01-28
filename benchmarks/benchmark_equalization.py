# benchmarks/benchmark_equalization.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from time import perf_counter
from turtle import delay
from typing import Callable, Dict, List, Tuple

import numpy as np

import pydaptivefiltering as pdf
from pydaptivefiltering._utils.equalization import (
    qam4_constellation_unit_var,
    wgn_complex,
    apply_channel,
    wiener_equalizer,
    best_phase_rotation,
    hard_decision_qam4,
    ser_qam4,
    evm_mse,
    cm_cost,
)


# =============================================================================
# Benchmark configuration
# =============================================================================

@dataclass
class Scenario:
    ensemble: int = 50
    K: int = 20000
    delay: int = 1
    discard: int = 3000
    N: int = 5  # number of coefficients (M+1)
    H: Tuple[complex, ...] = (1.1 + 1j * 0.5, 0.1 - 1j * 0.3, -0.2 - 1j * 0.1)
    sigma_x2: float = 1.0
    snr_dbs: Tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0)
    # if you want MATLAB-fixed noise, set use_snr_db=False and set sigma_n2_fixed
    use_snr_db: bool = True
    sigma_n2_fixed: float = 10 ** (-2.5)


@dataclass
class AlgoSpec:
    name: str
    make: Callable[[np.ndarray], object]  # takes w_init -> filter instance
    compute_cm_cost: bool = False


def build_algorithms(N: int, mu: float, L: int, gamma: float, p: float, q: float) -> Dict[str, AlgoSpec]:
    """Returns available algorithms based on pdf.* existence."""
    algos: Dict[str, AlgoSpec] = {}

    if hasattr(pdf, "CMA"):
        algos["cma"] = AlgoSpec(
            name="CMA",
            make=lambda w_init: pdf.CMA(filter_order=(N - 1), step_size=mu, w_init=w_init),
            compute_cm_cost=True,
        )

    if hasattr(pdf, "AffineProjectionCM"):
        algos["apcm"] = AlgoSpec(
            name="AffineProjectionCM",
            make=lambda w_init: pdf.AffineProjectionCM(
                filter_order=(N - 1), step_size=mu, memory_length=L, gamma=gamma, w_init=w_init
            ),
            compute_cm_cost=True,
        )

    if hasattr(pdf, "Godard"):
        algos["godard"] = AlgoSpec(
            name="Godard",
            make=lambda w_init: pdf.Godard(
                filter_order=(N - 1), step_size=mu, p_exponent=p, q_exponent=q, w_init=w_init
            ),
            compute_cm_cost=False,
        )

    if hasattr(pdf, "Sato"):
        algos["sato"] = AlgoSpec(
            name="Sato",
            make=lambda w_init: pdf.Sato(filter_order=(N - 1), step_size=mu, w_init=w_init),
            compute_cm_cost=False,
        )

    return algos


# =============================================================================
# Core benchmark logic
# =============================================================================

def trial_noise_variance_from_snr(x_clean: np.ndarray, snr_db: float) -> float:
    """Choose sigma_n2 so that mean(|x_clean|^2)/sigma_n2 = SNR."""
    p_sig = float(np.mean(np.abs(x_clean) ** 2))
    snr_lin = 10.0 ** (snr_db / 10.0)
    return p_sig / snr_lin


def run_one_trial(
    *,
    algo: AlgoSpec,
    rng: np.random.Generator,
    H: np.ndarray,
    constellation: np.ndarray,
    K: int,
    delay: int,
    discard: int,
    sigma_n2: float,
    wiener: np.ndarray,
    verbose_optimize: bool,
) -> Dict[str, float]:
    """Generates one realization, runs filter, computes metrics."""
    # generate symbols + channel output
    s = rng.choice(constellation, size=K)
    x_clean = apply_channel(s, H, K)
    n = wgn_complex((K,), sigma_n2, rng)
    x = x_clean + n

    # input to blind equalizer (matches MATLAB: x(1+delay:end))
    x_in = x[delay:]                  # length K-delay
    x_in = x_in / np.std(x_in) 
    s_ref = s[: K - delay]            # reference for metrics

    # init weights: Wiener + random/4 (MATLAB-ish)
    Ncoeff = int(wiener.size)
    w_init = wiener.ravel() + (rng.normal(0.0, 1.0, size=Ncoeff) + 1j * rng.normal(0.0, 1.0, size=Ncoeff)) / 4.0

    # run
    filt = algo.make(w_init)
    t0 = perf_counter()
    res = filt.optimize(x_in, verbose=verbose_optimize)
    t1 = perf_counter()

    y = np.asarray(res.outputs).ravel()  # length K-delay
    runtime_s = float(t1 - t0)
    samples_per_s = float(len(y) / max(runtime_s, 1e-12))

    # discard warmup
    d0 = int(min(max(discard, 0), len(y) - 1))
    y_eval = y[d0:]
    s_eval = s_ref[d0:]

    # phase ambiguity correction + SER/EVM
    y_rot, phi = best_phase_rotation(y_eval, s_eval)
    s_hat = hard_decision_qam4(y_rot)

    ser = ser_qam4(s_hat, s_eval)
    evm2 = evm_mse(y_rot, s_eval)

    # learning curve proxy: res.errors tail in dB
    e_arr = np.asarray(res.errors).ravel()
    e_pow = np.abs(e_arr) ** 2
    tail = e_pow[max(0, len(e_pow) - 500):]
    mse_tail_db = float(10.0 * np.log10(float(np.mean(tail)) + 1e-20))

    out: Dict[str, float] = {
        "runtime_s": runtime_s,
        "samples_per_s": samples_per_s,
        "ser": ser,
        "evm2": evm2,
        "phase_phi_rad": float(phi),
        "mse_tail_db_from_errors": mse_tail_db,
    }

    if algo.compute_cm_cost:
        out["cm_cost"] = cm_cost(y_eval, R=1.0)

    return out


def benchmark(
    *,
    scenario: Scenario,
    algo_keys: List[str],
    mu: float,
    L: int,
    gamma: float,
    p: float,
    q: float,
    out_csv_path: str,
    base_seed: int,
    quiet: bool,
) -> None:
    rng_master = np.random.default_rng(base_seed)

    H = np.array(scenario.H, dtype=np.complex128)
    constellation = qam4_constellation_unit_var()

    # Wiener computed with fixed sigma_n2 baseline (stable + MATLAB-like)
    wiener = wiener_equalizer(H, scenario.N, scenario.sigma_x2, scenario.sigma_n2_fixed, scenario.delay)

    algos = build_algorithms(scenario.N, mu, L, gamma, p, q)
    chosen: List[AlgoSpec] = []
    for k in algo_keys:
        if k not in algos:
            print(f"[WARN] Algorithm '{k}' not available in pydaptivefiltering. Skipping.")
        else:
            chosen.append(algos[k])

    if not chosen:
        raise RuntimeError("No algorithms available to benchmark. Check algo keys and library exports.")

    # Prepare CSV
    fieldnames = [
        "algo",
        "snr_db",
        "ensemble_id",
        "seed",
        "K",
        "delay",
        "discard",
        "Ncoeff",
        "mu",
        "L",
        "gamma",
        "p",
        "q",
        "sigma_n2",
        "runtime_s",
        "samples_per_s",
        "ser",
        "evm2",
        "phase_phi_rad",
        "mse_tail_db_from_errors",
        "cm_cost",
    ]

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for snr_db in scenario.snr_dbs:
            for algo in chosen:
                if not quiet:
                    print(f"\n=== {algo.name} | SNR={snr_db:.1f} dB | ensemble={scenario.ensemble} ===")

                for l in range(scenario.ensemble):
                    seed = int(rng_master.integers(0, 2**32 - 1))
                    rng = np.random.default_rng(seed)

                    # choose sigma_n2
                    if scenario.use_snr_db:
                        # estimate sigma_n2 from a clean signal for this trial (fair per-realization SNR)
                        s_tmp = rng.choice(constellation, size=scenario.K)
                        x_clean_tmp = apply_channel(s_tmp, H, scenario.K)
                        sigma_n2 = trial_noise_variance_from_snr(x_clean_tmp, snr_db)
                    else:
                        sigma_n2 = float(scenario.sigma_n2_fixed)

                    verbose_optimize = (l == 0) and (not quiet)

                    t_real0 = perf_counter()
                    metrics = run_one_trial(
                        algo=algo,
                        rng=rng,
                        H=H,
                        constellation=constellation,
                        K=scenario.K,
                        delay=scenario.delay,
                        discard=scenario.discard,
                        sigma_n2=sigma_n2,
                        wiener=wiener,
                        verbose_optimize=verbose_optimize,
                    )
                    t_real1 = perf_counter()

                    row = {
                        "algo": algo.name,
                        "snr_db": float(snr_db),
                        "ensemble_id": int(l),
                        "seed": int(seed),
                        "K": int(scenario.K),
                        "delay": int(scenario.delay),
                        "discard": int(scenario.discard),
                        "Ncoeff": int(scenario.N),
                        "mu": float(mu),
                        "L": int(L),
                        "gamma": float(gamma),
                        "p": float(p),
                        "q": float(q),
                        "sigma_n2": float(sigma_n2),
                        "runtime_s": metrics.get("runtime_s", float("nan")),
                        "samples_per_s": metrics.get("samples_per_s", float("nan")),
                        "ser": metrics.get("ser", float("nan")),
                        "evm2": metrics.get("evm2", float("nan")),
                        "phase_phi_rad": metrics.get("phase_phi_rad", float("nan")),
                        "mse_tail_db_from_errors": metrics.get("mse_tail_db_from_errors", float("nan")),
                        "cm_cost": metrics.get("cm_cost", float("nan")),
                    }
                    writer.writerow(row)

                    if not quiet and ((l + 1) % 10 == 0 or (l + 1) == scenario.ensemble):
                        print(
                            f"[{algo.name} | {snr_db:.1f} dB] "
                            f"{l+1:>3}/{scenario.ensemble} "
                            f"trial_time={(t_real1 - t_real0)*1e3:7.1f} ms | "
                            f"SER={row['ser']:.3e} | EVM2={row['evm2']:.3e} | "
                            f"samples/s={row['samples_per_s']:.2e}"
                        )

    print(f"\nSaved benchmark results to: {out_csv_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark blind equalizers (CMA/AP-CM/Godard/Sato) on 4-QAM channel equalization."
    )
    p.add_argument("--out", type=str, default="benchmarks/results_equalization.csv", help="Output CSV path.")
    p.add_argument("--ensemble", type=int, default=50, help="Number of realizations per (algo, snr).")
    p.add_argument("--K", type=int, default=20000, help="Number of samples per realization.")
    p.add_argument("--delay", type=int, default=1, help="Equalizer delay (matches MATLAB).")
    p.add_argument("--discard", type=int, default=3000, help="Warm-up samples to discard before SER/EVM.")
    p.add_argument("--snr-dbs", type=float, nargs="+", default=[10, 15, 20, 25, 30], help="List of SNR values in dB.")
    p.add_argument("--use-snr-db", action="store_true", help="If set, noise variance is chosen per trial to match SNR (dB).")
    p.add_argument("--sigma-n2", type=float, default=10 ** (-2.5), help="Fixed noise variance (used if --use-snr-db is not set).")

    p.add_argument("--N", type=int, default=5, help="Number of coefficients (M+1).")
    p.add_argument("--mu", type=float, default=0.001, help="Step size (mu).")
    p.add_argument("--L", type=int, default=2, help="Memory length for AP-CM (reuse factor).")
    p.add_argument("--gamma", type=float, default=1e-10, help="Gamma for AP-CM.")
    p.add_argument("--p-exp", type=float, default=2.2, help="Godard p exponent.")
    p.add_argument("--q-exp", type=float, default=1.5, help="Godard q exponent.")

    p.add_argument("--algos", type=str, nargs="+", default=["cma", "apcm", "godard", "sato"],
                   help="Algorithms to run: cma apcm godard sato")
    p.add_argument("--seed", type=int, default=123, help="Base seed for reproducibility.")
    p.add_argument("--quiet", action="store_true", help="Less printing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    scenario = Scenario(
        ensemble=args.ensemble,
        K=args.K,
        delay=args.delay,
        discard=args.discard,
        N=args.N,
        snr_dbs=tuple(float(x) for x in args.snr_dbs),
        use_snr_db=bool(args.use_snr_db),
        sigma_n2_fixed=float(args.sigma_n2),
    )

    benchmark(
        scenario=scenario,
        algo_keys=[a.lower() for a in args.algos],
        mu=float(args.mu),
        L=int(args.L),
        gamma=float(args.gamma),
        p=float(args.p_exp),
        q=float(args.q_exp),
        out_csv_path=str(args.out),
        base_seed=int(args.seed),
        quiet=bool(args.quiet),
    )


if __name__ == "__main__":
    main()
