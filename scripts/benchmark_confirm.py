#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import pydaptivefiltering as pdf
from pydaptivefiltering._utils.example_helper import (
    generate_sign_input,
    generate_qam4_input,
    build_desired_from_fir,
)
from pydaptivefiltering._utils.signal import align_by_xcorr_and_gain


REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "bench_reports"

# Perfis vencedores para confirmação
CONFIRM_PROFILES: Dict[str, List[Dict[str, Any]]] = {
    "LMS": [
        {"step_size": 0.02},
        {"step_size": 0.01},
    ],
    "NLMS": [
        {"step_size": 0.2},
    ],
    "AffineProjection": [
        {"step_size": 0.01, "L": 2, "gamma": 1e-4},
    ],
    "RLS": [
        {"forgetting_factor": 0.995, "delta": 1.0},
        {"forgetting_factor": 0.99, "delta": 1.0},
    ],
    "RLSAlt": [
        {"forgetting_factor": 0.995, "delta": 1.0},
        {"forgetting_factor": 0.99, "delta": 1.0},
    ],
    "FastRLS": [
        {"forgetting_factor": 0.995, "epsilon": 0.1},
        {"forgetting_factor": 0.99, "epsilon": 0.1},
    ],
}


@dataclass
class Row:
    algo: str
    family: str
    is_complex: bool
    params: Dict[str, Any]
    sigma_n2: float
    seed: int
    ensemble: int
    K: int
    N: int
    runtime_s: float
    runtime_per_sample_us: float
    mse_final: float
    msemin_final: float
    emse_final: float
    misadjustment: float


FAMILY_MAP = {
    "LMS": "lms",
    "NLMS": "lms",
    "AffineProjection": "lms",
    "RLS": "rls",
    "RLSAlt": "rls",
    "FastRLS": "fast_rls",
}


def get_algo_class(name: str):
    return getattr(pdf, name, None)


def supports_complex(cls) -> bool:
    return bool(getattr(cls, "supports_complex", False))


def _msemin_from_noise(n: np.ndarray) -> np.ndarray:
    return (np.abs(n) ** 2).astype(float, copy=False)


def run_system_id_once(
    algo_name: str,
    params: Dict[str, Any],
    *,
    seed: int,
    ensemble: int,
    K: int,
    sigma_n2: float,
    N: int,
) -> Tuple[float, float, float]:
    cls = get_algo_class(algo_name)
    if cls is None:
        raise RuntimeError(f"{algo_name} not found")

    is_cx = supports_complex(cls)
    rng_master = np.random.default_rng(seed)

    if is_cx:
        Wo = np.array([0.32 + 0.21j, -0.3 + 0.7j, 0.5 - 0.8j, 0.2 + 0.5j], dtype=np.complex128)
        gen_input = generate_qam4_input
        dtype = np.complex128
    else:
        Wo = np.array([0.32, -0.30, 0.50, 0.20], dtype=float)
        gen_input = generate_sign_input
        dtype = float

    mse_mat = np.zeros((K, ensemble), dtype=float)
    msemin_mat = np.zeros((K, ensemble), dtype=float)

    t0 = time.perf_counter()
    init_accepts_filter_order = "filter_order" in cls.__init__.__code__.co_varnames

    for l in range(ensemble):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
        x = gen_input(rng, K).astype(dtype, copy=False)
        d, n = build_desired_from_fir(x, Wo, sigma_n2, rng)

        if init_accepts_filter_order:
            flt = cls(filter_order=N - 1, **params)
        else:
            flt = cls(**params)

        res = flt.optimize(x, d, verbose=False)

        y = np.asarray(res.outputs).ravel()[:K]
        d0 = np.asarray(d).ravel()[:K]

        al = align_by_xcorr_and_gain(
            y=np.real(y) if is_cx else y.astype(float, copy=False),
            d=np.real(d0) if is_cx else d0.astype(float, copy=False),
            max_lag=min(256, max(0, K - 1)),
            remove_mean=True,
            fit_gain=True,
        )
        e = al["d_aligned"] - al["y_aligned"]
        mse = (np.abs(e) ** 2).astype(float, copy=False)

        mse_col = np.empty((K,), dtype=float)
        if mse.size == 0:
            mse_col[:] = 0.0
        else:
            mse_col[:mse.size] = mse
            mse_col[mse.size:] = mse[-1]

        mse_mat[:, l] = mse_col
        msemin_mat[:, l] = _msemin_from_noise(np.asarray(n).ravel()[:K])

    runtime_s = float(time.perf_counter() - t0)
    MSE_av = np.mean(mse_mat, axis=1)
    MSEmin_av = np.mean(msemin_mat, axis=1)

    return float(MSE_av[-1]), float(MSEmin_av[-1]), runtime_s


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Confirm benchmark winners across multiple noise levels.")
    ap.add_argument("--out", default=str(OUTDIR), help="output directory")
    ap.add_argument("--algos", default=",".join(CONFIRM_PROFILES.keys()), help="comma-separated algos to include")
    ap.add_argument("--sigmas", default="0.01,0.04,0.1", help="comma-separated sigma_n2 values")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    ap.add_argument("--ensemble", type=int, default=80)
    ap.add_argument("--K", type=int, default=3000)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    sigmas = parse_float_list(args.sigmas)
    seeds = parse_int_list(args.seeds)

    selected_profiles: Dict[str, List[Dict[str, Any]]] = {}
    for a in algos:
        if a in CONFIRM_PROFILES:
            selected_profiles[a] = CONFIRM_PROFILES[a]
        else:
            print(f"[WARN] {a} not in CONFIRM_PROFILES; skipping.")

    rows: List[Row] = []

    # total de execuções
    total_runs = sum(len(p_list) for p_list in selected_profiles.values()) * len(sigmas) * len(seeds)
    pbar = tqdm(total=total_runs, desc="confirm", unit="run", dynamic_ncols=True)

    for algo, plist in selected_profiles.items():
        cls = get_algo_class(algo)
        if cls is None:
            print(f"[SKIP] {algo} not found")
            pbar.update(len(plist) * len(sigmas) * len(seeds))
            continue

        family = FAMILY_MAP.get(algo, "unknown")
        is_cx = supports_complex(cls)

        for params in plist:
            for sigma_n2 in sigmas:
                for sd in seeds:
                    t_run0 = time.perf_counter()
                    try:
                        mse_final, msemin_final, runtime_s = run_system_id_once(
                            algo,
                            params,
                            seed=sd,
                            ensemble=args.ensemble,
                            K=args.K,
                            sigma_n2=sigma_n2,
                            N=args.N,
                        )
                        emse = max(mse_final - msemin_final, 0.0)
                        misadj = (emse / msemin_final) if msemin_final > 0 else float("nan")
                        total_samples = float(args.ensemble * args.K)
                        rps_us = float(runtime_s / total_samples * 1e6)

                        rows.append(
                            Row(
                                algo=algo,
                                family=family,
                                is_complex=is_cx,
                                params=dict(params),
                                sigma_n2=sigma_n2,
                                seed=sd,
                                ensemble=args.ensemble,
                                K=args.K,
                                N=args.N,
                                runtime_s=runtime_s,
                                runtime_per_sample_us=rps_us,
                                mse_final=mse_final,
                                msemin_final=msemin_final,
                                emse_final=emse,
                                misadjustment=misadj,
                            )
                        )

                        pbar.set_postfix(
                            {
                                "algo": algo,
                                "sigma": f"{sigma_n2:.3g}",
                                "mse": f"{mse_final:.3g}",
                                "us": f"{rps_us:.3g}",
                                "run_s": f"{time.perf_counter()-t_run0:.2f}",
                            }
                        )
                    except Exception as e:
                        pbar.write(f"[ERR] {algo} sigma={sigma_n2} seed={sd} params={params}: {type(e).__name__}: {e}")
                    finally:
                        pbar.update(1)

    pbar.close()

    # Agregação global por (algo, params), combinando todas as sigmas+seeds
    groups_global: Dict[Tuple[str, str], List[Row]] = {}
    for r in rows:
        k = (r.algo, json.dumps(r.params, sort_keys=True))
        groups_global.setdefault(k, []).append(r)

    agg_global = []
    for (algo, _pkey), rr in groups_global.items():
        mse = np.array([x.mse_final for x in rr], dtype=float)
        emse = np.array([x.emse_final for x in rr], dtype=float)
        mis = np.array([x.misadjustment for x in rr], dtype=float)
        rps = np.array([x.runtime_per_sample_us for x in rr], dtype=float)

        agg_global.append(
            {
                "algo": algo,
                "family": rr[0].family,
                "is_complex": rr[0].is_complex,
                "params": rr[0].params,
                "n_runs": len(rr),
                "mse_mean": float(np.mean(mse)),
                "mse_std": float(np.std(mse)),
                "emse_mean": float(np.mean(emse)),
                "misadj_mean": float(np.nanmean(mis)),
                "us_per_sample_mean": float(np.mean(rps)),
            }
        )

    agg_global.sort(key=lambda x: (x["mse_mean"], x["us_per_sample_mean"]))

    # Agregação por sigma (para robustez)
    groups_sigma: Dict[Tuple[str, str, float], List[Row]] = {}
    for r in rows:
        k = (r.algo, json.dumps(r.params, sort_keys=True), float(r.sigma_n2))
        groups_sigma.setdefault(k, []).append(r)

    agg_by_sigma = []
    for (algo, _pkey, sigma), rr in groups_sigma.items():
        mse = np.array([x.mse_final for x in rr], dtype=float)
        rps = np.array([x.runtime_per_sample_us for x in rr], dtype=float)
        agg_by_sigma.append(
            {
                "algo": algo,
                "sigma_n2": sigma,
                "params": rr[0].params,
                "n_runs": len(rr),
                "mse_mean": float(np.mean(mse)),
                "mse_std": float(np.std(mse)),
                "us_per_sample_mean": float(np.mean(rps)),
            }
        )

    agg_by_sigma.sort(key=lambda x: (x["sigma_n2"], x["mse_mean"], x["us_per_sample_mean"]))

    ts = int(time.time())
    json_path = outdir / f"bench_confirm_{ts}.json"
    csv_global_path = outdir / f"bench_confirm_global_{ts}.csv"
    csv_sigma_path = outdir / f"bench_confirm_by_sigma_{ts}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "sigmas": sigmas,
                    "seeds": seeds,
                    "ensemble": args.ensemble,
                    "K": args.K,
                    "N": args.N,
                    "algos": algos,
                },
                "rows": [asdict(r) for r in rows],
                "agg_global": agg_global,
                "agg_by_sigma": agg_by_sigma,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with csv_global_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "algo",
            "family",
            "is_complex",
            "n_runs",
            "mse_mean",
            "mse_std",
            "emse_mean",
            "misadj_mean",
            "us_per_sample_mean",
            "params",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for a in agg_global:
            w.writerow(
                {
                    "algo": a["algo"],
                    "family": a["family"],
                    "is_complex": a["is_complex"],
                    "n_runs": a["n_runs"],
                    "mse_mean": f"{a['mse_mean']:.12g}",
                    "mse_std": f"{a['mse_std']:.12g}",
                    "emse_mean": f"{a['emse_mean']:.12g}",
                    "misadj_mean": f"{a['misadj_mean']:.12g}",
                    "us_per_sample_mean": f"{a['us_per_sample_mean']:.12g}",
                    "params": json.dumps(a["params"], sort_keys=True),
                }
            )

    with csv_sigma_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sigma_n2",
            "algo",
            "n_runs",
            "mse_mean",
            "mse_std",
            "us_per_sample_mean",
            "params",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for a in agg_by_sigma:
            w.writerow(
                {
                    "sigma_n2": f"{a['sigma_n2']:.12g}",
                    "algo": a["algo"],
                    "n_runs": a["n_runs"],
                    "mse_mean": f"{a['mse_mean']:.12g}",
                    "mse_std": f"{a['mse_std']:.12g}",
                    "us_per_sample_mean": f"{a['us_per_sample_mean']:.12g}",
                    "params": json.dumps(a["params"], sort_keys=True),
                }
            )

    top = min(args.top, len(agg_global))
    print(f"\n===== TOP {top} GLOBAL (all sigmas + seeds) =====")
    for i in range(top):
        a = agg_global[i]
        print(
            f"{i+1:>2}. {a['algo']:<16} mse={a['mse_mean']:.6g} "
            f"std={a['mse_std']:.3g} us={a['us_per_sample_mean']:.4g} "
            f"params={a['params']}"
        )

    print("\n===== BEST PER SIGMA =====")
    for s in sigmas:
        cand = [x for x in agg_by_sigma if abs(x["sigma_n2"] - s) < 1e-15]
        if not cand:
            continue
        best = min(cand, key=lambda x: (x["mse_mean"], x["us_per_sample_mean"]))
        print(
            f"sigma_n2={s:.4g} -> {best['algo']} "
            f"mse={best['mse_mean']:.6g} us={best['us_per_sample_mean']:.4g} "
            f"params={best['params']}"
        )

    print(
        f"\nWrote:\n"
        f"- {json_path.relative_to(REPO)}\n"
        f"- {csv_global_path.relative_to(REPO)}\n"
        f"- {csv_sigma_path.relative_to(REPO)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
