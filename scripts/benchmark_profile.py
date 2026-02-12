# scripts/benchmark_profile.py
from __future__ import annotations

import argparse
import csv
import json
import time
import tracemalloc
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import pydaptivefiltering as pdf
from pydaptivefiltering._utils.example_helper import (
    generate_sign_input,
    generate_qam4_input,
    build_desired_from_fir,
)

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "bench_reports"
CATALOG_PATH = REPO / "pydaptivefiltering" / "_utils" / "algo_param_catalog.json"

# Grupo inicial sugerido (ajuste à vontade)
DEFAULT_ALGOS = [
    "LMS", "NLMS", "AffineProjection",
    "RLS", "RLSAlt", "FastRLS", "QRRLS",
    "SMNLMS", "SMAffineProjection",
    "LRLSPosteriori", "NormalizedLRLS",
    "OLSBLMS", "CMA", "Kalman",
]

# Defaults iniciais (substitua pelos seus benchmark_confirm quando quiser)
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "LMS": {"step_size": 0.02},
    "NLMS": {"step_size": 0.2},
    "AffineProjection": {"step_size": 0.01, "L": 2, "gamma": 1e-4},
    "RLS": {"forgetting_factor": 0.995, "delta": 1.0},
    "RLSAlt": {"forgetting_factor": 0.995, "delta": 1.0},
    "FastRLS": {"forgetting_factor": 0.995, "epsilon": 0.1},
    "QRRLS": {"forgetting_factor": 0.995, "epsilon": 0.1},
    "SMNLMS": {"gamma_bar": 0.05, "step_size": 0.8, "epsilon": 1e-3},
    "SMAffineProjection": {"gamma_bar": 0.05, "L": 2, "gamma": 1e-4},
    "LRLSPosteriori": {"forgetting_factor": 0.995, "epsilon": 1.0},
    "NormalizedLRLS": {"forgetting_factor": 0.995, "epsilon": 1.0},
    "OLSBLMS": {},
    "CMA": {},
    "Kalman": {},  # caso seu construtor exija params, ajuste aqui
}


@dataclass
class ProfileRow:
    algo: str
    params: Dict[str, Any]
    seed: int
    ensemble: int
    K: int
    N: int
    sigma_n2: float
    is_complex: bool

    init_time_s: float
    optimize_time_s: float
    post_time_s: float
    total_time_s: float
    us_per_sample: float

    peak_mem_mb: float
    mse_final: float
    msemin_final: float
    emse_final: float


def get_algo_class(name: str):
    return getattr(pdf, name, None)


def supports_complex(cls) -> bool:
    return bool(getattr(cls, "supports_complex", False))


def load_catalog() -> Dict[str, Any]:
    if not CATALOG_PATH.exists():
        return {}
    obj = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return obj.get("algorithms", {})


def filter_params_with_catalog(
    algo: str, params: Dict[str, Any], catalog: Dict[str, Any]
) -> Dict[str, Any]:
    info = catalog.get(algo, {})
    acc = set(info.get("required_params", [])) | set(info.get("optional_params", []))
    has_kwargs = bool(info.get("has_kwargs", False))
    if has_kwargs or not acc:
        return dict(params)
    return {k: v for k, v in params.items() if k in acc}


def build_kalman_case(K: int, rng: np.random.Generator):
    dt = 1.0
    x_true = np.zeros((K, 2), dtype=float)
    a = np.zeros(K, dtype=float)
    a[K // 6 : K // 3] = 0.08
    a[K // 2 : K // 2 + K // 8] = -0.12
    a[int(0.75 * K) : int(0.85 * K)] = 0.05

    x_true[0] = [0.0, 1.0]
    for k in range(1, K):
        pos_prev, vel_prev = x_true[k - 1]
        vel = vel_prev + a[k] * dt
        pos = pos_prev + vel_prev * dt + 0.5 * a[k] * dt**2
        x_true[k] = [pos, vel]

    sigma_meas = 0.8
    y_meas = x_true[:, 0] + sigma_meas * rng.standard_normal(K)

    A = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    C_T = np.array([[1.0, 0.0]], dtype=float)
    sigma_a = 0.15
    Q = (sigma_a**2) * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]], dtype=float)

    return {
        "input_signal": y_meas,
        "desired_signal": None,
        "kalman_ctor": {
            "A": A,
            "C_T": C_T,
            "Rn": Q,
            "Rn1": np.array([[sigma_meas**2]], dtype=float),
            "x_init": np.array([y_meas[0], 0.0], dtype=float),
            "Re_init": np.eye(2, dtype=float) * 50.0,
        },
        "x_true": x_true,
    }


def _msemin_from_noise(n: np.ndarray) -> np.ndarray:
    return (np.abs(n) ** 2).astype(float, copy=False)


def run_profile_once(
    algo_name: str,
    params: Dict[str, Any],
    *,
    seed: int,
    ensemble: int,
    K: int,
    sigma_n2: float,
    N: int,
) -> Tuple[float, float, float, float, float, float, float]:
    cls = get_algo_class(algo_name)
    if cls is None:
        raise RuntimeError(f"{algo_name} not found in pydaptivefiltering")

    is_cx = supports_complex(cls)
    rng_master = np.random.default_rng(seed)

    mse_mat = np.zeros((K, ensemble), dtype=float)
    msemin_mat = np.zeros((K, ensemble), dtype=float)

    init_acc = 0.0
    opt_acc = 0.0
    post_acc = 0.0

    init_accepts_filter_order = "filter_order" in cls.__init__.__code__.co_varnames

    tracemalloc.start()
    t_total0 = time.perf_counter()

    for l in range(ensemble):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))

        if algo_name == "Kalman":
            case = build_kalman_case(K, rng)
            ctor = case["kalman_ctor"]
            t0 = time.perf_counter()
            flt = cls(**ctor, **params) if params else cls(**ctor)
            init_acc += time.perf_counter() - t0

            t0 = time.perf_counter()
            res = flt.optimize(case["input_signal"], verbose=False)
            opt_acc += time.perf_counter() - t0

            t0 = time.perf_counter()
            x_hat = np.asarray(res.outputs)
            x_true = case["x_true"]
            err = (x_hat[:, 0] - x_true[:, 0])[:K]
            mse = (err**2).astype(float, copy=False)
            msemin = np.full((K,), np.var(case["input_signal"] - x_true[:, 0]), dtype=float)
            post_acc += time.perf_counter() - t0
        else:
            if is_cx:
                Wo = np.array([0.32 + 0.21j, -0.3 + 0.7j, 0.5 - 0.8j, 0.2 + 0.5j], dtype=np.complex128)
                gen_input = generate_qam4_input
                dtype = np.complex128
            else:
                Wo = np.array([0.32, -0.30, 0.50, 0.20], dtype=float)
                gen_input = generate_sign_input
                dtype = float

            x = gen_input(rng, K).astype(dtype, copy=False)
            d, n = build_desired_from_fir(x, Wo, sigma_n2, rng)

            t0 = time.perf_counter()
            if init_accepts_filter_order:
                flt = cls(filter_order=N - 1, **params)
            else:
                flt = cls(**params)
            init_acc += time.perf_counter() - t0

            t0 = time.perf_counter()
            res = flt.optimize(x, d, verbose=False)
            opt_acc += time.perf_counter() - t0

            t0 = time.perf_counter()
            e = np.asarray(res.errors).ravel()[:K]
            mse = (np.abs(e) ** 2).astype(float, copy=False)
            msemin = _msemin_from_noise(np.asarray(n).ravel()[:K])
            post_acc += time.perf_counter() - t0

        # pad para K
        if mse.size < K:
            mse_col = np.empty((K,), dtype=float)
            mse_col[:mse.size] = mse
            mse_col[mse.size:] = mse[-1] if mse.size else 0.0
            mse = mse_col

        if msemin.size < K:
            mm = np.empty((K,), dtype=float)
            mm[:msemin.size] = msemin
            mm[msemin.size:] = msemin[-1] if msemin.size else 0.0
            msemin = mm

        mse_mat[:, l] = mse[:K]
        msemin_mat[:, l] = msemin[:K]

    total_time = time.perf_counter() - t_total0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    MSE_av = np.mean(mse_mat, axis=1)
    MSEmin_av = np.mean(msemin_mat, axis=1)

    mse_final = float(MSE_av[-1])
    msemin_final = float(MSEmin_av[-1])
    emse_final = max(mse_final - msemin_final, 0.0)

    peak_mb = float(peak / (1024**2))
    us_per_sample = float(total_time / (ensemble * K) * 1e6)

    return (
        init_acc,
        opt_acc,
        post_acc,
        total_time,
        us_per_sample,
        peak_mb,
        mse_final,
        msemin_final,
        emse_final,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTDIR), help="output directory")
    ap.add_argument("--algos", default=",".join(DEFAULT_ALGOS), help="comma-separated algos")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated seeds")
    ap.add_argument("--ensemble", type=int, default=40)
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--sigma-n2", type=float, default=0.04)
    ap.add_argument("--N", type=int, default=4)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog()
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows: List[ProfileRow] = []

    for algo in algos:
        cls = get_algo_class(algo)
        if cls is None:
            print(f"[SKIP] {algo}: not found")
            continue

        is_cx = supports_complex(cls)
        base_params = DEFAULT_PARAMS.get(algo, {})
        params = filter_params_with_catalog(algo, base_params, catalog)

        print(f"\n=== Profiling {algo} | params={params} ===")
        for sd in seeds:
            try:
                (
                    init_t, opt_t, post_t, total_t, usps, peak_mb,
                    mse_f, msemin_f, emse_f
                ) = run_profile_once(
                    algo, params,
                    seed=sd,
                    ensemble=args.ensemble,
                    K=args.K,
                    sigma_n2=args.sigma_n2,
                    N=args.N,
                )

                rows.append(
                    ProfileRow(
                        algo=algo,
                        params=params,
                        seed=sd,
                        ensemble=args.ensemble,
                        K=args.K,
                        N=args.N,
                        sigma_n2=args.sigma_n2,
                        is_complex=is_cx,
                        init_time_s=init_t,
                        optimize_time_s=opt_t,
                        post_time_s=post_t,
                        total_time_s=total_t,
                        us_per_sample=usps,
                        peak_mem_mb=peak_mb,
                        mse_final=mse_f,
                        msemin_final=msemin_f,
                        emse_final=emse_f,
                    )
                )

                print(
                    f"[OK] {algo} seed={sd} total={total_t:.3f}s "
                    f"us/sample={usps:.3f} peak={peak_mb:.1f}MB "
                    f"mse={mse_f:.4g} emse={emse_f:.4g}"
                )
            except Exception as e:
                print(f"[ERR] {algo} seed={sd}: {type(e).__name__}: {e}")

    if not rows:
        print("No rows generated.")
        return 1

    ts = int(time.time())
    json_path = outdir / f"profile_raw_{ts}.json"
    csv_path = outdir / f"profile_raw_{ts}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": [asdict(r) for r in rows]}, f, indent=2, ensure_ascii=False)

    fields = list(asdict(rows[0]).keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = asdict(r)
            row["params"] = json.dumps(row["params"], sort_keys=True)
            w.writerow(row)

    print(f"\nWrote:\n- {json_path.relative_to(REPO)}\n- {csv_path.relative_to(REPO)}")
    print("Now run: python scripts/profile_report.py --csv", str(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
