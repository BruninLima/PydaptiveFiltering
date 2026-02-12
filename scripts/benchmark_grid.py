# scripts/benchmark_grid.py
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
CATALOG_PATH = REPO / "pydaptivefiltering" / "_utils" / "algo_param_catalog.json"


# =========================
# Benchmark targets
# =========================
SYSTEM_ID_ALGOS = [
    "LMS", "NLMS", "AffineProjection",
    "LMSNewton", "Power2ErrorLMS",
    "TDomainLMS", "TDomainDCT", "TDomainDFT",
    "RLS", "RLSAlt",
    "FastRLS", "StabFastRLS",
    "QRRLS",
    "SMNLMS", "SMBNLMS", "SMAffineProjection", "SimplifiedSMPUAP", "SimplifiedSMAP",
    "LRLSPosteriori", "LRLSErrorFeedback", "LRLSPriori", "NormalizedLRLS",
]

FAMILY_MAP = {
    "LMS": "lms", "NLMS": "lms", "AffineProjection": "lms",
    "LMSNewton": "lms", "Power2ErrorLMS": "lms",
    "TDomainLMS": "lms", "TDomainDCT": "lms", "TDomainDFT": "lms",
    "RLS": "rls", "RLSAlt": "rls",
    "FastRLS": "fast_rls", "StabFastRLS": "fast_rls",
    "QRRLS": "qr_rls",
    "SMNLMS": "set_membership", "SMBNLMS": "set_membership",
    "SMAffineProjection": "set_membership", "SimplifiedSMPUAP": "set_membership",
    "SimplifiedSMAP": "set_membership",
    "LRLSPosteriori": "lattice", "LRLSErrorFeedback": "lattice",
    "LRLSPriori": "lattice", "NormalizedLRLS": "lattice",
}

# Grid base (o catálogo remove parâmetros inválidos automaticamente)
GRID: Dict[str, Dict[str, List[Any]]] = {
    "LMS": {"step_size": [0.005, 0.01, 0.02, 0.05]},
    "NLMS": {"step_size": [0.2, 0.4, 0.8], "epsilon": [1e-3, 1e-2]},
    "AffineProjection": {"step_size": [0.002, 0.005, 0.01], "L": [2, 4], "gamma": [1e-4, 1e-3]},

    "LMSNewton": {"step_size": [1e-3, 3e-3, 1e-2], "forgetting_factor": [0.99, 0.995]},
    "Power2ErrorLMS": {"step_size": [5e-3, 1e-2], "bd": [8, 12], "tau": [0.15, 0.25]},
    "TDomainLMS": {"step_size": [2e-3, 5e-3], "gamma": [1e-3, 1e-2], "alpha": [0.98, 0.99]},
    "TDomainDCT": {"step_size": [2e-3, 5e-3], "gamma": [1e-3, 1e-2], "alpha": [0.98, 0.99]},
    "TDomainDFT": {"step_size": [2e-3, 5e-3], "gamma": [1e-3, 1e-2], "alpha": [0.98, 0.99]},

    "RLS": {"forgetting_factor": [0.99, 0.995], "delta": [1.0]},
    "RLSAlt": {"forgetting_factor": [0.99, 0.995], "delta": [1.0]},
    "FastRLS": {"forgetting_factor": [0.99, 0.995], "epsilon": [0.1, 1.0]},
    "StabFastRLS": {"forgetting_factor": [0.99, 0.995], "epsilon": [0.1, 1.0]},
    "QRRLS": {"forgetting_factor": [0.99, 0.995], "epsilon": [0.1, 1.0]},

    "SMNLMS": {"gamma_bar": [0.02, 0.05, 0.1], "epsilon": [1e-3, 1e-2], "step_size": [0.5, 0.8]},
    "SMBNLMS": {"gamma_bar": [0.02, 0.05, 0.1], "epsilon": [1e-3, 1e-2], "step_size": [0.5, 0.8]},
    "SMAffineProjection": {"gamma_bar": [0.02, 0.05], "L": [2, 4], "gamma": [1e-4, 1e-3]},
    "SimplifiedSMPUAP": {"gamma_bar": [0.5, 1.0], "L": [2, 4], "gamma": [1e-3, 1e-2]},
    "SimplifiedSMAP": {"gamma_bar": [0.5, 1.0], "L": [2, 4], "gamma": [1e-3, 1e-2]},

    "LRLSPosteriori": {"forgetting_factor": [0.99, 0.995], "epsilon": [1.0, 10.0]},
    "LRLSErrorFeedback": {"forgetting_factor": [0.99, 0.995], "epsilon": [1.0, 10.0]},
    "LRLSPriori": {"forgetting_factor": [0.99, 0.995], "epsilon": [1.0, 10.0]},
    "NormalizedLRLS": {"forgetting_factor": [0.99, 0.995], "epsilon": [1.0, 10.0]},
}


@dataclass
class Row:
    algo: str
    family: str
    is_complex: bool
    params: Dict[str, Any]

    seed: int
    ensemble: int
    K: int
    sigma_n2: float
    N: int

    runtime_s: float
    runtime_per_sample_us: float

    mse_final: float
    msemin_final: float
    emse_final: float
    misadjustment: float


# =========================
# Helpers
# =========================
def get_algo_class(name: str):
    return getattr(pdf, name, None)


def supports_complex(cls) -> bool:
    return bool(getattr(cls, "supports_complex", False))


def grid_iter(d: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    if not d:
        yield {}
        return
    keys = list(d.keys())
    vals = [d[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def _msemin_from_noise(n: np.ndarray) -> np.ndarray:
    return (np.abs(n) ** 2).astype(float, copy=False)


def load_catalog() -> Dict[str, Any]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found at {CATALOG_PATH}. "
            f"Generate it first or adjust CATALOG_PATH."
        )
    obj = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if "algorithms" not in obj:
        raise ValueError(f"Invalid catalog format in {CATALOG_PATH}: missing 'algorithms'")
    return obj["algorithms"]


def accepted_params_for(algo: str, catalog: Dict[str, Any]) -> Tuple[set[str], bool]:
    info = catalog.get(algo, {})
    acc = set(info.get("required_params", [])) | set(info.get("optional_params", []))
    has_kwargs = bool(info.get("has_kwargs", False))
    return acc, has_kwargs


def filter_params_with_catalog(algo: str, params: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
    acc, has_kwargs = accepted_params_for(algo, catalog)
    if has_kwargs:
        return dict(params)
    return {k: v for k, v in params.items() if k in acc}


def algo_family(algo: str, catalog: Dict[str, Any]) -> str:
    info = catalog.get(algo, {})
    fam = info.get("family")
    return str(fam) if fam else FAMILY_MAP.get(algo, "unknown")


def to_jsonable(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def build_agg(rows: List[Row]) -> List[Dict[str, Any]]:
    def key(r: Row):
        return (r.algo, json.dumps(r.params, sort_keys=True))

    groups: Dict[Tuple[str, str], List[Row]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    agg: List[Dict[str, Any]] = []
    for (algo, _), rr in groups.items():
        mse = np.array([x.mse_final for x in rr], dtype=float)
        emse = np.array([x.emse_final for x in rr], dtype=float)
        mis = np.array([x.misadjustment for x in rr], dtype=float)
        rps = np.array([x.runtime_per_sample_us for x in rr], dtype=float)

        agg.append({
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
        })

    agg.sort(key=lambda x: (x["mse_mean"], x["us_per_sample_mean"]))
    return agg


def pareto_frontier(agg: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Minimizar mse_mean e us_per_sample_mean
    items = sorted(agg, key=lambda a: (a["mse_mean"], a["us_per_sample_mean"]))
    frontier = []
    best_us = math.inf
    for a in items:
        us = float(a["us_per_sample_mean"])
        if us < best_us:
            frontier.append(a)
            best_us = us
    return frontier


def write_outputs(json_path: Path, csv_path: Path, rows: List[Row], agg: List[Dict[str, Any]]) -> None:
    payload = {
        "rows": [asdict(r) for r in rows],
        "agg": agg,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=to_jsonable)

    fieldnames = [
        "algo", "family", "is_complex", "n_runs",
        "mse_mean", "mse_std", "emse_mean", "misadj_mean",
        "us_per_sample_mean", "params",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for a in agg:
            w.writerow({
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
            })


def append_partial_row(partial_jsonl: Path, row: Row) -> None:
    with partial_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row), ensure_ascii=False, default=to_jsonable) + "\n")


# =========================
# Core run
# =========================
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
    """
    Returns: (mse_final, msemin_final, runtime_s)
    """
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

    init_accepts_filter_order = ("filter_order" in cls.__init__.__code__.co_varnames)

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

    mse_av = np.mean(mse_mat, axis=1)
    msemin_av = np.mean(msemin_mat, axis=1)

    return float(mse_av[-1]), float(msemin_av[-1]), runtime_s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTDIR), help="output directory")
    ap.add_argument("--algos", default=",".join(SYSTEM_ID_ALGOS), help="comma-separated list")
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated")
    ap.add_argument("--ensemble", type=int, default=80)
    ap.add_argument("--K", type=int, default=3000)
    ap.add_argument("--sigma-n2", type=float, default=0.04)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--max-combos", type=int, default=5000, help="cap total combos per algo")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=50, help="checkpoint frequency in successful runs")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    catalog = load_catalog()
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    run_id = str(int(time.time()))
    json_path = outdir / f"bench_grid_{run_id}.json"
    csv_path = outdir / f"bench_grid_{run_id}.csv"
    partial_jsonl = outdir / f"bench_grid_{run_id}.partial.jsonl"

    rows: List[Row] = []
    success_count = 0
    total_attempts = 0

    print(f"[INFO] run_id={run_id}")
    print(f"[INFO] outputs:\n- {json_path}\n- {csv_path}\n- {partial_jsonl}")

    for algo in algos:
        cls = get_algo_class(algo)
        if cls is None:
            print(f"[SKIP] {algo} (not found in pydaptivefiltering)")
            continue

        fam = algo_family(algo, catalog)
        is_cx = supports_complex(cls)

        raw_grid = list(grid_iter(GRID.get(algo, {})))
        filtered_grid: List[Dict[str, Any]] = []
        seen = set()

        for p in raw_grid:
            pf = filter_params_with_catalog(algo, p, catalog)
            k = json.dumps(pf, sort_keys=True)
            if k in seen:
                continue
            seen.add(k)
            filtered_grid.append(pf)

        if len(filtered_grid) > args.max_combos:
            filtered_grid = filtered_grid[:args.max_combos]

        total_runs_algo = len(filtered_grid) * len(seeds)
        print(
            f"\n=== {algo} | family={fam} | complex={is_cx} | combos={len(filtered_grid)} "
            f"| seeds={len(seeds)} | ensemble={args.ensemble} | K={args.K} ==="
        )

        pbar = tqdm(total=total_runs_algo, desc=algo, unit="run", dynamic_ncols=True)
        t_algo0 = time.perf_counter()

        for params in filtered_grid:
            for sd in seeds:
                total_attempts += 1
                t_run0 = time.perf_counter()

                try:
                    mse_final, msemin_final, runtime_s = run_system_id_once(
                        algo_name=algo,
                        params=params,
                        seed=sd,
                        ensemble=args.ensemble,
                        K=args.K,
                        sigma_n2=args.sigma_n2,
                        N=args.N,
                    )

                    emse = max(mse_final - msemin_final, 0.0)
                    misadj = (emse / msemin_final) if msemin_final > 0 else float("nan")
                    total_samples = float(args.ensemble * args.K)
                    rps_us = float(runtime_s / total_samples * 1e6)

                    row = Row(
                        algo=algo,
                        family=fam,
                        is_complex=is_cx,
                        params=dict(params),
                        seed=sd,
                        ensemble=args.ensemble,
                        K=args.K,
                        sigma_n2=args.sigma_n2,
                        N=args.N,
                        runtime_s=runtime_s,
                        runtime_per_sample_us=rps_us,
                        mse_final=mse_final,
                        msemin_final=msemin_final,
                        emse_final=emse,
                        misadjustment=misadj,
                    )
                    rows.append(row)
                    append_partial_row(partial_jsonl, row)
                    success_count += 1

                    pbar.set_postfix({
                        "mse": f"{mse_final:.4g}",
                        "us": f"{rps_us:.3g}",
                        "run_s": f"{time.perf_counter() - t_run0:.2f}",
                    })

                    if args.save_every > 0 and (success_count % args.save_every == 0):
                        agg_ckpt = build_agg(rows)
                        write_outputs(json_path, csv_path, rows, agg_ckpt)
                        pbar.write(
                            f"[CKPT] saved after {success_count} successful runs "
                            f"(attempts={total_attempts})"
                        )

                except Exception as e:
                    pbar.set_postfix_str(f"ERR {type(e).__name__}")
                    pbar.write(f"[ERR] {algo} seed={sd} params={params}: {type(e).__name__}: {e}")

                finally:
                    pbar.update(1)

        pbar.close()
        print(f"[{algo}] elapsed={time.perf_counter() - t_algo0:.2f}s | attempted={total_runs_algo}")

    # Final save
    agg = build_agg(rows)
    write_outputs(json_path, csv_path, rows, agg)

    # Terminal summary
    top = min(args.top, len(agg))
    print(f"\n===== TOP {top} (sorted by mse_mean, then speed) =====")
    for i in range(top):
        a = agg[i]
        print(
            f"{i+1:>2}. {a['algo']:<18} mse={a['mse_mean']:.6g} "
            f"std={a['mse_std']:.3g} us={a['us_per_sample_mean']:.4g} "
            f"params={a['params']}"
        )

    front = pareto_frontier(agg)
    print(f"\n===== Pareto frontier ({len(front)} points) =====")
    for a in front[: max(1, min(20, len(front)) )]:
        print(
            f"- {a['algo']:<18} mse={a['mse_mean']:.6g} us={a['us_per_sample_mean']:.4g} "
            f"params={a['params']}"
        )

    print(
        f"\nDone. Successful runs: {success_count} / attempted: {total_attempts}\n"
        f"Files written:\n- {json_path.relative_to(REPO)}\n"
        f"- {csv_path.relative_to(REPO)}\n"
        f"- {partial_jsonl.relative_to(REPO)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python scripts/benchmark_grid.py --algos LMS,NLMS,RLS --seeds 0 --ensemble 20 --K 1200 --save-every 20
# python scripts/benchmark_grid.py --algos LMS,NLMS,AffineProjection,RLS,RLSAlt,FastRLS,QRRLS --seeds 0,1,2 --ensemble 80 --K 3000 --save-every 50
# python scripts/benchmark_grid.py --algos RLS,RLSAlt,FastRLS,QRRLS --seeds 0,1,2,3 --ensemble 150 --K 5000 --no-keep-rows --save-every 25
# python scripts/benchmark_grid.py --algos LMS,NLMS,AffineProjection,RLS,RLSAlt,FastRLS,QRRLS --seeds 0,1,2,3,4 --ensemble 120 --K 5000 --save-every 50

