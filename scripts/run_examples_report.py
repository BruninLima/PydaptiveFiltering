# scripts/run_examples_report.py
from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import importlib.util
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "auto_examples"
REPORTS_DIR = REPO_ROOT / "auto_reports"


# -----------------------------
# datamodel (benchmark-grade)
# -----------------------------
@dataclass
class ExampleResult:
    relpath: str
    status: str                 # "ok" | "error" | "skipped"
    runtime_s: float

    # metadata (best-effort)
    algo: Optional[str] = None
    family: Optional[str] = None
    scenario: Optional[str] = None
    seed: Optional[int] = None
    ensemble: Optional[int] = None
    K: Optional[int] = None
    K_eff: Optional[int] = None
    L: Optional[int] = None
    sigma_n2: Optional[float] = None
    supports_complex: Optional[bool] = None

    # metrics
    mse_final: Optional[float] = None
    msemin_final: Optional[float] = None
    emse_final: Optional[float] = None
    misadjustment: Optional[float] = None
    conv_iter: Optional[int] = None
    runtime_per_sample_us: Optional[float] = None

    # only if error/skipped
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    traceback_tail: Optional[str] = None


# -----------------------------
# helpers
# -----------------------------
def _trim_traceback(tb: str, *, max_chars: int = 12000, last_lines: int = 120) -> str:
    if not tb:
        return ""
    tb = tb[-max_chars:]
    lines = tb.splitlines()
    return "\n".join(lines[-last_lines:])


def _as_1d_array(x: Any) -> np.ndarray:
    try:
        return np.asarray(x).ravel()
    except Exception:
        return np.asarray([], dtype=float)


def _last_scalar(x: Any) -> Optional[float]:
    arr = _as_1d_array(x)
    if arr.size == 0:
        return None
    try:
        return float(arr[-1])
    except Exception:
        return None


def _conv_iter(
    mse_curve: Any,
    floor_curve: Any,
    *,
    win: int = 30,
    tol: float = 0.20,
    start: int = 0,
) -> Optional[int]:
    """
    First index k such that for a window of length win,
    mse[k:k+win] <= floor[k:k+win] * (1+tol)
    """
    mse = _as_1d_array(mse_curve).astype(float, copy=False)
    floor = _as_1d_array(floor_curve).astype(float, copy=False)

    if mse.size == 0 or floor.size == 0:
        return None

    n = min(mse.size, floor.size)
    mse = mse[:n]
    floor = floor[:n]

    if n < win + 1:
        return None

    target = floor * (1.0 + float(tol))
    start = int(max(0, start))

    for k in range(start, n - win):
        if np.all(mse[k : k + win] <= target[k : k + win]):
            return int(k)
    return None


def load_module_from_path(path: Path):
    """Import a .py file by path (not as __main__)."""
    module_name = f"example_{path.stem}_{abs(hash(str(path))) % (10**8)}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def iter_example_files(examples_dir: Path) -> List[Path]:
    files = sorted(p for p in examples_dir.rglob("*.py") if p.is_file())
    out: List[Path] = []
    for p in files:
        if p.name == "__init__.py":
            continue
        if p.name.startswith("_"):
            continue
        rel = p.relative_to(REPO_ROOT).as_posix().lower()
        if "/kalman/" in rel or "kalman" in p.name.lower():
            continue
        out.append(p)
    return out


def _env_filter_match(path: Path) -> bool:
    """
    Optional filters:
      - PYDAF_INCLUDE: substring that must appear in relpath
      - PYDAF_EXCLUDE: substring that must NOT appear in relpath
    """
    rel = str(path.relative_to(REPO_ROOT)).lower()
    inc = os.getenv("PYDAF_INCLUDE", "").strip().lower()
    exc = os.getenv("PYDAF_EXCLUDE", "").strip().lower()

    if inc and inc not in rel:
        return False
    if exc and exc in rel:
        return False
    return True


def _env_filter_scenario_family(meta: Dict[str, Any]) -> bool:
    """
    Optional filters based on returned dict metadata:
      - PYDAF_ONLY_SCENARIO: exact match
      - PYDAF_ONLY_FAMILY: exact match
    """
    only_scen = os.getenv("PYDAF_ONLY_SCENARIO", "").strip()
    only_fam = os.getenv("PYDAF_ONLY_FAMILY", "").strip()

    scen = str(meta.get("scenario", "")).strip()
    fam = str(meta.get("family", "")).strip()

    if only_scen and scen != only_scen:
        return False
    if only_fam and fam != only_fam:
        return False
    return True


def _extract_meta(ret: Any) -> Dict[str, Any]:
    if isinstance(ret, dict):
        return ret
    return {}


def _calc_metrics_from_return(ret: Any) -> Dict[str, Any]:
    """
    Compute metrics when possible.
    Expected dict keys (best-effort):
      - MSE_av (vector)
      - MSEmin_av (vector)
      - K, K_eff, ensemble, total_s
    """
    meta = _extract_meta(ret)
    out: Dict[str, Any] = {}

    # MSE final
    if isinstance(meta, dict):
        if "MSE_av" in meta:
            out["mse_final"] = _last_scalar(meta.get("MSE_av"))
        elif "MSE" in meta:
            out["mse_final"] = _last_scalar(meta.get("MSE"))

        # Noise floor final
        if "MSEmin_av" in meta:
            out["msemin_final"] = _last_scalar(meta.get("MSEmin_av"))

        # EMSE & misadjustment
        mf = out.get("mse_final", None)
        mn = out.get("msemin_final", None)
        if mf is not None and mn is not None:
            out["emse_final"] = float(mf - mn)
            if mn > 0:
                out["misadjustment"] = float((mf - mn) / mn)

        # Convergence index
        if ("MSE_av" in meta) and ("MSEmin_av" in meta):
            out["conv_iter"] = _conv_iter(
                meta["MSE_av"],
                meta["MSEmin_av"],
                win=int(os.getenv("PYDAF_CONV_WIN", "30")),
                tol=float(os.getenv("PYDAF_CONV_TOL", "0.20")),
                start=int(os.getenv("PYDAF_CONV_START", "0")),
            )

    return out


def _runtime_per_sample_us(meta: Dict[str, Any], runtime_s: float) -> Optional[float]:
    """
    Prefer meta["total_s"] if present, else measured runtime_s.
    samples = ensemble * K_eff (or K)
    """
    total_s = meta.get("total_s", None)
    try:
        total_s = float(total_s) if total_s is not None else float(runtime_s)
    except Exception:
        total_s = float(runtime_s)

    ensemble = meta.get("ensemble", None)
    if ensemble is None:
        ensemble = 1
    try:
        ensemble = int(ensemble)
    except Exception:
        ensemble = 1

    K_eff = meta.get("K_eff", None)
    K = meta.get("K", None)
    try:
        n = int(K_eff) if K_eff is not None else int(K) if K is not None else None
    except Exception:
        n = None

    if not n or n <= 0 or ensemble <= 0:
        return None

    denom = float(ensemble * n)
    return float(total_s / denom * 1e6)


def run_one_example(path: Path, seed: int = 0) -> ExampleResult:
    t0 = time.perf_counter()
    rel = str(path.relative_to(REPO_ROOT))

    try:
        mod = load_module_from_path(path)

        if not hasattr(mod, "main") or not callable(mod.main):
            runtime_s = float(time.perf_counter() - t0)
            return ExampleResult(
                relpath=rel,
                status="skipped",
                runtime_s=runtime_s,
                error_msg="No callable main(seed=..., plot=...) found.",
            )

        os.environ.setdefault("MPLBACKEND", "Agg")

        # run
        ret = mod.main(seed=seed, plot=False)
        runtime_s = float(time.perf_counter() - t0)

        meta = _extract_meta(ret)

        # Optional filter by scenario/family (only possible after running)
        if isinstance(meta, dict) and not _env_filter_scenario_family(meta):
            return ExampleResult(
                relpath=rel,
                status="skipped",
                runtime_s=runtime_s,
                error_msg="Filtered out by PYDAF_ONLY_SCENARIO / PYDAF_ONLY_FAMILY.",
            )

        metrics = _calc_metrics_from_return(ret)

        res = ExampleResult(
            relpath=rel,
            status="ok",
            runtime_s=runtime_s,
            algo=meta.get("algo"),
            family=meta.get("family"),
            scenario=meta.get("scenario"),
            seed=meta.get("seed", seed),
            ensemble=meta.get("ensemble"),
            K=meta.get("K"),
            K_eff=meta.get("K_eff"),
            L=meta.get("L"),
            sigma_n2=meta.get("sigma_n2"),
            supports_complex=meta.get("supports_complex"),
            mse_final=metrics.get("mse_final"),
            msemin_final=metrics.get("msemin_final"),
            emse_final=metrics.get("emse_final"),
            misadjustment=metrics.get("misadjustment"),
            conv_iter=metrics.get("conv_iter"),
        )

        # runtime per sample
        if isinstance(meta, dict):
            res.runtime_per_sample_us = _runtime_per_sample_us(meta, runtime_s)

        # Close matplotlib figures if loaded
        try:
            import matplotlib.pyplot as plt  # type: ignore
            plt.close("all")
        except Exception:
            pass

        return res

    except Exception as e:
        runtime_s = float(time.perf_counter() - t0)
        tb = traceback.format_exc()
        return ExampleResult(
            relpath=rel,
            status="error",
            runtime_s=runtime_s,
            error_type=type(e).__name__,
            error_msg=str(e),
            traceback_tail=_trim_traceback(tb),
        )


def write_reports(results: List[ExampleResult], reports_dir: Path) -> Tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "examples_report.json"
    csv_path = reports_dir / "examples_report.csv"

    # JSON full
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    # CSV compact
    fieldnames = [
        "relpath", "status", "algo", "family", "scenario",
        "runtime_s", "runtime_per_sample_us",
        "ensemble", "K", "K_eff", "L", "sigma_n2",
        "mse_final", "msemin_final", "emse_final", "misadjustment",
        "conv_iter",
        "error_type", "error_msg",
    ]

    def fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, float):
            return f"{x:.12g}"
        return str(x)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {
                "relpath": r.relpath,
                "status": r.status,
                "algo": r.algo or "",
                "family": r.family or "",
                "scenario": r.scenario or "",
                "runtime_s": fmt(r.runtime_s),
                "runtime_per_sample_us": fmt(r.runtime_per_sample_us),
                "ensemble": fmt(r.ensemble),
                "K": fmt(r.K),
                "K_eff": fmt(r.K_eff),
                "L": fmt(r.L),
                "sigma_n2": fmt(r.sigma_n2),
                "mse_final": fmt(r.mse_final),
                "msemin_final": fmt(r.msemin_final),
                "emse_final": fmt(r.emse_final),
                "misadjustment": fmt(r.misadjustment),
                "conv_iter": fmt(r.conv_iter),
                "error_type": r.error_type or "",
                "error_msg": (r.error_msg or "")[:5000],
            }
            w.writerow(row)

    return json_path, csv_path


def print_summary(results: List[ExampleResult]) -> None:
    ok = sum(r.status == "ok" for r in results)
    err = sum(r.status == "error" for r in results)
    sk = sum(r.status == "skipped" for r in results)

    print("\n================ Example Run Summary ================")
    print(f"Total:   {len(results)}")
    print(f"OK:      {ok}")
    print(f"Errors:  {err}")
    print(f"Skipped: {sk}")

    if err:
        print("\n--- Errors ---")
        for r in results:
            if r.status == "error":
                print(f"* {r.relpath}: {r.error_type}: {r.error_msg}")

    if sk:
        print("\n--- Skipped ---")
        for r in results:
            if r.status == "skipped":
                print(f"* {r.relpath}: {r.error_msg}")


def _rank_table(results: List[ExampleResult], scenario: str) -> None:
    rows = [r for r in results if r.status == "ok" and (r.scenario == scenario)]
    if not rows:
        return

    print(f"\n================ Ranking ({scenario}) ================")

    # By mse_final
    rows_mse = [r for r in rows if r.mse_final is not None]
    if rows_mse:
        rows_mse.sort(key=lambda r: r.mse_final)  # smaller better
        print("\nTop by mse_final:")
        for r in rows_mse[:10]:
            print(
                f"  {r.algo:>18}  mse={r.mse_final:.4g}  "
                f"time/us={r.runtime_per_sample_us:.3g}" if r.runtime_per_sample_us else
                f"  {r.algo:>18}  mse={r.mse_final:.4g}"
            )

    # By misadjustment (only if available)
    rows_mis = [r for r in rows if r.misadjustment is not None]
    if rows_mis:
        rows_mis.sort(key=lambda r: r.misadjustment)
        print("\nTop by misadjustment:")
        for r in rows_mis[:10]:
            print(
                f"  {r.algo:>18}  misadj={r.misadjustment:.4g}  "
                f"emse={r.emse_final:.3g}  floor={r.msemin_final:.3g}"
            )

    # By convergence iteration (smaller is better)
    rows_conv = [r for r in rows if r.conv_iter is not None]
    if rows_conv:
        rows_conv.sort(key=lambda r: r.conv_iter)
        print("\nTop by conv_iter:")
        for r in rows_conv[:10]:
            print(
                f"  {r.algo:>18}  conv_iter={r.conv_iter}  "
                f"mse={r.mse_final:.3g}" if r.mse_final is not None else
                f"  {r.algo:>18}  conv_iter={r.conv_iter}"
            )


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")

    if not EXAMPLES_DIR.exists():
        print(f"[ERROR] auto_examples not found at: {EXAMPLES_DIR}")
        return 2

    # allow imports from repo
    sys.path.insert(0, str(REPO_ROOT))

    files = [p for p in iter_example_files(EXAMPLES_DIR) if _env_filter_match(p)]
    if not files:
        print("[WARN] No example .py files found after filters.")
        return 0

    seed = int(os.getenv("PYDAF_SEED", "0"))

    results: List[ExampleResult] = []
    t0 = time.perf_counter()

    for p in files:
        print(f"[RUN] {p.relative_to(REPO_ROOT)}")
        res = run_one_example(p, seed=seed)
        results.append(res)

    total_s = float(time.perf_counter() - t0)

    json_path, csv_path = write_reports(results, REPORTS_DIR)
    print_summary(results)

    # Rankings by scenario
    scenarios = sorted({r.scenario for r in results if r.scenario})
    for scen in scenarios:
        _rank_table(results, scen)

    print("\nReports written:")
    print(f" - {json_path.relative_to(REPO_ROOT)}")
    print(f" - {csv_path.relative_to(REPO_ROOT)}")
    print(f"Total wall time: {total_s:.2f} s")

    return 1 if any(r.status == "error" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
