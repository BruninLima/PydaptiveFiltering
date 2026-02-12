#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "bench_reports"


def _parse_params_cell(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    s = str(x).strip()
    if not s:
        return {}
    # tenta JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # tenta literal python
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def _latest_confirm_csv(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        return p

    cands = sorted(OUTDIR.glob("bench_confirm_global_*.csv"))
    if not cands:
        raise FileNotFoundError(
            "No bench_confirm_global_*.csv found in bench_reports/. "
            "Run benchmark_confirm.py first or pass --csv PATH."
        )
    return cands[-1]


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _pick_accuracy(group: pd.DataFrame) -> pd.Series:
    g = group.sort_values(["mse_mean", "us_per_sample_mean"], ascending=[True, True])
    return g.iloc[0]


def _pick_speed(group: pd.DataFrame, speed_mse_tol: float) -> pd.Series:
    # speed_mse_tol: aceita candidatos com mse <= best_mse * (1 + tol), escolhe menor us
    best_mse = group["mse_mean"].min()
    lim = best_mse * (1.0 + speed_mse_tol)
    cand = group[group["mse_mean"] <= lim]
    if cand.empty:
        cand = group
    cand = cand.sort_values(["us_per_sample_mean", "mse_mean"], ascending=[True, True])
    return cand.iloc[0]


def _pick_balanced(group: pd.DataFrame, balanced_mse_tol: float) -> pd.Series:
    # 1) restringe a uma "faixa de qualidade" perto do melhor mse
    best_mse = group["mse_mean"].min()
    lim = best_mse * (1.0 + balanced_mse_tol)
    cand = group[group["mse_mean"] <= lim]
    if cand.empty:
        cand = group.copy()

    # 2) score normalizado mse+speed, com pequeno peso em std
    mse_min, mse_max = cand["mse_mean"].min(), cand["mse_mean"].max()
    us_min, us_max = cand["us_per_sample_mean"].min(), cand["us_per_sample_mean"].max()
    std_min, std_max = cand["mse_std"].min(), cand["mse_std"].max()

    def norm(v, lo, hi):
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            return 0.0
        return (v - lo) / (hi - lo)

    score = (
        cand["mse_mean"].apply(lambda v: norm(v, mse_min, mse_max)) * 0.55
        + cand["us_per_sample_mean"].apply(lambda v: norm(v, us_min, us_max)) * 0.35
        + cand["mse_std"].apply(lambda v: norm(v, std_min, std_max)) * 0.10
    )
    cand = cand.assign(_score=score).sort_values(
        ["_score", "mse_mean", "us_per_sample_mean"], ascending=[True, True, True]
    )
    return cand.iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Choose default params per algorithm from bench_confirm_global CSV."
    )
    ap.add_argument("--csv", default=None, help="Path to bench_confirm_global_*.csv (optional).")
    ap.add_argument(
        "--algos",
        default="",
        help="Comma-separated list to restrict algorithms (optional). Example: LMS,NLMS,RLS",
    )
    ap.add_argument(
        "--speed-mse-tol",
        type=float,
        default=0.03,
        help="Speed profile: allow up to +X relative MSE over best (default 0.03 = +3%%).",
    )
    ap.add_argument(
        "--balanced-mse-tol",
        type=float,
        default=0.015,
        help="Balanced profile: shortlist up to +X relative MSE over best (default 0.015 = +1.5%%).",
    )
    ap.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Discard rows with n_runs < min-runs (default 1).",
    )
    ap.add_argument(
        "--out",
        default=str(OUTDIR),
        help="Output directory (default bench_reports).",
    )
    args = ap.parse_args()

    csv_path = _latest_confirm_csv(args.csv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    needed_cols = {"algo", "mse_mean", "mse_std", "us_per_sample_mean", "params", "n_runs"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing columns: {sorted(missing)}")

    df = _ensure_numeric(df, ["mse_mean", "mse_std", "us_per_sample_mean", "n_runs"])
    df = df.dropna(subset=["algo", "mse_mean", "us_per_sample_mean"])
    df = df[df["n_runs"] >= int(args.min_runs)].copy()

    if args.algos.strip():
        allow = {x.strip() for x in args.algos.split(",") if x.strip()}
        df = df[df["algo"].isin(allow)].copy()

    if df.empty:
        raise RuntimeError("No rows left after filtering. Check --algos / --min-runs / CSV content.")

    # parse params dict
    df["params_dict"] = df["params"].apply(_parse_params_cell)

    defaults = {
        "meta": {
            "source_csv": str(csv_path),
            "generated_at_unix": int(time.time()),
            "speed_mse_tol": float(args.speed_mse_tol),
            "balanced_mse_tol": float(args.balanced_mse_tol),
            "min_runs": int(args.min_runs),
        },
        "profiles": {
            "speed": {},
            "balanced": {},
            "accuracy": {},
        },
    }

    summary_rows: List[Tuple[str, str, float, float, float, Dict[str, Any]]] = []

    for algo, g in df.groupby("algo", sort=True):
        g = g.sort_values(["mse_mean", "us_per_sample_mean"], ascending=[True, True]).reset_index(drop=True)

        acc = _pick_accuracy(g)
        spd = _pick_speed(g, float(args.speed_mse_tol))
        bal = _pick_balanced(g, float(args.balanced_mse_tol))

        defaults["profiles"]["accuracy"][algo] = acc["params_dict"]
        defaults["profiles"]["speed"][algo] = spd["params_dict"]
        defaults["profiles"]["balanced"][algo] = bal["params_dict"]

        summary_rows.append((
            algo, "accuracy", float(acc["mse_mean"]), float(acc["mse_std"]), float(acc["us_per_sample_mean"]), acc["params_dict"]
        ))
        summary_rows.append((
            algo, "balanced", float(bal["mse_mean"]), float(bal["mse_std"]), float(bal["us_per_sample_mean"]), bal["params_dict"]
        ))
        summary_rows.append((
            algo, "speed", float(spd["mse_mean"]), float(spd["mse_std"]), float(spd["us_per_sample_mean"]), spd["params_dict"]
        ))

    ts = int(time.time())
    out_json = outdir / f"default_params_{ts}.json"
    out_md = outdir / f"default_params_{ts}.md"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=2, ensure_ascii=False)

    # relatório markdown curto
    lines = []
    lines.append("# Default Params (auto-selected)\n")
    lines.append(f"- Source CSV: `{csv_path}`")
    lines.append(f"- Generated at: `{ts}`")
    lines.append(f"- Speed tolerance: `{args.speed_mse_tol}`")
    lines.append(f"- Balanced tolerance: `{args.balanced_mse_tol}`")
    lines.append("")
    lines.append("## Per-algorithm picks\n")
    lines.append("| Algo | Profile | mse_mean | mse_std | us/sample | params |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for algo, profile, mse, std, us, params in sorted(summary_rows, key=lambda x: (x[0], x[1])):
        lines.append(
            f"| {algo} | {profile} | {mse:.6g} | {std:.3g} | {us:.6g} | `{json.dumps(params, sort_keys=True)}` |"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")

    # print amigável
    print("\n=== Auto-selected defaults ===")
    for algo in sorted(defaults["profiles"]["accuracy"].keys()):
        a = defaults["profiles"]["accuracy"][algo]
        b = defaults["profiles"]["balanced"][algo]
        s = defaults["profiles"]["speed"][algo]
        print(f"\n{algo}:")
        print(f"  accuracy -> {a}")
        print(f"  balanced -> {b}")
        print(f"  speed    -> {s}")

    print(f"\nWrote:\n- {out_json}\n- {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
