# scripts/profile_report.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def pct(x: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * x / total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to profile_raw_*.csv")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERR] CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)

    # params de volta para json legível
    if "params" in df.columns:
        df["params"] = df["params"].apply(lambda s: json.loads(s) if isinstance(s, str) and s.startswith("{") else s)

    grp = (
        df.groupby(["algo"], as_index=False)
        .agg(
            runs=("algo", "count"),
            total_time_mean=("total_time_s", "mean"),
            us_per_sample_mean=("us_per_sample", "mean"),
            peak_mem_mb_mean=("peak_mem_mb", "mean"),
            mse_mean=("mse_final", "mean"),
            mse_std=("mse_final", "std"),
            emse_mean=("emse_final", "mean"),
            init_time_mean=("init_time_s", "mean"),
            optimize_time_mean=("optimize_time_s", "mean"),
            post_time_mean=("post_time_s", "mean"),
        )
    )

    grp["optimize_pct"] = [
        pct(o, t) for o, t in zip(grp["optimize_time_mean"], grp["total_time_mean"])
    ]
    grp["init_pct"] = [pct(i, t) for i, t in zip(grp["init_time_mean"], grp["total_time_mean"])]
    grp["post_pct"] = [pct(p, t) for p, t in zip(grp["post_time_mean"], grp["total_time_mean"])]

    # ranking de gargalo por tempo
    rank_time = grp.sort_values(["us_per_sample_mean", "total_time_mean"], ascending=[False, False]).head(args.top)

    # ranking por memória
    rank_mem = grp.sort_values("peak_mem_mb_mean", ascending=False).head(args.top)

    # qualidade x velocidade (menor mse e menor us/sample)
    rank_quality = grp.sort_values(["mse_mean", "us_per_sample_mean"], ascending=[True, True]).head(args.top)

    print("\n=== TOP by speed cost (higher us/sample = worse) ===")
    print(rank_time[[
        "algo", "runs", "us_per_sample_mean", "total_time_mean",
        "optimize_pct", "init_pct", "post_pct", "peak_mem_mb_mean", "mse_mean"
    ]].to_string(index=False))

    print("\n=== TOP by memory footprint ===")
    print(rank_mem[[
        "algo", "runs", "peak_mem_mb_mean", "us_per_sample_mean", "mse_mean"
    ]].to_string(index=False))

    print("\n=== TOP quality-speed tradeoff (low mse, then low us/sample) ===")
    print(rank_quality[[
        "algo", "runs", "mse_mean", "mse_std", "us_per_sample_mean", "peak_mem_mb_mean"
    ]].to_string(index=False))

    # salvar relatório consolidado
    out_csv = csv_path.with_name(csv_path.stem.replace("profile_raw", "profile_summary") + ".csv")
    grp.to_csv(out_csv, index=False)
    print(f"\nWrote summary: {out_csv}")

    # heurística de decisão Rust
    # candidato se optimize_pct alto e us/sample alto
    rust_candidates = grp[
        (grp["optimize_pct"] >= 70.0) & (grp["us_per_sample_mean"] >= np.percentile(grp["us_per_sample_mean"], 70))
    ].sort_values("us_per_sample_mean", ascending=False)

    if len(rust_candidates):
        print("\n=== Potential Rust candidates (heuristic) ===")
        print(rust_candidates[["algo", "us_per_sample_mean", "optimize_pct", "peak_mem_mb_mean", "mse_mean"]].to_string(index=False))
    else:
        print("\nNo strong Rust candidates by current heuristic.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
