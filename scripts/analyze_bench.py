#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import re
import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO / "bench_reports"


def pick_latest_csv(bench_dir: Path, pattern: str = r"^bench_grid_\d+\.csv$") -> Path:
    rx = re.compile(pattern)
    cands = [p for p in bench_dir.glob("*.csv") if rx.match(p.name)]
    if not cands:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {bench_dir} com padrão {pattern}"
        )
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Minimização biobjetivo:
      - x_col (mse_mean): menor é melhor
      - y_col (us_per_sample_mean): menor é melhor
    Retorna pontos não-dominados.
    """
    d = df.dropna(subset=[x_col, y_col]).copy()
    if d.empty:
        return d

    arr = d[[x_col, y_col]].to_numpy(dtype=float)
    n = arr.shape[0]
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        xi, yi = arr[i]
        # j domina i se:
        # xj <= xi e yj <= yi e ao menos uma estritamente menor
        better_or_equal = (arr[:, 0] <= xi) & (arr[:, 1] <= yi)
        strictly_better = (arr[:, 0] < xi) | (arr[:, 1] < yi)
        dominators = better_or_equal & strictly_better
        dominators[i] = False
        if dominators.any():
            is_dominated[i] = True

    front = d.loc[~is_dominated].copy()
    front = front.sort_values([x_col, y_col], ascending=[True, True])
    return front


def format_params_short(s: str, max_len: int = 90) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def build_summary_md(
    source_csv: Path,
    df: pd.DataFrame,
    top_global: pd.DataFrame,
    top_by_algo: pd.DataFrame,
    pareto: pd.DataFrame,
    top_global_n: int,
    top_algo_n: int,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append("# Benchmark Summary")
    lines.append("")
    lines.append(f"- **Generated at:** {ts}")
    lines.append(f"- **Source CSV:** `{source_csv.relative_to(REPO)}`")
    lines.append(f"- **Rows (configs):** {len(df)}")
    lines.append("")

    # Quick stats
    algos = sorted(df["algo"].dropna().unique().tolist()) if "algo" in df else []
    lines.append("## Quick Stats")
    lines.append("")
    lines.append(f"- **Algorithms:** {', '.join(algos) if algos else 'N/A'}")
    if not df.empty:
        lines.append(f"- **Best MSE:** {df['mse_mean'].min():.6g}")
        lines.append(f"- **Fastest (us/sample):** {df['us_per_sample_mean'].min():.6g}")
    lines.append("")

    # Top global
    lines.append(f"## Top {top_global_n} Global (MSE then Speed)")
    lines.append("")
    lines.append("| Rank | Algo | mse_mean | mse_std | us/sample | params |")
    lines.append("|---:|---|---:|---:|---:|---|")
    for i, (_, r) in enumerate(top_global.iterrows(), start=1):
        lines.append(
            f"| {i} | {r['algo']} | {r['mse_mean']:.6g} | {r['mse_std']:.3g} | "
            f"{r['us_per_sample_mean']:.6g} | `{format_params_short(r['params'])}` |"
        )
    lines.append("")

    # Top by algo
    lines.append(f"## Top {top_algo_n} per Algorithm")
    lines.append("")
    lines.append("| Algo | Rank | mse_mean | mse_std | us/sample | params |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for _, r in top_by_algo.iterrows():
        lines.append(
            f"| {r['algo']} | {int(r['_rank_algo'])} | {r['mse_mean']:.6g} | "
            f"{r['mse_std']:.3g} | {r['us_per_sample_mean']:.6g} | "
            f"`{format_params_short(r['params'])}` |"
        )
    lines.append("")

    # Pareto
    lines.append("## Pareto Frontier (error vs speed)")
    lines.append("")
    lines.append("| Algo | mse_mean | us/sample | params |")
    lines.append("|---|---:|---:|---|")
    for _, r in pareto.iterrows():
        lines.append(
            f"| {r['algo']} | {r['mse_mean']:.6g} | {r['us_per_sample_mean']:.6g} | "
            f"`{format_params_short(r['params'])}` |"
        )
    lines.append("")
    lines.append(
        "> Pareto = configurações não-dominadas (não existe outra com erro <= e custo <= ao mesmo tempo)."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="", help="Path do CSV de benchmark (opcional)")
    ap.add_argument("--top-global", type=int, default=15, help="Top global")
    ap.add_argument("--top-per-algo", type=int, default=3, help="Top por algoritmo")
    ap.add_argument("--outdir", type=str, default=str(BENCH_DIR), help="Diretório de saída")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.csv.strip():
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = (REPO / csv_path).resolve()
    else:
        csv_path = pick_latest_csv(BENCH_DIR)

    df = pd.read_csv(csv_path)

    required = ["algo", "mse_mean", "mse_std", "us_per_sample_mean", "params"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV sem colunas esperadas: faltando {missing}")

    df = ensure_numeric(df, ["mse_mean", "mse_std", "us_per_sample_mean"])
    df = df.dropna(subset=["algo", "mse_mean", "us_per_sample_mean"]).copy()

    # Ordenação base
    df_sorted = df.sort_values(
        ["mse_mean", "us_per_sample_mean", "mse_std"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    top_global_n = max(1, int(args.top_global))
    top_algo_n = max(1, int(args.top_per_algo))

    top_global = df_sorted.head(top_global_n).copy()

    # rank por algoritmo
    by_algo = df_sorted.copy()
    by_algo["_rank_algo"] = by_algo.groupby("algo").cumcount() + 1
    top_by_algo = by_algo[by_algo["_rank_algo"] <= top_algo_n].copy()

    # Pareto global
    pareto = pareto_front(df_sorted, "mse_mean", "us_per_sample_mean")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_md_path = outdir / f"summary_{ts}.md"
    pareto_csv_path = outdir / f"pareto_{ts}.csv"
    top_algo_csv_path = outdir / f"top_by_algo_{ts}.csv"

    summary_md = build_summary_md(
        source_csv=csv_path,
        df=df_sorted,
        top_global=top_global,
        top_by_algo=top_by_algo,
        pareto=pareto,
        top_global_n=top_global_n,
        top_algo_n=top_algo_n,
    )
    summary_md_path.write_text(summary_md, encoding="utf-8")

    pareto.to_csv(pareto_csv_path, index=False)
    top_by_algo.to_csv(top_algo_csv_path, index=False)

    print("\n===== Analyze Bench Done =====")
    print(f"Source: {csv_path}")
    print(f"Wrote:  {summary_md_path}")
    print(f"Wrote:  {pareto_csv_path}")
    print(f"Wrote:  {top_algo_csv_path}")

    print("\nTop global:")
    cols = ["algo", "mse_mean", "mse_std", "us_per_sample_mean", "params"]
    print(top_global[cols].to_string(index=False))

    print("\nPareto frontier:")
    print(pareto[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
