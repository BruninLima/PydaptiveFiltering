# pydaptivefiltering/_utils/progress.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


__all__ = ["ProgressConfig", "report_progress"]


@dataclass
class ProgressConfig:
    verbose_progress: bool = True
    print_every: int = 10
    tail_window: int = 200


def report_progress(
    *,
    l: int,
    ensemble: int,
    t0: float,
    t_real0: float,
    mse_col: np.ndarray,
    cfg: ProgressConfig,
) -> None:
    if not cfg.verbose_progress:
        return

    is_time = ((l + 1) % cfg.print_every == 0) or ((l + 1) == ensemble)
    if not is_time:
        return

    t_real1 = perf_counter()
    elapsed = t_real1 - t0
    avg_per_real = elapsed / (l + 1)
    eta = avg_per_real * (ensemble - (l + 1))

    mse_final_db = 10.0 * np.log10(float(mse_col[-1]) + 1e-20)
    tail = mse_col[max(0, len(mse_col) - cfg.tail_window):]
    tail_db = 10.0 * np.log10(float(np.mean(tail)) + 1e-20)

    print(
        f"[Ensemble {l+1:>3}/{ensemble}] "
        f"time={(t_real1 - t_real0)*1e3:7.1f} ms | "
        f"mse_final={mse_final_db:7.2f} dB | tail_mse={tail_db:7.2f} dB | "
        f"elapsed={elapsed:6.1f}s | ETA={eta:6.1f}s"
    )
