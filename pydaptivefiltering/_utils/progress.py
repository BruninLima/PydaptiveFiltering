# pydaptivefiltering/_utils/progress.py
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np

from .metrics import safe_scalar_db10

__all__ = [
    "ProgressConfig",
    "report_progress",
    "ProgressConfigChannel",
    "report_progressChannel",
    "report_progress_channel",
]


# ------------------------- helpers -------------------------

def _db10(x: float, *, eps: float) -> float:
    """10*log10(max(x, eps)) with NaN-safe behavior."""
    if not np.isfinite(x):
        return float("nan")
    return 10.0 * float(np.log10(max(float(x), float(eps))))


def _tail_mse(arr: np.ndarray, *, tail_window: int) -> float:
    """Mean of last tail_window samples (or all if shorter)."""
    v = np.asarray(arr, dtype=float).ravel()
    if v.size == 0:
        return float("nan")
    w = int(tail_window)
    if w <= 0:
        return float(np.mean(v))
    return float(np.mean(v[-min(v.size, w) :]))


def _should_print(l: int, ensemble: int, print_every: int) -> bool:
    """Print every N realizations and on last."""
    is_last = (l + 1) >= int(ensemble)
    pe = int(print_every)
    if pe <= 0:
        return is_last
    return is_last or (((l + 1) % pe) == 0)


# ------------------------- system ID progress -------------------------

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
    algo_tag: str = "SystemID",
    eps: float = 1e-20,
) -> None:
    """
    Print progress line for system identification ensemble loops.

    Parameters
    ----------
    l : int
        Realization index (0-based).
    ensemble : int
        Total number of realizations.
    t0 : float
        perf_counter() at start of the whole ensemble loop.
    t_real0 : float
        perf_counter() at start of this realization.
    mse_col : ndarray
        Vector of per-sample MSE for this realization (typically |e|^2).
    cfg : ProgressConfig
        Printing configuration.
    algo_tag : str
        Prefix tag in print line.
    eps : float
        Floor for log10 stability (in linear scale).
    """
    if not cfg.verbose_progress:
        return
    if not _should_print(l, ensemble, cfg.print_every):
        return

    t1 = perf_counter()
    elapsed = float(t1 - t0)
    per_real = elapsed / float(l + 1)
    eta = per_real * float(int(ensemble) - (l + 1))

    mse_vec = np.asarray(mse_col, dtype=float).ravel()
    mse_final = float(mse_vec[-1]) if mse_vec.size else float("nan")
    tail_mse = _tail_mse(mse_vec, tail_window=cfg.tail_window)

    mse_final_db = _db10(mse_final, eps=eps)
    tail_db = _db10(tail_mse, eps=eps)
    one_ms = (t1 - float(t_real0)) * 1e3

    print(
        f"[{algo_tag}] {l+1:>3}/{int(ensemble)} | "
        f"one={one_ms:7.1f} ms | "
        f"mse_final={mse_final_db:7.2f} dB | tail_mse={tail_db:7.2f} dB | "
        f"elapsed={elapsed:6.1f}s | ETA={eta:6.1f}s"
    )


# ------------------------- channel equalization progress -------------------------

@dataclass
class ProgressConfigChannel:
    """
    Progress printing configuration for channel equalization ensemble loops.

    Parameters
    ----------
    verbose_progress : bool
        If True, prints progress.
    print_every : int
        Print every `print_every` realizations (also prints on the last one).
    tail_window : int
        Tail window (samples) used to compute tail MSE.
    optimize_verbose_first : bool
        Convenience flag used by examples to set optimize(verbose=True) for the first realization.
        Not used inside printing functions.
    """
    verbose_progress: bool = True
    print_every: int = 10
    tail_window: int = 200
    optimize_verbose_first: bool = True


def report_progress_channel(
    *,
    l: int,
    ensemble: int,
    t0: float,
    t_real0: float,
    mse_col: np.ndarray,
    cfg: ProgressConfigChannel,
    algo_tag: str = "Equalization",
    eps: float = 1e-20,
    k_eff: Optional[int] = None,
) -> None:
    """
    Print progress line for channel equalization ensemble loops.

    Parameters
    ----------
    mse_col : ndarray
        Per-sample MSE for this realization.
    k_eff : int, optional
        Effective number of valid samples (e.g., K - delay). If provided and > 0,
        tail is computed over min(tail_window, k_eff) last samples.
    """
    if not cfg.verbose_progress:
        return
    if not _should_print(l, ensemble, cfg.print_every):
        return

    t1 = perf_counter()

    v = np.asarray(mse_col, dtype=float).ravel()
    if v.size == 0:
        tail_mse = float("nan")
    else:
        w = int(cfg.tail_window)
        if k_eff is not None and int(k_eff) > 0:
            w = min(w if w > 0 else v.size, int(k_eff))
        if w <= 0:
            tail_mse = float(np.mean(v))
        else:
            tail_mse = float(np.mean(v[-min(v.size, w) :]))

    tail_db = _db10(tail_mse, eps=eps)
    one_s = float(t1 - t_real0)
    total_s = float(t1 - t0)

    print(
        f"[{algo_tag}] {l+1:4d}/{int(ensemble)} | "
        f"tail MSE={tail_db:7.2f} dB | one={one_s:6.2f}s | total={total_s:7.2f}s"
    )


def report_progressChannel(
    *,
    l: int,
    ensemble: int,
    t0: float,
    t_real0: float,
    MSE_col: np.ndarray,
    K_eff: int,
    cfg: ProgressConfig,  
    algo_tag: str = "Equalization",
    eps: float = 1e-300,
) -> None:
    if isinstance(cfg, ProgressConfigChannel):
        cfg_ch = cfg
    else:
        cfg_ch = ProgressConfigChannel(
            verbose_progress=bool(getattr(cfg, "verbose_progress", True)),
            print_every=int(getattr(cfg, "print_every", 10)),
            tail_window=int(getattr(cfg, "tail_window", 200)),
        )

    report_progress_channel(
        l=l,
        ensemble=ensemble,
        t0=t0,
        t_real0=t_real0,
        mse_col=np.asarray(MSE_col),
        cfg=cfg_ch,
        algo_tag=algo_tag,
        eps=eps,
        k_eff=int(K_eff) if K_eff is not None else None,
    )
