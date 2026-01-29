# pydaptivefiltering/_utils/subband_id.py
from __future__ import annotations
import inspect
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, Any

import numpy as np



__all__ = [
    "SubbandIDConfig",
    "run_subband_system_id",
]


@dataclass
class SubbandIDConfig:
    """
    Harness config for subband/block system identification examples.

    Notes
    -----
    - Wo MUST NOT be a mutable default (np.ndarray). Use default_factory.
    """
    ensemble: int = 50
    K: int = 4096
    sigma_n2: float = 1e-3

    # ✅ FIX: mutable default -> default_factory
    Wo: np.ndarray = field(default_factory=lambda: np.array([0.32, -0.30, 0.50, 0.20], dtype=float))

    # Progress/report settings
    verbose_progress: bool = True
    print_every: int = 5
    tail_window: int = 200


def _generate_real_input(rng: np.random.Generator, K: int) -> np.ndarray:
    # sinal excitante estável p/ identificação: branco gaussiano
    return rng.standard_normal(K).astype(float)


def _build_desired_from_fir_real(
    x: np.ndarray,
    Wo: np.ndarray,
    sigma_n2: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    d = FIR(Wo) * x + n  (real)
    Convenção: Wo tem tamanho N (N taps). Saída por convolução causal.
    """
    Wo = np.asarray(Wo, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    y_clean = np.convolve(x, Wo, mode="full")[: x.size]
    n = rng.normal(0.0, np.sqrt(float(sigma_n2)), size=x.size).astype(float)
    d = y_clean + n
    return d, n


def _report_progress(
    algo_tag: str,
    l: int,
    ensemble: int,
    t0: float,
    mse_col: np.ndarray,
    cfg: SubbandIDConfig,
) -> None:
    if not cfg.verbose_progress:
        return
    if (l + 1) % int(cfg.print_every) != 0 and (l + 1) != ensemble:
        return

    tail = int(min(cfg.tail_window, mse_col.size))
    tail_mse = float(np.mean(mse_col[-tail:])) if tail > 0 else float(np.mean(mse_col))
    elapsed = perf_counter() - t0
    print(f"[subband/{algo_tag}] {l+1:>4}/{ensemble} | tail_mse={tail_mse:.3e} | elapsed={elapsed:.2f}s")


def run_subband_system_id(
    make_filter: Callable[[], Any],
    L: int,
    cfg: SubbandIDConfig,
    seed: int = 0,
    verbose_first: bool = True,
) -> Dict[str, Any]:
    """
    Run ensemble-averaged subband/block system ID.

    Parameters
    ----------
    make_filter:
        Callable that returns a *fresh* filter object with .optimize(x, d).
        The filter is expected to produce L samples per iteration in outputs/errors.
    L:
        Block advance / decimation (samples per iteration).
    cfg:
        SubbandIDConfig.
    seed:
        RNG seed.
    verbose_first:
        If True, pass verbose=True only on the first realization (if optimize supports it).

    Returns
    -------
    dict with:
      - Wo
      - MSE_av: (K_eff,) ensemble-averaged output error power
      - MSEmin_av: (K_eff,) noise power baseline
      - K_eff: effective length used (multiple of L)
    """
    rng_master = np.random.default_rng(seed)

    ensemble = int(cfg.ensemble)
    K = int(cfg.K)
    L = int(L)
    if L <= 0:
        raise ValueError("L must be a positive integer.")

    # K_eff: múltiplo de L (evita blocos parciais)
    K_eff = (K // L) * L
    if K_eff <= 0:
        raise ValueError(f"K too small for L. Got K={K}, L={L} => K_eff={K_eff}.")

    mse_mat = np.zeros((K_eff, ensemble), dtype=float)
    msemin_mat = np.zeros((K_eff, ensemble), dtype=float)

    t0 = perf_counter()

    for l in range(ensemble):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))

        x = _generate_real_input(rng, K_eff)
        d, n = _build_desired_from_fir_real(x, cfg.Wo, cfg.sigma_n2, rng)

        flt = make_filter()

        # verbose only on first run if optimize supports it
        opt_kwargs = {}
        if verbose_first and l == 0:
            try:
                sig = inspect.signature(flt.optimize)
                if "verbose" in sig.parameters:
                    opt_kwargs["verbose"] = True
            except Exception:
                pass

        res = flt.optimize(x, d, **opt_kwargs)

        e = np.asarray(res.errors).ravel()
        if e.size < K_eff:
            # Alguns blocos podem retornar exatamente K_eff, outros podem retornar menor
            # se a implementação define n_iters diferente. Truncamos p/ alinhar.
            e = np.pad(e, (0, K_eff - e.size), mode="constant")
        else:
            e = e[:K_eff]

        mse = (np.abs(e) ** 2).astype(float, copy=False)
        mse_mat[:, l] = mse
        msemin_mat[:, l] = (np.abs(n[:K_eff]) ** 2).astype(float, copy=False)

        _report_progress(
            algo_tag=type(flt).__name__,
            l=l,
            ensemble=ensemble,
            t0=t0,
            mse_col=mse,
            cfg=cfg,
        )

    MSE_av = np.mean(mse_mat, axis=1)
    MSEmin_av = np.mean(msemin_mat, axis=1)

    return {
        "Wo": np.asarray(cfg.Wo, dtype=float).copy(),
        "MSE_av": MSE_av,
        "MSEmin_av": MSEmin_av,
        "K_eff": int(K_eff),
    }

