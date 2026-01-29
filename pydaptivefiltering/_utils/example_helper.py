
import numpy as np
from typing import Tuple
from time import perf_counter
from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Dict  
from pydaptivefiltering._utils.noise import wgn_complex, wgn_real
from pydaptivefiltering._utils.signal import align_by_xcorr_and_gain

def generate_sign_input(rng: np.random.Generator, K: int) -> np.ndarray:
    """Real sign(randn) like MATLAB."""
    return np.sign(rng.standard_normal(K)).astype(float)


def generate_qam4_input(rng: np.random.Generator, K: int) -> np.ndarray:
    """4-QAM/QPSK with unit average power: {±1±j}/sqrt(2)."""
    re = np.where(rng.standard_normal(K) >= 0, 1.0, -1.0)
    im = np.where(rng.standard_normal(K) >= 0, 1.0, -1.0)
    return (re + 1j * im).astype(np.complex128) / np.sqrt(2.0)

def build_desired_from_fir(
    x: np.ndarray,
    Wo: np.ndarray,
    sigma_n2: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    d = conv(x, Wo)[:K] + n
    Returns (d, n) with same length as x.
    """
    K = int(x.size)
    Wo = np.asarray(Wo)
    d_clean = np.convolve(x, Wo, mode="full")[:K]

    if np.iscomplexobj(d_clean) or np.iscomplexobj(Wo) or np.iscomplexobj(x):
        n = wgn_complex(rng, (K,), sigma_n2)
        d = d_clean.astype(np.complex128) + n
    else:
        n = wgn_real(rng, (K,), sigma_n2)
        d = d_clean.astype(float) + n

    return d, n

def _coeff_hist_to_array(coeff_hist, n_coeffs: int, K: int, dtype) -> np.ndarray:
    """
    Normalize coefficient history to shape (n_coeffs, K+1).
    Base class sometimes stores list of vectors, sometimes ndarray.
    """
    ch = np.asarray(coeff_hist)
    if ch.ndim == 2 and ch.shape[1] == n_coeffs:
        T = ch.shape[0]
        out = np.zeros((n_coeffs, K + 1), dtype=dtype)
        T_use = min(T, K + 1)
        out[:, :T_use] = ch[:T_use, :].T
        return out

    if ch.ndim == 2 and ch.shape[0] == n_coeffs:
        T = ch.shape[1]
        out = np.zeros((n_coeffs, K + 1), dtype=dtype)
        T_use = min(T, K + 1)
        out[:, :T_use] = ch[:, :T_use]
        return out

    try:
        out = np.zeros((n_coeffs, K + 1), dtype=dtype)
        T_use = min(len(coeff_hist), K + 1)
        for t in range(T_use):
            out[:, t] = np.asarray(coeff_hist[t]).ravel()[:n_coeffs]
        return out
    except Exception as e:
        raise ValueError(f"Unsupported coefficient history format: {type(coeff_hist)}") from e

def pack_theta_from_result(res, w_last: np.ndarray, n_coeffs: int, K: int) -> np.ndarray:
    """
    Returns theta trajectory with shape (n_coeffs, K+1).
    Uses res.coefficients history if present; ensures last column = w_last.
    """
    dtype = np.result_type(w_last)
    theta = _coeff_hist_to_array(res.coefficients, n_coeffs=n_coeffs, K=K, dtype=dtype)
    theta[:, -1] = np.asarray(w_last).ravel()[:n_coeffs]
    return theta





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

    Wo: np.ndarray = field(default_factory=lambda: np.array([0.32, -0.30, 0.50, 0.20], dtype=float))

    verbose_progress: bool = True
    print_every: int = 5
    tail_window: int = 200


def _generate_real_input(rng: np.random.Generator, K: int) -> np.ndarray:
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

        # --- aligned MSE (use outputs, not res.errors) ---
        y = np.asarray(res.outputs).ravel()[:K_eff]
        d0 = np.asarray(d).ravel()[:K_eff]

        al = align_by_xcorr_and_gain(
            y=y,
            d=d0,
            max_lag=256,
            remove_mean=True,
            fit_gain=True,
        )
        e_al = al["d_aligned"] - al["y_aligned"]
        mse_valid = (e_al ** 2).astype(float, copy=False)

        mse_col = np.empty((K_eff,), dtype=float)

        if mse_valid.size == 0:
            mse_col[:] = 0.0
        else:
            mse_col[: mse_valid.size] = mse_valid
            mse_col[mse_valid.size :] = mse_valid[-1] 

        mse_mat[:, l] = mse_col
        msemin_mat[:, l] = (np.abs(n[:K_eff]) ** 2).astype(float, copy=False)

        _report_progress(
            algo_tag=type(flt).__name__,
            l=l,
            ensemble=ensemble,
            t0=t0,
            mse_col=mse_col,
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


