# tests/test_subband.py
#
#       Unit tests for Subband adaptive filters:
#         - CFDLMS
#         - DLCLLMS
#         - OLSBLMS
#
#       These tests are intentionally "harness-like":
#         - they try to import the algorithms from common module paths
#         - they instantiate filters with safe/stable hyperparameters when possible
#         - they validate sanity (return type, keys, shapes, finiteness)
#         - they validate convergence via tail-MSE (relative to signal power)
#
#       Notes:
#         - Subband algorithms may return outputs/errors shorter than the input
#           because of analysis/synthesis filter transients and/or decimation.
#           The tests therefore align by trimming the reference signals to the
#           returned length when needed.
#         - OLSBLMS typically needs analysis/synthesis filterbanks; we create a
#           simple DFT-modulated FIR filterbank using a prototype lowpass.
#
#       Authors (tests):
#         . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com
#         . (test harness written with ChatGPT assistance)

from __future__ import annotations

import inspect
import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import pytest

try:
    from scipy import signal
except Exception:  # pragma: no cover
    signal = None


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    class_name: str


ALGOS: List[AlgoSpec] = [
    AlgoSpec("cfdlms", "CFDLMS"),
    AlgoSpec("dlcllms", "DLCLLMS"),
    AlgoSpec("olsblms", "OLSBLMS"),
]

# Candidate import locations (adjust if you refactor package structure)
IMPORT_CANDIDATES: Dict[str, List[str]] = {
    "CFDLMS": [
        "pydaptivefiltering.subband.cfdlms",
        "pydaptivefiltering.subband.CFDLMS",
        "pydaptivefiltering.subband.cfd_lms",
    ],
    "DLCLLMS": [
        "pydaptivefiltering.subband.dlcllms",
        "pydaptivefiltering.subband.DLCLLMS",
        "pydaptivefiltering.subband.dlcl_lms",
    ],
    "OLSBLMS": [
        "pydaptivefiltering.subband.olsblms",
        "pydaptivefiltering.subband.OLSBLMS",
        "pydaptivefiltering.subband.olsb_lms",
    ],
}


def _try_import_class(module_candidates: Sequence[str], class_name: str) -> Optional[Type[Any]]:
    for modname in module_candidates:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, class_name):
                return getattr(mod, class_name)
        except Exception:
            continue
    return None


def _is_complex_capable(cls: Type[Any]) -> bool:
    return bool(getattr(cls, "supports_complex", False))


def _run_optimize(filt: Any, x: np.ndarray, d: np.ndarray) -> Dict[str, Any]:
    sig = inspect.signature(filt.optimize)
    if "verbose" in sig.parameters:
        return filt.optimize(x, d, verbose=False)
    return filt.optimize(x, d)


def _tail_mse(e: np.ndarray, tail: int = 500) -> float:
    e = np.asarray(e)
    tail = min(tail, e.size)
    if tail <= 0:
        return float("inf")
    return float(np.mean(np.abs(e[-tail:]) ** 2))


def _align_to_length(arr: np.ndarray, L: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size == L:
        return arr
    if arr.size > L:
        return arr[:L]
    out = np.zeros(L, dtype=arr.dtype)
    out[: arr.size] = arr
    return out


def _get_errors_from_res(res: Dict[str, Any], d_ref: np.ndarray) -> np.ndarray:
    if "priori_errors" in res:
        return np.asarray(res["priori_errors"])
    if "errors" in res:
        return np.asarray(res["errors"])
    if "outputs" in res:
        y = np.asarray(res["outputs"])
        L = min(y.size, d_ref.size)
        return _align_to_length(d_ref, L) - _align_to_length(y, L)
    raise AssertionError("Result dict must contain one of: errors/priori_errors/outputs.")


def _debug_blob(cls: Type[Any], res: Dict[str, Any], e: np.ndarray) -> str:
    parts = [f" | class={cls.__name__}", f" | module={cls.__module__}"]
    if "outputs" in res:
        y = np.asarray(res["outputs"])
        parts.append(f" | y.shape={y.shape}")
        parts.append(f" | max|y|={np.max(np.abs(y)):.3e}")
        parts.append(f" | max|imag(y)|={np.max(np.abs(np.imag(y))):.3e}")
    parts.append(f" | e.shape={np.asarray(e).shape}")
    parts.append(f" | max|e|={np.max(np.abs(e)):.3e}")
    return "".join(parts)


def _build_dft_filterbank(
    n_subbands: int,
    numtaps: int = 64,
    cutoff: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if signal is None:
        raise RuntimeError("scipy is required for subband tests (scipy.signal).")

    if cutoff is None:
        cutoff = 1.0 / max(2, n_subbands)

    p = signal.firwin(numtaps, cutoff=cutoff, window="hamming", pass_zero="lowpass")
    n = np.arange(numtaps)

    H = np.zeros((n_subbands, numtaps), dtype=complex)
    G = np.zeros((n_subbands, numtaps), dtype=complex)
    for k in range(n_subbands):
        mod = np.exp(-1j * 2 * np.pi * k * n / n_subbands)
        H[k, :] = p * mod
        G[k, :] = p * np.conj(mod)
    return H, G


def _instantiate_filter(cls: Type[Any], order: int) -> Any:
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    kwargs: Dict[str, Any] = {}

    # Hiperparâmetros mais agressivos para convergência rápida em testes
    if "step" in params:
        kwargs["step"] = 0.05  # Aumentado de 1e-3 para 0.05
    if "mu" in params:
        kwargs["mu"] = 0.05
    if "gamma" in params:
        kwargs["gamma"] = 1e-2 # Valor típico para OLSBLMS
    if "a" in params:
        kwargs["a"] = 0.1      # Valor que permite adaptação mais rápida da energia
    if "decimation_factor" in params:
        kwargs["decimation_factor"] = 4 # Combinar com o M do cosmod
    if "n_subbands" in params:
        kwargs["n_subbands"] = 4

    # Lógica especial para OLSBLMS: 
    # Se for OLSBLMS, não passamos nada em analysis_filters para ele usar o embutido
    if cls.__name__ == "OLSBLMS":
        pass 
    elif "analysis_filters" in params and "synthesis_filters" in params:
        n_subbands = int(kwargs.get("n_subbands", 4))
        H, G = _build_dft_filterbank(n_subbands=n_subbands, numtaps=64)
        kwargs["analysis_filters"] = H
        kwargs["synthesis_filters"] = G

    if "filter_order" in params:
        kwargs["filter_order"] = order
        return cls(**kwargs)

    for name in ("m", "order"):
        if name in params:
            kwargs[name] = order
            return cls(**kwargs)

    try:
        return cls(order, **kwargs)
    except TypeError as e:
        raise TypeError(
            f"Could not instantiate {cls.__name__} with order={order}. __init__ params: {list(params.keys())}"
        ) from e


def _assert_realish(arr: np.ndarray, name: str, dbg: str) -> None:
    arr = np.asarray(arr)
    max_im = float(np.max(np.abs(np.imag(arr)))) if arr.size else 0.0
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    tol = 1e-8 + 1e-6 * max(1.0, max_abs)
    assert max_im < tol, f"{name} inesperadamente complexo em teste REAL (max_im={max_im:.3e}, tol={tol:.3e}){dbg}"


@pytest.mark.parametrize("algo", ALGOS, ids=lambda a: a.name)
def test_subband_real_sanity_and_convergence(algo: AlgoSpec, system_data_real):
    cls = _try_import_class(IMPORT_CANDIDATES[algo.class_name], algo.class_name)
    if cls is None:
        pytest.skip(f"Não encontrei {algo.class_name} nos imports candidatos. Ajuste IMPORT_CANDIDATES.")

    x = np.asarray(system_data_real["x"], dtype=float)
    d = np.asarray(system_data_real["d_ideal"], dtype=float)
    order = int(system_data_real["order"])

    f = _instantiate_filter(cls, order=order)
    res = _run_optimize(f, x, d)

    assert isinstance(res, dict), f"{algo.class_name}: optimize deve retornar dict"
    assert "coefficients" in res, f"{algo.class_name}: faltando 'coefficients' no retorno"

    if "outputs" in res:
        y = np.asarray(res["outputs"])
        assert y.ndim == 1, f"{algo.class_name}: outputs should be 1-D"
        assert y.size > 10, f"{algo.class_name}: outputs too short"
        assert np.all(np.isfinite(np.real(y))), f"{algo.class_name}: outputs NaN/Inf (real)"
        _assert_realish(y, f"{algo.class_name}.outputs", dbg=_debug_blob(cls, res, np.zeros(1)))

    e = _get_errors_from_res(res, d)
    assert e.ndim == 1, f"{algo.class_name}: errors should be 1-D"
    assert e.size > 10, f"{algo.class_name}: errors too short"
    assert np.all(np.isfinite(np.real(e))), f"{algo.class_name}: errors NaN/Inf (real)"
    _assert_realish(e, f"{algo.class_name}.errors", dbg=_debug_blob(cls, res, e))

    L = int(e.size)
    d_aligned = _align_to_length(d, L)

    tail = min(500, L // 2) if L > 50 else max(10, L // 3)
    mse_tail = _tail_mse(e, tail=tail)
    sig_pow = float(np.mean(np.abs(d_aligned) ** 2)) + 1e-12

    dbg = _debug_blob(cls, res, e)
    assert mse_tail < 0.10 * sig_pow, (
        f"{algo.class_name}: MSE tail alto (mse_tail={mse_tail:.3e}, sig_pow={sig_pow:.3e}, tail={tail}){dbg}"
    )


@pytest.mark.parametrize("algo", ALGOS, ids=lambda a: a.name)
def test_subband_complex_if_supported(algo: AlgoSpec, system_data):
    cls = _try_import_class(IMPORT_CANDIDATES[algo.class_name], algo.class_name)
    if cls is None:
        pytest.skip(f"Não encontrei {algo.class_name} nos imports candidatos. Ajuste IMPORT_CANDIDATES.")

    if not _is_complex_capable(cls):
        pytest.skip(f"{algo.class_name}: supports_complex=False")

    x = np.asarray(system_data["x"], dtype=complex)
    d = np.asarray(system_data["d_ideal"], dtype=complex)
    order = int(system_data["order"])

    f = _instantiate_filter(cls, order=order)
    res = _run_optimize(f, x, d)

    assert isinstance(res, dict), f"{algo.class_name}: optimize deve retornar dict"
    assert "coefficients" in res, f"{algo.class_name}: faltando 'coefficients' no retorno"

    if "outputs" in res:
        y = np.asarray(res["outputs"])
        assert y.ndim == 1, f"{algo.class_name}: outputs should be 1-D"
        assert y.size > 10, f"{algo.class_name}: outputs too short"
        assert np.all(np.isfinite(np.real(y))), f"{algo.class_name}: outputs NaN/Inf (real)"
        assert np.all(np.isfinite(np.imag(y))), f"{algo.class_name}: outputs NaN/Inf (imag)"

    e = _get_errors_from_res(res, d)
    assert e.ndim == 1, f"{algo.class_name}: errors should be 1-D"
    assert e.size > 10, f"{algo.class_name}: errors too short"
    assert np.all(np.isfinite(np.real(e))), f"{algo.class_name}: errors NaN/Inf (real)"
    assert np.all(np.isfinite(np.imag(e))), f"{algo.class_name}: errors NaN/Inf (imag)"

    L = int(e.size)
    d_aligned = _align_to_length(d, L)

    tail = min(500, L // 2) if L > 50 else max(10, L // 3)
    mse_tail = _tail_mse(e, tail=tail)
    sig_pow = float(np.mean(np.abs(d_aligned) ** 2)) + 1e-12

    dbg = _debug_blob(cls, res, e)
    assert mse_tail < 0.10 * sig_pow, (
        f"{algo.class_name}: MSE tail alto (mse_tail={mse_tail:.3e}, sig_pow={sig_pow:.3e}, tail={tail}){dbg}"
    )
