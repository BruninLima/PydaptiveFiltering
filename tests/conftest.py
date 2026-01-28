# tests/conftest.py

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from pydaptivefiltering.base import OptimizationResult


def _last_coefficients(obj):
    """
    Extract 'final' coefficients from several supported containers.

    Supported:
      - OptimizationResult: uses result.coefficients (last snapshot)
      - dict-like: expects key 'coefficients' (or legacy 'w_history'), uses last snapshot
      - np.ndarray/list: returned as-is (assumed final coefficients)

    Returns
    -------
    np.ndarray
        Final coefficient vector (or matrix for algorithms that store 2D coefficients).
    """
    if isinstance(obj, OptimizationResult):
        coeffs = np.asarray(obj.coefficients)
        if coeffs.size == 0:
            return coeffs
        # If coeffs is (N, n_coeffs) -> take last row
        if coeffs.ndim >= 2:
            return coeffs[-1]
        # If coeffs is 1D already, assume it is final
        return coeffs

    if isinstance(obj, dict):
        coeffs = obj.get("coefficients", None)
        if coeffs is None:
            coeffs = obj.get("w_history", None)
        if coeffs is None:
            # fallback: if dict stores final w directly
            if "w" in obj:
                return np.asarray(obj["w"])
            raise KeyError("Could not find 'coefficients' (or 'w_history'/'w') in dict result.")

        coeffs_arr = np.asarray(coeffs)
        if coeffs_arr.size == 0:
            return coeffs_arr
        if coeffs_arr.ndim >= 2:
            return coeffs_arr[-1]
        return coeffs_arr

    return np.asarray(obj)


@pytest.fixture
def calculate_msd():
    """
    Mean-square deviation (MSD) between true coefficients and an estimate.

    Accepts w_est as:
      - np.ndarray / list (final coefficient vector),
      - OptimizationResult (uses last entry of result.coefficients),
      - dict-like (uses last entry of 'coefficients' or 'w_history').
    """
    def _calc(w_true, w_est):
        w_true = np.asarray(w_true)
        w_hat = _last_coefficients(w_est)

        # Flatten for MSD comparison (works for vector or matrix final coeffs)
        w_true_flat = w_true.reshape(-1)
        w_hat_flat = np.asarray(w_hat).reshape(-1)

        if w_true_flat.shape != w_hat_flat.shape:
            raise ValueError(
                f"MSD shape mismatch: w_true has {w_true_flat.shape}, w_est has {w_hat_flat.shape}"
            )

        return float(np.mean(np.abs(w_true_flat - w_hat_flat) ** 2))

    return _calc


# -------------------------
# Complex-valued fixtures
# -------------------------

@pytest.fixture
def correlated_data():
    rng = np.random.default_rng(42)
    n_samples = 2000

    h_unknown = np.array([0.6, -0.3, 0.1, 0.05], dtype=np.complex128)
    order = int(len(h_unknown) - 1)

    white_noise = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    white_noise = white_noise.astype(np.complex128, copy=False)

    x = np.convolve(white_noise, np.array([1.0, 0.8, 0.5], dtype=float), mode="full")[:n_samples]
    x = x.astype(np.complex128, copy=False)

    d = np.convolve(x, h_unknown, mode="full")[:n_samples]
    d = d.astype(np.complex128, copy=False)

    return x, d, h_unknown, order


@pytest.fixture
def system_data():
    rng = np.random.default_rng(42)
    n_samples = 5000

    w_optimal = np.array([0.4, -0.2, 0.1], dtype=np.complex128)
    order = int(len(w_optimal) - 1)

    x = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2.0)
    x = x.astype(np.complex128, copy=False)

    d_ideal = signal.lfilter(w_optimal, 1, x).astype(np.complex128, copy=False)

    return {
        "x": x,
        "d_ideal": d_ideal,
        "w_optimal": w_optimal,
        "order": order,
        "n_samples": n_samples,
    }


@pytest.fixture
def lms_data(correlated_data):
    # Alias kept for backward compatibility with older tests
    x, d, h_unknown, order = correlated_data
    return {"x": x, "d": d, "h_unknown": h_unknown, "order": order, "n_samples": int(len(x))}


@pytest.fixture
def rls_test_data():
    rng = np.random.default_rng(42)
    n_samples = 1000

    w_optimal = np.array([0.5, -0.4, 0.2], dtype=np.complex128)
    order = int(len(w_optimal) - 1)

    u = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    u = u.astype(np.complex128, copy=False)

    x = np.zeros(n_samples, dtype=np.complex128)
    for i in range(1, n_samples):
        x[i] = 0.9 * x[i - 1] + u[i]

    d = np.convolve(x, w_optimal, mode="full")[:n_samples].astype(np.complex128, copy=False)

    return {"x": x, "d": d, "w_optimal": w_optimal, "order": order, "n_samples": n_samples}


# -------------------------
# Nonlinear fixtures (REAL)
# -------------------------

@pytest.fixture
def quadratic_system_data():
    """Gera dados de um sistema não-linear quadrático para Volterra (REAL)."""
    rng = np.random.default_rng(42)
    n_samples = 1500

    x = rng.standard_normal(n_samples).astype(np.float64, copy=False)
    d = np.zeros(n_samples, dtype=np.float64)

    # Modelo: d(k) = -0.76x(k) - 1.0x(k-1) + 0.5x(k)^2 + ruído
    for k in range(1, n_samples):
        d[k] = -0.76 * x[k] - 1.0 * x[k - 1] + 0.5 * (x[k] ** 2) + 0.01 * rng.standard_normal()

    return {"x": x, "d": d, "n_samples": n_samples}


@pytest.fixture
def rbf_mapping_data():
    """Gera uma função seno para mapeamento não-linear (RBF/MLP) (REAL)."""
    n_samples = 1000
    x_axis = np.linspace(0.0, 4.0 * np.pi, n_samples, dtype=np.float64)
    input_sig = np.sin(x_axis).astype(np.float64, copy=False)
    desired = (input_sig ** 2).astype(np.float64, copy=False)

    return {"x": input_sig, "d": desired}


# -------------------------
# Real-valued fixtures
# -------------------------

@pytest.fixture
def correlated_data_real():
    rng = np.random.default_rng(42)
    n_samples = 2000

    h_unknown = np.array([0.6, -0.3, 0.1, 0.05], dtype=np.float64)
    order = int(len(h_unknown) - 1)

    white_noise = rng.standard_normal(n_samples).astype(np.float64, copy=False)
    x = np.convolve(white_noise, np.array([1.0, 0.8, 0.5], dtype=np.float64), mode="full")[:n_samples]
    x = x.astype(np.float64, copy=False)

    d = np.convolve(x, h_unknown, mode="full")[:n_samples].astype(np.float64, copy=False)

    return x, d, h_unknown, order


@pytest.fixture
def system_data_real():
    rng = np.random.default_rng(42)
    n_samples = 5000

    w_optimal = np.array([0.4, -0.2, 0.1], dtype=np.float64)
    order = int(len(w_optimal) - 1)

    x = rng.standard_normal(n_samples).astype(np.float64, copy=False)
    d_ideal = signal.lfilter(w_optimal, 1, x).astype(np.float64, copy=False)

    return {
        "x": x,
        "d_ideal": d_ideal,
        "w_optimal": w_optimal,
        "order": order,
        "n_samples": n_samples,
    }


@pytest.fixture
def rls_test_data_real():
    rng = np.random.default_rng(42)
    n_samples = 1000

    w_optimal = np.array([0.5, -0.4, 0.2], dtype=np.float64)
    order = int(len(w_optimal) - 1)

    u = rng.standard_normal(n_samples).astype(np.float64, copy=False)

    x = np.zeros(n_samples, dtype=np.float64)
    for i in range(1, n_samples):
        x[i] = 0.9 * x[i - 1] + u[i]

    d = np.convolve(x, w_optimal, mode="full")[:n_samples].astype(np.float64, copy=False)

    return {"x": x, "d": d, "w_optimal": w_optimal, "order": order, "n_samples": n_samples}


@pytest.fixture
def lms_data_real(correlated_data_real):
    x, d, h_unknown, order = correlated_data_real
    return {"x": x, "d": d, "h_unknown": h_unknown, "order": order, "n_samples": int(len(x))}
