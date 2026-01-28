# tests/test_rls_real_equivalence.py

from __future__ import annotations

import numpy as np
import pytest

from pydaptivefiltering import FastRLS, StabFastRLS
from pydaptivefiltering.base import OptimizationResult


def generate_real_fir_data(
    n_samples: int = 2000,
    w_true: np.ndarray | None = None,
    noise_std: float = 0.01,
    seed: int = 0,
):
    """
    Gera x real i.i.d e d = FIR(w_true) * x + ruido.
    Convenção: x_k = [x[k], x[k-1], ..., x[k-M]]  (amostra mais recente primeiro)
    """
    rng = np.random.default_rng(seed)

    if w_true is None:
        w_true = np.array([0.5, -0.4, 0.2], dtype=float)

    x = rng.standard_normal(n_samples).astype(float)

    m = len(w_true) - 1  # order (taps-1)
    x_pad = np.zeros(n_samples + m, dtype=float)
    x_pad[m:] = x

    d = np.zeros(n_samples, dtype=float)
    for k in range(n_samples):
        xk = x_pad[k : k + m + 1][::-1]
        d[k] = float(np.dot(w_true, xk))

    d += noise_std * rng.standard_normal(n_samples)
    return x, d, w_true


def rel_rmse(a, b, eps: float = 1e-12) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    num = np.sqrt(np.mean(np.abs(a - b) ** 2))
    den = np.sqrt(np.mean(np.abs(a) ** 2)) + eps
    return float(num / den)


def last_w(res: OptimizationResult) -> np.ndarray:
    coeffs = np.asarray(res.coefficients)
    if coeffs.ndim != 2 or coeffs.shape[0] == 0:
        raise AssertionError("coefficients history is empty or has unexpected shape.")
    return coeffs[-1]


@pytest.fixture
def real_rls_test_data():
    x, d, w_true = generate_real_fir_data(
        n_samples=2000,
        w_true=np.array([0.5, -0.4, 0.2], dtype=float),
        noise_std=0.01,
        seed=123,
    )
    return {
        "x": x,
        "d": d,
        "w_true": w_true,
        "n_taps": int(w_true.size),
        "filter_order": int(w_true.size - 1),
    }


def test_stab_vs_fast_equivalence_real_short(real_rls_test_data):
    """
    Em poucas iterações, StabFastRLS deve produzir saída muito próxima do FastRLS
    (equivalência prática), sem exigir igualdade elemento-a-elemento.
    """
    data = real_rls_test_data
    n_short = 30

    x = data["x"][:n_short]
    d = data["d"][:n_short]
    order = data["filter_order"]

    f_fast = FastRLS(filter_order=order, forgetting_factor=0.99, epsilon=0.1)
    f_stab = StabFastRLS(filter_order=order, forgetting_factor=0.99, epsilon=0.1)

    res_fast = f_fast.optimize(x, d)
    res_stab = f_stab.optimize(x, d)

    y_fast = np.asarray(res_fast.outputs)
    y_stab = np.asarray(res_stab.outputs)

    # Para dados reais, se algum retornar complexo, imag deve ser ~0
    if np.iscomplexobj(y_fast):
        assert np.max(np.abs(np.imag(y_fast))) < 1e-6
        y_fast = np.real(y_fast)

    if np.iscomplexobj(y_stab):
        assert np.max(np.abs(np.imag(y_stab))) < 1e-6
        y_stab = np.real(y_stab)

    assert np.isrealobj(y_stab)

    rrmse = rel_rmse(y_fast, y_stab)
    assert rrmse < 0.02  # 2%


def test_stab_converges_to_true_system_real(real_rls_test_data):
    """Testa convergência do StabFastRLS REAL para um FIR conhecido."""
    data = real_rls_test_data
    x = data["x"]
    d = data["d"]
    w_true = data["w_true"]
    order = data["filter_order"]

    f_stab = StabFastRLS(filter_order=order, forgetting_factor=0.995, epsilon=0.1)
    res = f_stab.optimize(x, d)

    w_est = last_w(res)
    if np.iscomplexobj(w_est):
        assert np.max(np.abs(np.imag(w_est))) < 1e-8
        w_est = np.real(w_est)

    np.testing.assert_allclose(w_est, w_true, rtol=0.10, atol=0.05)

    e = np.asarray(res.errors)
    if np.iscomplexobj(e):
        assert np.max(np.abs(np.imag(e))) < 1e-8
        e = np.real(e)

    mse_tail = float(np.mean(e[-200:] ** 2))
    assert mse_tail < 5e-3


def test_fast_converges_to_true_system_real(real_rls_test_data):
    """Testa convergência do FastRLS para dados reais."""
    data = real_rls_test_data
    x = data["x"]
    d = data["d"]
    w_true = data["w_true"]
    order = data["filter_order"]

    f_fast = FastRLS(filter_order=order, forgetting_factor=0.995, epsilon=0.1)
    res = f_fast.optimize(x, d)

    w_est = last_w(res)
    if np.iscomplexobj(w_est):
        assert np.max(np.abs(np.imag(w_est))) < 1e-6
        w_est = np.real(w_est)

    np.testing.assert_allclose(w_est, w_true, rtol=0.15, atol=0.08)

    e = np.asarray(res.errors)
    if np.iscomplexobj(e):
        assert np.max(np.abs(np.imag(e))) < 1e-6
        e = np.real(e)

    mse_tail = float(np.mean(e[-200:] ** 2))
    assert mse_tail < 1e-2


def test_real_input_produces_real_outputs_policy(real_rls_test_data):
    """
    Política recomendada com API padronizada (e algoritmos complex):
    - Se entrada/desired são reais, outputs/errors devem ter parte imaginária ~0.
    """
    data = real_rls_test_data
    x = data["x"][:200]
    d = data["d"][:200]
    order = data["filter_order"]

    f_stab = StabFastRLS(filter_order=order, forgetting_factor=0.99, epsilon=0.1)
    res = f_stab.optimize(x, d)

    y = np.asarray(res.outputs)
    e = np.asarray(res.errors)

    if np.iscomplexobj(y):
        assert np.max(np.abs(np.imag(y))) < 1e-6
    if np.iscomplexobj(e):
        assert np.max(np.abs(np.imag(e))) < 1e-6


def test_long_term_stability_comparison():
    """Verifica se o StabFastRLS é numericamente estável em execução longa."""
    x_long, d_long, w_true = generate_real_fir_data(n_samples=20000, seed=42)
    order = int(w_true.size - 1)

    f_stab = StabFastRLS(filter_order=order, forgetting_factor=0.999, epsilon=0.5)
    res_stab = f_stab.optimize(x_long, d_long)

    y = np.asarray(res_stab.outputs)
    if np.iscomplexobj(y):
        y = np.real(y)

    assert np.all(np.isfinite(y))
    mse_end = float(np.mean((y[-500:] - d_long[-500:]) ** 2))
    assert mse_end < 1.0  # não divergiu


def test_tracking_performance_abrupt_change():
    """Testa se o StabFastRLS consegue rastrear uma mudança no sistema."""
    x1, d1, w_true = generate_real_fir_data(n_samples=1000, seed=1)
    w_new = -w_true
    x2, d2, _ = generate_real_fir_data(n_samples=1000, w_true=w_new, seed=2)

    x_total = np.concatenate([x1, x2])
    d_total = np.concatenate([d1, d2])

    order = int(w_true.size - 1)

    f_stab = StabFastRLS(filter_order=order, forgetting_factor=0.995, epsilon=0.5)
    res = f_stab.optimize(x_total, d_total)

    w_final = last_w(res)
    if np.iscomplexobj(w_final):
        w_final = np.real(w_final)

    np.testing.assert_allclose(w_final, w_new, atol=0.2)
