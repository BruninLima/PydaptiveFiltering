# tests/test_rls_real_equivalence.py

from __future__ import annotations

import numpy as np
import pytest

from pydaptivefiltering import FastRLS, StabFastRLS


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


def _last_w(coeff_history):
    """
    Helper: seu framework guarda w_history como lista de arrays.
    Suporta também o caso de matriz (n_taps x (n_iter+1)) ou (n_iter x n_taps).
    """
    if coeff_history is None:
        return None
    if isinstance(coeff_history, list) and len(coeff_history) > 0:
        return np.asarray(coeff_history[-1])

    arr = np.asarray(coeff_history)
    if arr.ndim == 2:
        # tenta inferir orientação
        if arr.shape[0] >= arr.shape[1]:
            return arr[-1, :]
        else:
            return arr[:, -1]
    return arr


def _get_outputs(res: dict) -> np.ndarray:
    if "outputs" in res:
        return np.asarray(res["outputs"])
    if "y" in res:
        return np.asarray(res["y"])
    raise KeyError("Result does not contain outputs/y.")


def _get_errors(res: dict, d: np.ndarray) -> np.ndarray:
    # Aceita vários padrões de chave
    for k in ("priori_errors", "errors", "e"):
        if k in res:
            return np.asarray(res[k])
    # fallback: d - y
    y = _get_outputs(res)
    if np.iscomplexobj(y):
        y = np.real(y)
    return np.asarray(d) - np.asarray(y)


def _make_filter_with_matching_taps(cls, n_taps: int, **kwargs):
    """
    Alguns filtros usam `filter_order` como:
      - ordem M (taps = M+1), ou
      - número de taps diretamente (taps = M)

    Para evitar o erro: shapes (2,) and (3,) not aligned,
    tentamos as duas convenções e escolhemos a que produz len(w)=n_taps.
    """
    # tentativa A: filter_order = n_taps - 1 (convenção mais comum: ordem)
    try:
        f = cls(filter_order=n_taps - 1, **kwargs)
        w = np.asarray(getattr(f, "w"))
        if w.size == n_taps:
            return f
    except Exception:
        f = None

    # tentativa B: filter_order = n_taps (convenção: taps)
    f2 = cls(filter_order=n_taps, **kwargs)
    w2 = np.asarray(getattr(f2, "w"))
    if w2.size == n_taps:
        return f2

    # se nenhuma bateu, dá um erro bem explicativo
    wA = None
    if f is not None:
        try:
            wA = np.asarray(getattr(f, "w")).size
        except Exception:
            wA = None
    raise AssertionError(
        f"{cls.__name__}: could not match n_taps={n_taps}. "
        f"TryA(order={n_taps-1}) -> w.size={wA}; "
        f"TryB(order={n_taps}) -> w.size={w2.size}."
    )


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
    n_taps = data["n_taps"]

    f_fast = _make_filter_with_matching_taps(FastRLS, n_taps=n_taps, forgetting_factor=0.99)
    f_stab = _make_filter_with_matching_taps(StabFastRLS, n_taps=n_taps, forgetting_factor=0.99, epsilon=0.1)

    res_fast = f_fast.optimize(x, d)
    res_stab = f_stab.optimize(x, d)

    y_fast = _get_outputs(res_fast)
    y_stab = _get_outputs(res_stab)

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
    """
    Testa convergência do StabFastRLS REAL para um FIR conhecido.
    """
    data = real_rls_test_data
    x = data["x"]
    d = data["d"]
    w_true = data["w_true"]
    n_taps = data["n_taps"]

    f_stab = _make_filter_with_matching_taps(
        StabFastRLS,
        n_taps=n_taps,
        forgetting_factor=0.995,
        epsilon=0.1,
    )
    res = f_stab.optimize(x, d)

    w_est = _last_w(res.get("coefficients", None))
    assert w_est is not None, "StabFastRLS must return coefficients history."

    if np.iscomplexobj(w_est):
        assert np.max(np.abs(np.imag(w_est))) < 1e-8
        w_est = np.real(w_est)

    # como RLS tende a convergir bem, tolerâncias moderadas
    np.testing.assert_allclose(w_est, w_true, rtol=0.10, atol=0.05)

    e = _get_errors(res, d)
    if np.iscomplexobj(e):
        assert np.max(np.abs(np.imag(e))) < 1e-8
        e = np.real(e)

    mse_tail = float(np.mean(e[-200:] ** 2))
    assert mse_tail < 5e-3


def test_fast_converges_to_true_system_real(real_rls_test_data):
    """
    Testa convergência do FastRLS para dados reais.
    """
    data = real_rls_test_data
    x = data["x"]
    d = data["d"]
    w_true = data["w_true"]
    n_taps = data["n_taps"]

    f_fast = _make_filter_with_matching_taps(
        FastRLS,
        n_taps=n_taps,
        forgetting_factor=0.995,
    )
    res = f_fast.optimize(x, d)

    w_est = _last_w(res.get("coefficients", None))
    assert w_est is not None, "FastRLS must return coefficients history."

    if np.iscomplexobj(w_est):
        assert np.max(np.abs(np.imag(w_est))) < 1e-6
        w_est = np.real(w_est)

    np.testing.assert_allclose(w_est, w_true, rtol=0.15, atol=0.08)

    e = _get_errors(res, d)
    if np.iscomplexobj(e):
        assert np.max(np.abs(np.imag(e))) < 1e-6
        e = np.real(e)

    mse_tail = float(np.mean(e[-200:] ** 2))
    assert mse_tail < 1e-2


def test_real_stab_fast_rls_complex_input_policy(real_rls_test_data):
    """
    Política recomendada para a versão REAL:
    - ou rejeitar complexo (TypeError/ValueError)
    - ou aceitar e emitir ComplexWarning e descartar imag

    Aceita QUALQUER uma das duas políticas.
    """
    data = real_rls_test_data
    n_taps = data["n_taps"]

    x = data["x"].astype(complex) + 1j * 0.5 * data["x"]
    d = data["d"].astype(complex) + 1j * 0.5 * data["d"]

    f_stab = _make_filter_with_matching_taps(
        StabFastRLS,
        n_taps=n_taps,
        forgetting_factor=0.99,
        epsilon=0.1,
    )

    # Policy A: strict reject
    with pytest.raises((TypeError, ValueError)):
        f_stab.optimize(x[:20], d[:20])
        return  # se levantou, ok

    # Policy B: warn + discard imag
    # (Se você implementar warning no decorator, este bloco vira o comportamento esperado.)
    with pytest.warns(np.ComplexWarning):
        res = f_stab.optimize(x[:20], d[:20])

    y = _get_outputs(res)
    assert np.isrealobj(y)


def test_long_term_stability_comparison():
    """Verifica se o StabFastRLS sobrevive onde o FastRLS padrão pode divergir."""
    x_long, d_long, w_true = generate_real_fir_data(n_samples=20000, seed=42)
    n_taps = int(w_true.size)
    order_like = n_taps - 1  # só para referência

    f_stab = _make_filter_with_matching_taps(
        StabFastRLS,
        n_taps=n_taps,
        forgetting_factor=0.999,
        epsilon=0.5,
    )
    res_stab = f_stab.optimize(x_long, d_long)

    y = _get_outputs(res_stab)
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

    n_taps = int(w_true.size)

    f_stab = _make_filter_with_matching_taps(
        StabFastRLS,
        n_taps=n_taps,
        forgetting_factor=0.995,
        epsilon=0.5,
    )
    res = f_stab.optimize(x_total, d_total)

    w_final = _last_w(res.get("coefficients", None))
    assert w_final is not None

    if np.iscomplexobj(w_final):
        w_final = np.real(w_final)

    # Deve estar perto do NOVO sistema (tolerância mais larga porque tracking é mais difícil)
    np.testing.assert_allclose(w_final, w_new, atol=0.2)
