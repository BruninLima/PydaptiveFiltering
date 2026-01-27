# tests/test_rls_real_equivalence.py

import numpy as np
import pytest

# Ajuste os imports para o caminho correto no seu projeto
from pydaptivefiltering import FastRLS
from pydaptivefiltering import StabFastRLS


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

    m = len(w_true) - 1
    x_pad = np.zeros(n_samples + m, dtype=float)
    x_pad[m:] = x

    d = np.zeros(n_samples, dtype=float)
    for k in range(n_samples):
        xk = x_pad[k : k + m + 1][::-1]
        d[k] = float(np.dot(w_true, xk))

    d += noise_std * rng.standard_normal(n_samples)
    return x, d, w_true


def rel_rmse(a, b, eps: float = 1e-12) -> float:
    """
    RMSE relativo robusto para comparar saídas entre algoritmos
    que não são aritmeticamente idênticos.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    num = np.sqrt(np.mean(np.abs(a - b) ** 2))
    den = np.sqrt(np.mean(np.abs(a) ** 2)) + eps
    return float(num / den)


def _last_w(coeff_history):
    """
    Helper: seu framework guarda w_history como lista de arrays.
    Suporta também o caso de matriz (n_taps x (n_iter+1)).
    """
    if isinstance(coeff_history, list):
        return np.asarray(coeff_history[-1])
    arr = np.asarray(coeff_history)
    if arr.ndim == 2:
        return arr[:, -1]
    return arr


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
        "order": 2,
        "w_true": w_true,
    }


def test_stab_vs_fast_equivalence_real_short(real_rls_test_data):
    """
    Em poucas iterações, StabFastRLS deve produzir saída muito próxima do FastRLS
    (equivalência prática), sem exigir igualdade elemento-a-elemento.

    Por quê?
    - StabFastRLS aplica estabilização numérica (clamps/floors) e pode alterar levemente a aritmética
    - FastRLS pode manter dtype complexo internamente mesmo com dados reais
    """
    data = real_rls_test_data
    n_short = 30

    x = data["x"][:n_short]
    d = data["d"][:n_short]

    f_fast = FastRLS(filter_order=data["order"], forgetting_factor=0.99)
    f_stab = StabFastRLS(filter_order=data["order"], forgetting_factor=0.99, epsilon=0.1)

    res_fast = f_fast.optimize(x, d)
    res_stab = f_stab.optimize(x, d)

    y_fast = np.asarray(res_fast["outputs"])
    y_stab = np.asarray(res_stab["outputs"])

    # Para dados reais, se o FastRLS retornar complexo, a parte imag deve ser ~0
    if np.iscomplexobj(y_fast):
        assert np.max(np.abs(np.imag(y_fast))) < 1e-6
        y_fast = np.real(y_fast)

    # StabFastRLS REAL deve ser real
    assert np.isrealobj(y_stab)

    rrmse = rel_rmse(y_fast, y_stab)
    assert rrmse < 0.02  # 2% (ajuste conforme seu cenário)


def test_stab_converges_to_true_system_real(real_rls_test_data):
    """
    Testa convergência do StabFastRLS REAL para um FIR conhecido.
    """
    data = real_rls_test_data
    x = data["x"]
    d = data["d"]
    w_true = data["w_true"]

    f_stab = StabFastRLS(
        filter_order=data["order"],
        forgetting_factor=0.995,
        epsilon=0.1,
    )
    res = f_stab.optimize(x, d)

    w_est = _last_w(res["coefficients"])
    if np.iscomplexobj(w_est):
        assert np.max(np.abs(np.imag(w_est))) < 1e-8
        w_est = np.real(w_est)

    np.testing.assert_allclose(w_est, w_true, rtol=0.10, atol=0.05)

    e = np.asarray(res["priori_errors"], dtype=float)
    mse_tail = float(np.mean(e[-200:] ** 2))
    assert mse_tail < 5e-3


def test_fast_converges_to_true_system_real(real_rls_test_data):
    """
    Testa convergência do FastRLS para dados reais.
    (Se ele divergir, isso dá diagnóstico útil sobre estabilidade numérica.)
    """
    data = real_rls_test_data
    x = data["x"]
    d = data["d"]
    w_true = data["w_true"]

    f_fast = FastRLS(
        filter_order=data["order"],
        forgetting_factor=0.995,
    )
    res = f_fast.optimize(x, d)

    w_est = _last_w(res["coefficients"])
    if np.iscomplexobj(w_est):
        assert np.max(np.abs(np.imag(w_est))) < 1e-6
        w_est = np.real(w_est)

    np.testing.assert_allclose(w_est, w_true, rtol=0.15, atol=0.08)

    # Se FastRLS não expõe priori_errors, reconstruímos via d - y
    if "priori_errors" in res:
        e = np.asarray(res["priori_errors"])
    else:
        y = np.asarray(res["outputs"])
        if np.iscomplexobj(y):
            y = np.real(y)
        e = d - y

    e = np.asarray(e, dtype=float)
    mse_tail = float(np.mean(e[-200:] ** 2))
    assert mse_tail < 1e-2


def test_real_stab_fast_rls_complex_input_policy(real_rls_test_data):
    """
    Política recomendada para a versão REAL:
    - ou rejeitar complexo (TypeError)
    - ou aceitar e emitir ComplexWarning

    Este teste aceita QUALQUER uma das duas políticas, para não quebrar seu CI
    enquanto você decide qual comportamento quer.

    Se você decidir por uma só, simplifique este teste.
    """
    data = real_rls_test_data
    x = data["x"].astype(complex) + 1j * 0.5 * data["x"]
    d = data["d"].astype(complex) + 1j * 0.5 * data["d"]

    f_stab = StabFastRLS(filter_order=data["order"], forgetting_factor=0.99, epsilon=0.1)

    # Policy A: strict
    try:
        with pytest.raises(TypeError):
            f_stab.optimize(x[:20], d[:20])
        return
    except AssertionError:
        # Not strict, so we check warning-based behavior
        pass

    # Policy B: warn + discard imag
    with pytest.warns(np.ComplexWarning):
        res = f_stab.optimize(x[:20], d[:20])

    assert "outputs" in res
    assert np.isrealobj(res["outputs"])
