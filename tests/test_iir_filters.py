# tests/test_iir_filters.py

import pytest
import numpy as np

from pydaptivefiltering.iir import (
    ErrorEquation,
    GaussNewton,
    GaussNewtonGradient,
    RLSIIR,
    SteiglitzMcBride,
)

from pydaptivefiltering.base import OptimizationResult


# Parâmetros conservadores para estabilidade nos testes
IIR_CLASSES = [
    (ErrorEquation, {"lambda_hat": 0.999, "delta": 10.0}),
    (GaussNewton, {"alpha": 0.001, "step": 0.01, "delta": 10.0}),
    (GaussNewtonGradient, {"step": 0.001}),
    (RLSIIR, {"lambda_hat": 0.999, "delta": 10.0}),
    (SteiglitzMcBride, {"step": 0.001}),
]


@pytest.mark.parametrize("filter_class, params", IIR_CLASSES)
def test_iir_structure_and_output_shapes(filter_class, params, lms_data_real):
    """
    Sanidade de API:
    - roda optimize(x, d) com sinais reais
    - retorna OptimizationResult
    - outputs/errors com mesmo shape
    - outputs/errors finitos
    - coefficients não vazio
    """
    x = np.asarray(lms_data_real["x"], dtype=float) * 0.1
    d = np.asarray(lms_data_real["d"], dtype=float) * 0.1

    flt = filter_class(**params)
    res = flt.optimize(x, d)

    assert isinstance(res, OptimizationResult)

    assert np.asarray(res.outputs).shape == x.shape
    assert np.asarray(res.errors).shape == x.shape

    assert np.all(np.isfinite(np.asarray(res.outputs)))
    assert np.all(np.isfinite(np.asarray(res.errors)))

    coeffs = res.coefficients
    assert coeffs is not None
    # aceita lista ou array empilhado
    if isinstance(coeffs, list):
        assert len(coeffs) > 0
    else:
        coeffs_arr = np.asarray(coeffs)
        assert coeffs_arr.size > 0


def test_error_equation_convergence(lms_data_real):
    """Verifica se o ErrorEquation reduz o erro em um sistema simples (FIR equivalente)."""
    x = np.asarray(lms_data_real["x"], dtype=float)[:1000]
    d = 0.5 * x

    flt = ErrorEquation(lambda_hat=0.99, delta=1.0)
    res = flt.optimize(x, d)

    e = np.asarray(res.errors, dtype=float)
    initial_mse = float(np.mean(e[5:50] ** 2))
    final_mse = float(np.mean(e[-50:] ** 2))

    assert np.isfinite(initial_mse)
    assert np.isfinite(final_mse)
    assert final_mse < initial_mse


def test_steiglitz_mcbride_stability(lms_data_real):
    """Verifica se o filtro permanece finito com parâmetros estáveis."""
    x = np.asarray(lms_data_real["x"], dtype=float)[:500] * 0.01
    d = np.asarray(lms_data_real["d"], dtype=float)[:500] * 0.01

    flt = SteiglitzMcBride(step=0.001)
    res = flt.optimize(x, d)

    assert np.all(np.isfinite(np.asarray(res.outputs)))
    assert np.all(np.isfinite(np.asarray(res.errors)))


def test_gauss_newton_vs_gradient_based(lms_data_real):
    """Gauss-Newton não deve divergir e deve ao menos melhorar vs o começo (sanidade)."""
    x = np.asarray(lms_data_real["x"], dtype=float)[:1000] * 0.1
    d = np.asarray(lms_data_real["d"], dtype=float)[:1000] * 0.1

    gn_full = GaussNewton(alpha=0.01, step=0.1, delta=100.0)
    gn_grad = GaussNewtonGradient(step=0.001)

    res_full = gn_full.optimize(x, d)
    res_grad = gn_grad.optimize(x, d)

    e_full = np.asarray(res_full.errors, dtype=float)
    e_grad = np.asarray(res_grad.errors, dtype=float)

    mse_full_tail = float(np.mean(e_full[-100:] ** 2))
    mse_grad_tail = float(np.mean(e_grad[-100:] ** 2))

    assert np.isfinite(mse_full_tail)
    assert np.isfinite(mse_grad_tail)

    # sanidade: GN full deve melhorar vs início (evita teste frágil)
    assert mse_full_tail < float(np.mean(e_full[:10] ** 2))


@pytest.mark.parametrize(
    "filter_class, params",
    [
        (RLSIIR, {"lambda_hat": 0.99, "delta": 10.0}),
        (ErrorEquation, {"lambda_hat": 0.98, "delta": 10.0}),
    ],
)
def test_iir_input_validation(filter_class, params):
    flt = filter_class(**params)
    with pytest.raises(ValueError):
        flt.optimize(np.zeros(10, dtype=float), np.zeros(5, dtype=float))
