# tests/test_iir_accuracy.py

import pytest
import numpy as np
from scipy import signal

from pydaptivefiltering.iir import (
    ErrorEquation,
    GaussNewton,
    GaussNewtonGradient,
    RLSIIR,
    SteiglitzMcBride,
)


def generate_iir_system(n_samples, b_true, a_true, noise_std=0.01, seed=42):
    """Gera dados de um sistema IIR: d = (B/A)x + ruído."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples).astype(float)
    d_clean = signal.lfilter(b_true, a_true, x).astype(float)
    d = d_clean + float(noise_std) * rng.standard_normal(n_samples).astype(float)
    return x, d


@pytest.fixture
def target_system():
    """Define um sistema IIR simples de 1ª ordem para teste de convergência."""
    # H(z) = 0.5 / (1 - 0.2 z^-1) -> b=[0.5], a=[1, -0.2]
    # Convenção theta = [a1, b0] -> [0.2, 0.5]
    return {
        "b": np.array([0.5], dtype=float),
        "a": np.array([1.0, -0.2], dtype=float),
        "theta_true": np.array([0.2, 0.5], dtype=float),
        "zeros_order": 0,
        "poles_order": 1,
    }


def test_rls_iir_parameter_accuracy(target_system):
    """Verifica se o RLSIIR consegue recuperar os coeficientes exatos do sistema (ruído zero)."""
    x, d = generate_iir_system(
        2000, target_system["b"], target_system["a"], noise_std=0.0, seed=42
    )

    flt = RLSIIR(
        zeros_order=target_system["zeros_order"],
        poles_order=target_system["poles_order"],
        forgetting_factor=1.0,
        delta=10.0,
    )
    res = flt.optimize(x, d)

    w_final = np.asarray(res.coefficients[-1], dtype=float)
    msd = float(np.sum((w_final - target_system["theta_true"]) ** 2))

    assert msd < 1e-5


def test_gauss_newton_convergence_speed(target_system):
    """Compara velocidade de convergência: GaussNewton vs GaussNewtonGradient."""
    x, d = generate_iir_system(
        1000, target_system["b"], target_system["a"], noise_std=0.01, seed=42
    )

    gn_full = GaussNewton(
        zeros_order=0,
        poles_order=1,
        alpha=0.05,
        step_size=0.2,
        delta=1.0,
    )
    gn_grad = GaussNewtonGradient(
        zeros_order=0,
        poles_order=1,
        step_size=0.01,
    )

    res_full = gn_full.optimize(x, d)
    res_grad = gn_grad.optimize(x, d)

    mse_full = float(np.mean(np.asarray(res_full.errors[250:300], dtype=float) ** 2))
    mse_grad = float(np.mean(np.asarray(res_grad.errors[250:300], dtype=float) ** 2))

    assert mse_full < mse_grad


def test_steiglitz_mcbride_bias_reduction():
    """
    Steiglitz-McBride tende a reduzir viés em presença de erro na saída
    em comparação com formulações diretas.
    """
    b_true = [1.0, 0.5]
    a_true = [1.0, -0.8, 0.2]

    x, d = generate_iir_system(3000, b_true, a_true, noise_std=0.05, seed=42)

    sm = SteiglitzMcBride(
        zeros_order=1,
        poles_order=2,
        step_size=0.005,
    )
    res = sm.optimize(x, d)

    final_mse = float(np.mean(np.asarray(res.errors[-200:], dtype=float) ** 2))
    assert final_mse < 0.01


@pytest.mark.parametrize("filter_class", [RLSIIR, ErrorEquation, GaussNewton])
def test_iir_transfer_function_match(filter_class, target_system):
    """
    Sanity: resposta ao degrau do filtro estimado deve ser próxima da do sistema alvo.
    """
    x, d = generate_iir_system(
        2000, target_system["b"], target_system["a"], noise_std=0.01, seed=42
    )

    params = {"zeros_order": 0, "poles_order": 1}

    if filter_class is GaussNewton:
        params.update({"alpha": 0.1, "step_size": 0.05, "delta": 1.0})
    elif filter_class is RLSIIR:
        params.update({"forgetting_factor": 0.999, "delta": 10.0})
    elif filter_class is ErrorEquation:
        # ErrorEquation usa "epsilon" (regularização), não delta
        params.update({"forgetting_factor": 0.999, "epsilon": 1e-3})

    flt = filter_class(**params)
    res = flt.optimize(x, d)

    w_est = np.asarray(res.coefficients[-1], dtype=float)

    # Reconstrói H(z) a partir de theta = [a1, b0]
    a_est = [1.0, -float(w_est[0])]
    b_est = [float(w_est[1])]

    _, y_true = signal.step((target_system["b"], target_system["a"]))
    _, y_est = signal.step((b_est, a_est))

    dist = float(np.linalg.norm(y_true[:20] - y_est[:20]))
    assert dist < 0.5


def test_iir_stability_check():
    """Verifica se o filtro estimado permanece estável (polo dentro do círculo unitário)."""
    x, d = generate_iir_system(1000, [1.0], [1.0, -0.9], noise_std=0.01, seed=42)

    flt = GaussNewton(zeros_order=0, poles_order=1, step_size=0.1)
    res = flt.optimize(x, d)

    w_final = np.asarray(res.coefficients[-1], dtype=float)
    pole = float(w_final[0])  # para 1ª ordem, polo = a1

    assert abs(pole) < 1.0, f"Filtro instável com polo em {pole}"


def test_bias_comparison_sm_vs_ee():
    """Steiglitz-McBride deve ter menos viés que ErrorEquation com ruído alto."""
    b_true, a_true = [0.5], [1.0, -0.6]
    theta_true = np.array([0.6, 0.5], dtype=float)

    x, d = generate_iir_system(4000, b_true, a_true, noise_std=0.2, seed=42)

    ee = ErrorEquation(
        zeros_order=0,
        poles_order=1,
        forgetting_factor=0.99,
        epsilon=1e-3,
    )
    sm = SteiglitzMcBride(
        zeros_order=0,
        poles_order=1,
        step_size=0.01,
    )

    res_ee = ee.optimize(x, d)
    res_sm = sm.optimize(x, d)

    w_ee = np.asarray(res_ee.coefficients[-1], dtype=float)
    w_sm = np.asarray(res_sm.coefficients[-1], dtype=float)

    msd_ee = float(np.sum((w_ee - theta_true) ** 2))
    msd_sm = float(np.sum((w_sm - theta_true) ** 2))

    assert msd_sm < msd_ee
