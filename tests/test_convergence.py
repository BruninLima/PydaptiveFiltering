from __future__ import annotations

import numpy as np
import pytest

import pydaptivefiltering as pdf

WO = np.array([0.5, -0.3, 0.2, 0.1])
NOISE_STD = 1e-3


def _system_id_data(n_samples: int, complex_data: bool = False, seed: int = 0):
    rng = np.random.default_rng(seed)
    if complex_data:
        x = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / 2
        w = WO + 1j * np.array([0.1, 0.2, 0.0, -0.05])
        x_pad = np.concatenate([np.zeros(WO.size - 1), x])
        d = np.array([np.vdot(w, x_pad[k : k + WO.size][::-1]) for k in range(n_samples)])
        d = d + NOISE_STD * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    else:
        x = 0.5 * rng.standard_normal(n_samples)
        d = np.convolve(x, WO)[:n_samples] + NOISE_STD * rng.standard_normal(n_samples)
    return x, d


@pytest.mark.parametrize(
    "make",
    [
        lambda: pdf.RLS(3, 1.0, 0.98),
        lambda: pdf.RLSAlt(3, 1.0, 0.98),
        lambda: pdf.QRRLS(3, 0.98),
        lambda: pdf.StabFastRLS(3, 0.98, 0.1),
        lambda: pdf.LRLSPosteriori(3, 0.98, 0.1),
        lambda: pdf.LRLSPriori(3, 0.98, 0.1),
        lambda: pdf.LRLSErrorFeedback(3, 0.98, 0.1),
        lambda: pdf.NormalizedLRLS(3, 0.98, 0.1),
    ],
)
def test_rls_family_stays_converged_on_long_runs(make):
    x, d = _system_id_data(15000)
    res = make().optimize(x, d)
    tail_mse = np.mean(np.abs(res.errors[-3000:]) ** 2)
    assert tail_mse < 5 * NOISE_STD**2


@pytest.mark.parametrize("complex_data", [False, True])
def test_lattice_rls_matches_conventional_rls(complex_data):
    x, d = _system_id_data(4000, complex_data=complex_data, seed=1)
    ref = pdf.RLS(3, 10.0, 0.98).optimize(x, d, return_internal_states=True)
    for cls in (pdf.LRLSPosteriori, pdf.LRLSErrorFeedback, pdf.NormalizedLRLS, pdf.LRLSPriori):
        res = cls(3, 0.98, 0.1).optimize(x, d)
        expected = ref.errors if res.error_type == "a_priori" else ref.extra["errors_posteriori"]
        # Initializations differ slightly, so compare after the transient.
        np.testing.assert_allclose(res.errors[1000:], expected[1000:], atol=1e-8)


def test_nonlinear_rls_stays_converged_on_long_runs():
    rng = np.random.default_rng(2)
    n = 15000
    x = 0.5 * rng.standard_normal(n)
    x_prev = np.concatenate([[0.0], x[:-1]])
    d = 0.5 * x - 0.2 * x_prev + 0.1 * x * x_prev + NOISE_STD * rng.standard_normal(n)
    res = pdf.VolterraRLS(2, 0.98, 1.0).optimize(x, d)
    assert np.mean(res.errors[-3000:] ** 2) < 5 * NOISE_STD**2


@pytest.mark.parametrize(
    "cls", [pdf.ErrorEquation, pdf.GaussNewton, pdf.GaussNewtonGradient, pdf.RLSIIR, pdf.SteiglitzMcBride]
)
def test_iir_filters_honor_w_init(cls):
    w0 = [0.1, 0.2, 0.3]
    f = cls(1, 1, w_init=w0)
    np.testing.assert_allclose(f.w, w0)
    np.testing.assert_allclose(f.w_history[0], w0)


def test_gauss_newton_default_step_converges():
    from scipy.signal import lfilter

    rng = np.random.default_rng(3)
    x = 0.5 * rng.standard_normal(15000)
    d = lfilter([1.0, 0.5], [1.0, -0.6], x) + NOISE_STD * rng.standard_normal(x.size)
    res = pdf.GaussNewton(1, 1).optimize(x, d)
    assert np.mean(res.errors[-3000:] ** 2) < 5 * NOISE_STD**2


def test_dlcllms_default_converges():
    x, d = _system_id_data(20000)
    res = pdf.DLCLLMS(3, 4).optimize(x, d)
    assert np.mean(np.abs(res.errors[-2000:]) ** 2) < 5 * NOISE_STD**2


def test_fast_rls_warns_instead_of_crashing():
    x, d = _system_id_data(20000)
    with pytest.warns(RuntimeWarning, match="StabFastRLS"):
        res = pdf.FastRLS(3, 0.98, 0.1).optimize(x, d)
    assert res.errors.shape == (20000,)


@pytest.mark.parametrize("cls", [pdf.CMA, pdf.Godard, pdf.Sato])
def test_blind_dispersion_constant_sets_output_level(cls):
    rng = np.random.default_rng(4)
    n = 30000
    s = (rng.choice([-1, 1], n) + 1j * rng.choice([-1, 1], n)) / np.sqrt(2)
    x = np.convolve(s, [1.0, 0.3 + 0.2j, -0.1])[:n]
    f = cls(5, 1e-3, w_init=np.r_[1.0, 0, 0, 0, 0, 0], dispersion_constant=1.0)
    res = f.optimize(x)
    assert abs(np.mean(np.abs(res.outputs[-3000:]) ** 2) - 1.0) < 0.05
