import pytest
import numpy as np
from scipy import signal
from pydaptivefiltering import LMS, NLMS, AffineProjection, SignData, SignError
from pydaptivefiltering import LMSNewton
from pydaptivefiltering import TDomainDFT
from pydaptivefiltering import TDomainDCT
from pydaptivefiltering import Power2ErrorLMS

# --- FAMÍLIA PADRÃO ---

def test_lms_implementation(system_data, calculate_msd):
    filt = LMS(filter_order=system_data["order"], step=0.01)
    filt.optimize(system_data["x"], system_data["d_ideal"])
    assert calculate_msd(system_data["w_optimal"], filt.w) < 1e-4

def test_nlms_implementation(system_data, calculate_msd):
    # Teste de robustez com escala
    x_high = system_data["x"] * 5
    d_high = signal.lfilter(system_data["w_optimal"], 1, x_high)
    filt = NLMS(filter_order=system_data["order"], step=0.1, gamma=1e-6)
    filt.optimize(x_high, d_high)
    assert calculate_msd(system_data["w_optimal"], filt.w) < 1e-4

def test_affine_projection_implementation(system_data, calculate_msd):
    filt = AffineProjection(filter_order=system_data["order"], step=0.1, L=2, gamma=1e-3)
    filt.optimize(system_data["x"], system_data["d_ideal"])
    assert calculate_msd(system_data["w_optimal"], filt.w) < 1e-4

# --- FAMÍLIA AVANÇADA / TRANSFORM DOMAIN ---

def test_lms_newton_accuracy(system_data, calculate_msd):
    delta = 1.0
    init_inv = delta * np.eye(system_data["order"] + 1)
    newton = LMSNewton(system_data["order"], alpha=0.99, initial_inv_rx=init_inv, step=0.1)
    newton.optimize(system_data["x"], system_data["d_ideal"])
    assert calculate_msd(system_data["w_optimal"], newton.w) < 1e-5

def test_transform_domain_dft_accuracy(system_data, calculate_msd):
    dft = TDomainDFT(system_data["order"], step=0.1, gamma=1e-4, alpha=0.1, initial_power=1.0)
    dft.optimize(system_data["x"], system_data["d_ideal"])
    assert calculate_msd(system_data["w_optimal"], dft.w) < 1e-3

def test_transform_domain_dct_accuracy(system_data, calculate_msd):
    x_r, d_r = np.real(system_data["x"]), np.real(system_data["d_ideal"])
    dct_filt = TDomainDCT(system_data["order"], step=0.1, gamma=1e-4, alpha=0.1, initial_power=1.0)
    dct_filt.optimize(x_r, d_r)
    msd = calculate_msd(np.real(system_data["w_optimal"]), dct_filt.w)
    assert msd < 1e-3

# --- ALGORITMOS QUANTIZADOS (SIGN / POWER2) ---

def test_sign_data_implementation(system_data, calculate_msd):
    x_r, d_r = np.real(system_data["x"]), np.real(system_data["d_ideal"])
    filt = SignData(filter_order=system_data["order"], step=0.001)
    filt.optimize(x_r, d_r)
    assert calculate_msd(np.real(system_data["w_optimal"]), filt.w) < 0.05

def test_sign_error_implementation(system_data, calculate_msd):
    x_r, d_r = np.real(system_data["x"]), np.real(system_data["d_ideal"])
    filt = SignError(filter_order=system_data["order"], step=0.001)
    filt.optimize(x_r, d_r)
    assert calculate_msd(np.real(system_data["w_optimal"]), filt.w) < 0.05

def test_power2_quantization_stability(system_data, calculate_msd):
    x_r, d_r = np.real(system_data["x"]), np.real(system_data["d_ideal"])
    p2 = Power2ErrorLMS(filter_order=system_data["order"], step=0.01, bd=8, tau=1e-3)
    res = p2.optimize(x_r, d_r)
    mse_final = np.mean(np.abs(res['errors'][-500:])**2)
    assert mse_final < 0.05

def test_input_validation_all(system_data):
    filters = [
        LMS(system_data["order"], 0.01),
        NLMS(system_data["order"], 0.1, gamma=1e-6),
        AffineProjection(system_data["order"], 0.1, L=2, gamma=1e-3),
        SignData(system_data["order"], 0.001),
        SignError(system_data["order"], 0.001)
    ]
    for f in filters:
        with pytest.raises(ValueError):
            f.optimize(np.array([1, 2]), np.array([1]))