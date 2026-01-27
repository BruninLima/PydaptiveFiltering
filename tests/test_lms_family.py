import pytest
import numpy as np
from pydaptivefiltering import LMS, NLMS, AffineProjection

def test_lms_convergence(lms_data, calculate_msd):
    # lms_data é o dicionário retornado pela fixture no conftest.py
    model = LMS(filter_order=lms_data["order"], step=0.05)
    model.optimize(lms_data["x"], lms_data["d"])
    assert calculate_msd(lms_data["h_unknown"], model.w) < 1e-3

def test_nlms_robustness(lms_data, calculate_msd):
    """Testa o NLMS com sinal de entrada de escala variada."""
    # O NLMS deve ser invariante à escala da entrada devido à normalização
    x_large = lms_data["x"] * 100.0
    d_large = np.convolve(x_large, lms_data["h_unknown"], mode='full')[:lms_data["n_samples"]]
    
    model = NLMS(filter_order=lms_data["order"], step=0.1, gamma=1e-6)
    model.optimize(x_large, d_large)
    assert calculate_msd(lms_data["h_unknown"], model.w) < 1e-3



def test_affine_projection_colored_noise(lms_data, calculate_msd):
    """AP deve convergir bem com sinal altamente correlacionado (ruído colorido)."""
    x = lms_data["x"]
    # Gerando um processo AR(1) para correlacionar a entrada
    x_colored = np.zeros_like(x)
    for i in range(1, len(x)):
        x_colored[i] = 0.9 * x_colored[i-1] + x[i]
    
    d_colored = np.convolve(x_colored, lms_data["h_unknown"], mode='full')[:lms_data["n_samples"]]
    
    # O AP usa L vetores passados para "branquear" a entrada e acelerar a convergência
    model = AffineProjection(filter_order=lms_data["order"], step=0.2, L=4)
    model.optimize(x_colored, d_colored)
    assert calculate_msd(lms_data["h_unknown"], model.w) < 5e-3



def test_weight_initialization(lms_data):
    """Garante que o w_init personalizado seja respeitado."""
    w_init = np.array([1.0, 1.0, 1.0], dtype=complex)
    model = LMS(filter_order=lms_data["order"], w_init=w_init)
    np.testing.assert_array_equal(model.w, w_init)

def test_input_validation(lms_data):
    """Garante que o erro de dimensão seja disparado corretamente."""
    model = LMS(filter_order=lms_data["order"])
    with pytest.raises(ValueError):
        # x e d com tamanhos diferentes devem levantar ValueError
        model.optimize(lms_data["x"], lms_data["d"][:100])