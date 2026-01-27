import pytest
import numpy as np
from scipy.signal import lfilter
from pydaptivefiltering.LatticeRLS.LRLS_pos import LatticeRLS
from pydaptivefiltering.LatticeRLS.LRLS_priori import LatticeRLS_Priori
from pydaptivefiltering.LatticeRLS.NLRLS_pos import NormalizedLatticeRLS
from pydaptivefiltering.LatticeRLS.LRLS_EF import LatticeRLSErrorFeedback

@pytest.fixture
def identification_scenario():
    """Gera dados para identificação de um sistema FIR de ordem 2."""
    np.random.seed(42)
    # Aumentado para 5000 para garantir que o NLRLS estabilize a energia xi
    n_samples = 5000 
    w_target = np.array([0.5, -0.3, 0.2]) 
    
    x = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    d_clean = lfilter(w_target, [1.0], x)
    
    # Ruído complexo (ajustado para variância que não mascare a convergência)
    noise = 0.005 * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    d = d_clean + noise
    
    return x, d, w_target

@pytest.mark.parametrize("model_class, threshold", [
    (LatticeRLS, 1e-3), 
    (LatticeRLS_Priori, 1e-3), 
    (NormalizedLatticeRLS, 6e-2), # NLRLS tem erro residual maior em sinais brancos
    (LatticeRLSErrorFeedback, 5e-3) 
])
def test_mse_convergence(model_class, threshold, identification_scenario):
    """Verifica se o MSE final é compatível com o ruído para todas as variantes."""
    x, d, _ = identification_scenario
    order = 2
    
    # Epsilon maior (1e-1) ajuda a "amortecer" o transiente do EF e NLRLS
    model = model_class(filter_order=order, lambda_factor=0.99, epsilon=1e-1)
    res = model.optimize(x, d)
    
    errors = res["errors"]
    # Analisamos as últimas 500 amostras (regime permanente)
    mse_steady_state = np.mean(np.abs(errors[-500:])**2)
    
    assert mse_steady_state < threshold, f"MSE alto para {model_class.__name__}: {mse_steady_state}"

def test_lattice_performance_color_noise():
    """LRLS deve convergir bem com sinais correlacionados (onde brilha frente ao LMS)."""
    np.random.seed(1)
    n = 4000
    white_noise = np.random.randn(n)
    x_correlated = lfilter([1.0], [1.0, -0.9], white_noise) 
    
    w_target = np.array([0.8, -0.5])
    d = lfilter(w_target, [1.0], x_correlated) + 0.01 * np.random.randn(n)
    
    model = LatticeRLSErrorFeedback(filter_order=1, lambda_factor=0.995, epsilon=0.1)
    res = model.optimize(x_correlated, d)
    
    mse_final = np.mean(np.abs(res["errors"][-500:])**2)
    assert mse_final < 5e-3, f"LRLS_EF falhou em sinal correlacionado. MSE: {mse_final}"

def test_normalized_lattice_tracking():
    """Verifica se o NLRLS rastreia mudanças bruscas."""
    np.random.seed(0)
    n = 4000 # Aumentado para dar tempo ao NLRLS de re-normalizar a energia
    x = np.random.randn(n)
    
    d = np.zeros(n)
    d[:2000] = 0.5 * x[:2000]
    d[2000:] = -0.8 * x[2000:] # Degrau na amostra 2000
    
    model = NormalizedLatticeRLS(filter_order=1, lambda_factor=0.98)
    res = model.optimize(x, d)
    
    # Medimos o erro bem depois do degrau para garantir convergência
    error_before = np.mean(np.abs(res["errors"][1700:2000])**2)
    error_after = np.mean(np.abs(res["errors"][3700:4000])**2)
    
    # Limiares ajustados para a sensibilidade da normalização do NLRLS
    assert error_before < 8e-2, f"Erro pré-degrau alto: {error_before}"
    assert error_after < 8e-2, f"Erro pós-degrau alto: {error_after}"