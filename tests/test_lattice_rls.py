import pytest
import numpy as np
from pydaptivefiltering import LRLSPosteriori
from pydaptivefiltering import LRLSPriori
from pydaptivefiltering import NormalizedLRLS
from pydaptivefiltering import LRLSErrorFeedback

# Parâmetros globais para os testes
LAMBDA = 0.999
EPSILON = 0.2

@pytest.mark.parametrize("model_class", [
    LRLSPosteriori,
    LRLSPriori,
    LRLSErrorFeedback,
    NormalizedLRLS
])
def test_lattice_variants_execution(model_class, system_data):
    """Garante que todas as variantes executam e retornam as chaves corretas."""
    x = system_data["x"]
    d = system_data["d_ideal"]
    order = system_data["order"]
    
    model = model_class(filter_order=order, lambda_factor=LAMBDA, epsilon=EPSILON)
    res = model.optimize(x, d)
    
    assert "outputs" in res
    assert "errors" in res
    assert len(res["outputs"]) == len(x)
    assert not np.any(np.isnan(res["outputs"])), f"NaN detectado em {model_class.__name__}"

@pytest.mark.parametrize("model_class", [
    LRLSPosteriori,
    LRLSPriori,
    LRLSErrorFeedback
])
def test_lattice_convergence(model_class, system_data):
    """Verifica se as variantes RLS convergem para um Erro Quadrático Médio baixo."""
    x, d = system_data["x"], system_data["d_ideal"]
    
    model = model_class(filter_order=system_data["order"], lambda_factor=LAMBDA, epsilon=EPSILON)
    res = model.optimize(x, d)
    
    # Cálculo do erro médio nas últimas 100 amostras
    mse_final = np.mean(np.abs(res["errors"][-100:])**2)
    assert mse_final < 1e-4, f"Falha na convergência de {model_class.__name__}: MSE {mse_final}"

def test_nlrls_normalization_stability(system_data):
    """Teste específico para a versão normalizada com sinais de alta amplitude."""
    x_high = system_data["x"] * 100
    d_high = system_data["d_ideal"] * 100
    
    model = NormalizedLRLS(filter_order=system_data["order"], epsilon=EPSILON)
    res = model.optimize(x_high, d_high)
    
    # O NLRLS deve ser estável mesmo com ganhos altos
    assert np.all(np.isfinite(res["errors"]))

def test_lrls_error_feedback_numeric_integrity(system_data):
    """Garante que o Error Feedback mantém as energias xi_f e xi_b positivas."""
    model = LRLSErrorFeedback(filter_order=system_data["order"])
    model.optimize(system_data["x"], system_data["d_ideal"])
    
    # Energias de predição forward e backward não podem ser negativas ou nulas
    assert np.all(model.xi_f > 0)
    assert np.all(model.xi_b > 0)

def test_compare_priori_vs_posteriori(system_data):
    """Valida que A Priori e A Posteriori tendem ao mesmo resultado em regime permanente."""
    # Usamos o sinal completo (5000 amostras) para garantir convergência total com lambda=0.999
    x, d = system_data["x"], system_data["d_ideal"]
    
    m_post = LRLSPosteriori(system_data["order"], lambda_factor=LAMBDA, epsilon=EPSILON)
    m_prio = LRLSPriori(system_data["order"], lambda_factor=LAMBDA, epsilon=EPSILON)
    
    res_post = m_post.optimize(x, d)
    res_prio = m_prio.optimize(x, d)
    
    # Compara as últimas 100 amostras do regime permanente
    np.testing.assert_allclose(
        res_post["outputs"][-100:], 
        res_prio["outputs"][-100:], 
        atol=2e-3
    )

def test_complex_signal_support():
    """Verifica suporte a sinais complexos (comum em comunicações/radar)."""
    n = 200
    x_c = (np.random.randn(n) + 1j*np.random.randn(n))
    # Sistema simples: d[k] = x[k] * (0.5 + 0.5j)
    d_c = x_c * (0.5 + 0.5j)
    
    model = LRLSPosteriori(filter_order=1)
    res = model.optimize(x_c, d_c)
    
    assert np.iscomplexobj(res["outputs"])
    assert np.mean(np.abs(res["errors"][-50:])**2) < 1e-5