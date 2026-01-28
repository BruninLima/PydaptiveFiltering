import pytest
from pydaptivefiltering import SMNLMS, SMAffineProjection, SMBNLMS, SimplifiedSMAP

def test_sm_nlms_basic(system_data):
    """Valida a estrutura básica e ocorrência de atualizações no SM-NLMS."""
    x, d = system_data["x"], system_data["d_ideal"]
    order = system_data["order"]
    
    filt = SMNLMS(filter_order=order, gamma_bar=0.1, gamma=1e-6)
    res = filt.optimize(x, d)
    
    assert filt.w.shape == (order + 1,)
    assert res['n_updates'] > 0

def test_sm_bnlms_accuracy(system_data, calculate_msd):
    """Valida a precisão do SM-BNLMS (Binormalizado)."""
    x, d = system_data["x"], system_data["d_ideal"]
    w_true = system_data["w_optimal"]
    
    filt = SMBNLMS(filter_order=system_data["order"], gamma_bar=0.15, gamma=1e-6)
    filt.optimize(x, d)
    
    # Usando calculate_msd do conftest em vez de calcular MSE manual com numpy
    assert calculate_msd(w_true, filt.w) < 0.05

def test_sm_ap_structure():
    """Verifica se a matriz de dados interna do SM-AP tem as dimensões corretas."""
    L = 2
    order = 4
    # Note: gamma_bar_vector pode ser passado como lista, evitando import do numpy
    filt = SMAffineProjection(filter_order=order, gamma_bar=0.1, gamma_bar_vector=[0.0, 0.0, 0.0], L=L, gamma=1e-3)
    
    # X_matrix armazena L+1 vetores regressores de tamanho M+1
    assert filt.X_matrix.shape == (order + 1, L + 1)



def test_sm_simp_ap_performance(system_data, calculate_msd):
    """Testa a convergência da versão simplificada do SM-AP."""
    x, d = system_data["x"], system_data["d_ideal"]
    w_true = system_data["w_optimal"]
    
    filt = SimplifiedSMAP(filter_order=system_data["order"], gamma_bar=0.1, L=2, gamma=1e-3)
    filt.optimize(x, d)
    
    assert calculate_msd(w_true, filt.w) < 0.05

def test_sm_zero_update_logic():
    """
    Verifica se o filtro economiza processamento (zero updates) 
    quando o erro está dentro do limite (gamma_bar).
    """
    # Sinais muito pequenos para garantir que o erro fique abaixo do gamma_bar
    x = [0.001] * 100
    d = [0.001] * 100
    
    filt = SimplifiedSMAP(filter_order=2, gamma_bar=1.0, L=1, gamma=1e-3)
    res = filt.optimize(x, d)
    
    # O coração do Set-Membership: se o erro é pequeno, não gasta energia atualizando
    assert res['n_updates'] == 0

def test_sm_mismatched_dimensions():
    """Garante que o erro de dimensão entre entrada e desejado seja disparado."""
    filt = SMBNLMS(filter_order=3, gamma_bar=0.1, gamma=1e-6)
    
    # Necessário importar pytest apenas para este teste de exceção
    with pytest.raises(ValueError):
        filt.optimize([0.0] * 10, [0.0] * 9)