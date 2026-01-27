import numpy as np
from pydaptivefiltering import LMS
from pydaptivefiltering import LMSNewton
from pydaptivefiltering import TDomainDFT
from pydaptivefiltering import TDomainDCT
from pydaptivefiltering import Power2ErrorLMS

def test_lms_newton_convergence(correlated_data, calculate_msd):
    """Verifica a convergência do LMS-Newton em sinais correlacionados."""
    x, d, h_true, order = correlated_data
    # Matriz identidade escalada para o chute inicial da inversa
    init_inv = 10.0 * np.eye(order + 1)
    
    model = LMSNewton(filter_order=order, alpha=0.95, initial_inv_rx=init_inv, step=0.1)
    model.optimize(x, d)
    
    # O MSD deve ser baixo após processar o sinal completo
    assert calculate_msd(h_true, model.w) < 1e-2

def test_newton_superiority_on_colored_noise(correlated_data, calculate_msd):
    """LMS-Newton deve convergir mais rápido que o LMS padrão (medido em 500 amostras)."""
    x, d, h_true, order = correlated_data
    init_inv = np.eye(order + 1)
    
    # Avaliamos apenas as primeiras 500 amostras para ver quem converge mais rápido
    x_short, d_short = x[:500], d[:500]
    
    model_lms = LMS(filter_order=order, step=0.01)
    model_lms.optimize(x_short, d_short)
    msd_lms = calculate_msd(h_true, model_lms.w)
    
    # LMS-Newton com alpha=0.9 para adaptação rápida da matriz de correlação
    model_newton = LMSNewton(filter_order=order, initial_inv_rx=init_inv, step=0.05, alpha=0.9)
    model_newton.optimize(x_short, d_short)
    msd_newton = calculate_msd(h_true, model_newton.w)
    
    # Newton deve descorrelacionar a entrada e atingir erro menor que o LMS puro
    assert msd_newton < msd_lms

def test_tdomain_dft_vs_dct(correlated_data, calculate_msd):
    """Compara algoritmos de domínio transformado (DFT vs DCT)."""
    x, d, h_true, order = correlated_data
    
    # Parâmetros conforme o __init__ da sua classe TDomain_DFT
    model_dft = TDomainDFT(order, gamma=1e-4, alpha=0.1, initial_power=1.0, step=0.1)
    model_dct = TDomainDCT(order, gamma=1e-4, alpha=0.1, initial_power=1.0, step=0.1)
    
    model_dft.optimize(x, d)
    model_dct.optimize(x, d)
    
    assert calculate_msd(h_true, model_dft.w) < 5e-3
    assert calculate_msd(h_true, model_dct.w) < 5e-3

def test_tdomain_power_normalization(correlated_data):
    """Valida a estabilidade do vetor de potência (power_vector)."""
    x, d, _, order = correlated_data
    model = TDomainDFT(order, gamma=1e-4, alpha=0.1, initial_power=1.0, step=0.1)
    model.optimize(x[:100], d[:100])
    
    # O atributo na sua classe é 'power_vector'
    assert hasattr(model, 'power_vector'), "Atributo 'power_vector' não encontrado."
    assert np.all(model.power_vector > 0), "Existem bins com potência negativa ou zero."
    assert model.power_vector.shape == (order + 1,)

def test_power2_error_logic(correlated_data, calculate_msd):
    """Verifica se o Power2_Error funciona corretamente com sinais reais."""
    x, d, h_true, order = correlated_data
    x_real, d_real = np.real(x).astype(float), np.real(d).astype(float)
    
    model = Power2ErrorLMS(filter_order=order, bd=8, tau=1e-3, step=0.01)
    model.optimize(x_real, d_real)
    
    # MSD para sinais reais
    assert calculate_msd(np.real(h_true), model.w) < 0.1

def test_newton_matrix_stability(correlated_data):
    """Garante que a matriz inversa de correlação (inv_rx) permanece definida positiva."""
    x, d, _, order = correlated_data
    init_inv = np.eye(order + 1)
    
    model = LMSNewton(filter_order=order, alpha=0.9, initial_inv_rx=init_inv)
    model.optimize(x[:100], d[:100])
    
    # Verificação de autovalores para garantir que é definida positiva
    eigenvalues = np.linalg.eigvals(model.inv_rx)
    assert np.all(eigenvalues.real > 0)

def test_newton_tracking_performance(system_data, calculate_msd):
    """Testa o rastreamento (tracking) de mudança abrupta usando LMS-Newton."""
    x = system_data["x"]
    w_opt = system_data["w_optimal"]
    order = system_data["order"]
    init_inv = np.eye(order + 1)
    
    # O sistema inverte os coeficientes na metade das amostras
    d_part1 = np.convolve(x[:500], w_opt, mode='full')[:500]
    d_part2 = np.convolve(x[500:1000], -w_opt, mode='full')[:500]
    d_total = np.concatenate([d_part1, d_part2])
    
    model = LMSNewton(filter_order=order, alpha=0.95, initial_inv_rx=init_inv, step=0.1)
    model.optimize(x[:1000], d_total)
    
    # Deve convergir para o novo estado (-w_opt)
    assert calculate_msd(-w_opt, model.w) < 1e-2