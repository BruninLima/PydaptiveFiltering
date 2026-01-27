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

def test_newton_scale_invariance(system_data_real, calculate_msd):
    """
    LMS-Newton deve convergir mesmo se escalarmos o sinal de entrada,
    desde que a matriz inversa compense o ganho.
    """
    x = system_data_real["x"]
    d = system_data_real["d_ideal"]
    order = system_data_real["order"]
    
    # Escala o sinal por 10
    x_scaled = x * 10.0
    d_scaled = d * 10.0
    
    init_inv = np.eye(order + 1)
    model = LMSNewton(order, initial_inv_rx=init_inv, step=0.1, alpha=0.95)
    model.optimize(x_scaled, d_scaled)
    
    # O vetor de pesos w deve ser o mesmo, independente da escala do sinal
    # (Pois d_ideal também foi escalado proporcionalmente)
    assert calculate_msd(system_data_real["w_optimal"], model.w) < 1e-2

def test_algorithms_robustness_to_zero_input(system_data):
    """Verifica se os algoritmos evitam divisão por zero usando regularização."""
    order = system_data["order"]
    x_zeros = np.zeros(100, dtype=complex)
    d_zeros = np.zeros(100, dtype=complex)

    # CORREÇÃO: Adicionados alpha e initial_power exigidos pelo seu código
    model_td = TDomainDFT(
        filter_order=order, 
        gamma=1e-6, 
        alpha=0.1, 
        initial_power=1.0, 
        step=0.1
    )
    model_td.optimize(x_zeros, d_zeros)

    init_inv = np.eye(order + 1)
    model_newton = LMSNewton(order, initial_inv_rx=init_inv, alpha=0.9)
    model_newton.optimize(x_zeros, d_zeros)

    assert not np.isnan(model_td.w).any(), "TDomainDFT falhou com entrada zero (NaN detected)"
    assert not np.isnan(model_newton.w).any(), "LMSNewton falhou com entrada zero (NaN detected)"

def test_tdomain_convergence_speed_colored_noise(correlated_data_real, calculate_msd):
    """Em ruído colorido, o TDomainDFT deve superar o LMS padrão."""
    x, d, h_true, order = correlated_data_real

    lms = LMS(order, step=0.01)
    lms.optimize(x[:1000], d[:1000])

    # CORREÇÃO: Adicionados gamma e initial_power exigidos pelo seu código
    td_dft = TDomainDFT(
        filter_order=order, 
        gamma=1e-4, 
        alpha=0.1, 
        initial_power=1.0, 
        step=0.1
    )
    td_dft.optimize(x[:1000], d[:1000])

    msd_lms = calculate_msd(h_true, lms.w)
    msd_dft = calculate_msd(h_true, td_dft.w)

    assert msd_dft < msd_lms