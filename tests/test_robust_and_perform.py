import warnings
import numpy as np
from pydaptivefiltering import LMS, NLMS, AffineProjection, SignError

def test_nlms_zero_input_regularization(system_data):
    """Garante que o NLMS não sofre divisão por zero com entrada nula devido ao gamma."""
    order = system_data["order"]
    x_zero = np.zeros(500)
    d_zero = np.zeros(500)
    
    # O gamma deve impedir a divergência por divisão por zero
    filt = NLMS(filter_order=order, step=0.1, gamma=1e-8)
    filt.optimize(x_zero, d_zero)
    
    assert not np.any(np.isnan(filt.w)), "NLMS gerou NaN com entrada zero (falha na regularização)"
    assert not np.any(np.isinf(filt.w)), "NLMS gerou Inf com entrada zero (falha na regularização)"

def test_affine_projection_decorrelation_advantage(correlated_data, calculate_msd):
    """Valida se o APA (L>1) converge mais rápido que o NLMS (L=1) para sinais coloridos."""
    x, d, h_true, order = correlated_data
    
    # Usamos apenas o início do sinal para medir velocidade de convergência
    x_short, d_short = x[:400], d[:400]
    
    # APA com L=1 é matematicamente equivalente ao NLMS
    apa_l1 = AffineProjection(filter_order=order, step=0.2, L=1, gamma=1e-3)
    apa_l1.optimize(x_short, d_short)
    msd_l1 = calculate_msd(h_true, apa_l1.w)
    
    # APA com L=4 deve lidar melhor com a correlação do sinal
    apa_l4 = AffineProjection(filter_order=order, step=0.2, L=4, gamma=1e-3)
    apa_l4.optimize(x_short, d_short)
    msd_l4 = calculate_msd(h_true, apa_l4.w)
    
    # Em sinais coloridos, o erro do APA com maior ordem de projeção deve ser menor
    assert msd_l4 < msd_l1



def test_sign_error_robustness_to_impulsive_noise(system_data, calculate_msd):
    """Verifica se o Sign-Error é mais robusto (sofre menos desvio) que o LMS sob outliers."""
    x, d = np.real(system_data["x"]), np.real(system_data["d_ideal"])
    h_true = np.real(system_data["w_optimal"])
    
    # Vamos usar poucas amostras para capturar a sensibilidade ao outlier
    # e colocar o outlier bem próximo do fim do teste
    n_test = 500
    x_short, d_short = x[:n_test], d[:n_test]
    
    d_corrupted = d_short.copy()
    # Inserindo um outlier massivo na amostra 450
    d_corrupted[450] += 100.0 
    
    filt_lms = LMS(filter_order=system_data["order"], step=0.01)
    filt_sign = SignError(filter_order=system_data["order"], step=0.01)
    
    filt_lms.optimize(x_short, d_corrupted)
    filt_sign.optimize(x_short, d_corrupted)
    
    msd_lms = calculate_msd(h_true, filt_lms.w)
    msd_sign = calculate_msd(h_true, filt_sign.w)
    
    # O outlier de 100.0 multiplicado pelo step 0.01 fará o LMS deslocar seus coeficientes em 1.0
    # Já o Sign-Error deslocará apenas 0.01 (step * sign(e)), sendo muito mais robusto.
    assert msd_sign < msd_lms

def test_lms_stability_limit(system_data):
    """Verifica se o filtro diverge conforme esperado sem poluir o console."""
    filt = LMS(filter_order=system_data["order"], step=100.0)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Silencia o overflow esperado
        filt.optimize(system_data["x"], system_data["d_ideal"])
    
    assert np.any(np.isnan(filt.w)) or np.any(np.isinf(filt.w))

def test_algorithm_invariance_to_scaling(system_data, calculate_msd):
    """NLMS deve ser invariante à escala do sinal de entrada, ao contrário do LMS."""
    x, d = system_data["x"], system_data["d_ideal"]
    h_true, order = system_data["w_optimal"], system_data["order"]
    
    # Sinal escalado por 10
    x_scaled = x * 10
    d_scaled = d * 10
    
    # NLMS deve convergir com o mesmo step
    filt_nlms = NLMS(filter_order=order, step=0.5)
    filt_nlms.optimize(x_scaled, d_scaled)
    
    assert calculate_msd(h_true, filt_nlms.w) < 1e-3