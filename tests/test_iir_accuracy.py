# test_iir_accuracy.py
import pytest
import numpy as np
from scipy import signal
from pydaptivefiltering.IIR_Filters import (
    ErrorEquation, 
    GaussNewton, 
    GaussNewton_GradientBased, 
    RLS_IIR, 
    Steiglitz_McBride
)

def generate_iir_system(n_samples, b_true, a_true, noise_std=0.01):
    """Gera dados de um sistema IIR: d = (B/A)x + ruído."""
    np.random.seed(42)
    x = np.random.randn(n_samples)
    # d[k] = b0*x[k] + b1*x[k-1] + a1*d[k-1]...
    # Nota: No livro do Diniz, d[k] = theta.T * regressor
    # onde regressor = [d(k-1)...d(k-N), x(k)...x(k-M)]
    d_clean = signal.lfilter(b_true, a_true, x)
    d = d_clean + noise_std * np.random.randn(n_samples)
    return x, d

@pytest.fixture
def target_system():
    """Define um sistema IIR simples de 1ª ordem para teste de convergência."""
    # H(z) = 0.5 / (1 - 0.2z^-1) -> b=[0.5], a=[1, -0.2]
    # No formato do vetor theta: [a1, b0] -> [0.2, 0.5]
    return {
        "b": np.array([0.5]),
        "a": np.array([1.0, -0.2]),
        "theta_true": np.array([0.2, 0.5]), 
        "M": 0,
        "N": 1
    }

def test_rls_iir_parameter_accuracy(target_system):
    """Verifica se o RLS_IIR consegue recuperar os coeficientes exatos do sistema."""
    x, d = generate_iir_system(2000, target_system["b"], target_system["a"], noise_std=0.0)
    
    flt = RLS_IIR(M=target_system["M"], N=target_system["N"], lambda_hat=1.0, delta=10.0)
    result = flt.optimize(x, d)
    
    w_final = result["coefficients"][-1]
    # MSD (Mean Square Deviation) entre os coeficientes
    msd = np.sum((w_final - target_system["theta_true"])**2)
    
    # RLS deve ser extremamente preciso com ruído zero
    assert msd < 1e-5

def test_gauss_newton_convergence_speed(target_system):
    """Compara a velocidade de convergência do Gauss-Newton vs Gradient-Based."""
    x, d = generate_iir_system(1000, target_system["b"], target_system["a"], noise_std=0.01)
    
    # Gauss-Newton Full (mais rápido por usar matriz de correlação)
    gn_full = GaussNewton(M=0, N=1, alpha=0.05, step=0.2, delta=1.0)
    # Gauss-Newton Gradient (mais lento, similar ao LMS)
    gn_grad = GaussNewton_GradientBased(M=0, N=1, step=0.01)
    
    res_full = gn_full.optimize(x, d)
    res_grad = gn_grad.optimize(x, d)
    
    # MSE na metade do caminho (amostra 300)
    mse_full = np.mean(res_full["errors"][250:300]**2)
    mse_grad = np.mean(res_grad["errors"][250:300]**2)
    
    # GN Full deve convergir muito mais rápido
    assert mse_full < mse_grad

def test_steiglitz_mcbride_bias_reduction():
    """
    Testa a capacidade do Steiglitz-McBride de lidar com erro de saída.
    Este algoritmo é famoso por ter menos viés que o Error Equation em IIR.
    """
    # Sistema de 2ª ordem mais complexo
    b_true = [1.0, 0.5]
    a_true = [1.0, -0.8, 0.2]
    # theta_true = [0.8, -0.2, 1.0, 0.5] (N=2, M=1)
    
    x, d = generate_iir_system(3000, b_true, a_true, noise_std=0.05)
    
    sm = Steiglitz_McBride(M=1, N=2, step=0.005)
    result = sm.optimize(x, d)
    
    # Verifica se o erro final é comparável ao nível do ruído injetado (0.05**2 = 0.0025)
    final_mse = np.mean(result["errors"][-200:]**2)
    assert final_mse < 0.01 

@pytest.mark.parametrize("filter_class", [RLS_IIR, ErrorEquation, GaussNewton])
def test_iir_transfer_function_match(filter_class, target_system):
    """
    Teste de sanidade: Verifica se a resposta em frequência do filtro 
    estimado se aproxima da do sistema original.
    """
    x, d = generate_iir_system(2000, target_system["b"], target_system["a"])
    
    params = {"M": 0, "N": 1}
    if filter_class == GaussNewton:
        params.update({"alpha": 0.1, "step": 0.05, "delta": 1.0})
    elif filter_class == RLS_IIR or filter_class == ErrorEquation:
        params.update({"lambda_hat": 0.999, "delta": 10})
        
    flt = filter_class(**params)
    result = flt.optimize(x, d)
    
    w_est = result["coefficients"][-1]
    # Reconstrói H(z) a partir de w_est = [a1, b0]
    # H_est = b0 / (1 - a1*z^-1)
    a_est = [1.0, -w_est[0]]
    b_est = [w_est[1]]
    
    # Compara a resposta ao degrau
    _, y_true = signal.step((target_system["b"], target_system["a"]))
    _, y_est = signal.step((b_est, a_est))
    
    # A diferença entre as respostas temporais deve ser pequena
    dist = np.linalg.norm(y_true[:20] - y_est[:20])
    assert dist < 0.5