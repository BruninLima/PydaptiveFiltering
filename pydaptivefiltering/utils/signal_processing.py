import numpy as np

def coefficients_to_ba(w, M, N):
    """
    Converte o vetor de pesos adaptativos theta para os vetores b, a do SciPy.
    theta = [a1, a2, ..., aN, b0, b1, ..., bM]
    H(z) = (b0 + b1*z^-1) / (1 - a1*z^-1 - a2*z^-2)
    """
    a = np.concatenate(([1.0], -w[:N]))
    b = w[N:]
    return b, a

def check_stability(w, N):
    """
    Verifica a estabilidade BIBO do filtro IIR.
    Retorna True se estável, False se instável.
    """
    if N == 0: return True 
    _, a = coefficients_to_ba(w, 0, N) 
    poles = np.roots(a)
    return np.all(np.abs(poles) < 1.0)

def project_stability(w, N, margin=0.98):
    """
    Se o filtro for instável, projeta os polos para dentro do círculo unitário.
    Útil para algoritmos como Steiglitz-McBride e Gauss-Newton.
    """
    b_part = w[N:]
    a_coeffs = w[:N]
    
    a_poly = np.concatenate(([1.0], -a_coeffs))
    poles = np.roots(a_poly)
    
    for i in range(len(poles)):
        if np.abs(poles[i]) >= 1.0:
            poles[i] = (poles[i] / np.abs(poles[i])) * margin
            
    new_a_poly = np.poly(poles)
    new_a_coeffs = -new_a_poly[1:]
    
    return np.concatenate((new_a_coeffs, b_part))