import numpy as np
from .typing import ArrayLike

__all__ = ["toeplitz", "make_channel_matrix"]


def toeplitz(c: ArrayLike, r: ArrayLike) -> np.ndarray:
    """Minimal Toeplitz (no SciPy)."""
    c, r = np.asarray(c), np.asarray(r)
    m, n = c.size, r.size
    out = np.empty((m, n), dtype=np.result_type(c, r))
    for i in range(m):
        for j in range(n):
            k = j - i
            out[i, j] = r[k] if k >= 0 else c[-k]
    return out

def make_channel_matrix(H: ArrayLike, N: int) -> ArrayLike:
    """Toeplitz([H0, 0..], [H, 0..])."""
    H = np.asarray(H, dtype=np.complex128).ravel()
    first_col = np.concatenate(([H[0]], np.zeros(N - 1, dtype=np.complex128)))
    first_row = np.concatenate((H, np.zeros(N - 1, dtype=np.complex128)))
    return toeplitz(first_col, first_row)


def coefficients_to_ba(w, N):
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

def dft_matrix(M: int) -> np.ndarray:
    """
    Equivalent to MATLAB dftmtx(M):

        F[m,n] = exp(-j*2*pi*m*n/M), for m,n = 0..M-1
    """
    m = np.arange(M, dtype=float)
    n = np.arange(M, dtype=float)
    return np.exp(-1j * 2.0 * np.pi * np.outer(m, n) / float(M))
