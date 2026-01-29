import numpy as np 
from typing import Tuple
from .typing import ArrayLike
from .linalg import make_channel_matrix
from .noise import wgn_complex

__all__ = [
    "qam4_constellation_unit_var",
    "simulate_constellations",
    "best_phase_rotation",
    "hard_decision_qam4",
    "ser_qam4",
    "evm_mse",
    "cm_cost",
]

def qam4_constellation_unit_var() -> ArrayLike:
    """MATLAB: qammod(0:3,4)/sqrt(2) => {±1±j}/sqrt(2), unit average power."""
    return np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128) / np.sqrt(2.0)

def simulate_constellations(
    rng: np.random.Generator,
    H: np.ndarray,
    N: int,
    Ksim: int,
    sigma_n2: float,
    constellation: np.ndarray,
    w_final: np.ndarray,
    w_wiener: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      equalizerInputMatrix (N,Ksim),
      equalizerOutputVector (Ksim,),
      equalizerOutputVectorWiener (Ksim,)
    """
    HMatrix = make_channel_matrix(H, N)
    inputMatrix = rng.choice(constellation, size=(N + len(H) - 1, Ksim))
    noiseMatrix = wgn_complex(rng, (N, Ksim), sigma_n2)
    equalizerInputMatrix = HMatrix @ inputMatrix + noiseMatrix

    w_final_v = np.asarray(w_final, dtype=np.complex128).reshape(-1)
    w_wiener_v = np.asarray(w_wiener, dtype=np.complex128).reshape(-1)

    y_eq = (np.conj(w_final_v) @ equalizerInputMatrix).ravel()
    y_w = (np.conj(w_wiener_v) @ equalizerInputMatrix).ravel()
    return equalizerInputMatrix, y_eq, y_w


def best_phase_rotation(y: np.ndarray, s_ref: np.ndarray) -> Tuple[np.ndarray, float]:
    """Find phi minimizing || y*exp(-j phi) - s_ref || via LS phase."""
    phi = float(np.angle(np.vdot(s_ref, y)))
    y_rot = y * np.exp(-1j * phi)
    return y_rot, phi

def hard_decision_qam4(y: np.ndarray) -> np.ndarray:
    """Hard-decision for QAM4/QPSK points {±1±j}/sqrt(2)."""
    re = np.where(np.real(y) >= 0, 1.0, -1.0)
    im = np.where(np.imag(y) >= 0, 1.0, -1.0)
    return (re + 1j * im) / np.sqrt(2.0)


def ser_qam4(y_hat: np.ndarray, s_ref: np.ndarray) -> float:
    return float(np.mean(y_hat != s_ref))


def evm_mse(y_rot: np.ndarray, s_ref: np.ndarray) -> float:
    """Mean squared error after phase alignment (often called EVM^2)."""
    return float(np.mean(np.abs(y_rot - s_ref) ** 2))


def cm_cost(y: np.ndarray, R: float = 1.0) -> float:
    """Constant-modulus cost (useful for CMA/AP-CM outputs)."""
    return float(np.mean((np.abs(y) ** 2 - R) ** 2))
