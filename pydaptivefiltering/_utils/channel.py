import numpy as np 
from typing import Tuple
from .typing import ArrayLike
from .signal import apply_channel
from .linalg import make_channel_matrix
from .noise import wgn_complex

__all__ = [
    "wiener_equalizer",
    "generate_channel_data",
]


def wiener_equalizer(H: ArrayLike, N: int, sigma_x2: float, sigma_n2: float, delay: int) -> ArrayLike:
    """
    Matches MATLAB:
      HMatrix = toeplitz([H(1) zeros(1,N-1)],[H zeros(1,N-1)]);
      Rx = sigma_x2*eye(N+len(H)-1);
      Rn = sigma_n2*eye(N);
      Ry = HMatrix*Rx*HMatrix' + Rn;
      RxDeltaY = [zeros(1,delay) sigma_x2 zeros(1,N+len(H)-2-delay)]*(HMatrix');
      Wiener = (RxDeltaY*inv(Ry)).';
    Returns Wiener as shape (N,1).
    """
    HMatrix = make_channel_matrix(H, N)
    Rx = sigma_x2 * np.eye(N + len(H) - 1, dtype=np.complex128)
    Rn = sigma_n2 * np.eye(N, dtype=np.complex128)
    Ry = HMatrix @ Rx @ np.conj(HMatrix.T) + Rn

    rx_deltay_row = np.concatenate(
        (
            np.zeros(delay, dtype=np.complex128),
            np.array([sigma_x2], dtype=np.complex128),
            np.zeros(N + len(H) - 2 - delay, dtype=np.complex128),
        )
    )
    RxDeltaY = rx_deltay_row @ np.conj(HMatrix.T)  # (N,)
    return (RxDeltaY @ np.linalg.inv(Ry)).reshape(-1, 1)


def generate_channel_data(
    rng: np.random.Generator,
    K: int,
    H: ArrayLike,
    sigma_n2: float,
    constellation: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Generate s, x = channel(s)+noise, n (all length K)."""
    s = rng.choice(constellation, size=K)
    n = wgn_complex(rng, (K,), sigma_n2)
    x = apply_channel(s, H, K) + n
    return s, x, n
