import numpy as np 
from .typing import ArrayLike
__all__ = ["wgn_real", "wgn_complex"]


def wgn_real(rng: np.random.Generator, shape, sigma_n2: float) -> ArrayLike:
    return rng.normal(0.0, np.sqrt(sigma_n2), size=shape).astype(float)


def wgn_complex(rng: np.random.Generator, shape, sigma_n2: float) -> ArrayLike:
    return (
        rng.normal(0.0, np.sqrt(sigma_n2 / 2.0), size=shape)
        + 1j * rng.normal(0.0, np.sqrt(sigma_n2 / 2.0), size=shape)
    ).astype(np.complex128)