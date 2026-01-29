import numpy as np
from .typing import ArrayLike



def db10(x: ArrayLike, *, eps: float = 1e-20) -> np.ndarray:
    """10*log10(x) with numerical guard."""
    x = np.asarray(x, dtype=float)
    return 10.0 * np.log10(np.maximum(x, eps))

def safe_scalar_db10(x: float, *, eps: float) -> float:
    """
    Return scalar dB for a scalar linear value, robust to NaN/inf.
    Uses db10 under the hood.
    """
    if not np.isfinite(x):
        return float("nan")
    return float(db10(np.array([x], dtype=float), eps=eps)[0])

def db20(x: ArrayLike, *, eps: float = 1e-12) -> np.ndarray:
    """20*log10(|x|) with numerical guard."""
    x = np.asarray(x)
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))
