# ._utils.shapes.py
from __future__ import annotations
from typing import Sequence, Union
import numpy as np

__all__ = ["as_2d_col", "as_meas_matrix", "mat_at_k"]

def as_2d_col(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    if x.ndim == 2 and x.shape[0] == 1:
        return x.reshape(-1, 1)
    raise ValueError(f"Expected a vector compatible with (n,1). Got shape={x.shape}.")

def as_meas_matrix(y_seq: np.ndarray) -> np.ndarray:
    y_seq = np.asarray(y_seq)
    if y_seq.ndim == 1:
        return y_seq.reshape(-1, 1)
    if y_seq.ndim == 2:
        return y_seq
    if y_seq.ndim == 3 and y_seq.shape[-1] == 1:
        return y_seq[..., 0]
    raise ValueError(f"input_signal must have shape (N,), (N,p) or (N,p,1). Got {y_seq.shape}.")

def mat_at_k(mat_or_seq: Union[np.ndarray, Sequence[np.ndarray]], k: int) -> np.ndarray:
    if isinstance(mat_or_seq, (list, tuple)):
        return np.asarray(mat_or_seq[k])
    return np.asarray(mat_or_seq)
