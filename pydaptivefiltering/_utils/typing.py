# ._utils.typing.py
from __future__ import annotations
from typing import Union, Sequence
import numpy as np

ArrayLike = Union[np.ndarray, Sequence[Union[int, float, complex]]]

RealArrayLike = Union[np.ndarray, Sequence[Union[int, float]]]
ComplexArrayLike = Union[np.ndarray, Sequence[Union[int,float,complex]]]

__all__ = ["ArrayLike", "RealArrayLike", "ComplexArrayLike"]
