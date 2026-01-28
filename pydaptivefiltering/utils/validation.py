# .utils.validation.py

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional, Tuple

import numpy as np


def _extract_signals(args: tuple, kwargs: dict) -> Tuple[Optional[Any], Optional[Any]]:
    """Extract (x, d) from args/kwargs supporting both naming conventions.

    Supported:
      - positional: (x, d, ...)
      - named: input_signal / desired_signal
      - named legacy: x / d
    """
    x = None
    d = None

    # positional
    if len(args) >= 1:
        x = args[0]
    if len(args) >= 2:
        d = args[1]

    # named preferred
    if "input_signal" in kwargs:
        x = kwargs["input_signal"]
    if "desired_signal" in kwargs:
        d = kwargs["desired_signal"]

    # named legacy
    if "x" in kwargs:
        x = kwargs["x"]
    if "d" in kwargs:
        d = kwargs["d"]

    return x, d


def ensure_real_signals(func: Callable[..., Any]) -> Callable[..., Any]:
    """Ensure input signals are real-valued.

    Works for both calling conventions:
      optimize(input_signal=..., desired_signal=...)
      optimize(x=..., d=...)
      optimize(x, d)

    Raises
    ------
    TypeError:
        If complex data is detected.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        x, d = _extract_signals(args, kwargs)

        if x is None:
            raise TypeError(f"{self.__class__.__name__}: missing input signal (input_signal/x).")

        if np.iscomplexobj(x):
            raise TypeError(
                f"{self.__class__.__name__} does not support complex inputs for input_signal/x."
            )

        if d is not None and np.iscomplexobj(d):
            raise TypeError(
                f"{self.__class__.__name__} does not support complex inputs for desired_signal/d."
            )

        return func(self, *args, **kwargs)

    return wrapper


def ensure_complex_signals(func: Callable[..., Any]) -> Callable[..., Any]:
    """Ensure input signals are complex-valued (casts to complex if needed).

    Useful when an algorithm is defined for complex data and you want consistent dtype.

    Notes
    -----
    - This decorator does NOT reshape/flatten; `validate_input` in base.py does that.
    - It mainly ensures dtype is complex.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        x, d = _extract_signals(args, kwargs)

        if x is None:
            raise TypeError(f"{self.__class__.__name__}: missing input signal (input_signal/x).")

        # If user passed real, we allow it but cast to complex for consistency.
        # We'll reconstruct kwargs if needed; otherwise we just call and let validate_input cast.
        if not np.iscomplexobj(np.asarray(x)):
            # try to update kwargs without breaking positional usage
            if "input_signal" in kwargs:
                kwargs["input_signal"] = np.asarray(x, dtype=complex)
            elif "x" in kwargs:
                kwargs["x"] = np.asarray(x, dtype=complex)

        if d is not None and not np.iscomplexobj(np.asarray(d)):
            if "desired_signal" in kwargs:
                kwargs["desired_signal"] = np.asarray(d, dtype=complex)
            elif "d" in kwargs:
                kwargs["d"] = np.asarray(d, dtype=complex)

        return func(self, *args, **kwargs)

    return wrapper


def ensure_same_length(func: Callable[..., Any]) -> Callable[..., Any]:
    """Ensure x and d have the same number of samples (1D length).

    This is mostly redundant if you're already using @validate_input,
    but can be used for functions that don't use it.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        x, d = _extract_signals(args, kwargs)

        if x is None:
            raise TypeError(f"{self.__class__.__name__}: missing input signal (input_signal/x).")
        if d is None:
            return func(self, *args, **kwargs)

        x_arr = np.ravel(np.asarray(x))
        d_arr = np.ravel(np.asarray(d))
        if x_arr.shape[0] != d_arr.shape[0]:
            raise ValueError(
                f"Tamanhos inconsistentes: input({x_arr.shape[0]}) != desired({d_arr.shape[0]})"
            )

        return func(self, *args, **kwargs)

    return wrapper
#EOF