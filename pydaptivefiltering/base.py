# base.py

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union, Callable

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[complex]]


@dataclass
class OptimizationResult:
    """Standard output container for optimization/adaptation runs.

    Attributes
    ----------
    outputs:
        Estimated output signal y[k] produced by the adaptive filter.
    errors:
        Error signal (definition depends on `error_type`), typically e[k] = d[k] - y[k].
    coefficients:
        Coefficient history over time. Recommended shape: (N, n_coeffs).
    algorithm:
        Algorithm name (usually class name).
    runtime_ms:
        Runtime in milliseconds.
    error_type:
        Error semantics tag, e.g. "a_priori", "a_posteriori", "output_error".
    extra:
        Optional container for internal states / debug info.
    """

    outputs: np.ndarray
    errors: np.ndarray
    coefficients: np.ndarray
    algorithm: str
    runtime_ms: float
    error_type: str = "a_priori"
    extra: Optional[Dict[str, Any]] = None

    def mse(self) -> np.ndarray:
        """Instantaneous squared error (magnitude-squared for complex)."""
        return np.abs(self.errors) ** 2

    def __repr__(self) -> str:
        return f"<OptimizationResult algo={self.algorithm} samples={len(self.outputs)}>"


def validate_input(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to validate and normalize `optimize` inputs.

    Accepts all of the following calling styles:

    1) Standard, preferred:
        optimize(input_signal=..., desired_signal=..., **kwargs)
        optimize(input_signal, desired_signal, **kwargs)

    2) Legacy aliases:
        optimize(x=..., d=..., **kwargs)
        optimize(x, d, **kwargs)

    Notes
    -----
    - Signals are converted with `np.asarray` and flattened to 1D (ravel).
    - If either signal is complex, both are cast to complex; otherwise float.
    - Length mismatch raises ValueError.
    """
    sig = inspect.signature(method)
    param_names = set(sig.parameters.keys())

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        input_signal = None
        desired_signal = None

        if len(args) >= 1:
            input_signal = args[0]
        if len(args) >= 2:
            desired_signal = args[1]

        if "input_signal" in kwargs:
            input_signal = kwargs.pop("input_signal")
        if "desired_signal" in kwargs:
            desired_signal = kwargs.pop("desired_signal")

        if "x" in kwargs:
            input_signal = kwargs.pop("x")
        if "d" in kwargs:
            desired_signal = kwargs.pop("d")

        if input_signal is None:
            raise TypeError("Missing input signal: pass input_signal (or alias x).")

        x = np.asarray(input_signal)
        d = None if desired_signal is None else np.asarray(desired_signal)

        x = np.ravel(x)
        if d is not None:
            d = np.ravel(d)
            if x.shape[0] != d.shape[0]:
                raise ValueError(
                    f"Inconsistent lengths: input({x.shape[0]}) != desired({d.shape[0]})"
                )

        dtype = complex if np.iscomplexobj(x) or (d is not None and np.iscomplexobj(d)) else float
        x = x.astype(dtype, copy=False)
        if d is not None:
            d = d.astype(dtype, copy=False)

        if d is None:
            if "input_signal" in param_names:
                return method(self, input_signal=x, **kwargs)
            if "x" in param_names:
                return method(self, x=x, **kwargs)
            return method(self, x, **kwargs)

        if "input_signal" in param_names and "desired_signal" in param_names:
            return method(self, input_signal=x, desired_signal=d, **kwargs)
        if "x" in param_names and "d" in param_names:
            return method(self, x=x, d=d, **kwargs)

        return method(self, x, d, **kwargs)

    return wrapper


class AdaptiveFilter(ABC):
    """Abstract base class for all adaptive filters.

    Parameters
    ----------
    filter_order:
        Order in the FIR sense (number of taps - 1). For non-FIR structures, it can be used
        as a generic size indicator for base allocation.
    w_init:
        Initial coefficient vector. If None, initialized to zeros.

    Notes
    -----
    - Subclasses should set `supports_complex = True` if they support complex-valued data.
    - Subclasses are expected to call `_record_history()` every iteration (or use helper methods)
      if they want coefficient trajectories.
    """

    supports_complex: bool = False

    def __init__(self, filter_order: int, w_init: Optional[ArrayLike] = None) -> None:
        self.filter_order: int = int(filter_order)
        self._dtype = complex if self.supports_complex else float

        self.regressor: np.ndarray = np.zeros(self.filter_order + 1, dtype=self._dtype)

        if w_init is not None:
            self.w: np.ndarray = np.asarray(w_init, dtype=self._dtype)
        else:
            self.w = np.zeros(self.filter_order + 1, dtype=self._dtype)

        self.w_history: List[np.ndarray] = []
        self._record_history()

    def _record_history(self) -> None:
        """Store a snapshot of current coefficients."""
        self.w_history.append(np.asarray(self.w).copy())

    def _final_coeffs(self, coefficients: Any) -> Any:
        """Return last coefficients from a history container (list or 2D array)."""
        if coefficients is None:
            return None
        if isinstance(coefficients, list) and len(coefficients) > 0:
            return coefficients[-1]
        try:
            a = np.asarray(coefficients)
            if a.ndim == 2:
                return a[-1, :]
        except Exception:
            pass
        return coefficients

    def _pack_results(
        self,
        outputs: np.ndarray,
        errors: np.ndarray,
        runtime_s: float,
        error_type: str = "a_priori",
        extra: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Centralized output packaging to standardize results."""
        return OptimizationResult(
            outputs=np.asarray(outputs),
            errors=np.asarray(errors),
            coefficients=np.asarray(self.w_history),
            algorithm=self.__class__.__name__,
            runtime_ms=float(runtime_s) * 1000.0,
            error_type=str(error_type),
            extra=extra,
        )

    def filter_signal(self, input_signal: ArrayLike) -> np.ndarray:
        """Filter an input signal using current coefficients.

        Default implementation assumes an FIR structure with taps `self.w` and
        regressor convention:
            x_k = [x[k], x[k-1], ..., x[k-m]]
        and output:
            y[k] = w^H x_k   (Hermitian for complex)
        """
        x = np.asarray(input_signal, dtype=self._dtype)
        n_samples = x.size
        y = np.zeros(n_samples, dtype=self._dtype)

        x_padded = np.zeros(n_samples + self.filter_order, dtype=self._dtype)
        x_padded[self.filter_order:] = x

        for k in range(n_samples):
            x_k = x_padded[k : k + self.filter_order + 1][::-1]
            y[k] = np.dot(self.w.conj(), x_k)

        return y

    @classmethod
    def default_test_init_kwargs(cls, order: int) -> dict:
        """Override in subclasses to provide init kwargs for standardized tests."""
        return {}

    @abstractmethod
    def optimize(
        self,
        input_signal: ArrayLike,
        desired_signal: ArrayLike,
        **kwargs: Any,
    ) -> Any:
        """Run the adaptation procedure.

        Subclasses should return either:
        - OptimizationResult (recommended), or
        - dict-like with standardized keys, if you are migrating older code.
        """
        raise NotImplementedError

    def reset_filter(self, w_new: Optional[ArrayLike] = None) -> None:
        """Reset coefficients and history."""
        if w_new is not None:
            self.w = np.asarray(w_new, dtype=self._dtype)
        else:
            self.w = np.zeros(self.filter_order + 1, dtype=self._dtype)
        self.w_history = []
        self._record_history()


def display_library_info() -> None:
    """Displays information about pydaptivefiltering chapters."""
    chapters: Dict[str, str] = {
        "3 & 4": "LMS Algorithms",
        "5": "RLS Algorithms",
        "6": "Set-Membership Algorithms",
        "7": "Lattice-based RLS Algorithms",
        "8": "Fast Transversal RLS Algorithms",
        "9": "QR Decomposition Based RLS Algorithms",
        "10": "IIR Adaptive Filters",
        "11": "Nonlinear Adaptive Filters",
        "12": "Subband Adaptive Filters",
        "13": "Blind Adaptive Filtering",
    }

    print("\n--- Pydaptive Filtering (Based on Paulo S. R. Diniz) ---")
    print(f"{'Chapter':<10} | {'Algorithm Area'}")
    print("-" * 50)
    for ch, area in chapters.items():
        print(f"Chapter {ch:<3} | {area}")
    print("-" * 50)


if __name__ == "__main__":
    display_library_info()
#EOF