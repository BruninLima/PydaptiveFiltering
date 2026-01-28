#  subband.cfdlms.py
#
#       Implements the Constrained Frequency-Domain LMS (CFDLMS) algorithm
#       for REAL valued data.
#       (Algorithm 12.4 - book: Adaptive Filtering: Algorithms and Practical
#                                                        Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import time
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input
from pydaptivefiltering.utils.validation import ensure_real_signals


class CFDLMS(AdaptiveFilter):
    """
    Implements the Constrained Frequency-Domain LMS (CFDLMS) algorithm for real-valued data.
    (Algorithm 12.4, Diniz)

    Notes
    -----
    - This algorithm is block-based: each iteration produces L time-domain outputs.
    - Internally it uses complex FFT processing; outputs/errors returned are real.
    - Coefficients are a subband matrix ww with shape (M, Nw+1).
    - For compatibility with the base class, `self.w` stores a flattened view of `ww`.
      The returned OptimizationResult.coefficients still comes from `self.w_history`
      (flattened), and the full matrix history is provided in `extra["ww_history"]`.
    """
    supports_complex: bool = False

    M: int
    L: int
    Nw: int
    step: float
    gamma: float
    smoothing: float

    def __init__(
        self,
        filter_order: int = 5,
        n_subbands: int = 64,
        decimation: Optional[int] = None,
        step: float = 0.1,
        gamma: float = 1e-2,
        smoothing: float = 0.01,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        if n_subbands <= 0:
            raise ValueError("n_subbands (M) must be a positive integer.")
        if filter_order < 0:
            raise ValueError("filter_order (Nw) must be >= 0.")
        if decimation is None:
            decimation = n_subbands // 2
        if decimation <= 0 or decimation > n_subbands:
            raise ValueError("decimation (L) must satisfy 1 <= L <= M.")
        if gamma <= 0:
            raise ValueError("gamma must be > 0.")
        if not (0.0 < smoothing <= 1.0):
            raise ValueError("smoothing must be in (0, 1].")

        self.M = int(n_subbands)
        self.L = int(decimation)
        self.Nw = int(filter_order)

        self.step = float(step)
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)

        n_params = self.M * (self.Nw + 1)
        super().__init__(filter_order=n_params - 1, w_init=None)

        self.ww: np.ndarray = np.zeros((self.M, self.Nw + 1), dtype=np.complex128)
        if w_init is not None:
            w0 = np.asarray(w_init)
            if w0.ndim == 2 and w0.shape == (self.M, self.Nw + 1):
                self.ww = w0.astype(np.complex128, copy=True)
            else:
                w0 = w0.reshape(-1)
                if w0.size != n_params:
                    raise ValueError(
                        f"w_init has incompatible size. Expected {n_params} "
                        f"or shape ({self.M},{self.Nw+1}), got {w0.size}."
                    )
                self.ww = w0.reshape(self.M, self.Nw + 1).astype(np.complex128, copy=True)

        self.uu: np.ndarray = np.zeros((self.M, self.Nw + 1), dtype=np.complex128)
        self.sig: np.ndarray = np.zeros(self.M, dtype=np.float64)

        self.w = self.ww.reshape(-1).astype(float, copy=False)
        self.w_history = []
        self._record_history()

        self.ww_history: list[np.ndarray] = []

    def reset_filter(self, w_new: Optional[Union[np.ndarray, list]] = None) -> None:
        """
        Reset coefficients/history.

        If w_new is:
          - None: zeros
          - shape (M, Nw+1): used directly
          - flat of length M*(Nw+1): reshaped
        """
        n_params = self.M * (self.Nw + 1)

        if w_new is None:
            self.ww = np.zeros((self.M, self.Nw + 1), dtype=np.complex128)
        else:
            w0 = np.asarray(w_new)
            if w0.ndim == 2 and w0.shape == (self.M, self.Nw + 1):
                self.ww = w0.astype(np.complex128, copy=True)
            else:
                w0 = w0.reshape(-1)
                if w0.size != n_params:
                    raise ValueError(
                        f"w_new has incompatible size. Expected {n_params} "
                        f"or shape ({self.M},{self.Nw+1}), got {w0.size}."
                    )
                self.ww = w0.reshape(self.M, self.Nw + 1).astype(np.complex128, copy=True)

        self.uu = np.zeros((self.M, self.Nw + 1), dtype=np.complex128)
        self.sig = np.zeros(self.M, dtype=np.float64)

        self.ww_history = []
        self.w = self.ww.reshape(-1).astype(float, copy=False)
        self.w_history = []
        self._record_history()

    @ensure_real_signals
    @validate_input
    def optimize(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        verbose: bool = False,
        return_internal_states: bool = False,
    ) -> OptimizationResult:
        """
        Executes the CFDLMS weight update process.

        Parameters
        ----------
        input_signal:
            Input signal x[n] (real-valued).
        desired_signal:
            Desired signal d[n] (real-valued).
        verbose:
            If True, prints runtime.
        return_internal_states:
            If True, returns additional internal trajectories in result.extra.

        Returns
        -------
        OptimizationResult
            outputs:
                Estimated output signal (real), length = n_iters * L.
            errors:
                Output error signal (real), same length as outputs.
            coefficients:
                Flattened coefficient history (from base `w_history`).
            error_type:
                "output_error".

        Extra (always)
        -------------
        extra["ww_history"]:
            List of coefficient matrices ww over iterations; each entry has shape (M, Nw+1).
        extra["n_iters"]:
            Number of block iterations.

        Extra (when return_internal_states=True)
        --------------------------------------
        extra["sig"]:
            Final smoothed energy per bin (M,).
        extra["sig_history"]:
            Energy history per iteration (n_iters, M).
        """
        tic: float = time()

        x = np.asarray(input_signal, dtype=np.float64).ravel()
        d = np.asarray(desired_signal, dtype=np.float64).ravel()

        M = self.M
        L = self.L
        Nw = self.Nw

        max_iters_from_x = int(np.floor((x.size + L - M) / L) + 1) if (x.size + L) >= M else 0
        max_iters_from_d = int(d.size // L)
        n_iters = max(0, min(max_iters_from_x, max_iters_from_d))

        out_len = n_iters * L
        outputs = np.zeros(out_len, dtype=np.float64)
        errors  = np.zeros(out_len, dtype=np.float64)

        xpad = np.concatenate([np.zeros(L, dtype=np.float64), x])

        self.ww_history = []

        sig_hist: Optional[np.ndarray] = np.zeros((n_iters, M), dtype=np.float64) if return_internal_states else None

        uu = self.uu
        ww = self.ww
        sig = self.sig

        a = self.smoothing
        u_step = self.step
        gamma = self.gamma
        sqrtM = np.sqrt(M)

        for k in range(n_iters):
            start = k * L
            seg_x = xpad[start : start + M]

            x_p = seg_x[::-1].astype(np.complex128, copy=False)

            d_seg = d[start : start + L]
            d_p = d_seg[::-1].astype(np.complex128, copy=False)

            ui = np.fft.fft(x_p) / sqrtM

            uu[:, 1:] = uu[:, :-1]
            uu[:, 0] = ui

            uy = np.sum(uu * ww, axis=1)

            y_block = np.fft.ifft(uy) * sqrtM
            y_firstL = y_block[:L]

            e_rev = d_p - y_firstL

            y_time = np.real(y_firstL[::-1])
            e_time = d_seg - y_time

            outputs[start : start + L] = y_time
            errors[start : start + L] = e_time

            e_pad = np.concatenate([e_rev, np.zeros(M - L, dtype=np.complex128)])
            et = np.fft.fft(e_pad) / sqrtM
            sig[:] = (1.0 - a) * sig + a * (np.abs(ui) ** 2)

            denom = gamma + (Nw + 1) * sig
            gain = u_step / denom

            wwc = (gain[:, None] * np.conj(uu) * et[:, None]).astype(np.complex128, copy=False)

            waux = np.fft.fft(wwc, axis=0) / sqrtM
            waux[L:, :] = 0.0
            wwc_c = np.fft.ifft(waux, axis=0) * sqrtM

            ww = ww + wwc_c

            self.ww_history.append(ww.copy())

            self.w = np.real(ww.reshape(-1)).astype(float, copy=False)
            self._record_history()

            if return_internal_states and sig_hist is not None:
                sig_hist[k, :] = sig

        self.uu = uu
        self.ww = ww
        self.sig = sig

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[CFDLMS] Completed in {runtime_s * 1000:.03f} ms | iters={n_iters} | out_len={out_len}")

        extra: Dict[str, Any] = {
            "ww_history": self.ww_history,
            "n_iters": int(n_iters),
        }
        if return_internal_states:
            extra["sig"] = sig.copy()
            extra["sig_history"] = sig_hist

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="output_error",
            extra=extra,
        )
# EOF