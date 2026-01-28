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
    Constrained Frequency-Domain LMS (CFDLMS) for real-valued signals (block adaptive).

    Implements the Constrained Frequency-Domain LMS algorithm (Algorithm 12.4, Diniz)
    for identifying/estimating a real-valued FIR system in a block-wise frequency-domain
    framework with a time-domain constraint (to control circular convolution / enforce
    effective FIR support).

    Block structure and main variables
    ----------------------------------
    Let:
        - M: number of subbands / FFT size (also the block length in frequency domain),
        - L: decimation / number of fresh time samples per iteration (block advance),
        - Nw: time-support (per subband) of the adaptive filters, so each subband filter
              has length (Nw+1) in the *time-lag* axis (columns of `ww`).

    Internal coefficient representation
    -----------------------------------
    The adaptive parameters are stored as a complex matrix:

        ww  in C^{M x (Nw+1)}

    where each row corresponds to one frequency bin (subband), and each column is a
    delay-tap in the *block* (overlap) dimension.

    For compatibility with the base API:
        - `self.w` stores a flattened real view of `ww` (real part only),
        - `OptimizationResult.coefficients` comes from the base `w_history` (flattened),
        - the full matrix trajectory is returned in `result.extra["ww_history"]`.

    Signal processing conventions (as implemented)
    ----------------------------------------------
    Per iteration k (block index):
    - Build an M-length time vector from the most recent input segment (reversed):
          x_p = [x[kL+M-1], ..., x[kL]]^T
      then compute a *unitary* FFT:
          ui = FFT(x_p) / sqrt(M)

    - Maintain a regressor matrix `uu` with shape (M, Nw+1) containing the most recent
      Nw+1 frequency-domain regressors (columns shift right each iteration).

    - Compute frequency-domain output per bin:
          uy = sum_j uu[:, j] * ww[:, j]
      and return to time domain:
          y_block = IFFT(uy) * sqrt(M)

      Only the first L samples are used as the “valid” output of this block.

    Error, energy smoothing, and update
    -----------------------------------
    The algorithm forms an L-length error (in the reversed time order used internally),
    zero-pads it to length M, and FFTs it (unitary) to obtain `et`.

    A smoothed energy estimate per bin is kept:
        sig[k] = (1-a) sig[k-1] + a |ui|^2
    where `a = smoothing`.

    The normalized per-bin step is:
        gain = step / (gamma + (Nw+1) * sig)

    A preliminary frequency-domain correction is built:
        wwc = gain[:,None] * conj(uu) * et[:,None]

    Constrained / time-domain projection
    ------------------------------------
    The “constraint” is applied by transforming wwc along axis=0 (FFT across bins),
    zeroing time indices >= L (i.e., enforcing an L-sample time support),
    and transforming back (IFFT). This is the standard “constrained” step that reduces
    circular-convolution artifacts.

    Returned sequences
    ------------------
    - `outputs`: real-valued estimated output, length = n_iters * L
    - `errors`:  real-valued output error (d - y), same length as outputs
    - `error_type="output_error"` (block output error, not a priori scalar error)

    Parameters
    ----------
    filter_order : int, default=5
        Subband filter order Nw (number of taps is Nw+1 along the overlap dimension).
    n_subbands : int, default=64
        FFT size M (number of subbands / frequency bins).
    decimation : int, optional
        Block advance L (samples per iteration). If None, defaults to M//2.
    step_size : float, default=0.1
        Global step size (mu).
    gamma : float, default=1e-2
        Regularization constant in the normalization denominator (>0).
    smoothing : float, default=0.01
        Exponential smoothing factor a in (0,1].
    w_init : array_like, optional
        Initial coefficients. Can be either:
        - matrix shape (M, Nw+1), or
        - flat length M*(Nw+1), reshaped internally.

    Notes
    -----
    - Real-valued interface: input_signal and desired_signal are enforced real.
      Internally complex arithmetic is used due to FFT processing.
    - This is a block algorithm: one iteration produces L output samples.
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
        step_size: float = 0.1,
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

        self.step_size = float(step_size)
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
        Run CFDLMS adaptation over real-valued (x[n], d[n]) in blocks.

        Parameters
        ----------
        input_signal : array_like of float
            Input sequence x[n], shape (N,).
        desired_signal : array_like of float
            Desired sequence d[n], shape (N,).
        verbose : bool, default=False
            If True, prints runtime and basic iteration stats.
        return_internal_states : bool, default=False
            If True, includes additional internal trajectories in result.extra.

        Returns
        -------
        OptimizationResult
            outputs : ndarray of float, shape (n_iters * L,)
                Concatenated block outputs (L per iteration).
            errors : ndarray of float, shape (n_iters * L,)
                Output error sequence e[n] = d[n] - y[n].
            coefficients : ndarray
                Flattened coefficient history (from base class; real part of ww).
            error_type : str
                "output_error".
            extra : dict
                Always contains:
                    - "ww_history": list of ndarray, each shape (M, Nw+1)
                    - "n_iters": int
                If return_internal_states=True, also contains:
                    - "sig": ndarray, shape (M,) final smoothed per-bin energy
                    - "sig_history": ndarray, shape (n_iters, M)
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
        u_step = self.step_size
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