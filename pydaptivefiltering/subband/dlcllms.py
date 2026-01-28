#  subband.dlcllms.py
#
#       Implements the Delayless Closed-Loop Subband (LMS) Adaptive-Filtering Algorithm
#       for REAL valued data.
#       (Algorithm 12.3 - book: Adaptive Filtering: Algorithms and Practical
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


ArrayLike = Union[np.ndarray, list]


def _dft_matrix(M: int) -> np.ndarray:
    """
    Equivalent to MATLAB dftmtx(M):

        F[m,n] = exp(-j*2*pi*m*n/M), for m,n = 0..M-1
    """
    m = np.arange(M, dtype=float)
    n = np.arange(M, dtype=float)
    return np.exp(-1j * 2.0 * np.pi * np.outer(m, n) / float(M))


def _design_polyphase_nyquist_bank(M: int, nfd: int) -> np.ndarray:
    """
    Polyphase decomposition of the Nyquist (fractional-delay) prototype exactly as in dlcllms.m.

    MATLAB reference (with L=M):
        K=(Nfd-1)/2;
        n=-K:K;
        b=sinc(n/L);
        win=hamming(Nfd);
        h=b.*win';
        h=[zeros(1,ceil(Nfd/M)*M-Nfd) h];
        Ed=reshape(h,M,length(h)/M);  % column-major
    """
    if nfd <= 0:
        raise ValueError("nyquist_len (Nfd) must be a positive integer.")
    n = np.arange(nfd, dtype=float) - (float(nfd) - 1.0) / 2.0
    L = float(M)
    h = np.sinc(n / L) * np.hamming(nfd)
    P = int(np.ceil(nfd / float(M)))
    pad = P * M - nfd
    if pad > 0:
        h = np.concatenate([np.zeros(pad, dtype=float), h.astype(float)])
    Ed = h.reshape((M, P), order="F").astype(float)
    return Ed


class DLCLLMS(AdaptiveFilter):
    """
    Delayless Closed-Loop Subband LMS (DLCLLMS) for real-valued fullband signals.

    Implements the Delayless Closed-Loop Subband LMS adaptive filtering algorithm
    (Algorithm 12.3, Diniz) using:
      - a DFT analysis bank (complex subband signals),
      - a polyphase Nyquist / fractional-delay prototype (Ed) to realize the delayless
        closed-loop structure,
      - and an equivalent fullband FIR mapping (GG) used to generate the output in the
        time domain.

    High-level operation (as implemented)
    -------------------------------------
    Processing is block-based with block length:
        L = M   (M = number of subbands / DFT size)

    For each block k:
      1) Form a reversed block x_p and pass each sample through a per-branch fractional-delay
         structure (polyphase) driven by `Ed`, producing x_frac (length M).
      2) Compute subband input:
            x_sb = F @ x_frac
         where F is the (non-unitary) DFT matrix (MATLAB dftmtx convention).
      3) Map current subband coefficients to an equivalent fullband FIR:
            GG = equivalent_fullband(w_sb)
         and filter the fullband input block through GG (with state) to produce y_block.
      4) Compute fullband error e_block = d_block - y_block.
      5) Pass the reversed error block through the same fractional-delay structure to get e_frac,
         then compute subband error:
            e_sb = F @ e_frac
      6) Update subband coefficients with an LMS-like recursion using a subband delay line x_cl
         and a smoothed power estimate sig[m]:
            sig[m] = (1-a) sig[m] + a |x_sb[m]|^2
            mu_n  = step / (gamma + (Nw+1) * sig[m])
            w_sb[m,:] <- w_sb[m,:] + 2 * mu_n * conj(e_sb[m]) * x_cl[m,:]

    Coefficient representation and mapping
    --------------------------------------
    - Subband coefficients are stored in:
          w_sb : complex ndarray, shape (M, Nw+1)

    - For output synthesis and for the base API, an equivalent fullband FIR is built:
          GG : real ndarray, length (M*Nw)

      The mapping matches the provided MATLAB logic:
        * Compute ww = real(F^H w_sb) / M
        * For branch m=0: take ww[0, :Nw]
        * For m>=1: convolve ww[m,:] with Ed[m-1,:] and extract a length-Nw segment
          starting at (Dint+1), where Dint=(P-1)//2 and P is the polyphase length.

    - The base-class coefficient vector `self.w` stores GG (float), and
      `OptimizationResult.coefficients` contains the history of GG recorded **once per block**
      (plus the initial entry).

    Parameters
    ----------
    filter_order : int, default=5
        Subband filter order Nw (number of taps per subband delay line is Nw+1).
    n_subbands : int, default=4
        Number of subbands M (DFT size). Also equals the processing block length L.
    step_size : float, default=0.1
        Global LMS step size.
    gamma : float, default=1e-2
        Regularization constant in the normalized step denominator (>0 recommended).
    a : float, default=1e-2
        Exponential smoothing factor for subband power sig in (0,1].
    nyquist_len : int, default=2
        Length Nfd of the Nyquist (fractional-delay) prototype used to build Ed.
    w_init : array_like, optional
        Initial subband coefficient matrix. Can be either:
          - shape (M, Nw+1), or
          - flat length M*(Nw+1), reshaped internally.

    Notes
    -----
    - Real-valued interface (input_signal and desired_signal enforced real). Internal
      computations use complex subband signals.
    - This implementation processes only `n_used = floor(N/M)*M` samples. Any tail
      samples (N - n_used) are left with output=0 and error=d in that region.
    - The reported `error_type` is "output_error" (fullband output error sequence).
    """
    supports_complex: bool = False

    def __init__(
        self,
        filter_order: int = 5,
        n_subbands: int = 4,
        step_size: float = 0.1,
        gamma: float = 1e-2,
        a: float = 1e-2,
        nyquist_len: int = 2,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        self.M: int = int(n_subbands)
        if self.M <= 0:
            raise ValueError("n_subbands must be a positive integer.")

        self.Nw: int = int(filter_order)
        if self.Nw <= 0:
            raise ValueError("filter_order must be a positive integer.")

        self.step_size: float = float(step_size)
        self.gamma: float = float(gamma)
        self.a: float = float(a)

        self.nyquist_len: int = int(nyquist_len)
        if self.nyquist_len <= 0:
            raise ValueError("nyquist_len must be a positive integer.")

        self._full_len: int = int(self.M * self.Nw)

        super().__init__(filter_order=self._full_len - 1, w_init=None)

        self.Ed: np.ndarray = _design_polyphase_nyquist_bank(self.M, self.nyquist_len)
        self._P: int = int(self.Ed.shape[1])
        self._Dint: int = int((self._P - 1) // 2)

        self.F: np.ndarray = _dft_matrix(self.M)

        self.w_sb: np.ndarray = np.zeros((self.M, self.Nw + 1), dtype=complex)
        if w_init is not None:
            w0 = np.asarray(w_init)
            if w0.ndim == 2 and w0.shape == (self.M, self.Nw + 1):
                self.w_sb = w0.astype(complex, copy=True)
            else:
                w0 = w0.reshape(-1)
                if w0.size != self.M * (self.Nw + 1):
                    raise ValueError(
                        f"w_init has incompatible size. Expected {self.M*(self.Nw+1)} "
                        f"or shape ({self.M},{self.Nw+1}), got {w0.size}."
                    )
                self.w_sb = w0.reshape((self.M, self.Nw + 1)).astype(complex, copy=True)

        self.x_cl: np.ndarray = np.zeros((self.M, self.Nw + 1), dtype=complex)

        self.sig: np.ndarray = np.zeros((self.M,), dtype=float)

        self._xx_frac: np.ndarray = np.zeros((self._P, self.M), dtype=float)
        self._ee_frac: np.ndarray = np.zeros((self._P, self.M), dtype=float)

        self._x_state: np.ndarray = np.zeros((max(self._full_len - 1, 0),), dtype=float)

        self.w_history = []
        self._record_history()

    def reset_filter(self, w_new: Optional[Union[np.ndarray, list]] = None) -> None:
        """
        Reset coefficients and history.

        - If w_new is provided:
            * If shape (M, Nw+1): interpreted as subband coefficients.
            * If flat of length M*(Nw+1): reshaped as subband coefficients.
        - Resets internal states (x_cl, sig, fractional-delay, FIR state).
        """
        if w_new is None:
            self.w_sb = np.zeros((self.M, self.Nw + 1), dtype=complex)
        else:
            w0 = np.asarray(w_new)
            if w0.ndim == 2 and w0.shape == (self.M, self.Nw + 1):
                self.w_sb = w0.astype(complex, copy=True)
            else:
                w0 = w0.reshape(-1)
                if w0.size != self.M * (self.Nw + 1):
                    raise ValueError(
                        f"w_new has incompatible size. Expected {self.M*(self.Nw+1)} "
                        f"or shape ({self.M},{self.Nw+1}), got {w0.size}."
                    )
                self.w_sb = w0.reshape((self.M, self.Nw + 1)).astype(complex, copy=True)

        self.x_cl = np.zeros((self.M, self.Nw + 1), dtype=complex)
        self.sig = np.zeros((self.M,), dtype=float)
        self._xx_frac = np.zeros((self._P, self.M), dtype=float)
        self._ee_frac = np.zeros((self._P, self.M), dtype=float)
        self._x_state = np.zeros((max(self._full_len - 1, 0),), dtype=float)

        GG = self._equivalent_fullband()
        self.w = GG.astype(float, copy=True)
        self.w_history = []
        self._record_history()

    def _equivalent_fullband(self) -> np.ndarray:
        """
        Build the equivalent fullband FIR GG (length M*Nw) from current subband coefficients,
        matching the MATLAB mapping.

        Returns
        -------
        GG : np.ndarray, shape (M*Nw,), dtype=float
        """
        ww = np.real(self.F.conj().T @ self.w_sb) / float(self.M)

        G = np.zeros((self.M, self.Nw), dtype=float)
        G[0, :] = ww[0, : self.Nw]

        for m in range(1, self.M):
            aux = np.convolve(self.Ed[m - 1, :], ww[m, :], mode="full")
            start = self._Dint + 1
            stop = start + self.Nw
            G[m, :] = aux[start:stop]

        GG = G.reshape(-1, order="F")
        return GG

    def _fir_block(self, b: np.ndarray, x_block: np.ndarray) -> np.ndarray:
        """
        FIR filtering with state, matching MATLAB `filter(b,1,x,zi)` block-by-block.
        """
        Lb = int(b.size)
        if Lb == 0:
            return np.zeros_like(x_block, dtype=float)
        if Lb == 1:
            return float(b[0]) * x_block

        y = np.zeros_like(x_block, dtype=float)
        state = self._x_state

        for i, x_n in enumerate(x_block):
            acc = float(b[0]) * float(x_n)
            if Lb > 1 and state.size > 0:
                acc += float(np.dot(b[1:], state[: Lb - 1]))
            y[i] = acc

            if state.size > 0:
                state[1:] = state[:-1]
                state[0] = float(x_n)

        self._x_state = state
        return y

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
        Run DLCLLMS adaptation block-by-block.

        Parameters
        ----------
        input_signal : array_like of float
            Fullband input x[n], shape (N,).
        desired_signal : array_like of float
            Fullband desired d[n], shape (N,).
        verbose : bool, default=False
            If True, prints runtime and block stats.
        return_internal_states : bool, default=False
            If True, returns additional internal trajectories in result.extra.

        Returns
        -------
        OptimizationResult
            outputs : ndarray of float, shape (N,)
                Estimated fullband output y[n]. Only the first `n_used` samples are
                produced by block processing; remaining tail (if any) is zero.
            errors : ndarray of float, shape (N,)
                Fullband error e[n] = d[n] - y[n]. Tail (if any) equals d[n] there.
            coefficients : ndarray
                History of equivalent fullband FIR vectors GG (length M*Nw), stored
                once per processed block (plus initial entry).
            error_type : str
                "output_error".

            extra : dict
                Always contains:
                    - "n_blocks": number of processed blocks
                    - "block_len": block length (equals M)
                    - "n_used": number of processed samples (multiple of M)
                If return_internal_states=True, also contains:
                    - "sig_history": ndarray (n_blocks, M) of smoothed subband power
                    - "w_sb_final": final subband coefficient matrix (M, Nw+1)
        """
        tic: float = time()

        x = np.asarray(input_signal, dtype=float).ravel()
        d = np.asarray(desired_signal, dtype=float).ravel()

        n_samples: int = int(x.size)
        M: int = int(self.M)
        L: int = M

        n_blocks: int = int(n_samples // L)
        n_used: int = int(n_blocks * L)

        outputs = np.zeros((n_samples,), dtype=float)
        errors = np.zeros((n_samples,), dtype=float)

        sig_hist: Optional[np.ndarray] = np.zeros((n_blocks, M), dtype=float) if return_internal_states else None

        self.w_history = []
        self._record_history()

        if n_blocks == 0:
            errors = d - outputs
            runtime_s: float = float(time() - tic)
            extra: Dict[str, Any] = {"n_blocks": 0, "block_len": L, "n_used": 0}
            return self._pack_results(
                outputs=outputs,
                errors=errors,
                runtime_s=runtime_s,
                error_type="output_error",
                extra=extra,
            )

        for k in range(n_blocks):
            i0 = k * L
            i1 = i0 + L

            x_block = x[i0:i1]
            d_block = d[i0:i1]

            x_p = x_block[::-1]

            x_frac = np.zeros((M,), dtype=float)
            for m in range(M):
                self._xx_frac[1:, m] = self._xx_frac[:-1, m]
                self._xx_frac[0, m] = x_p[m]
                x_frac[m] = float(np.dot(self.Ed[m, :], self._xx_frac[:, m]))

            xsb = self.F @ x_frac.astype(complex)

            GG = self._equivalent_fullband()
            y_block = self._fir_block(GG, x_block)

            outputs[i0:i1] = y_block
            e_block = d_block - y_block
            errors[i0:i1] = e_block

            self.w = GG.astype(float, copy=True)
            self._record_history()

            e_p = e_block[::-1]
            e_frac = np.zeros((M,), dtype=float)
            for m in range(M):
                self._ee_frac[1:, m] = self._ee_frac[:-1, m]
                self._ee_frac[0, m] = e_p[m]
                e_frac[m] = float(np.dot(self.Ed[m, :], self._ee_frac[:, m]))

            esb = self.F @ e_frac.astype(complex)

            for m in range(M):
                self.x_cl[m, 1:] = self.x_cl[m, :-1]
                self.x_cl[m, 0] = xsb[m]

                self.sig[m] = (1.0 - self.a) * self.sig[m] + self.a * (np.abs(xsb[m]) ** 2)

                mu_n = self.step_size / (self.gamma + (self.Nw + 1) * self.sig[m])

                self.w_sb[m, :] = self.w_sb[m, :] + 2.0 * mu_n * np.conj(esb[m]) * self.x_cl[m, :]

            if return_internal_states and sig_hist is not None:
                sig_hist[k, :] = self.sig

        if n_used < n_samples:
            outputs[n_used:] = 0.0
            errors[n_used:] = d[n_used:] - outputs[n_used:]

        runtime_s: float = float(time() - tic)
        if verbose:
            print(f"[DLCLLMS] Completed in {runtime_s * 1000:.03f} ms | blocks={n_blocks} | used={n_used}/{n_samples}")

        extra: Dict[str, Any] = {
            "n_blocks": int(n_blocks),
            "block_len": int(L),
            "n_used": int(n_used),
        }
        if return_internal_states:
            extra.update(
                {
                    "sig_history": sig_hist,
                    "w_sb_final": self.w_sb.copy(),
                }
            )

        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="output_error",
            extra=extra,
        )
# EOF