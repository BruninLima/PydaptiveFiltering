#  subband.cfdlms.py
#
#       Implements the Constrained Frequency-Domain LMS (CFDLMS) algorithm
#       for REAL valued data.
#       (Algorithm 12.4 - book: Adaptive Filtering: Algorithms and Practical
#                                                        Implementation, Diniz)
#
#       Notes
#       -----
#       This implementation follows the reference Matlab script `cfdlms.m`
#       provided by the user (toolbox baseline). The algorithm processes the
#       input in overlapped blocks of length M, advancing by L samples per
#       iteration (typically L = M/2). The adaptive filter is implemented in
#       the frequency domain with one sub-filter per frequency bin.
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
from typing import Optional, Union, List, Dict, Any

from pydaptivefiltering.base import AdaptiveFilter
from pydaptivefiltering.utils.validation import ensure_real_signals


class CFDLMS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Constrained Frequency-Domain LMS (CFDLMS) algorithm for REAL valued data.
        (Algorithm 12.4 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Overview
    --------
        The CFDLMS algorithm operates in the frequency domain using M subbands (frequency bins)
        and updates an adaptive filter of length (Nw+1) *per bin*. Each iteration consumes L new
        time samples (decimation/interpolation factor) and uses an overlapped block of length M
        (usually L = M/2) for FFT processing.

        The Matlab reference (`cfdlms.m`) uses:
            - Block length: M
            - Hop size: L
            - Frequency-domain regressor buffer: uu (M x (Nw+1))
            - Per-bin weights: ww (M x (Nw+1))
            - Smoothed subband energy estimate: sig (M,)

        A key feature is the *constraint* step, implemented by FFT across the subband index (m)
        and zeroing bins m = L..M-1 (keeping only the first L bins), then IFFT back. This matches
        the reference script:
            waux = fft(wwc)/sqrt(M)
            wwc  = sqrt(M)*ifft([waux(1:L,:); zeros(M-L, Nw+1)])

    Parameters
    ----------
        filter_order : int
            Subband adaptive filter order Nw (the filter has Nw+1 taps per subband).
        n_subbands : int
            Number of subbands (FFT size) M.
        decimation : int | None
            Hop size L (number of time-domain output samples produced per iteration).
            If None, defaults to M//2 (as in the reference Matlab script).
        step : float
            Convergence factor u.
        gamma : float
            Small positive constant to prevent the update normalization from getting too large.
        smoothing : float
            Exponential smoothing factor 'a' for subband energy estimate sig.
        w_init : array_like, optional
            Initial coefficient matrix. Accepted shapes:
              - (M, Nw+1) : direct initialization
              - (M*(Nw+1),) : will be reshaped to (M, Nw+1)

    Attributes
    ----------
        ww : np.ndarray
            Current subband coefficient matrix, shape (M, Nw+1), complex dtype (internal).
        uu : np.ndarray
            Regressor buffer, shape (M, Nw+1), complex dtype (internal).
        sig : np.ndarray
            Smoothed energy per bin, shape (M,), float dtype.

    """
    supports_complex: bool = False

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
        # AdaptiveFilter expects a "filter_order" concept; here we map it to Nw (subband order).
        super().__init__(filter_order, w_init=None)

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

        self.M: int = int(n_subbands)
        self.L: int = int(decimation)
        self.Nw: int = int(filter_order)

        self.step: float = float(step)
        self.gamma: float = float(gamma)
        self.smoothing: float = float(smoothing)

        # Initialize weights ww (M x (Nw+1))
        self.ww: np.ndarray = np.zeros((self.M, self.Nw + 1), dtype=np.complex128)
        if w_init is not None:
            w0 = np.asarray(w_init)
            if w0.ndim == 2 and w0.shape == (self.M, self.Nw + 1):
                self.ww = w0.astype(np.complex128, copy=True)
            else:
                w0 = w0.reshape(-1)
                if w0.size != self.M * (self.Nw + 1):
                    raise ValueError(
                        f"w_init has incompatible size. Expected {self.M*(self.Nw+1)} "
                        f"or shape ({self.M},{self.Nw+1}), got {w0.size}."
                    )
                self.ww = w0.reshape(self.M, self.Nw + 1).astype(np.complex128, copy=True)

        # Internal state buffers
        self.uu: np.ndarray = np.zeros((self.M, self.Nw + 1), dtype=np.complex128)
        self.sig: np.ndarray = np.zeros(self.M, dtype=np.float64)

        # History (kept separate because ww is matrix-shaped)
        self.ww_history: List[np.ndarray] = []

    @ensure_real_signals
    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Union[np.ndarray, list],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Description
        -----------
            Executes the CFDLMS weight update process.

        Inputs
        -------
            input_signal   : np.ndarray | list
                Input signal x[n] (REAL valued).
            desired_signal : np.ndarray | list
                Desired signal d[n] (REAL valued).
            verbose        : bool
                Verbose boolean.

        Outputs
        -------
            dictionary:
                outputs      : np.ndarray
                    Estimated output signal (REAL), aligned to the desired signal
                    for the processed portion (length = n_iters * L).
                errors       : np.ndarray
                    Output error signal (REAL), same length as outputs.
                coefficients : list[np.ndarray]
                    History of coefficient matrices ww over iterations; each entry has
                    shape (M, Nw+1).

        Main Variables (matching the Matlab reference)
        ----------------------------------------------
            M              : number of subbands (FFT length).
            L              : hop size / decimation factor.
            Nw             : subband filter order (taps per bin = Nw+1).
            x_p            : time-domain block of length M (reversed as in Matlab code).
            ui             : FFT(x_p)/sqrt(M), shape (M,).
            uu             : regressor buffer, shape (M, Nw+1).
            ww             : coefficient matrix, shape (M, Nw+1).
            uy             : per-bin output in frequency domain, shape (M,).
            y_block        : ifft(uy)*sqrt(M), time-domain block, shape (M,).
            e_rev          : error samples in reversed order (length L), as in Matlab.
            et             : FFT([e_rev; zeros(M-L)])/sqrt(M), shape (M,).
            sig            : smoothed subband energy estimate.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br
        """
        tic: float = time()

        x = np.asarray(input_signal, dtype=np.float64)
        d = np.asarray(desired_signal, dtype=np.float64)

        self._validate_inputs(x, d)

        M = self.M
        L = self.L
        Nw = self.Nw

        # Number of iterations limited by both desired length and available input for M-long blocks
        # Need: k*L + M <= (len(x) + L) because we pad L zeros at the beginning.
        max_iters_from_x = int(np.floor((x.size + L - M) / L) + 1) if (x.size + L) >= M else 0
        max_iters_from_d = d.size // L
        n_iters = max(0, min(max_iters_from_x, max_iters_from_d))

        # Outputs (time-aligned), length = n_iters * L
        out_len = n_iters * L
        y_out = np.zeros(out_len, dtype=np.float64)
        e_out = np.zeros(out_len, dtype=np.float64)

        # Pad input with L zeros (as in Matlab xin = [zeros(1,L) xin])
        xpad = np.concatenate([np.zeros(L, dtype=np.float64), x])

        # Reset/initialize histories
        self.ww_history = []

        # Local references to avoid attribute lookups in loop
        uu = self.uu
        ww = self.ww
        sig = self.sig
        a = self.smoothing
        u_step = self.step
        gamma = self.gamma
        sqrtM = np.sqrt(M)

        for k in range(n_iters):
            # Time indices for this iteration
            start = k * L
            seg_x = xpad[start: start + M]  # length M (oldest->newest)

            # Matlab reference reverses the block:
            x_p = seg_x[::-1].astype(np.complex128, copy=False)

            # Desired block (L samples) and its reversed version (as in Matlab dsb)
            d_seg = d[start: start + L]
            d_p = d_seg[::-1].astype(np.complex128, copy=False)

            # Analysis transform
            ui = np.fft.fft(x_p) / sqrtM  # shape (M,)

            # Update regressor buffer uu = [ui, uu(:,1:end-1)]
            uu[:, 1:] = uu[:, :-1]
            uu[:, 0] = ui

            # Frequency-domain output per bin: uy[m] = uu[m,:] * ww[m,:].'
            uy = np.sum(uu * ww, axis=1)  # shape (M,)

            # Synthesis transform
            y_block = np.fft.ifft(uy) * sqrtM  # shape (M,)

            # Keep first L time samples (Matlab uses y(1:L,k))
            y_firstL = y_block[:L]  # reversed-time convention relative to d_p

            # Error in Matlab's reversed convention
            e_rev = d_p - y_firstL  # length L

            # Store outputs/errors in *time order* (reverse back)
            y_time = np.real(y_firstL[::-1])
            e_time = d_seg - y_time

            y_out[start: start + L] = y_time
            e_out[start: start + L] = e_time

            # FFT of padded error: et = fft([e; zeros(M-L,1)])/sqrt(M)
            e_pad = np.concatenate([e_rev, np.zeros(M - L, dtype=np.complex128)])
            et = np.fft.fft(e_pad) / sqrtM  # shape (M,)

            # Unconstrained increment wwc (M x (Nw+1))
            # wwc[m,:] = u/(gamma+(Nw+1)*sig[m]) * conj(uu[m,:]) * et[m]
            # Update sig first: sig[m] = (1-a)*sig[m] + a*|ui[m]|^2
            sig[:] = (1.0 - a) * sig + a * (np.abs(ui) ** 2)

            denom = gamma + (Nw + 1) * sig  # shape (M,)
            gain = u_step / denom           # shape (M,)

            # Broadcasting: (M,1) * (M,Nw+1) * (M,1)
            wwc = (gain[:, None] * np.conj(uu) * et[:, None]).astype(np.complex128, copy=False)

            # Constraint step (as in Matlab):
            # waux = fft(wwc)/sqrt(M)    [FFT across subband index m]
            # wwc  = sqrt(M)*ifft([waux(1:L,:); zeros(M-L,Nw+1)])
            waux = np.fft.fft(wwc, axis=0) / sqrtM
            waux[L:, :] = 0.0
            wwc_c = np.fft.ifft(waux, axis=0) * sqrtM

            # Update weights
            ww = ww + wwc_c

            # Record history (store a copy to avoid mutation)
            self.ww_history.append(ww.copy())

        # Write back updated states
        self.uu = uu
        self.ww = ww
        self.sig = sig

        if verbose:
            print(f"CFDLMS Adaptation completed in {(time() - tic)*1000:.03f} ms")

        return {
            "outputs": y_out,
            "errors": e_out,
            "coefficients": self.ww_history,
        }

# EOF
