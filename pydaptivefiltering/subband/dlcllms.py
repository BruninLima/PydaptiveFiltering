#  subband.dlcllms.py
#
#       Implements the Delayless Closed-Loop Subband (LMS) Adaptive-Filtering Algorithm
#       for REAL valued data.
#       (Algorithm 12.3 - book: Adaptive Filtering: Algorithms and Practical
#                                                        Implementation, Diniz)
#
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

# Imports 
from __future__ import annotations

import numpy as np
from time import time
from typing import Optional, Union, List, Dict

from pydaptivefiltering.base import AdaptiveFilter

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

    Parameters
    ----------
    M : int
        Number of subbands (also the block length L=M).
    nfd : int
        Nyquist filter length Nfd.

    Returns
    -------
    Ed : np.ndarray, shape (M, P), dtype=float
        Polyphase matrix (analysis bank used in the fractional-delay path),
        where P = ceil(Nfd/M).
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
    Description
    -----------
        Implements the Delayless Closed-Loop Subband LMS Adaptive-Filtering Algorithm
        for REAL valued data (DLCLLMS).
        (Algorithm 12.3 - book: Adaptive Filtering: Algorithms and Practical
        Implementation, Diniz)

        This class is a faithful translation of the MATLAB reference `dlcllms.m`:
        - block processing with block length L = M (number of subbands)
        - DFT analysis bank (dftmtx)
        - fractional-delay (Nyquist) polyphase structure Ed (built from Nfd)
        - delayless fullband output produced by mapping the subband coefficients
            into an equivalent fullband FIR GG, then filtering the current block
        - closed-loop: the fullband error is split into subbands and used to
            update the subband filters with an NLMS-like normalized step.

    Notes
    -----
        * The algorithm targets REAL-valued fullband signals (input and desired).
        Internally, subband signals become complex due to the DFT analysis bank.
        * The coefficient mapping step follows MATLAB exactly:
            ww = real(F' * w_cl) / M
        which forces a real-valued equivalent fullband filter GG.

    Attributes
    ----------
        supports_complex : bool
            False (algorithm is specified for REAL-valued data).
    """

    supports_complex: bool = False

    def __init__(
        self,
        filter_order: int = 5,
        n_subbands: int = 4,
        step: float = 0.1,
        gamma: float = 1e-2,
        a: float = 1e-2,
        nyquist_len: int = 2,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Inputs
        -------
            filter_order : int
                Order of the adaptive filter in each subband (Nw in MATLAB).
                Each subband has length (Nw+1).
            n_subbands : int
                Number of subbands (M in MATLAB). Also the block length (L=M).
            step : float
                Convergence factor (u in MATLAB).
            gamma : float
                Small constant to prevent the updating factor from getting too large.
            a : float
                Smoothing factor for the input power estimate in each subband.
            nyquist_len : int
                Nyquist filter length (Nfd in MATLAB) used in the fractional-delay path.
            w_init : array_like, optional
                Optional initial *subband* coefficients w_cl of shape (M, Nw+1) or
                a flat vector of length M*(Nw+1). If provided, it is used to
                initialize the subband coefficient matrix.

        Notes
        -----
            - The equivalent fullband FIR GG has length M*Nw (as in MATLAB).
            - `AdaptiveFilter` base is initialized with order (M*Nw - 1) so that
            `self.w` can expose the current mapped fullband coefficients.
        """
        self.M = int(n_subbands)
        if self.M <= 0:
            raise ValueError("n_subbands must be a positive integer.")

        self.Nw = int(filter_order)
        if self.Nw <= 0:
            raise ValueError("filter_order must be a positive integer.")

        self.step = float(step)
        self.gamma = float(gamma)
        self.a = float(a)

        self.nyquist_len = int(nyquist_len)
        if self.nyquist_len <= 0:
            raise ValueError("nyquist_len must be a positive integer.")

        self._full_len = self.M * self.Nw 
        super().__init__(self._full_len - 1, w_init=None)

        self.Ed = _design_polyphase_nyquist_bank(self.M, self.nyquist_len)
        self._P = int(self.Ed.shape[1])
        self._Dint = int((self._P - 1) // 2)

        self.F = _dft_matrix(self.M)

        self.w_sb = np.zeros((self.M, self.Nw + 1), dtype=complex)

        if w_init is not None:
            w0 = np.asarray(w_init)
            if w0.ndim == 2 and w0.shape == (self.M, self.Nw + 1):
                self.w_sb = w0.astype(complex)
            else:
                w0 = w0.reshape(-1)
                if w0.size == self.M * (self.Nw + 1):
                    self.w_sb = w0.reshape((self.M, self.Nw + 1)).astype(complex)

        self.x_cl = np.zeros((self.M, self.Nw + 1), dtype=complex)

        self.sig = np.zeros((self.M,), dtype=float)

        self._xx_frac = np.zeros((self._P, self.M), dtype=float)
        self._ee_frac = np.zeros((self._P, self.M), dtype=float)

        self._x_state = np.zeros((max(self._full_len - 1, 0),), dtype=float)

        self.w_history: List[np.ndarray] = []

    def _validate_inputs_real_1d(self, x: np.ndarray, d: np.ndarray) -> None:
        if x.ndim != 1 or d.ndim != 1:
            raise ValueError("input_signal and desired_signal must be 1-D arrays.")
        if x.size != d.size:
            raise ValueError("input_signal and desired_signal must have the same length.")
        if x.size == 0:
            raise ValueError("Signals cannot be empty.")

    def _equivalent_fullband(self) -> np.ndarray:
        """
        Build the equivalent fullband FIR GG (length M*Nw) from current subband coefficients,
        exactly as in MATLAB:

            ww = real(F' * w_cl) / M         # ww: M x (Nw+1)
            G(1,:) = ww(1,1:end-1)           # length Nw
            for m=2..M:
                aux = conv(Ed(m-1,:), ww(m,:))
                G(m,:) = aux(Dint+2:Dint+1+size(ww,2)-1)   # length Nw
            GG = reshape(G, 1, M*Nw)         # column-major

        Returns
        -------
        GG : np.ndarray, shape (M*Nw,), dtype=float
        """
        ww = np.real(self.F.conj().T @ self.w_sb) / float(self.M)  # (M, Nw+1), real

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
        FIR filtering with state, matching MATLAB `filter(b,1,x,zi)` block by block.

        Parameters
        ----------
        b : np.ndarray
            FIR coefficients (length Lb).
        x_block : np.ndarray
            Input block (1-D).

        Returns
        -------
        y_block : np.ndarray
            Output block (same length as x_block).
        """
        Lb = int(b.size)
        if Lb == 0:
            return np.zeros_like(x_block, dtype=float)
        if Lb == 1:
            return float(b[0]) * x_block

        y = np.zeros_like(x_block, dtype=float)
        state = self._x_state

        for i, x_n in enumerate(x_block):
            acc = b[0] * x_n
            if Lb > 1:
                acc += float(np.dot(b[1:], state[: Lb - 1]))
            y[i] = acc

            if state.size > 0:
                state[1:] = state[:-1]
                state[0] = x_n

        self._x_state = state
        return y

    def optimize(
        self,
        input_signal: ArrayLike,
        desired_signal: ArrayLike,
        verbose: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the adaptation process for the DLCLLMS algorithm (delayless closed-loop subband LMS),
            faithfully matching the MATLAB reference `dlcllms.m`.

        Inputs
        -------
            input_signal : np.ndarray | list
                Fullband input signal x[n] (REAL-valued).
            desired_signal : np.ndarray | list
                Fullband desired signal d[n] (REAL-valued).
            verbose : bool
                Verbose boolean.

        Outputs
        -------
            dictionary:
                outputs : np.ndarray
                    Estimated fullband output y[n] (REAL).
                errors : np.ndarray
                    Fullband error e[n] = d[n] - y[n] (REAL).
                coefficients : list[np.ndarray]
                    History of equivalent fullband FIR vectors GG (each has length M*Nw),
                    stored once per processed block (k = 1..Nblocks).

        Main Variables (MATLAB names)
        -----------------------------
            M, L           : number of subbands and block length (L=M)
            Nw             : subband adaptive filter order
            Ed             : polyphase Nyquist (fractional-delay) bank
            F              : DFT matrix
            x_p            : reversed block of input samples (phase alignment)
            x_frac         : fractional-delay outputs per phase
            xsb            : subband input samples (DFT output)
            ww, G, GG       : mapping from subband coefficients to equivalent fullband FIR
            y_dl           : delayless fullband output for the block
            e_aux1, e_p     : fullband error block and its reversed version
            e_frac, esb     : fractional-delay outputs for error and subband errors
            sig_cl, unlms   : smoothed subband power and normalized step

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br
        """
        tic = time()

        x = np.asarray(input_signal, dtype=float).reshape(-1)
        d = np.asarray(desired_signal, dtype=float).reshape(-1)
        self._validate_inputs_real_1d(x, d)

        n_samples = int(x.size)
        M = self.M
        L = M

        n_blocks = n_samples // L
        n_used = n_blocks * L

        y = np.zeros((n_samples,), dtype=float)
        e = np.zeros((n_samples,), dtype=float)

        if n_blocks == 0:
            e = d - y
            return {"outputs": y, "errors": e, "coefficients": self.w_history}

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
            self.w = GG.astype(float, copy=True)
            self.w_history.append(self.w.copy())

            y_block = self._fir_block(GG, x_block)
            y[i0:i1] = y_block

            e_block = d_block - y_block
            e[i0:i1] = e_block

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

                mu_n = self.step / (self.gamma + (self.Nw + 1) * self.sig[m])

                self.w_sb[m, :] = self.w_sb[m, :] + 2.0 * mu_n * np.conj(esb[m]) * self.x_cl[m, :]

        if n_used < n_samples:
            y[n_used:] = 0.0
            e[n_used:] = d[n_used:] - y[n_used:]

        if verbose:
            print(f"DLCLLMS Adaptation completed in {(time() - tic) * 1000:.03f} ms")

        return {"outputs": y, "errors": e, "coefficients": self.w_history}
# EOF