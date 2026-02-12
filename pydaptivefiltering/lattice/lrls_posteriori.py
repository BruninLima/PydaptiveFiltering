# lattice.lrls_posteriori.py
#
#      Implements the Lattice RLS algorithm based on a posteriori errors.
#      (Algorithm 7.1 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, Diniz)
#
#      Authors:
#       . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#       . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#       . Markus Vinícius Santos Lima    - mvsl20@gmail.com          & markus@lps.ufrj.br
#       . Wallace Alves Martins          - wallace@lps.ufrj.br
#       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#       . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

from __future__ import annotations

import numpy as np
from time import perf_counter
from typing import Any, Dict, Optional, Union

from pydaptivefiltering.base import AdaptiveFilter, OptimizationResult, validate_input


class LRLSPosteriori(AdaptiveFilter):
    """
    Lattice RLS using a posteriori errors (LRLS, a posteriori form), complex-valued.

    Implements Diniz (Algorithm 7.1) in a lattice/ladder structure:

    1) **Lattice prediction stage** (order ``M``):
       Updates forward/backward a posteriori prediction errors and energy terms
       using exponentially weighted recursions.

    2) **Ladder (joint-process) stage** (length ``M+1``):
       Updates the ladder coefficients ``v`` and produces the **a posteriori**
       output error by progressively "whitening" the desired sample through the
       backward-error vector.

    Library conventions
    -------------------
    - Complex-valued implementation (``supports_complex=True``).
    - Ladder coefficients are stored in ``self.v`` with length ``M+1``.
    - For compatibility with :class:`~pydaptivefiltering.base.AdaptiveFilter`,
      ``self.w`` mirrors ``self.v`` at each iteration and the base-class history
      corresponds to the ladder coefficient trajectory.

    Parameters
    ----------
    filter_order : int
        Lattice order ``M`` (number of sections). The ladder has ``M+1`` coefficients.
    lambda_factor : float, optional
        Forgetting factor ``lambda`` used in the exponentially weighted recursions.
        Default is 0.99.
    epsilon : float, optional
        Initialization/regularization constant for the energy variables
        (forward/backward). Default is 0.1.
    w_init : array_like of complex, optional
        Optional initial ladder coefficients of length ``M+1``. If None, initializes
        with zeros.
    denom_floor : float, optional
        Small positive floor used to avoid division by (near) zero in normalization
        terms (``gamma`` variables and energy denominators). Default is 1e-12.
    xi_floor : float, optional
        Floor applied to energy variables to keep them positive. If None, defaults
        to ``epsilon``.

    Notes
    -----
    Signals and dimensions
    ~~~~~~~~~~~~~~~~~~~~~~
    For lattice order ``M``:

    - ``delta`` has shape ``(M,)`` (lattice delta state)
    - ``xi_f`` and ``xi_b`` have shape ``(M+1,)`` (forward/backward energies)
    - ``error_b_prev`` and the per-sample ``curr_err_b`` have shape ``(M+1,)``
      (backward-error vectors)
    - ``v`` and ``delta_v`` have shape ``(M+1,)`` (ladder state and coefficients)

    A posteriori error (as returned)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The ladder stage starts with ``e_post = d[k]`` and updates it as:

    .. math::
        e_{post}(k) \\leftarrow e_{post}(k) - v_m^*(k)\\, b_m(k),

    where :math:`b_m(k)` are the components of the backward-error vector.
    The final ``e_post`` is the **a posteriori error** returned in ``errors[k]``,
    while the output estimate is returned as ``outputs[k] = d[k] - e_post``.

    References
    ----------
    .. [1] P. S. R. Diniz, *Adaptive Filtering: Algorithms and Practical
       Implementation*, Algorithm 7.1.
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int,
        lambda_factor: float = 0.99,
        epsilon: float = 0.1,
        w_init: Optional[Union[np.ndarray, list]] = None,
        denom_floor: float = 1e-12,
        xi_floor: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        filter_order:
            Number of lattice sections M. Ladder has M+1 coefficients.
        lambda_factor:
            Forgetting factor λ.
        epsilon:
            Energy initialization / regularization.
        w_init:
            Optional initial ladder coefficient vector (length M+1). If None, zeros.
        denom_floor:
            Floor used to avoid division by (near) zero in normalization terms.
        xi_floor:
            Floor used to keep energies positive (defaults to epsilon).
        """
        super().__init__(filter_order=filter_order, w_init=w_init)

        self.lam = float(lambda_factor)
        self.epsilon = float(epsilon)
        self.n_sections = int(filter_order)

        self._tiny = float(denom_floor)
        self._xi_floor = float(xi_floor) if xi_floor is not None else float(self.epsilon)

        self.delta = np.zeros(self.n_sections, dtype=complex)
        self.xi_f = np.ones(self.n_sections + 1, dtype=float) * self.epsilon
        self.xi_b = np.ones(self.n_sections + 1, dtype=float) * self.epsilon
        self.error_b_prev = np.zeros(self.n_sections + 1, dtype=complex)

        if w_init is not None:
            v0 = np.asarray(w_init, dtype=complex).reshape(-1)
            if v0.size != self.n_sections + 1:
                raise ValueError(
                    f"w_init must have length {self.n_sections + 1}, got {v0.size}"
                )
            self.v = v0
        else:
            self.v = np.zeros(self.n_sections + 1, dtype=complex)

        self.delta_v = np.zeros(self.n_sections + 1, dtype=complex)

        self.w = self.v.copy()
        self.w_history = []
        self._record_history()


    @validate_input
    def optimize(
    self,
    input_signal: np.ndarray,
    desired_signal: np.ndarray,
    verbose: bool = False,
    return_internal_states: bool = False,
    store_history: bool = True,
    history_stride: int = 1,
) -> OptimizationResult:
        
        """
        Executes LRLS adaptation (a posteriori form) over paired sequences ``x[k]`` and ``d[k]``.

        Parameters
        ----------
        input_signal : array_like of complex
            Input sequence ``x[k]`` with shape ``(N,)``.
        desired_signal : array_like of complex
            Desired/reference sequence ``d[k]`` with shape ``(N,)``.
        verbose : bool, optional
            If True, prints the total runtime after completion.
        return_internal_states : bool, optional
            If True, returns selected *final* internal states in ``result.extra``
            (not full trajectories).

        Returns
        -------
        OptimizationResult
            Result object with fields:
            - outputs : ndarray of complex, shape ``(N,)``
                Estimated output sequence. In this implementation:
                ``outputs[k] = d[k] - e_post[k]``.
            - errors : ndarray of complex, shape ``(N,)``
                A posteriori error produced by the ladder stage (final ``e_post``).
            - coefficients : ndarray
                Ladder coefficient history (mirrors ``self.v`` via ``self.w``).
            - error_type : str
                Set to ``"a_posteriori"``.
            - extra : dict, optional
                Present only if ``return_internal_states=True`` (see below).

        Extra (when return_internal_states=True)
        --------------------------------------
        xi_f : ndarray of float, shape ``(M+1,)``
            Final forward energies.
        xi_b : ndarray of float, shape ``(M+1,)``
            Final backward energies.
        delta : ndarray of complex, shape ``(M,)``
            Final lattice delta state.
        delta_v : ndarray of complex, shape ``(M+1,)``
            Final ladder delta state used to compute ``v``.
        """
        
        t0 = perf_counter()

        x_in = np.asarray(input_signal, dtype=complex).ravel()
        d_in = np.asarray(desired_signal, dtype=complex).ravel()

        n_samples = int(d_in.size)
        outputs = np.empty(n_samples, dtype=complex)
        errors = np.empty(n_samples, dtype=complex)

        M = self.n_sections
        lam = self.lam
        tiny = self._tiny
        xi_floor = self._xi_floor

        delta = self.delta
        xi_f = self.xi_f
        xi_b = self.xi_b
        delta_v = self.delta_v
        v = self.v

        err_b_prev = self.error_b_prev.copy()
        err_b_curr = np.zeros(M + 1, dtype=complex)

        hs = 1 if history_stride is None or history_stride < 1 else int(history_stride)

        for k in range(n_samples):
            xk = x_in[k]
            err_f = xk

            err_b_curr.fill(0.0)
            err_b_curr[0] = xk

            energy_x = err_f.real * err_f.real + err_f.imag * err_f.imag
            xi_f0 = lam * xi_f[0] + energy_x
            if xi_f0 < xi_floor:
                xi_f0 = xi_floor
            xi_f[0] = xi_f0
            xi_b[0] = xi_f0

            gamma_m = 1.0

            for m in range(M):
                denom_g = gamma_m if gamma_m > tiny else tiny

                ebpm = err_b_prev[m]
                dm = lam * delta[m] + (ebpm * np.conj(err_f)) / denom_g
                delta[m] = dm

                denom_xib = xi_b[m] + tiny
                denom_xif = xi_f[m] + tiny

                kappa_f = np.conj(dm) / denom_xib
                kappa_b = dm / denom_xif

                new_err_f = err_f - kappa_f * ebpm
                eb_next = ebpm - kappa_b * err_f
                err_b_curr[m + 1] = eb_next

                e_nf = new_err_f.real * new_err_f.real + new_err_f.imag * new_err_f.imag
                e_bn = eb_next.real * eb_next.real + eb_next.imag * eb_next.imag

                xif_next = lam * xi_f[m + 1] + e_nf / denom_g
                xib_next = lam * xi_b[m + 1] + e_bn / denom_g
                xi_f[m + 1] = xif_next if xif_next > xi_floor else xi_floor
                xi_b[m + 1] = xib_next if xib_next > xi_floor else xi_floor

                ebm = err_b_curr[m]
                energy_b_curr = ebm.real * ebm.real + ebm.imag * ebm.imag
                gamma_m_next = gamma_m - (energy_b_curr / denom_xib)
                gamma_m = gamma_m_next if gamma_m_next > tiny else tiny

                err_f = new_err_f

            e_post = d_in[k]
            gamma_ladder = 1.0

            for m in range(M + 1):
                denom_gl = gamma_ladder if gamma_ladder > tiny else tiny
                cbm = err_b_curr[m]

                dvm = lam * delta_v[m] + (cbm * np.conj(e_post)) / denom_gl
                delta_v[m] = dvm

                denom_xib_m = xi_b[m] + tiny
                vm = dvm / denom_xib_m
                v[m] = vm

                e_post = e_post - np.conj(vm) * cbm

                energy_b_l = cbm.real * cbm.real + cbm.imag * cbm.imag
                gamma_ladder_next = gamma_ladder - (energy_b_l / denom_xib_m)
                gamma_ladder = gamma_ladder_next if gamma_ladder_next > tiny else tiny

            outputs[k] = d_in[k] - e_post
            errors[k] = e_post

            err_b_prev, err_b_curr = err_b_curr, err_b_prev

            self.w[...] = v

            if store_history and (k % hs == 0):
                self._record_history()

        self.error_b_prev[...] = err_b_prev

        runtime_s = float(perf_counter() - t0)
        if verbose:
            print(f"[LRLSPosteriori] Completed in {runtime_s * 1000:.02f} ms")

        extra: Optional[Dict[str, Any]] = None
        if return_internal_states:
            extra = {
                "xi_f": xi_f.copy(),
                "xi_b": xi_b.copy(),
                "delta": delta.copy(),
                "delta_v": delta_v.copy(),
            }
        return self._pack_results(
            outputs=outputs,
            errors=errors,
            runtime_s=runtime_s,
            error_type="a_posteriori",
            extra=extra,
        )
# EOF