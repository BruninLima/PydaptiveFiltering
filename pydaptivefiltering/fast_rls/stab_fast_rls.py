#  RLS.StabFastRLS_real.py
#
#       Implements the Stabilized Fast Transversal RLS algorithm for REAL valued data.
#       (Algorithm 8.2 - book: Adaptive Filtering: Algorithms and Practical
#                                                       Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@gmail.com
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom
#        . Wallace Alves Martins          - wallace.wam@gmail.com
#        . Luiz Wagner Pereira Biscainho  - cpneqs@gmail.com
#        . Paulo Sergio Ramirez Diniz     - diniz@lps.ufrj.br

#Imports
import numpy as np
from time import time
from typing import Optional, Union, List, Dict, Any
from pydaptivefiltering.base import AdaptiveFilter
from pydaptivefiltering.utils.validation import ensure_real_signals

class StabFastRLS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Stabilized Fast Transversal RLS algorithm for REAL valued data.
        (Algorithm 8.2 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Notes
    -----
    - This implementation is REAL-only (float64).
    - Includes numerical safeguards against division by zero / NaN:
        * denominator clamping in all critical inversions
        * floor on xiMin_f / xiMin_b
        * optional clipping on gamma values
    - Avoids per-iteration array concatenations (pre-allocation and slicing instead).

    Parameters
    ----------
        filter_order : int
            N in the textbook (adaptive filter has N+1 taps).
        forgetting_factor : float
            lambda, with 0 < lambda < 1.
        epsilon : float
            Initialization for xiMin_backward and xiMin_forward (small positive).
        kappa1, kappa2, kappa3 : float
            Stabilization constants (default: 1.5, 2.5, 1.0).
        w_init : array_like, optional
            Initial coefficients for the joint-process estimation weight vector w(k,N).

    Authors
    -------
        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@gmail.com
        . Markus Vinícius Santos Lima    - mvsl20@gmailcom
        . Wallace Alves Martins          - wallace.wam@gmail.com
        . Luiz Wagner Pereira Biscainho  - cpneqs@gmail.com
        . Paulo Sergio Ramirez Diniz     - diniz@lps.ufrj.br
    """
    supports_complex: bool = False
    def __init__(
        self,
        filter_order: int,
        forgetting_factor: float = 0.99,
        epsilon: float = 1e-1,
        kappa1: float = 1.5,
        kappa2: float = 2.5,
        kappa3: float = 1.0,
        w_init: Optional[Union[np.ndarray, list]] = None,
        denom_floor: Optional[float] = None,
        xi_floor: Optional[float] = None,
        gamma_clip: Optional[float] = None,
    ) -> None:
        super().__init__(filter_order, w_init)

        if not (0.0 < forgetting_factor < 1.0):
            raise ValueError("forgetting_factor (lambda) must satisfy 0 < lambda < 1.")
        if epsilon <= 0:
            raise ValueError("epsilon must be a small positive constant.")

        self.lambda_: float = float(forgetting_factor)
        self.epsilon: float = float(epsilon)

        self.kappa1: float = float(kappa1)
        self.kappa2: float = float(kappa2)
        self.kappa3: float = float(kappa3)

        finfo = np.finfo(np.float64)
        self.denom_floor: float = float(denom_floor) if denom_floor is not None else float(finfo.tiny * 1e3)
        self.xi_floor: float = float(xi_floor) if xi_floor is not None else float(finfo.tiny * 1e6)
        self.gamma_clip: Optional[float] = float(gamma_clip) if gamma_clip is not None else None

        # Ensure joint-process weights are real
        self.w = np.asarray(self.w, dtype=np.float64)

    @staticmethod
    def _clamp_denom(den: float, floor: float) -> float:
        """Clamp denominator away from 0 while preserving sign."""
        if not np.isfinite(den):
            # fallback to signed floor
            return np.copysign(floor, den if den != 0 else 1.0)
        if abs(den) < floor:
            return np.copysign(floor, den if den != 0 else 1.0)
        return den

    def _safe_inv(self, den: float, floor: float, clamp_counter: Dict[str, int], key: str) -> float:
        """Compute 1/den with clamping; increments clamp counter when used."""
        den2 = self._clamp_denom(den, floor)
        if den2 != den:
            clamp_counter[key] = clamp_counter.get(key, 0) + 1
        return 1.0 / den2

    def _maybe_clip_gamma(self, g: float) -> float:
        """Optionally clip gamma magnitude."""
        if self.gamma_clip is None:
            return g
        return float(np.clip(g, -self.gamma_clip, self.gamma_clip))

    @ensure_real_signals
    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Union[np.ndarray, list],
        verbose: bool = False,
        return_internal_states: bool = False,
        return_debug_info: bool = True,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray], Dict[str, Any]]]:
        """
        Description
        -----------
            Executes the weight update process for the Stabilized Fast Transversal RLS algorithm
            (Algorithm 8.2 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz).

        Inputs
        -------
            input_signal   : np.ndarray | list
                Input vector x (REAL).
            desired_signal : np.ndarray | list
                Desired vector d (REAL).
            verbose        : bool
                Verbose boolean (prints runtime).
            return_internal_states : bool
                If True, returns internal scalar trajectories (xiMin_f, xiMin_b, gamma_N_3).
            return_debug_info : bool
                If True, returns clamp counters and final internal values for diagnostics.

        Outputs
        -------
            dictionary:
                outputs            : Store the estimated output y(k) for each iteration (REAL).
                priori_errors      : Store the a priori error e(k) for each iteration (REAL).
                posteriori_errors  : Store the a posteriori error ε(k) for each iteration (REAL).
                coefficients       : Store the estimated coefficients for each iteration (list of arrays).
                internal_states    : (optional) trajectories of internal scalars.
                debug_info         : (optional) clamp counters and final internal scalar values.

        Main Variables
        --------------
            regressor (r)     : Vector containing the tapped delay line with MATLAB-matching reversal (length N+2).
            w (self.w)        : Joint-process coefficient vector (length N+1).
            w_f, w_b          : Forward/backward predictor coefficient vectors (length N+1).
            phiHatN, phiHatNp1: Gain-related vectors in Algorithm 8.2.
            gamma_*           : Scalar normalization terms (susceptible to numerical issues).
            xiMin_*           : Scalar forward/backward minimum prediction error powers (must remain positive).

        Numerical Safeguards
        --------------------
            - All critical inversions are protected by denominator clamping:
                * gamma_Np1_1 update
                * xiMin_f update
                * gamma_N_2 update
                * gamma_N_3 update
            - xiMin_f and xiMin_b are floored to remain positive.
            - Optional |gamma| clipping via gamma_clip (disabled by default).

        Misc Variables
        --------------
            tic               : Initial time for runtime calculation.
            n_samples         : Number of iterations based on signal size.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@gmail.com
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom
            . Wallace Alves Martins          - wallace.wam@gmail.com
            . Luiz Wagner Pereira Biscainho  - cpneqs@gmail.com
            . Paulo Sergio Ramirez Diniz     - diniz@lps.ufrj.br
        """
        tic = time()

        x = np.asarray(input_signal)
        d = np.asarray(desired_signal)

        if np.iscomplexobj(x):
            x = np.real(x)
        if np.iscomplexobj(d):
            d = np.real(d)

        x = x.astype(np.float64, copy=False)
        d = d.astype(np.float64, copy=False)

        self._validate_inputs(x, d)
        n_samples = x.size

        n_taps = self.m + 1       # N+1
        reg_len = self.m + 2      # N+2 (matches MATLAB)

        y = np.zeros(n_samples, dtype=np.float64)
        e_priori = np.zeros(n_samples, dtype=np.float64)
        e_post = np.zeros(n_samples, dtype=np.float64)

        xiMin_f = float(self.epsilon)
        xiMin_b = float(self.epsilon)

        gamma_Np1_1 = 0.0
        gamma_N_2 = 0.0
        gamma_N_3 = 1.0

        w_f = np.zeros(n_taps, dtype=np.float64)
        w_b = np.zeros(n_taps, dtype=np.float64)

        phiHatN = np.zeros(n_taps, dtype=np.float64)
        phiHatNp1 = np.zeros(reg_len, dtype=np.float64)  

        self.w = np.asarray(self.w, dtype=np.float64).reshape(-1)
        if self.w.size != n_taps:
            raise ValueError(f"w_init must have length {n_taps} (got {self.w.size}).")

        x_padded = np.zeros(n_samples + n_taps, dtype=np.float64)
        x_padded[n_taps:] = x

        lam = self.lambda_
        k1, k2, k3 = self.kappa1, self.kappa2, self.kappa3

        clamp_counter: Dict[str, int] = {}  

        internal: Dict[str, Any] = {}
        if return_internal_states:
            internal = {
                "xiMin_f": np.zeros(n_samples, dtype=np.float64),
                "xiMin_b": np.zeros(n_samples, dtype=np.float64),
                "gamma_N_3": np.zeros(n_samples, dtype=np.float64),
            }

        for k in range(n_samples):
            r = x_padded[k: k + reg_len][::-1].copy()

            error_f_line = float(r[0] - np.dot(w_f, r[1:]))

            error_f = float(error_f_line * gamma_N_3)

            phiHatNp1[0] = 0.0
            phiHatNp1[1:] = phiHatN

            den_scale = lam * xiMin_f
            scale = self._safe_inv(den_scale, self.denom_floor, clamp_counter, "inv(lam*xiMin_f)")

            phiHatNp1[0] += scale * error_f_line
            phiHatNp1[1:] += scale * (-w_f) * error_f_line

            inv_gammaN3 = self._safe_inv(gamma_N_3, self.denom_floor, clamp_counter, "inv(gamma_N_3)")
            den_gNp1 = inv_gammaN3 + phiHatNp1[0] * error_f_line
            gamma_Np1_1 = self._safe_inv(den_gNp1, self.denom_floor, clamp_counter, "inv(den_gamma_Np1_1)")
            gamma_Np1_1 = self._maybe_clip_gamma(gamma_Np1_1)

            inv_xi_f_lam = self._safe_inv(xiMin_f * lam, self.denom_floor, clamp_counter, "inv(xiMin_f*lam)")
            den_xi_f = inv_xi_f_lam - gamma_Np1_1 * (phiHatNp1[0] ** 2)
            xiMin_f = self._safe_inv(den_xi_f, self.denom_floor, clamp_counter, "inv(den_xiMin_f)")
            if xiMin_f < self.xi_floor:
                xiMin_f = self.xi_floor
                clamp_counter["floor(xiMin_f)"] = clamp_counter.get("floor(xiMin_f)", 0) + 1

            w_f += phiHatN * error_f

            error_b_line_1 = float(lam * xiMin_b * phiHatNp1[-1])
            error_b_line_2 = float(-np.dot(w_b, r[:-1]) + r[-1])

            eb3_line_1 = error_b_line_2 * k1 + error_b_line_1 * (1.0 - k1)
            eb3_line_2 = error_b_line_2 * k2 + error_b_line_1 * (1.0 - k2)
            eb3_line_3 = error_b_line_2 * k3 + error_b_line_1 * (1.0 - k3)

            inv_gammaNp1 = self._safe_inv(gamma_Np1_1, self.denom_floor, clamp_counter, "inv(gamma_Np1_1)")
            den_gN2 = inv_gammaNp1 - phiHatNp1[-1] * eb3_line_3
            gamma_N_2 = self._safe_inv(den_gN2, self.denom_floor, clamp_counter, "inv(den_gamma_N_2)")
            gamma_N_2 = self._maybe_clip_gamma(gamma_N_2)

            eb3_1 = eb3_line_1 * gamma_N_2
            eb3_2 = eb3_line_2 * gamma_N_2

            xiMin_b = lam * xiMin_b + eb3_2 * eb3_line_2
            if xiMin_b < self.xi_floor:
                xiMin_b = self.xi_floor
                clamp_counter["floor(xiMin_b)"] = clamp_counter.get("floor(xiMin_b)", 0) + 1


            phiHatN = phiHatNp1[:-1] + phiHatNp1[-1] * w_b

            w_b += phiHatN * eb3_1

            den_gN3 = 1.0 + float(np.dot(phiHatN, r[:-1]))
            gamma_N_3 = self._safe_inv(den_gN3, self.denom_floor, clamp_counter, "inv(den_gamma_N_3)")
            gamma_N_3 = self._maybe_clip_gamma(gamma_N_3)

            y[k] = float(np.dot(self.w, r[:-1]))
            e_priori[k] = float(d[k] - y[k])
            e_post[k] = float(e_priori[k] * gamma_N_3)

            self.w = self.w + phiHatN * e_post[k]
            self._record_history()

            if return_internal_states:
                internal["xiMin_f"][k] = xiMin_f
                internal["xiMin_b"][k] = xiMin_b
                internal["gamma_N_3"][k] = gamma_N_3

        out: Dict[str, Union[np.ndarray, List[np.ndarray], Dict[str, Any]]] = {
            "outputs": y,
            "priori_errors": e_priori,
            "posteriori_errors": e_post,
            "coefficients": self.w_history,
        }

        if return_internal_states:
            out["internal_states"] = internal

        if return_debug_info:
            out["debug_info"] = {
                "clamp_counter": clamp_counter,
                "final_values": {
                    "xiMin_f": float(xiMin_f),
                    "xiMin_b": float(xiMin_b),
                    "gamma_N_3": float(gamma_N_3),
                    "gamma_Np1_1": float(gamma_Np1_1),
                    "gamma_N_2": float(gamma_N_2),
                },
                "settings": {
                    "denom_floor": float(self.denom_floor),
                    "xi_floor": float(self.xi_floor),
                    "gamma_clip": None if self.gamma_clip is None else float(self.gamma_clip),
                },
            }

        if verbose:
            print(f"StabFastRLS (REAL) completed in {(time() - tic)*1000:.03f} ms")

        return out


# EOF
