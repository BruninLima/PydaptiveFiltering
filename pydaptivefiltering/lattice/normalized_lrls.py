# lattice.normalized_lrls.py
#
#      Implements the Normalized Lattice RLS algorithm based on a posteriori error.
#      (Algorithm 7.6 - book: Adaptive Filtering: Algorithms and Practical
#                               Implementation, Diniz)
#
#      Authors:
#       . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#       . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#       . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#       . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#       . Paulo Sergio Ramirez Diniz     -                             diniz@lps.ufrj.br

# Imports
import numpy as np
from time import time
from typing import Optional, Union, Dict, List
from pydaptivefiltering.base import AdaptiveFilter


class NormalizedLRLS(AdaptiveFilter):
    """
    Normalized Lattice RLS (NLRLS) algorithm based on a posteriori error.

    Implements Algorithm 7.6 (Diniz), which is designed to have superior numerical
    properties: internal normalized variables (errors and reflection coefficients)
    are magnitude-bounded by unity, improving stability.

    Notes on implementation / conventions
    -------------------------------------
    - Complex-valued implementation (supports_complex = True).
    - This algorithm is not a standard FIR-tapped delay line; it uses a lattice
      prediction stage and a ladder-like estimation stage (here expressed via rho_v).
    - For library consistency, we map the "main coefficient vector" to self.w
      by using rho_v as a coefficient-like vector (length M+1). This provides a
      coherent API (w, w_history, returned coefficients) across pydaptivefiltering.

    Returned dictionary follows the project convention:
        outputs, errors, coefficients

    Attributes
    ----------
    lam : float
        Forgetting factor λ.
    epsilon : float
        Regularization term used in normalization and safety denominators.
    n_sections : int
        Number of lattice sections M.
    rho : np.ndarray
        Normalized reflection coefficient vector for prediction stage (length M).
    rho_v : np.ndarray
        Normalized coefficient-like vector for estimation stage (length M+1).
        Used as self.w for consistency and recorded in w_history.
    bar_b_prev : np.ndarray
        Previous backward normalized errors (length M+1).
    xi_half : float
        Square-root energy term (scalar) used for normalization of x and e.
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int,
        lambda_factor: float = 0.99,
        epsilon: float = 1e-6,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Inputs
        -------
            filter_order  : int (The order of the filter M)
            lambda_factor : float (Forgetting factor Lambda)
            epsilon       : float (Regularization factor for initialization)
            w_init        : array_like, optional (Initial coefficients)

        Notes
        -----
        - The normalized lattice recursion uses rho and rho_v. If w_init is provided,
          we interpret it as an initialization for rho_v (length M+1). This provides
          a consistent "coefficient vector" interpretation inside the library.
        """
        super().__init__(filter_order, w_init)
        self.lam = float(lambda_factor)
        self.epsilon = float(epsilon)
        self.n_sections = int(filter_order)

        self.rho = np.zeros(self.n_sections, dtype=complex)

        if w_init is not None:
            self.rho_v = np.asarray(w_init, dtype=complex).reshape(-1)
            if self.rho_v.size != self.n_sections + 1:
                raise ValueError(
                    f"w_init must have length {self.n_sections + 1}, got {self.rho_v.size}"
                )
        else:
            self.rho_v = np.zeros(self.n_sections + 1, dtype=complex)

        self.bar_b_prev = np.zeros(self.n_sections + 1, dtype=complex)

        self.xi_half = float(np.sqrt(self.epsilon))

        self._tiny = 1e-12

        self.w = self.rho_v.copy()
        self.w_history = []
        self._record_history()

    def _safe_sqrt(self, value: float) -> float:
        """
        Ensures the value for sqrt is not negative due to precision errors.

        Parameters
        ----------
        value : float
            Value that should be >= 0, but may become slightly negative due to
            floating-point rounding.

        Returns
        -------
        float
            sqrt(max(value, 0.0))
        """
        return float(np.sqrt(max(0.0, float(value))))

    def optimize(
        self,
        input_signal: Union[np.ndarray, list],
        desired_signal: Union[np.ndarray, list],
        verbose: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the Normalized Lattice RLS (NLRLS)
            algorithm (Algorithm 7.6 - Diniz), based on a posteriori error.

        Inputs
        -------
            input_signal   : np.ndarray | list
                Input vector x (length N).
            desired_signal : np.ndarray | list
                Desired vector d (length N).
            verbose        : bool
                Verbose boolean (prints runtime if True).

        Outputs
        -------
            dictionary:
                outputs      : np.ndarray
                    Estimated output y[k] for each iteration.
                errors       : np.ndarray
                    A posteriori error e[k] for each iteration.
                coefficients : List[np.ndarray]
                    History of coefficient vectors (here, rho_v exposed as self.w).

        Main Variables (normalized lattice)
        -----------------------------------
            xi_half   : sqrt energy accumulator for normalization of x and e.
            bar_f     : normalized forward error
            bar_b     : normalized backward errors vector
            rho       : normalized reflection coefficients for prediction stage
            bar_e     : normalized a posteriori estimation error
            rho_v     : normalized "ladder-like" coefficients for estimation stage

        Numerical Notes
        ---------------
        - All internal normalized variables are intended to be magnitude-bounded by 1.
        - Additional epsilon and tiny constants are used to avoid division by zero and
          to protect against negative sqrt arguments due to rounding.

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

        x_in = np.asarray(input_signal, dtype=complex).reshape(-1)
        d_in = np.asarray(desired_signal, dtype=complex).reshape(-1)
        self._validate_inputs(x_in, d_in)

        n_samples = d_in.size
        y = np.zeros(n_samples, dtype=complex)
        e = np.zeros(n_samples, dtype=complex)

        sqrt_lam = float(np.sqrt(self.lam))

        for k in range(n_samples):

            xi_sq = self.xi_half * self.xi_half
            xi_sq = self.lam * xi_sq + (np.abs(x_in[k]) ** 2) + self.epsilon
            self.xi_half = float(np.sqrt(max(xi_sq, self._tiny)))

            bar_f = x_in[k] / (self.xi_half + self.epsilon)
            abs_bf = np.abs(bar_f)
            if abs_bf > 1.0:
                bar_f = bar_f / (abs_bf + self.epsilon)

            bar_b_curr = np.zeros(self.n_sections + 1, dtype=complex)
            bar_b_curr[0] = bar_f

            for m in range(self.n_sections):
                cos_f = self._safe_sqrt(1.0 - (np.abs(bar_f) ** 2))
                cos_b_prev = self._safe_sqrt(1.0 - (np.abs(self.bar_b_prev[m]) ** 2))

                self.rho[m] = (
                    sqrt_lam * cos_f * cos_b_prev * self.rho[m]
                    + np.conj(bar_f) * self.bar_b_prev[m]
                )

                abs_rho = np.abs(self.rho[m])
                if abs_rho >= 1.0:
                    self.rho[m] = self.rho[m] / (abs_rho + self.epsilon)

                cos_rho = self._safe_sqrt(1.0 - (np.abs(self.rho[m]) ** 2))

                denom_f = (cos_rho * cos_b_prev) + self.epsilon
                denom_b = (cos_rho * cos_f) + self.epsilon

                f_next = (bar_f - self.rho[m] * self.bar_b_prev[m]) / denom_f
                b_next = (self.bar_b_prev[m] - np.conj(self.rho[m]) * bar_f) / denom_b

                bar_f = f_next
                bar_b_curr[m + 1] = b_next

            bar_e = d_in[k] / (self.xi_half + self.epsilon)
            abs_be = np.abs(bar_e)
            if abs_be > 1.0:
                bar_e = bar_e / (abs_be + self.epsilon)

            for m in range(self.n_sections + 1):
                cos_e = self._safe_sqrt(1.0 - (np.abs(bar_e) ** 2))
                cos_b = self._safe_sqrt(1.0 - (np.abs(bar_b_curr[m]) ** 2))

                self.rho_v[m] = (
                    sqrt_lam * cos_e * cos_b * self.rho_v[m]
                    + np.conj(bar_e) * bar_b_curr[m]
                )

                abs_rv = np.abs(self.rho_v[m])
                if abs_rv >= 1.0:
                    self.rho_v[m] = self.rho_v[m] / (abs_rv + self.epsilon)

                cos_rho_v = self._safe_sqrt(1.0 - (np.abs(self.rho_v[m]) ** 2))

                denom = (cos_rho_v * cos_b) + self.epsilon
                bar_e = (bar_e - self.rho_v[m] * bar_b_curr[m]) / denom

            e[k] = bar_e * self.xi_half
            y[k] = d_in[k] - e[k]

            self.bar_b_prev = bar_b_curr.copy()

            self.w = self.rho_v.copy()
            self._record_history()

        if verbose:
            print(f"[NLRLS] Completed in {(time() - tic) * 1000:.02f} ms")

        return {
            "outputs": y,
            "errors": e,
            "coefficients": self.w_history,
        }
# EOF