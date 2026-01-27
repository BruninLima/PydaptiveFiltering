# IIR_Filters.GaussNewton.py
#
#       Implements the Gauss-Newton algorithm for REAL valued data.
#       (Modified version of Algorithm 10.1 - book: Adaptive Filtering: Algorithms and 
#                                                          Practical Implementation, 3rd Ed., Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto   - guilhermepinto7@gmail.com  &  guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima   - mvsl20@gmailcom            &  markus@lps.ufrj.br
#        . Wallace Alves Martins         - wallace.wam@gmail.com      &  wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           &  wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br

# Imports
import numpy as np
from time import time
from typing import Optional, Union, List, Dict
from pydaptivefiltering.base import AdaptiveFilter
from pydaptivefiltering.utils.validation import ensure_real_signals

class GaussNewton(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Gauss-Newton algorithm for IIR filters with REAL valued data.
        (Modified version of Algorithm 10.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = False
    def __init__(
        self, 
        M: int, 
        N: int, 
        alpha: float = 0.05, 
        step: float = 1.0,
        delta: float = 1e-3, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            M      : int (Adaptive filter numerator order)
            N      : int (Adaptive filter denominator order)
            alpha  : float (Scalar factor similar to 1 - Forgetting factor)
            step   : float (Step-size mu)
            delta  : float (Regularization factor)
            w_init : array_like, optional (Initial coefficients)
        """
        total_order = M + N
        super().__init__(total_order, w_init)
        
        self.M: int = M
        self.N: int = N
        self.alpha: float = alpha
        self.step: float = step
        self.delta: float = delta
        
        self.n_coeffs: int = M + 1 + N
        self.w: np.ndarray = np.zeros(self.n_coeffs, dtype=float)
        
        self.Sd: np.ndarray = (1.0 / delta) * np.eye(self.n_coeffs)
        
        self.y_buffer: np.ndarray = np.zeros(N, dtype=float)
        
        max_buffer = max(M + 1, N)
        self.x_line_buffer: np.ndarray = np.zeros(max_buffer, dtype=float)
        self.y_line_buffer: np.ndarray = np.zeros(max_buffer, dtype=float)

    def _stability_procedure(self, a_coeffs: np.ndarray) -> np.ndarray:
        """
        Ensures IIR filter stability by reflecting poles outside the unit circle.
        """
        poly_coeffs = np.concatenate(([1], -a_coeffs))
        poles = np.roots(poly_coeffs)
        mask = np.abs(poles) > 1
        poles[mask] = 1.0 / np.conj(poles[mask])
        new_poly = np.poly(poles)
        return -np.real(new_poly[1:])

    @ensure_real_signals
    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the Gauss-Newton algorithm.

        Inputs
        -------
            input_signal   : np.ndarray | list (Input vector x)
            desired_signal : np.ndarray | list (Desired vector d)
            verbose        : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs      : Store the estimated output for each iteration.
                errors       : Store the error for each iteration.
                coefficients : Store the estimated coefficients for each iteration.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom            & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com      & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br
        """
        tic: float = time()
        
        x: np.ndarray = np.asarray(input_signal, dtype=float)
        d: np.ndarray = np.asarray(desired_signal, dtype=float)
        
        self._validate_inputs(x, d)
        n_samples: int = x.size
        
        y: np.ndarray = np.zeros(n_samples, dtype=float)
        e: np.ndarray = np.zeros(n_samples, dtype=float)
        
        x_padded: np.ndarray = np.zeros(n_samples + self.M, dtype=float)
        x_padded[self.M:] = x

        for k in range(n_samples):
            reg_x = x_padded[k : k + self.M + 1][::-1]
            regressor = np.concatenate((self.y_buffer, reg_x))
            
            y[k] = np.dot(self.w, regressor)
            e[k] = d[k] - y[k]
            
            a_coeffs = self.w[:self.N]
            
            x_line_k = x[k] + np.dot(a_coeffs, self.x_line_buffer[:self.N])
            y_line_k = 0.0
            if self.N > 0:
                prev_y = y[k-1] if k > 0 else 0.0
                y_line_k = -prev_y + np.dot(a_coeffs, self.y_line_buffer[:self.N])
            
            self.x_line_buffer = np.concatenate(([x_line_k], self.x_line_buffer[:-1]))
            self.y_line_buffer = np.concatenate(([y_line_k], self.y_line_buffer[:-1]))
            
            phi = np.concatenate((self.y_line_buffer[:self.N], -self.x_line_buffer[:self.M + 1]))
            
            inv_alpha = 1.0 - self.alpha
            num = self.Sd @ np.outer(phi, phi) @ self.Sd
            den = (inv_alpha / self.alpha) + phi.T @ self.Sd @ phi
            self.Sd = (1.0 / inv_alpha) * (self.Sd - (num / den))
            
            self.w = self.w - self.step * self.Sd @ phi * e[k]
            
            if self.N > 0:
                self.w[:self.N] = self._stability_procedure(self.w[:self.N])
                self.y_buffer = np.concatenate(([y[k]], self.y_buffer[:-1]))
            
            self._record_history()

        if verbose:
            print(f"Gauss-Newton Adaptation completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history
        }

# EOF