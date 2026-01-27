# iir.steiglitz_mcbride.py
#
#       Implements the Steiglitz-McBride algorithm for REAL valued data.
#       (Algorithm 10.4 - book: Adaptive Filtering: Algorithms and Practical
#                                                       Implementation, 3rd Ed., Diniz)
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

class SteiglitzMcBride(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Steiglitz-McBride algorithm for REAL valued data.
        (Algorithm 10.4 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = False
    def __init__(
        self, 
        M: int, 
        N: int, 
        step: float = 1e-3, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            M      : int (Adaptive filter numerator order)
            N      : int (Adaptive filter denominator order)
            step   : float (Step-size mu)
            w_init : array_like, optional (Initial coefficients)
        """
        total_order = M + N + 1
        super().__init__(total_order, w_init)
        
        self.M: int = M
        self.N: int = N
        self.step: float = step
        
        self.n_coeffs: int = M + 1 + N
        self.w: np.ndarray = np.zeros(self.n_coeffs, dtype=float)
        
        self.y_buffer: np.ndarray = np.zeros(N, dtype=float)
        
        max_buffer = max(M + 1, N + 1)
        self.xf_buffer: np.ndarray = np.zeros(max_buffer, dtype=float)
        self.df_buffer: np.ndarray = np.zeros(max_buffer, dtype=float)

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
            Executes the weight update process for the Steiglitz-McBride algorithm.

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
                errors_s     : Store the auxiliary error (errorVector_s).

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
        e_s: np.ndarray = np.zeros(n_samples, dtype=float)
        
        x_padded: np.ndarray = np.zeros(n_samples + self.M, dtype=float)
        x_padded[self.M:] = x

        for k in range(n_samples):
            reg_x = x_padded[k : k + self.M + 1][::-1]
            regressor = np.concatenate((self.y_buffer, reg_x))
            
            y[k] = np.dot(self.w, regressor)
            e[k] = d[k] - y[k]
            
            a_coeffs = self.w[:self.N]
            
            xf_k = x[k] + np.dot(a_coeffs, self.xf_buffer[:self.N])
            df_k = d[k] + np.dot(a_coeffs, self.df_buffer[:self.N])
            
            self.xf_buffer = np.concatenate(([xf_k], self.xf_buffer[:-1]))
            self.df_buffer = np.concatenate(([df_k], self.df_buffer[:-1]))
            
            regressor_s = np.concatenate((self.df_buffer[1:self.N+1], self.xf_buffer[:self.M+1]))
            if self.N == 0:
                regressor_s = self.xf_buffer[:self.M+1]
            else:
                regressor_s = np.concatenate((self.df_buffer[1:self.N+1], self.xf_buffer[:self.M+1]))

            e_s[k] = df_k - np.dot(self.w, regressor_s)
            
            self.w = self.w + 2 * self.step * regressor_s * e[k]
            
            if self.N > 0:
                self.w[:self.N] = self._stability_procedure(self.w[:self.N])
                self.y_buffer = np.concatenate(([y[k]], self.y_buffer[:-1]))
            
            self._record_history()

        if verbose:
            print(f"Steiglitz-McBride completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history,
            'errors_s': e_s
        }

# EOF