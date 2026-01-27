# IIR_Filters.ErrorEquation.py
#
#       Implements the Equation Error RLS algorithm for REAL valued data.
#       (Algorithm 10.3 - book: Adaptive Filtering: Algorithms and Practical
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

class ErrorEquation(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Equation Error RLS algorithm for REAL valued data.
        (Algorithm 10.3 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = False
    def __init__(
        self, 
        M: int, 
        N: int, 
        lambda_hat: float = 0.99, 
        delta: float = 1e-3, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            M          : int (Order of the numerator/zeros)
            N          : int (Order of the denominator/poles)
            lambda_hat : float (Forgetting factor)
            delta      : float (Regularization factor)
            w_init     : array_like, optional (Initial coefficients)
        """
        total_order = M + N 
        super().__init__(total_order, w_init)
        
        self.M: int = M
        self.N: int = N
        self.lambda_hat: float = lambda_hat
        self.delta: float = delta
        
        self.n_coeffs: int = M + 1 + N
        self.w: np.ndarray = np.zeros(self.n_coeffs, dtype=float)
        self.Sd: np.ndarray = (1.0 / delta) * np.eye(self.n_coeffs)
        
        self.y_buffer: np.ndarray = np.zeros(N, dtype=float)
        self.d_buffer: np.ndarray = np.zeros(N, dtype=float)

    def _stability_procedure(self, a_coeffs: np.ndarray) -> np.ndarray:
        """
        Garante a estabilidade do filtro IIR refletindo polos externos.
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
            Executes the weight update process for the Equation Error RLS algorithm.

        Inputs
        -------
            input_signal   : np.ndarray | list (Input vector x)
            desired_signal : np.ndarray | list (Desired vector d)
            verbose         : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs      : Store the estimated output of each iteration.
                errors       : Store the error for each iteration.
                coefficients : Store the estimated coefficients for each iteration.
                errors_aux   : Store the auxiliary error (e_e).

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
        e_aux: np.ndarray = np.zeros(n_samples, dtype=float)
        
        x_padded: np.ndarray = np.zeros(n_samples + self.M, dtype=float)
        x_padded[self.M:] = x

        for k in range(n_samples):
            regressor_x = x_padded[k : k + self.M + 1][::-1]
            
            regressor = np.concatenate((self.y_buffer, regressor_x))
            
            regressor_e = np.concatenate((self.d_buffer, regressor_x))
            
            y[k] = np.dot(self.w, regressor)
            y_e = np.dot(self.w, regressor_e)
            
            e[k] = d[k] - y[k]
            e_aux[k] = d[k] - y_e
            
            num = self.Sd @ np.outer(regressor_e, regressor_e) @ self.Sd
            den = self.lambda_hat + regressor_e.T @ self.Sd @ regressor_e
            self.Sd = (1.0 / self.lambda_hat) * (self.Sd - (num / den))
            
            self.w = self.w + self.Sd @ regressor_e * e_aux[k]
            
            if self.N > 0:
                self.w[:self.N] = self._stability_procedure(self.w[:self.N])
                self.y_buffer = np.concatenate(([y[k]], self.y_buffer[:-1]))
                self.d_buffer = np.concatenate(([d[k]], self.d_buffer[:-1]))
            
            self._record_history()

        if verbose:
            print(f"Error Equation Adaptation completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history,
            'errors_aux': e_aux
        }

# EOF