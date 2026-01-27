#  nonlinear.Bilinear_RLS.py
#
#       Implements the Bilinear RLS algorithm for REAL valued data.
#       (Algorithm 11.3 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

# Imports
import numpy as np
from time import time
from typing import Optional, Union, List, Dict
from pydaptivefiltering.base import AdaptiveFilter
from pydaptivefiltering.utils.validation import ensure_real_signals

class BilinearRLS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Bilinear RLS algorithm for REAL valued data.
        (Algorithm 11.3 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = False
    def __init__(
        self, 
        filter_order: int = 4, 
        forgetting_factor: float = 0.98, 
        delta: float = 1.0,
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order             : int (Number of coefficients in the bilinear regressor, default 4)
            forgetting_factor        : float (Exponential weighting factor/forgetting factor)
            delta                    : float (Regularization factor for P matrix initialization)
            w_init                   : array_like, optional (Initial coefficients)
        """
        super().__init__(filter_order - 1, w_init)
        self.forgetting_factor: float = forgetting_factor
        self.delta: float = delta
        
        # P matrix initialization (Sd in your Matlab code)
        self.P: np.ndarray = np.eye(filter_order) / delta

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
            Executes the weight update process for the Bilinear RLS algorithm.

        Inputs
        -------
            input_signal   : np.ndarray | list (Input vector x)
            desired_signal : np.ndarray | list (Desired vector d)
            verbose        : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs      : Store the estimated output of each iteration.
                errors       : Store the error for each iteration.
                coefficients : Store the estimated coefficients for each iteration.

        Main Variables
        --------- 
            uxl            : Bilinear regressor [x(k), d(k-1), x(k)d(k-1), x(k-1)d(k-1)]'.
            P              : Inverse of the correlation matrix (Sd).
            y              : Output at iteration k.
            e              : Output error at iteration k.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br
        """
        tic: float = time()
        
        x: np.ndarray = np.asarray(input_signal)
        d_ref: np.ndarray = np.asarray(desired_signal)

        self._validate_inputs(x, d_ref)
        n_samples: int = x.size
        
        y: np.ndarray = np.zeros(n_samples, dtype=complex)
        e: np.ndarray = np.zeros(n_samples, dtype=complex)
        
        x_prev = 0.0
        d_prev = 0.0

        for k in range(n_samples):
            # Construção do regressor uxl
            uxl = np.array([
                x[k], 
                d_prev, 
                x[k] * d_prev, 
                x_prev * d_prev
            ], dtype=complex)
            
            # Erro a priori: d(k) - w(k)' * uxl
            # Usando .conj() para manter o padrão Diniz de números complexos
            e_priori = d_ref[k] - np.dot(self.w.conj(), uxl)
            
            # Atualização da matriz P (ganho)
            psi = np.dot(self.P, uxl)
            den = self.forgetting_factor + np.dot(uxl.conj(), psi)
            self.P = (1.0 / self.forgetting_factor) * (self.P - np.outer(psi, psi.conj()) / den)
            
            # Atualização dos pesos usando o ganho a posteriori
            k_gain = np.dot(self.P, uxl)
            self.w = self.w + e_priori * k_gain
            
            # Saída e erro atuais
            y[k] = np.dot(self.w.conj(), uxl)
            e[k] = d_ref[k] - y[k]
            
            # Atualização das memórias para k+1
            x_prev = x[k]
            d_prev = d_ref[k]
            
            self._record_history()

        if verbose:
            print(f"Bilinear RLS Adaptation completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history
        }
# EOF