#  rls.rls_alt.py
#
#       Implements the Alternative RLS algorithm for COMPLEX valued data.
#       RLS_Alt differs from RLS in the number of computations. The RLS_Alt
#       uses an auxiliar variable (psi) in order to reduce the computational burden.
#       (Algorithm 5.4 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@gmail.com
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

#Imports
import numpy as np
from time import time
from typing import Optional, Union, List, Dict
from pydaptivefiltering.base import AdaptiveFilter

class RLSAlt(AdaptiveFilter):
    """
    Description
    -----------
    Implements the Alternative RLS algorithm for COMPLEX valued data.
    This version (Algorithm 5.4) optimizes the computation of the inverse 
    correlation matrix using an auxiliary vector psi.
    """
    supports_complex: bool = True
    def __init__(
        self, 
        filter_order: int, 
        delta: float, 
        lamb: float, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order : int (Filter order M)
            delta        : float (Initialization: S_d(0) = delta^-1 * I)
            lamb         : float (Forgetting factor)
            w_init       : array_like, optional (Initial weights)
        """
        super().__init__(filter_order, w_init)
        self.lamb = lamb
        self.delta = delta
        self.S_d = (1.0 / delta) * np.eye(self.m + 1, dtype=complex)

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the Alternative RLS algorithm.

        Inputs
        -------
            input_signal   : np.ndarray | list (Input vector x)
            desired_signal : np.ndarray | list (Desired vector d)
            verbose        : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs            : Estimated output (a priori)
                errors             : Estimation error (a priori)
                outputs_posteriori : Estimated output (a posteriori)
                errors_posteriori  : Estimation error (a posteriori)
                coefficients       : History of estimated coefficients

        Main Variables
        --------- 
            regressor : Tapped-delay line vector.
            S_d       : Inverse of the deterministic autocorrelation matrix.
            psi       : Auxiliary vector (S_d(k-1) * regressor) to reduce computations.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom
            . Wallace Alves Martins          - wallace.wam@gmail.com
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com
            . Paulo Sergio Ramirez Diniz    - diniz@lps.ufrj.br
        """
        tic = time()
        x_in = np.asarray(input_signal, dtype=complex)
        d_in = np.asarray(desired_signal, dtype=complex)
        self._validate_inputs(x_in, d_in)

        n_iterations = d_in.size
        self.outputs = np.zeros(n_iterations, dtype=complex)
        self.errors = np.zeros(n_iterations, dtype=complex)
        outputs_post = np.zeros(n_iterations, dtype=complex)
        errors_post = np.zeros(n_iterations, dtype=complex)

        for k in range(n_iterations):
            self.regressor = np.roll(self.regressor, 1)
            self.regressor[0] = x_in[k]

            self.outputs[k] = np.dot(self.w.conj(), self.regressor)
            self.errors[k] = d_in[k] - self.outputs[k]

            psi = self.S_d @ self.regressor
            
            den = self.lamb + np.dot(self.regressor.conj().T, psi)
            
            self.S_d = (1.0 / self.lamb) * (self.S_d - np.outer(psi, psi.conj()) / den)

            self.w += self.errors[k].conj() * (self.S_d @ self.regressor)

            outputs_post[k] = np.dot(self.w.conj(), self.regressor)
            errors_post[k] = d_in[k] - outputs_post[k]
            
            self._record_history()

        if verbose:
            print(f'Total runtime {(time() - tic)*1000:.03f} ms')

        return {
            'outputs': self.outputs,
            'errors': self.errors,
            'outputs_posteriori': outputs_post,
            'errors_posteriori': errors_post,
            'coefficients': self.w_history
        }

# EOF