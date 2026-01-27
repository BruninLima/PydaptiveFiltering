#  set_membership.nlms.py
#
#       Implements the Set-membership Normalized LMS algorithm for COMPLEX valued data.
#       (Algorithm 6.1 - book: Adaptive Filtering: Algorithms and Practical
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

class SMNLMS(AdaptiveFilter):
    """
    Description
    -----------
    Implements the Set-membership Normalized LMS algorithm for COMPLEX valued data.
    Algorithm 6.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz.
    
    Set-membership filtering implies that the coefficients are only updated if the 
    magnitude of the error is greater than gamma_bar.
    """
    supports_complex: bool = True
    def __init__(
        self, 
        filter_order: int, 
        gamma_bar: float, 
        gamma: float, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order : int (Filter order M)
            gamma_bar    : float (Upper bound for the error modulus)
            gamma        : float (Regularization factor to avoid division by zero)
            w_init       : array_like, optional (Initial weights)
        """
        super().__init__(filter_order, w_init)
        self.gamma_bar = gamma_bar
        self.gamma = gamma
        self.n_updates = 0

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray], int]]:
        """
        Description
        -----------
            Executes the optimization process for the SM-NLMS algorithm.

        Inputs
        -------
            input_signal   : np.ndarray | list (Input vector x)
            desired_signal : np.ndarray | list (Desired vector d)
            verbose        : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs      : Estimated output of each iteration.
                errors       : Error for each iteration.
                coefficients : History of estimated coefficients.
                n_updates    : Total number of coefficient updates performed.

        Main Variables
        --------- 
            regressor : Vector containing the tapped delay line.
            gamma_bar : Error threshold for triggering updates.
            mu        : Adaptive step-size computed based on error magnitude.

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
        self.n_updates = 0

        for k in range(n_iterations):
            self.regressor = np.roll(self.regressor, 1)
            self.regressor[0] = x_in[k]

            self.outputs[k] = np.dot(self.w.conj(), self.regressor)
            self.errors[k] = d_in[k] - self.outputs[k]
            
            error_abs = np.abs(self.errors[k])

            if error_abs > self.gamma_bar:
                self.n_updates += 1
                mu = 1.0 - (self.gamma_bar / error_abs)
                
                norm_sq = np.dot(self.regressor.conj().T, self.regressor)
                den = self.gamma + norm_sq
                
                self.w += (mu / den) * (self.errors[k].conj() * self.regressor)
            
            self._record_history()

        if verbose:
            runtime = (time() - tic) * 1000
            print(f"[SM-NLMS] Completed. Updates: {self.n_updates}/{n_iterations} ({100*self.n_updates/n_iterations:.1f}%)")
            print(f"Total runtime {runtime:.03f} ms")

        return {
            'outputs': self.outputs,
            'errors': self.errors,
            'coefficients': self.w_history,
            'n_updates': self.n_updates
        }
# EOF