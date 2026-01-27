#  lms.tdomain.py
#
#       Implements the Transform-Domain LMS algorithm for COMPLEX valued data.
#       (Algorithm 4.4 - book: Adaptive Filtering: Algorithms and Practical
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

class TDomainLMS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Transform-Domain LMS algorithm for COMPLEX valued data.
        (Algorithm 4.4 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = True
    def __init__(
        self, 
        filter_order: int, 
        gamma: float, 
        alpha: float, 
        initial_power: float, 
        transform_matrix: np.ndarray, 
        step: float = 1e-2, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order     : int (The order of the filter M)
            gamma            : float (Small positive constant to avoid singularity)
            alpha            : float (Smoothing factor for power estimation: 0 < alpha < 0.1)
            initial_power    : float (Initial power estimate for all bins)
            transform_matrix : np.ndarray (Unitary matrix T of size M+1 x M+1)
            step             : float (Convergence factor mu)
            w_init           : array_like, optional (Initial coefficients)
        """
        super().__init__(filter_order, w_init)
        self.gamma: float = gamma
        self.alpha: float = alpha
        self.step: float = step
        self.T: np.ndarray = np.asarray(transform_matrix, dtype=complex)

        self.power_vector: np.ndarray = np.full(filter_order + 1, initial_power, dtype=float)

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the TD-LMS algorithm.

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
                coefficients : Store the estimated coefficients (transform domain).

        Main Variables
        --------- 
            regressor      : Vector containing the tapped delay line (x_k).
            regressorT     : Transformed regressor (z_k = T * x_k).
            power_vector   : Recursive estimate of the power in each transform bin.
            error_it       : Estimation error at iteration k.

        Misc Variables
        --------------
            tic            : Initial time for runtime calculation.
            n_samples      : Number of iterations based on signal size.

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
        
        x_in: np.ndarray = np.asarray(input_signal, dtype=complex)
        d_in: np.ndarray = np.asarray(desired_signal, dtype=complex)

        self._validate_inputs(x_in, d_in)
        n_samples: int = d_in.size
        
        y: np.ndarray = np.zeros(n_samples, dtype=complex)
        e: np.ndarray = np.zeros(n_samples, dtype=complex)

        

        for k in range(n_samples):
            self.regressor = np.roll(self.regressor, 1)
            self.regressor[0] = x_in[k]

            regressorT = self.T @ self.regressor

            self.power_vector = (self.alpha * np.real(regressorT * regressorT.conj()) + 
                                (1.0 - self.alpha) * self.power_vector)

            y[k] = np.dot(self.w.conj(), regressorT)

            e[k] = d_in[k] - y[k]

            self.w = self.w + self.step * e[k].conj() * regressorT / (self.gamma + self.power_vector)
            
            self._record_history()

        if verbose:
            runtime: float = (time() - tic) * 1000
            print(f"[TD-LMS] Adaptation completed in {runtime:.3f} ms.")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history
        }
# EOF