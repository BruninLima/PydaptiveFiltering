#  lms.tdomain_dct.py
#
#       Implements the Transform-Domain LMS algorithm, based on the Discrete
#       Cossine Transform (DCT) Matrix, for COMPLEX valued data.
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
from scipy.fftpack import dct
from typing import Optional, Union, List, Dict
from pydaptivefiltering.base import AdaptiveFilter

class TDomainDCT(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Transform-Domain LMS algorithm, based on the Discrete
        Cossine Transform (DCT), for COMPLEX valued data.
        (Algorithm 4.4 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = True
    def __init__(
        self, 
        filter_order: int, 
        gamma: float, 
        alpha: float, 
        initial_power: float, 
        step: float = 1e-2, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order  : int (The order of the filter M)
            gamma         : float (Regularization factor to avoid singularity)
            alpha         : float (Smoothing factor for power estimation)
            initial_power : float (Initial power estimate for all bins)
            step          : float (Convergence factor mu)
            w_init        : array_like, optional (Initial coefficients)
        """
        super().__init__(filter_order, w_init)
        self.gamma: float = gamma
        self.alpha: float = alpha
        self.step: float = step
        
        self.N = filter_order + 1
        self.T = dct(np.eye(self.N), norm='ortho', axis=0)
        
        self.w_dct = self.T @ self.w
        self.power_vector = np.full(self.N, initial_power, dtype=float)
        self.w_history_dct = [self.w_dct.copy()]

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the TDomain-DCT algorithm.

        Inputs
        -------
            input_signal   : np.ndarray | list (Input vector x)
            desired_signal : np.ndarray | list (Desired vector d)
            verbose        : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs         : Store the estimated output of each iteration.
                errors          : Store the error for each iteration.
                coefficients    : Store the estimated coefficients (Original Domain).
                coefficientsDCT : Store the estimated coefficients (Transform Domain).

        Main Variables
        --------- 
            regressor    : Vector containing the tapped delay line (x_k).
            regressorDCT : DCT of the regressor (z_k = T * x_k).
            power_vector : Recursive estimate of the power in each DCT bin.

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

            regressorDCT = self.T @ self.regressor

            self.power_vector = (self.alpha * np.real(regressorDCT * regressorDCT.conj()) + 
                                (1.0 - self.alpha) * self.power_vector)

            y[k] = np.dot(self.w_dct.conj(), regressorDCT)
            e[k] = d_in[k] - y[k]

            self.w_dct = self.w_dct + self.step * e[k].conj() * regressorDCT / (self.gamma + self.power_vector)
            
            self.w = self.T.T @ self.w_dct
            
            self._record_history()
            self.w_history_dct.append(self.w_dct.copy())

        if verbose:
            print(f"[TD-LMS-DCT] Completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history,
            'coefficientsDCT': self.w_history_dct
        }

# EOF