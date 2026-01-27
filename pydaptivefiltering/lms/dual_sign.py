#  lms.dual_sign.py
#
#       Implements the DualSign LMS algorithm for REAL valued data.
#       (Modified version of Algorithm 4.1 - book: Adaptive Filtering: Algorithms
#                                              and Practical Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

#Imports
import numpy as np
from time import time
from typing import Optional, Union, List, Dict
from pydaptivefiltering.base import AdaptiveFilter
from pydaptivefiltering.utils.validation import ensure_real_signals

class DualSign(AdaptiveFilter):
    """
    Description
    -----------
    Implements the DualSign LMS algorithm for REAL valued data.
    This algorithm uses a sign-error approach with two different gains 
    controlled by a threshold (rho) on the error magnitude.

    Parameters
    ----------
    filter_order : int
        Order of the filter (number of coefficients - 1).
    rho : float
        Error modulus threshold that decides which gain factor to use.
    gamma : int
        Gain factor (must be > 1). Usually a power of two to facilitate 
        hardware implementation.
    step : float, optional
        Convergence (relaxation) factor, also known as mu. Default is 1e-2.
    w_init : np.ndarray | list, optional
        Initial weights of the filter. If None, initialized with zeros.
    """
    supports_complex: bool = False
    def __init__(
        self, 
        filter_order: int, 
        rho: float, 
        gamma: int, 
        step: float = 1e-2, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        super().__init__(filter_order, w_init)
        self.rho: float = rho
        self.gamma: int = gamma
        self.step: float = step
        
    @ensure_real_signals
    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Executes the DualSign LMS adaptation process.

        Parameters
        ----------
        input_signal : np.ndarray | list
            Input signal vector x(k).
        desired_signal : np.ndarray | list
            Desired signal vector d(k).
        verbose : bool, optional
            If True, prints the execution time and status. Default is False.

        Returns
        -------
        dict
            A dictionary containing:
            - 'outputs': np.ndarray of the estimated output signal.
            - 'errors': np.ndarray of the estimation error signal.
            - 'coefficients': list of the weight vectors at each iteration.
        """
        tic: float = time()
        
        x_in: np.ndarray = np.asarray(input_signal, dtype=float)
        d_in: np.ndarray = np.asarray(desired_signal, dtype=float)
        
        self._validate_inputs(x_in, d_in)
        n_iterations: int = d_in.size
        
        self.errors: np.ndarray = np.zeros(n_iterations, dtype=float)
        self.outputs: np.ndarray = np.zeros(n_iterations, dtype=float)

        

        for k in range(n_iterations):
            self.regressor: np.ndarray = np.roll(self.regressor, 1)
            self.regressor[0] = x_in[k]

            self.outputs[k] = np.dot(self.w, self.regressor)

            self.errors[k] = d_in[k] - self.outputs[k]

            if np.abs(self.errors[k]) > self.rho:
                dual_sign_error: float = self.gamma * np.sign(self.errors[k])
            else:
                dual_sign_error: float = np.sign(self.errors[k])

            self.w = self.w + 2 * self.step * dual_sign_error * self.regressor
            
            self.w_history.append(self.w.copy())

        if verbose:
            runtime: float = (time() - tic) * 1000
            print(f"\n[DualSign] Optimization completed in {runtime:.3f} ms.")

        return {
            'outputs': self.outputs,
            'errors': self.errors,
            'coefficients': self.w_history
        }

# EOF