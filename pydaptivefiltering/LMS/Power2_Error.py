#  lms.power2_error.py
#
#       Implements the Power-of-Two Error LMS algorithm for REAL valued data.
#       (Modified version of Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical
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

class Power2ErrorLMS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Power-of-Two Error LMS algorithm for REAL valued data.
        (Modified version of Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = False
    def __init__(
        self, 
        filter_order: int, 
        bd: int, 
        tau: float, 
        step: float = 1e-2, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order : int (The order of the filter M)
            bd           : int (Word length - signal bits)
            tau          : float (Gain Factor)
            step         : float (Convergence factor mu)
            w_init       : array_like, optional (Initial coefficients)
        """
        super().__init__(filter_order, w_init)
        self.bd: int = bd
        self.tau: float = tau
        self.step: float = step

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
            Executes the weight update process for the Power-of-Two Error LMS algorithm.

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
            regressor      : Vector containing the tapped delay line (x_k).
            outputs_vector : Represents the output at iteration k (y).
            error_vector   : Represents the output errors at iteration k (e).
            power2Error    : Error value quantized to the nearest power of two.

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
        
        x_in = np.asarray(input_signal, dtype=float)
        d_in = np.asarray(desired_signal, dtype=float)

        self.w = self.w.real.astype(float)
        self.regressor = self.regressor.real.astype(float)

        self._validate_inputs(x_in, d_in)
        n_samples: int = x_in.size
        
        y = np.zeros(n_samples, dtype=float)
        e = np.zeros(n_samples, dtype=float)
        
        x_padded: np.ndarray = np.zeros(n_samples + self.m, dtype=x_in.dtype)
        x_padded[self.m:] = x_in
        d = d_in

        for k in range(n_samples):
            x_k: np.ndarray = x_padded[k : k + self.m + 1][::-1]
            
            y[k] = np.dot(self.w, x_k)
            
            e[k] = d[k] - y[k]
            
            abs_error = abs(e[k])
            if abs_error >= 1:
                power2Error = np.sign(e[k])
            elif abs_error < 2**(-self.bd + 1):
                power2Error = self.tau * np.sign(e[k])
            else:
                power2Error = (2**(np.floor(np.log2(abs_error)))) * np.sign(e[k])

            self.w = self.w + 2 * self.step * power2Error * x_k
            
            self._record_history()

        if verbose:
            print(f"Power-of-Two Error LMS completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history
        }

# EOF