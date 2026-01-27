# blind.cma.py
#
#       Implements the Constant-Modulus algorithm for COMPLEX valued data.
#       (Algorithm 13.2 - book: Adaptive Filtering: Algorithms and Practical
#                                                        Implementation, Diniz)
#
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto   - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima   - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins         - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com          & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

# Imports
from __future__ import annotations

import numpy as np
from time import time
from typing import Optional, Union, List, Dict

from pydaptivefiltering.base import AdaptiveFilter

ArrayLike = Union[np.ndarray, list]

class CMA(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Constant-Modulus Algorithm (CMA) for blind adaptive 
        filtering with complex or real valued data.
        (Algorithm 13.2 - book: Adaptive Filtering: Algorithms and Practical
        Implementation, Diniz)

    Attributes
    ----------
        supports_complex : bool
            True (The algorithm supports complex-valued data).
    """

    supports_complex: bool = True

    def __init__(
        self,
        filter_order: int = 5,
        step: float = 0.01,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Inputs
        -------
            filter_order : int
                Order of the FIR filter (N). Number of coefficients is filter_order + 1.
            step : float
                Convergence (relaxation) factor (mu).
            w_init : array_like, optional
                Initial filter coefficients. If None, initialized with zeros.
        """
        super().__init__(filter_order, w_init=w_init)
        self.step = float(step)
        self.n_coeffs = int(filter_order + 1)

    def optimize(
        self,
        input_signal: ArrayLike,
        verbose: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the adaptation process for the CMA algorithm.
            As a blind algorithm, it targets a constant modulus property
            of the output signal rather than a known desired signal.

        Inputs
        -------
            input_signal : np.ndarray | list
                Signal fed into the adaptive filter.
            verbose : bool
                Verbose boolean.

        Outputs
        -------
            dictionary:
                outputs : np.ndarray
                    Estimated output y[n] of each iteration.
                errors : np.ndarray
                    CMA error e[n] for each iteration.
                coefficients : list[np.ndarray]
                    History of estimated coefficient vectors.

        Authors
        -------
            . Guilherme de Oliveira Pinto   - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
            . Markus Vinícius Santos Lima   - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins         - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com          & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br
        """
        tic = time()

        x = np.asarray(input_signal).reshape(-1)
        n_iterations = int(x.size)
        
        desired_level = np.mean(np.abs(x)**4) / np.mean(np.abs(x)**2)

        y = np.zeros(n_iterations, dtype=x.dtype)
        e = np.zeros(n_iterations, dtype=x.dtype)
        self.w_history = []

        x_state = np.zeros(self.n_coeffs, dtype=x.dtype)

        for it in range(n_iterations):
            x_state[1:] = x_state[:-1]
            x_state[0] = x[it]

            self.w_history.append(self.w.copy())

            y[it] = np.dot(np.conj(self.w), x_state)

            e[it] = np.abs(y[it])**2 - desired_level

            phi = 2.0 * e[it] * np.conj(y[it])
            
            self.w = self.w - self.step * phi * x_state

        if verbose:
            print(f"CMA Adaptation completed in {(time() - tic) * 1000:.03f} ms")

        return {
            "outputs": y,
            "errors": e,
            "coefficients": self.w_history,
        }
# EOF