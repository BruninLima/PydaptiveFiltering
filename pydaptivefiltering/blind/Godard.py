# blind.godard.py
#
#       Implements the Godard algorithm for COMPLEX valued data.
#       (Algorithm 13.1 - book: Adaptive Filtering: Algorithms and Practical
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

class Godard(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Godard algorithm for blind adaptive filtering 
        with complex or real valued data.
        (Algorithm 13.1 - book: Adaptive Filtering: Algorithms and Practical
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
        p_exponent: int = 2,
        q_exponent: int = 2,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Inputs
        -------
            filter_order : int
                Order of the FIR filter (N). Number of coefficients is filter_order + 1.
            step : float
                Convergence (relaxation) factor (mu).
            p_exponent : int
                Godard-error's exponent (p).
            q_exponent : int
                Exponent used to define the desired output level (q).
            w_init : array_like, optional
                Initial filter coefficients. If None, initialized with zeros.
        """
        super().__init__(filter_order, w_init=w_init)
        self.step = float(step)
        self.p = int(p_exponent)
        self.q = int(q_exponent)
        self.n_coeffs = int(filter_order + 1)

    def optimize(
        self,
        input_signal: ArrayLike,
        verbose: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the adaptation process for the Godard algorithm.
            This is a blind equalization algorithm, thus it does not require 
            a desired signal.

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
                    Godard error e[n] for each iteration.
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
        
        desired_level = np.mean(np.abs(x)**(2 * self.q)) / np.mean(np.abs(x)**self.q)

        y = np.zeros(n_iterations, dtype=x.dtype)
        e = np.zeros(n_iterations, dtype=x.dtype)
        self.w_history = []

        x_state = np.zeros(self.n_coeffs, dtype=x.dtype)

        for it in range(n_iterations):
            x_state[1:] = x_state[:-1]
            x_state[0] = x[it]

            self.w_history.append(self.w.copy())

            y[it] = np.dot(np.conj(self.w), x_state)

            e[it] = np.abs(y[it])**self.q - desired_level

            phi = (self.p * self.q * (e[it]**(self.p - 1)) * (np.abs(y[it])**(self.q - 2)) * np.conj(y[it]))
            
            self.w = self.w - (self.step * phi * x_state) / 2.0

        if verbose:
            print(f"Godard Adaptation completed in {(time() - tic) * 1000:.03f} ms")

        return {
            "outputs": y,
            "errors": e,
            "coefficients": self.w_history,
        }
# EOF