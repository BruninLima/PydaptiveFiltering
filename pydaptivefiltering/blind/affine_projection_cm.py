# blind.affine_projection_cm.py
#
#       Implements the Complex Affine-Projection Constant-Modulus algorithm 
#       for COMPLEX valued data.
#       (Algorithm 13.4 - book: Adaptive Filtering: Algorithms and Practical
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

from __future__ import annotations

import numpy as np
from time import time
from typing import Optional, Union, List, Dict

from pydaptivefiltering.base import AdaptiveFilter

ArrayLike = Union[np.ndarray, list]

class AffineProjectionCM(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Affine-Projection Constant-Modulus (AP-CM) algorithm 
        for blind adaptive filtering with complex or real valued data.
        (Algorithm 13.4 - book: Adaptive Filtering: Algorithms and Practical
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
        step: float = 0.1,
        memory_length: int = 2,
        gamma: float = 1e-6,
        w_init: Optional[Union[np.ndarray, list]] = None,
    ) -> None:
        """
        Inputs
        -------
            filter_order : int
                Order of the FIR filter (N). Number of coefficients is filter_order + 1.
            step : float
                Convergence (relaxation) factor (mu).
            memory_length : int
                Reuse data factor (referred as L in the textbook).
            gamma : float
                Regularization factor to avoid singularity in matrix inversion.
            w_init : array_like, optional
                Initial filter coefficients. If None, initialized with zeros.
        """
        super().__init__(filter_order, w_init=w_init)
        self.step = float(step)
        self.L = int(memory_length)
        self.gamma = float(gamma)
        self.n_coeffs = int(filter_order + 1)

    def optimize(
        self,
        input_signal: ArrayLike,
        verbose: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the adaptation process for the AP-CM algorithm.
            This algorithm uses multiple past regressors to accelerate 
            the blind equalization process.

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
                    Estimated output y[n] (first element of the output vector).
                errors : np.ndarray
                    Blind error e[n] (first element of the error vector).
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
        
        y_vec = np.zeros(n_iterations, dtype=x.dtype)
        e_vec = np.zeros(n_iterations, dtype=x.dtype)
        self.w_history = []

        regressor_matrix = np.zeros((self.n_coeffs, self.L + 1), dtype=x.dtype)
        
        I_reg = self.gamma * np.eye(self.L + 1)

        for it in range(n_iterations):
            regressor_matrix[:, 1:] = regressor_matrix[:, :-1]
            
            current_x = np.zeros(self.n_coeffs, dtype=x.dtype)
            for i in range(self.n_coeffs):
                if it - i >= 0:
                    current_x[i] = x[it - i]
            regressor_matrix[:, 0] = current_x

            self.w_history.append(self.w.copy())

            output_ap = np.dot(np.conj(regressor_matrix).T, self.w)
            
            desired_level_conj = np.zeros_like(output_ap)
            for i in range(len(output_ap)):
                if np.abs(output_ap[i]) > 0:
                    desired_level_conj[i] = output_ap[i] / np.abs(output_ap[i])
            
            error_ap = desired_level_conj - output_ap

            inv_part = np.dot(np.conj(regressor_matrix).T, regressor_matrix) + I_reg
            update_factor = np.linalg.solve(inv_part, error_ap)
            
            self.w = self.w + self.step * np.dot(regressor_matrix, update_factor)

            y_vec[it] = output_ap[0]
            e_vec[it] = error_ap[0]

        if verbose:
            print(f"AP-CM Adaptation completed in {(time() - tic) * 1000:.03f} ms")

        return {
            "outputs": y_vec,
            "errors": e_vec,
            "coefficients": self.w_history,
        }
# EOF