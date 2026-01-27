#  LMS.AffineProjection.py
#
#       Implements the Complex Affine-Projection algorithm for COMPLEX valued data.
#       (Algorithm 4.6 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
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
from pydaptivefiltering.main import AdaptiveFilter

class AffineProjection(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Complex Affine-Projection algorithm for COMPLEX valued data.
        (Algorithm 4.6 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """

    def __init__(
        self, 
        filter_order: int, 
        step: float = 1e-2, 
        gamma: float = 1e-6, 
        L: int = 2, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            filter_order : int
                The order of the filter (M).
            step : float
                Convergence (relaxation) factor (mu).
            gamma : float
                Regularization factor to ensure matrix invertibility.
            L : int
                Data reuse factor (number of vectors in the affine projection).
            w_init : array_like, optional
                Initial filter coefficients.
        """
        super().__init__(filter_order, w_init)
        self.step: float = step
        self.gamma: float = gamma
        self.L: int = L 

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the weight update process for the Affine-Projection algorithm.

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
            regressor      : Matrix containing past input vectors.
            outputs_vector : Represents the output at iteration k.
            error_vector   : Represents the output errors at iteration k.

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
        
        x: np.ndarray = np.asarray(input_signal, dtype=complex)
        d: np.ndarray = np.asarray(desired_signal, dtype=complex)
        
        self._validate_inputs(x, d)
        n_samples: int = x.size
        
        y_out: np.ndarray = np.zeros(n_samples, dtype=complex)
        e_out: np.ndarray = np.zeros(n_samples, dtype=complex)
        
        x_padded: np.ndarray = np.zeros(n_samples + self.m, dtype=complex)
        x_padded[self.m:] = x

        X_matrix: np.ndarray = np.zeros((self.L + 1, self.m + 1), dtype=complex)
        D_vector: np.ndarray = np.zeros(self.L + 1, dtype=complex)

        

        for k in range(n_samples):
            X_matrix[1:] = X_matrix[:-1]
            X_matrix[0] = x_padded[k : k + self.m + 1][::-1]
            
            D_vector[1:] = D_vector[:-1]
            D_vector[0] = d[k]
            
            Y_vector: np.ndarray = X_matrix @ self.w.conj()
            y_out[k] = Y_vector[0] 
            
            E_vector: np.ndarray = D_vector - Y_vector
            e_out[k] = E_vector[0] 
            
            corr_matrix: np.ndarray = X_matrix @ X_matrix.conj().T + self.gamma * np.eye(self.L + 1)
            
            try:
                update_term: np.ndarray = np.linalg.solve(corr_matrix, E_vector)
                self.w = self.w + self.step * (X_matrix.T @ update_term.conj())
            except np.linalg.LinAlgError:
                aux_inv: np.ndarray = np.linalg.pinv(corr_matrix)
                self.w = self.w + self.step * (X_matrix.T @ (aux_inv @ E_vector).conj())
            
            self._record_history()

        if verbose:
            print(f"Affine Projection completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y_out,
            'errors': e_out,
            'coefficients': self.w_history
        }

# EOF