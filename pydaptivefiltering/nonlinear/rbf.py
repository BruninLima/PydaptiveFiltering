#  nonlinear.rbf.py
#
#       Implements the Radial Basis Function algorithm for REAL valued data.
#       (Algorithm 11.5 - book: Adaptive Filtering: Algorithms and Practical
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

class RBF(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Radial Basis Function algorithm for REAL valued data.
        (Algorithm 11.5 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = False
    def __init__(
        self, 
        n_neurons: int, 
        input_dim: int,
        ur: float = 0.01, 
        uw: float = 0.01, 
        us: float = 0.01,
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            n_neurons    : int (Number of neurons in the hidden layer)
            input_dim    : int (Dimension of the input vector/regressor)
            ur           : float (Convergence factor for reference vector/centers)
            uw           : float (Convergence factor for weight vector)
            us           : float (Convergence factor for spread/sigma)
            w_init       : array_like, optional (Initial weights for the neurons)
        """
        super().__init__(n_neurons, w_init)
        
        self.n_neurons: int = n_neurons
        self.input_dim: int = input_dim
        self.ur: float = ur
        self.uw: float = uw
        self.us: float = us

        if w_init is None:
            self.w = np.random.randn(n_neurons)
        
        self.vet: np.ndarray = 0.5 * np.random.randn(n_neurons, input_dim)
        
        self.sigma: np.ndarray = np.ones(n_neurons, dtype=float)

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
            Executes the update process for weights, centers, and spreads of the RBF.

        Inputs
        -------
            input_signal   : np.ndarray | list (Input matrix/vector)
            desired_signal : np.ndarray | list (Desired vector d)
            verbose        : bool (Verbose boolean)

        Outputs
        -------
            dictionary:
                outputs      : Store the estimated output of each iteration.
                errors       : Store the error for each iteration.
                coefficients : Store the estimated weights (w) for each iteration.

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
        
        x_in: np.ndarray = np.asarray(input_signal, dtype=float)
        d: np.ndarray = np.asarray(desired_signal, dtype=float)

        self._validate_inputs(x_in, d)
        
        if x_in.ndim == 1:
            n_samples = x_in.size
            x_padded = np.zeros(n_samples + self.input_dim - 1)
            x_padded[self.input_dim - 1:] = x_in
            regressors = np.array([x_padded[k:k + self.input_dim][::-1] for k in range(n_samples)])
        else:
            n_samples = x_in.shape[0]
            regressors = x_in

        y: np.ndarray = np.zeros(n_samples, dtype=float)
        e: np.ndarray = np.zeros(n_samples, dtype=float)

        for k in range(n_samples):
            uxl_k = regressors[k]
            
            diff = uxl_k - self.vet
            dis_sq = np.sum(diff**2, axis=1)
            
            fdis = np.exp(-dis_sq / (self.sigma**2))
            
            y[k] = np.dot(self.w, fdis)
            e[k] = d[k] - y[k]
            
            self.w = self.w + 2 * self.uw * e[k] * fdis
            
            self.sigma = self.sigma + 2 * self.us * e[k] * fdis * self.w * dis_sq / (self.sigma**3)
            
            for p in range(self.n_neurons):
                self.vet[p] = self.vet[p] + 2 * self.ur * fdis[p] * e[k] * self.w[p] * (uxl_k - self.vet[p]) / (self.sigma[p]**2)

            self._record_history()

        if verbose:
            print(f"RBF Adaptation completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history
        }
# EOF