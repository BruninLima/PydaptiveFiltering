#  nonlinear.Complex_Radial_Basis_Function.py
#
#       Implements the Complex Radial Basis Function algorithm for COMPLEX valued data.
#       (Algorithm 11.6 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

import numpy as np
from time import time
from typing import Optional, Union, List, Dict
from pydaptivefiltering.base import AdaptiveFilter

class ComplexRBF(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Complex Radial Basis Function algorithm for COMPLEX valued data.
        (Algorithm 11.6 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = True
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
            input_dim    : int (Dimension of the input vector, e.g., number of taps)
            ur           : float (Convergence factor for reference vector/centers)
            uw           : float (Convergence factor for coefficient vector/weights)
            us           : float (Convergence factor for spread/sigma)
            w_init       : array_like, optional (Initial weights for the neurons)
        """
        super().__init__(n_neurons, w_init)
        
        self.n_neurons: int = n_neurons
        self.input_dim: int = input_dim
        self.ur: float = ur
        self.uw: float = uw
        self.us: float = us

        # Initialization as per the Matlab algorithm
        # Weights are complex
        if w_init is None:
            self.w = np.random.randn(n_neurons).astype(complex) + 1j * np.random.randn(n_neurons)
        
        # Reference vectors (Centers) - Initialized with small random values
        self.vet: np.ndarray = 0.5 * (np.random.randn(n_neurons, input_dim) + 1j * np.random.randn(n_neurons, input_dim))
        
        # Spreads (Sigma) - Initialized to ones
        self.sigma: np.ndarray = np.ones(n_neurons, dtype=float)

    def optimize(
        self, 
        input_signal: Union[np.ndarray, list], 
        desired_signal: Union[np.ndarray, list], 
        verbose: bool = False
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """
        Description
        -----------
            Executes the update process for weights, centers, and spreads of the CRBF.

        Inputs
        -------
            input_signal   : np.ndarray | list (Input matrix where each row/column is a regressor)
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
        
        # Ensure signals are complex
        x: np.ndarray = np.asarray(input_signal, dtype=complex)
        d: np.ndarray = np.asarray(desired_signal, dtype=complex)

        self._validate_inputs(x, d)
        
        # In RBF, the input is often a matrix of regressors. 
        # If it's a 1D signal, we assume it's the already constructed regressor stream.
        if x.ndim == 1:
            n_samples = x.size
            # Assuming a simple delay line if input_dim > 1 but x is 1D
            x_padded = np.zeros(n_samples + self.input_dim - 1, dtype=complex)
            x_padded[self.input_dim - 1:] = x
            regressors = np.array([x_padded[k:k + self.input_dim][::-1] for k in range(n_samples)])
        else:
            n_samples = x.shape[0]
            regressors = x

        y: np.ndarray = np.zeros(n_samples, dtype=complex)
        e: np.ndarray = np.zeros(n_samples, dtype=complex)

        for k in range(n_samples):
            uxl_k = regressors[k]
            
            # 1. Compute Euclidean Distance (dis) between input and centers
            # Equivalent to Matlab's dist(uxl, vet')
            diff = uxl_k - self.vet
            dis_sq = np.sum(np.real(diff)**2 + np.imag(diff)**2, axis=1)
            
            # 2. Activation Function (Gaussian fdis)
            fdis = np.exp(-dis_sq / (self.sigma**2))
            
            # 3. Output and Error
            y[k] = np.dot(self.w.conj(), fdis)
            e[k] = d[k] - y[k]
            
            # 4. Update Weights (uw)
            self.w = self.w + 2 * self.uw * e[k] * fdis
            
            # 5. Update Spread (us)
            # Derivative involves real and imag parts as per Algorithm 11.6
            grad_sigma = (2 * self.us * fdis * (e[k].real * self.w.real + e[k].imag * self.w.imag) * dis_sq / (self.sigma**3))
            self.sigma = self.sigma + grad_sigma
            
            # 6. Update Reference Vectors / Centers (ur)
            for p in range(self.n_neurons):
                term_real = e[k].real * self.w[p].real * (uxl_k - self.vet[p]).real
                term_imag = 1j * (e[k].imag * self.w[p].imag * (uxl_k - self.vet[p]).imag)
                
                self.vet[p] = self.vet[p] + 2 * self.ur * fdis[p] * (term_real + term_imag) / (self.sigma[p]**2)

            self._record_history()

        if verbose:
            print(f"Complex RBF Adaptation completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history
        }

# EOF