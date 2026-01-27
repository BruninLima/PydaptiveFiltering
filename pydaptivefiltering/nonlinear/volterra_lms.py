#  Volterra_LMS.py
#
#       Implements the Volterra LMS algorithm for REAL valued data.
#       (Algorithm 11.1 - book: Adaptive Filtering: Algorithms and Practical
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
from pydaptivefiltering.utils.validation import ensure_real_signals

class VolterraLMS(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Volterra LMS algorithm for REAL valued data.
        (Algorithm 11.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = False
    def __init__(
        self, 
        memory: int = 3, 
        step: Union[float, np.ndarray, list] = 1e-2, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            memory       : int (The linear memory length L. 
                           Total coefficients for 2nd order: L + L*(L+1)/2)
            step         : float | array_like (Convergence factor mu. 
                           Can be a scalar or a vector for individual term adjustment)
            w_init       : array_like, optional (Initial coefficients)
        """
        self.memory: int = memory
        
        n_coeffs = memory + (memory * (memory + 1)) // 2
        
        super().__init__(m=n_coeffs - 1, w_init=w_init)
        
        # Step can be a scalar or a vector for individual term adjustment
        self.step: Union[float, np.ndarray] = np.asarray(step) if isinstance(step, (list, np.ndarray)) else step

    def _create_volterra_regressor(self, x_lin: np.ndarray) -> np.ndarray:
        """
        Constructs the second-order Volterra regressor from the linear delay line.
        
        Sequence: [x(k), ..., x(k-L+1), x(k)^2, x(k)x(k-1), ..., x(k-L+1)^2]
        """
        # Quadratic terms (i <= j to avoid redundant terms and match Matlab uxl)
        quad_terms = []
        for i in range(self.memory):
            for j in range(i, self.memory):
                quad_terms.append(x_lin[i] * x_lin[j])
        
        return np.concatenate([x_lin, np.array(quad_terms)])
    
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
            Executes the weight update process for the Volterra LMS algorithm.

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
            uxl            : Volterra regressor containing linear and non-linear terms.
            y              : Represents the output at iteration k.
            e              : Represents the output errors at iteration k.

        Authors
        -------
            . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
            . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilhermepinto7@gmail.com
            . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
            . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
            . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
            . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br
        """
        tic: float = time()
        
        x: np.ndarray = np.asarray(input_signal)
        d: np.ndarray = np.asarray(desired_signal)

        self._validate_inputs(x, d)
        n_samples: int = x.size
        
        y: np.ndarray = np.zeros(n_samples, dtype=complex)
        e: np.ndarray = np.zeros(n_samples, dtype=complex)
        
        # Padding para lidar com a linha de atraso de tamanho 'memory'
        x_padded: np.ndarray = np.zeros(n_samples + self.memory - 1, dtype=complex)
        x_padded[self.memory - 1:] = x

        for k in range(n_samples):
            # 1. Extração da linha de atraso linear: [x(k), x(k-1), ..., x(k-L+1)]
            x_lin = x_padded[k : k + self.memory][::-1]
            
            # 2. Construção do regressor expandido
            uxl = self._create_volterra_regressor(x_lin)
            
            # 3. Cálculo da saída (usando conjugado para compatibilidade complexa)
            y[k] = np.dot(self.w.conj(), uxl)
            e[k] = d[k] - y[k]

            # 4. Atualização dos pesos: w(k+1) = w(k) + 2 * mu * e * uxl
            # Se mu for vetor, a multiplicação elemento a elemento funciona aqui
            self.w = self.w + 2 * self.step * e[k] * uxl
            
            self._record_history()

        if verbose:
            print(f"Volterra LMS Adaptation completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': self.w_history
        }
# EOF