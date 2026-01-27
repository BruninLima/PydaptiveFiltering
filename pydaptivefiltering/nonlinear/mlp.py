#  Multilayer_Perceptron.py
#
#       Implements the Multilayer Perceptron algorithm for REAL valued data.
#       (Algorithm 11.4 - book: Adaptive Filtering: Algorithms and Practical
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

class MultilayerPerceptron(AdaptiveFilter):
    """
    Description
    -----------
        Implements the Multilayer Perceptron algorithm for REAL valued data.
        (Algorithm 11.4 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)
    """
    supports_complex: bool = False
    def __init__(
        self, 
        n_neurons: int = 10, 
        input_dim: int = 3,
        step: float = 0.01, 
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            n_neurons    : int (Number of neurons in each hidden layer)
            input_dim    : int (Dimension of the input vector, default 3: [x(k), d(k-1), x(k-1)])
            step         : float (Convergence factor mu)
            w_init       : array_like, optional (Initial coefficients)
        """
        super().__init__(n_neurons, w_init)
        
        self.n_neurons: int = n_neurons
        self.input_dim: int = input_dim
        self.step: float = step

        self.w1: np.ndarray = 0.2 * np.random.randn(n_neurons, input_dim)
        self.w2: np.ndarray = 0.2 * np.random.randn(n_neurons, n_neurons)
        self.w3: np.ndarray = 0.2 * np.random.randn(n_neurons) 
        
        self.b1: np.ndarray = 0.1 * np.random.randn(n_neurons)
        self.b2: np.ndarray = 0.1 * np.random.randn(n_neurons)
        self.b3: float = 0.1 * np.random.randn()

    def _sigmoid(self, v: np.ndarray) -> np.ndarray:
        """ Standard logistic sigmoid function. """
        return 1.0 / (1.0 + np.exp(-v))

    def _sigmoid_derivative(self, v: np.ndarray) -> np.ndarray:
        """ Derivative of the standard logistic sigmoid function. """
        s = self._sigmoid(v)
        return s * (1.0 - s)

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
            Executes the weight update process for the MLP algorithm.

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
                coefficients : Store a dictionary containing w1, w2, w3 history.

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
        d_in: np.ndarray = np.asarray(desired_signal, dtype=float)

        # Detecta se o input é matriz (amostras, dim) ou vetor (amostras,)
        is_multidim = x_in.ndim > 1
        n_samples = x_in.shape[0]
        
        # Validação manual para suportar matrizes
        if n_samples != d_in.size:
            raise ValueError(f"Tamanhos incompatíveis: input ({n_samples}) e desired ({d_in.size})")

        y = np.zeros(n_samples, dtype=float)
        e = np.zeros(n_samples, dtype=float)
        
        w1_hist, w2_hist, w3_hist = [], [], []

        x_prev = 0.0
        d_prev = 0.0

        for k in range(n_samples):
            # LÓGICA DO REGRESSOR:
            # Se for multidimensional, usa a linha atual como vetor de entrada.
            # Se for unidimensional, monta o regressor padrão [x(k), d(k-1), x(k-1)]
            if is_multidim:
                uxl = x_in[k]
            else:
                uxl = np.array([x_in[k], d_prev, x_prev], dtype=float)
            
            # --- Forward Pass ---
            v1 = np.dot(self.w1, uxl) - self.b1
            y1 = self._sigmoid(v1)
            
            v2 = np.dot(self.w2, y1) - self.b2
            y2 = self._sigmoid(v2)
            
            y[k] = np.dot(y2, self.w3) - self.b3
            e[k] = d_in[k] - y[k]
            
            # --- Backward Pass (Backpropagation) ---
            # Erro na camada de saída (linear) refletido para a oculta 2
            er_hid2 = e[k] * self.w3 * self._sigmoid_derivative(v2)
            
            # Erro da oculta 2 refletido para a oculta 1
            er_hid1 = np.dot(self.w2.T, er_hid2) * self._sigmoid_derivative(v1)
            
            # --- Weight Updates ---
            self.w3 += 2 * self.step * e[k] * y2
            self.b3 -= 2 * self.step * e[k]
            
            self.w2 += 2 * self.step * np.outer(er_hid2, y1)
            self.b2 -= 2 * self.step * er_hid2
            
            self.w1 += 2 * self.step * np.outer(er_hid1, uxl)
            self.b1 -= 2 * self.step * er_hid1
            
            # Atualização dos estados para o próximo k (configuração série-paralela)
            # x_prev recebe o valor escalar se x_in for 1D
            x_prev = x_in[k] if not is_multidim else x_in[k, 0]
            d_prev = d_in[k]
            
            if self._record_history:
                w1_hist.append(self.w1.copy())
                w2_hist.append(self.w2.copy())
                w3_hist.append(self.w3.copy())

        if verbose:
            print(f"MLP Adaptation completed in {(time() - tic)*1000:.03f} ms")

        return {
            'outputs': y,
            'errors': e,
            'coefficients': {
                'w1': w1_hist,
                'w2': w2_hist,
                'w3': w3_hist
            }
        }
# EOF