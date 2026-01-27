#  Multilayer_Perceptron.py
#
#       Implements the Multilayer Perceptron algorithm for REAL valued data.
#       (Algorithm 11.4 - book: Adaptive Filtering: Algorithms and Practical
#                                                              Implementation, Diniz)
#       Modified to include Momentum and selectable Activation Functions.
#
#       Authors:
#        . Bruno Ramos Lima Netto         - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#        . Guilherme de Oliveira Pinto    - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#        . Markus Vinícius Santos Lima    - mvsl20@gmailcom           & markus@lps.ufrj.br
#        . Wallace Alves Martins          - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           & wagner@lps.ufrj.br
#        . Paulo Sergio Ramirez Diniz    -                            diniz@lps.ufrj.br

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
        
        Includes improvements:
        1. Momentum term for gradient descent stabilization.
        2. Selectable activation functions ('tanh' or 'sigmoid').
    """
    supports_complex: bool = False

    def __init__(
        self, 
        n_neurons: int = 10, 
        input_dim: int = 3,
        step: float = 0.01, 
        momentum: float = 0.9,
        activation: str = 'tanh',
        w_init: Optional[Union[np.ndarray, list]] = None
    ) -> None:
        """
        Inputs
        -------
            n_neurons    : int (Number of neurons in each hidden layer)
            input_dim    : int (Dimension of the input vector, default 3)
            step         : float (Convergence factor mu)
            momentum     : float (Momentum factor alpha, typically 0.0 to 0.9)
            activation   : str ('tanh' for hyperbolic tangent, 'sigmoid' for logistic)
            w_init       : array_like, optional (Initial coefficients)
        """
        super().__init__(n_neurons, w_init)
        
        self.n_neurons: int = n_neurons
        self.input_dim: int = input_dim
        self.step: float = step
        self.momentum: float = momentum
        
        # --- Configuração das Funções de Ativação ---
        # Usamos clip para evitar overflow numérico em exponenciais
        self._activation_map = {
            'tanh': (
                lambda v: np.tanh(np.clip(v, -50, 50)), 
                lambda v: 1.0 - np.tanh(np.clip(v, -50, 50))**2
            ),
            'sigmoid': (
                lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -50, 50))),
                lambda v: (1.0 / (1.0 + np.exp(-np.clip(v, -50, 50)))) * (1.0 - (1.0 / (1.0 + np.exp(-np.clip(v, -50, 50)))))
            )
        }
        
        if activation not in self._activation_map:
            raise ValueError(f"Activation '{activation}' not supported. Choose 'tanh' or 'sigmoid'.")
            
        self.act_func, self.act_deriv = self._activation_map[activation]

        # --- Inicialização de Pesos (Xavier/Glorot Initialization) ---
        # Melhor que random puro para evitar saturação inicial em Tanh/Sigmoid
        limit_w1 = np.sqrt(6 / (input_dim + n_neurons))
        limit_w2 = np.sqrt(6 / (n_neurons + n_neurons))
        limit_w3 = np.sqrt(6 / (n_neurons + 1))

        self.w1 = np.random.uniform(-limit_w1, limit_w1, (n_neurons, input_dim))
        self.w2 = np.random.uniform(-limit_w2, limit_w2, (n_neurons, n_neurons))
        self.w3 = np.random.uniform(-limit_w3, limit_w3, n_neurons)
        
        self.b1 = np.zeros(n_neurons)
        self.b2 = np.zeros(n_neurons)
        self.b3 = 0.0

        # --- Buffers para Momentum (Histórico dos Gradientes) ---
        self.prev_dw1 = np.zeros_like(self.w1)
        self.prev_dw2 = np.zeros_like(self.w2)
        self.prev_dw3 = np.zeros_like(self.w3)
        self.prev_db1 = np.zeros_like(self.b1)
        self.prev_db2 = np.zeros_like(self.b2)
        self.prev_db3 = 0.0

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
            Executes the weight update process for the MLP algorithm with Momentum.

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
        """
        tic: float = time()
        
        x_in: np.ndarray = np.asarray(input_signal, dtype=float)
        d_in: np.ndarray = np.asarray(desired_signal, dtype=float)

        is_multidim = x_in.ndim > 1
        n_samples = x_in.shape[0]
        
        if n_samples != d_in.size:
            raise ValueError(f"Shape mismatch: input ({n_samples}) and desired ({d_in.size})")

        y = np.zeros(n_samples, dtype=float)
        e = np.zeros(n_samples, dtype=float)
        
        w1_hist, w2_hist, w3_hist = [], [], []

        x_prev = 0.0
        d_prev = 0.0

        for k in range(n_samples):
            # Construção do regressor
            if is_multidim:
                uxl = x_in[k]
            else:
                uxl = np.array([x_in[k], d_prev, x_prev], dtype=float)
            
            # --- Forward Pass ---
            v1 = np.dot(self.w1, uxl) - self.b1
            y1 = self.act_func(v1)
            
            v2 = np.dot(self.w2, y1) - self.b2
            y2 = self.act_func(v2)
            
            # Camada de saída linear
            y[k] = np.dot(y2, self.w3) - self.b3
            e[k] = d_in[k] - y[k]
            
            # --- Backward Pass ---
            # Derivada da saída (linear) é 1, então propaga o erro direto
            er_hid2 = e[k] * self.w3 * self.act_deriv(v2)
            er_hid1 = np.dot(self.w2.T, er_hid2) * self.act_deriv(v1)
            
            # --- Weight Updates with Momentum ---
            # Atualização camada 3
            dw3 = 2 * self.step * e[k] * y2
            self.w3 += dw3 + self.momentum * self.prev_dw3
            self.prev_dw3 = dw3 # Salva para o próximo passo

            db3 = -2 * self.step * e[k]
            self.b3 += db3 + self.momentum * self.prev_db3
            self.prev_db3 = db3
            
            # Atualização camada 2
            dw2 = 2 * self.step * np.outer(er_hid2, y1)
            self.w2 += dw2 + self.momentum * self.prev_dw2
            self.prev_dw2 = dw2

            db2 = -2 * self.step * er_hid2
            self.b2 += db2 + self.momentum * self.prev_db2
            self.prev_db2 = db2
            
            # Atualização camada 1
            dw1 = 2 * self.step * np.outer(er_hid1, uxl)
            self.w1 += dw1 + self.momentum * self.prev_dw1
            self.prev_dw1 = dw1

            db1 = -2 * self.step * er_hid1
            self.b1 += db1 + self.momentum * self.prev_db1
            self.prev_db1 = db1
            
            # Atualização das memórias para k+1
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