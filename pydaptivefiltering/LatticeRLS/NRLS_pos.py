#  Lattice_RLS.NLRLS_pos.py
#
#      Implements the Normalized Lattice RLS algorithm based on a posteriori error.
#      (Algorithm 7.1 - book: Adaptive Filtering: Algorithms and Practical
#                                                       Implementation, Diniz)
#
#      Authors:
#       . Bruno Ramos Lima Netto        - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
#       . Guilherme de Oliveira Pinto   - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
#       . Markus Vinícius Santos Lima   - mvsl20@gmailcom           & markus@lps.ufrj.br
#       . Wallace Alves Martins         - wallace.wam@gmail.com     & wallace@lps.ufrj.br
#       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com          & wagner@lps.ufrj.br
#       . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

# Imports
import numpy as np
from time import time


def NLRLS_pos(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, Lambda: float, N: int, Epsilon: float,  verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the Normalized Lattice RLS algorithm based on a posteriori error.
        (Algorithm 7.2 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = NLRLS_pos(Filter, desired_signal, input_signal, Delta, Lambda, verbose)

    Inputs
    -------
        filter  : Adaptive Filter                                      filter object
        desired_signal : desired_signal signal                                       numpy array (row vector)
        input   : Input signal to feed filter                          numpy array (row vector)
        Lambda  : Forgetting factor                                    float
        N       : Number of sections of the lattice filter             int
        Epsilon : Regularization factor                                float
        verbose : Verbose boolean                                      bool

    Outputs
    -------
        dictionary:
            ladderVector : Store the ladder coefficients for each iteration      numpy array (column vector)
            kappaVector  : Store the reflection coefficients for each iteration  numpy array (column vector)
            posterioriErrorMatrix: Store the posteriori error for each iteration numpy array (column vector)

    Main Variables
    --------- 
        regressor
        outputs_vector[k] represents the output errors at iteration k    
        FIR error vectors. 
        error_vector[k] represents the output errors at iteration k.

    Misc Variables
    --------------
        tic
        nIterations


    Authors
    -------
        . Bruno Ramos Lima Netto        - brunolimanetto@gmail.com  & brunoln@cos.ufrj.br
        . Guilherme de Oliveira Pinto   - guilhermepinto7@gmail.com & guilherme@lps.ufrj.br
        . Markus Vinícius Santos Lima   - mvsl20@gmailcom           & markus@lps.ufrj.br
        . Wallace Alves Martins         - wallace.wam@gmail.com     & wallace@lps.ufrj.br
        . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com          & wagner@lps.ufrj.br
        . Paulo Sergio Ramirez Diniz    -                             diniz@lps.ufrj.br

    """

    # Data Initialization
    tic = time()
    nCoefficients = N + 1
    nIterations = input_signal.size()

    # Pre Allocations
    deltaVector = np.zeros((1, N + 1))
    deltaVector_D = np.zeros((1, N + 1))
    error_b_Curr = np.zeros((1, N + 2))
    error_b_Prev = np.zeros((1, N + 2))
    error_f = np.zeros((1, N + 2))
    ladderVector = np.zeros((nCoefficients, nIterations))
    kappaVector = np.zeros((nCoefficients, nIterations))
    posterioriErrorMatrix = np.zeros((N + 2, nIterations))

    # Initialize Parameters
    deltaVector = np.zeros((1, N + 1))
    deltaVector_D = np.zeros((1, N + 1))
    error_b_Prev = np.zeros((1, N + 2))
    sigma_d_2 = Epsilon
    sigma_x_2 = Epsilon

    for it in range(nIterations):
        # Feedforward Filtering
        regressor = np.append(input_signal[it], np.zeros((1, N)))
        outputs_vector = np.zeros((1, N + 2))
        outputs_vector[0] = np.dot(regressor, ladderVector[:, it])
        error_vector = desired_signal[it] - outputs_vector

        # Error Backward
        error_b_Curr[0] = error_vector[0]
        for i in range(1, N + 2):
            error_b_Curr[i] = error_vector[i] + \
                kappaVector[i-1, it] * error_b_Curr[i-1]

        # Error Forward
        error_f[N + 1] = error_b_Curr[N + 1]

        for i in range(N, 0, -1):
            error_f[i] = error_b_Curr[i] + kappaVector[i-1, it] * error_f[i+1]

        # Update Parameters
        sigma_d_2 = Lambda * sigma_d_2 + (1 - Lambda) * error_f[1] ** 2
        sigma_x_2 = Lambda * sigma_x_2 + (1 - Lambda) * regressor[0] ** 2

        xiMin_backward = sigma_d_2 / sigma_x_2

        for i in range(N):
            deltaVector[i] = error_f[i+1] / error_f[i]
            deltaVector_D[i] = (1 - Lambda) * \
                deltaVector[i] + Lambda * deltaVector_D[i]
            kappaVector[i, it] = deltaVector_D[i] / \
                (1 - deltaVector_D[i] * deltaVector_D[i] * xiMin_backward)
            ladderVector[i, it+1] = ladderVector[i, it] + \
                kappaVector[i, it] * error_f[i] * xiMin_backward

        ladderVector[N, it + 1] = ladderVector[N, it] + \
            error_f[N] * xiMin_backward
        # Save Posteriori Error
        posterioriErrorMatrix[:, it] = error_b_Curr

    return ladderVector, kappaVector, posterioriErrorMatrix
