#  Lattice_RLS.LRLS_pos.py
#
#      Implements the RLS algorithm for COMPLEX valued data.
#      (Algorithm 5.3 - book: Adaptive Filtering: Algorithms and Practical
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


def LRLS_pos(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, Lambda: float, N: int, Epsilon: float,  verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the RLS algorithm for COMPLEX valued data. 
        (Algorithm 5.3 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = RLS(Filter, desired_signal, input_signal, Delta, Lambda, verbose)

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

    # Initialization
    tic = time()
    nCoefficients = N + 1
    nIterations = input_signal.size()

    # Pre Allocations
    deltaVector = np.zeros((1, N + 1))
    deltaVector_D = np.zeros((1, N + 1))
    xiMin_f = np.zeros((1, N + 2))
    xiMin_b_Curr = np.zeros((1, N + 2))
    xiMin_b_Prev = np.zeros((1, N + 2))
    gammaVector_Curr = np.zeros((1, N + 2))
    gammaVector_Prev = np.ones((1, N + 2))
    error_b_Curr = np.zeros((1, N + 2))
    error_b_Prev = np.zeros((1, N + 2))
    error_f = np.zeros((1, N + 2))
    ladderVector = np.zeros((nCoefficients, nIterations))
    kappaVector = np.zeros((nCoefficients, nIterations))
    posterioriErrorMatrix = np.zeros((N + 2, nIterations))
    kappa_f = 0
    kappa_b = 0

    xiMin_f = np.ones((1, N + 2)) * Epsilon
    xiMin_b_Prev = np.ones((1, N + 2)) * Epsilon

    for it in range(nIterations):
        # Set Values for Section 0(Zero)
        gammaVector_Curr[0] = 1
        error_b_Curr[0] = input_signal[it]
        error_f[0] = input_signal[it]
        xiMin_f[0] = input_signal[it] ** 2 + Lambda * xiMin_f[0]
        xiMin_b_Curr[0] = xiMin_f[0]
        posterioriErrorMatrix[0, it] = desired_signal[it]

        for ot in range(N + 1):
            # Delta Time Update
            deltaVector[ot] = Lambda * deltaVector[ot] + (
                error_b_Prev[ot] / gammaVector_Prev[ot]
            ) * error_f[ot]

            # Order Update Equations
            gammaVector_Curr[ot + 1] = (
                gammaVector_Curr[ot] - error_b_Curr[ot] ** 2 / xiMin_b_Curr[ot]
            )

            kappa_b = deltaVector[ot] / xiMin_f[ot]
            kappa_f = deltaVector[ot] / xiMin_b_Prev[ot]

            error_b_Curr[ot + 1] = (
                error_b_Prev[ot + 1] - kappa_f * error_f[ot]
            )
            error_f[ot + 1] = error_f[ot] - kappa_b * error_b_Curr[ot]

            xiMin_f[ot + 1] = Lambda * xiMin_f[ot] + error_f[ot + 1] ** 2
            xiMin_b_Curr[ot + 1] = Lambda * xiMin_b_Curr[ot] + error_b_Curr[ot + 1] ** 2

            ladderVector[ot, it] = kappa_b
            kappaVector[ot, it] = kappa_f

            posterioriErrorMatrix[ot + 1, it] = error_b_Curr[ot + 1]

        error_b_Prev = error_b_Curr
        gammaVector_Prev = gammaVector_Curr
        xiMin_b_Prev = xiMin_b_Curr

    return ladderVector, kappaVector, posterioriErrorMatrix

#EOF