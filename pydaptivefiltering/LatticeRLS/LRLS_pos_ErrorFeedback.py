#  Lattice_RLS.LRLS_pos_ErrorFeedback.py
#
#      Implements the Lattice RLS algorithm based on a posteriori errors.
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


def LRLS_pos_ErrorFeedback(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, Lambda: float, N: int, Epsilon: float,  verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the Lattice RLS algorithm based on a posteriori errors. 
        (Algorithm 7.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = LRLS_pos_ErrorFeedback(Filter, desired_signal, input_signal, Delta, Lambda, verbose)

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
    nCoefficients = N + 1
    nIterations = input_signal.size()

    # Pre Allocations
    deltaVector = np.zeros(N + 1)
    xiMin_f_Curr = np.zeros(N + 2)
    xiMin_f_Prev = np.zeros(N + 2)
    xiMin_b_Curr = np.zeros(N + 2)
    xiMin_b_Prev = np.zeros(N + 2)
    xiMin_b_PPrev = np.zeros(N + 2)
    error_b_Curr = np.zeros(N + 2)
    error_b_Prev = np.zeros(N + 2)
    error_f = np.zeros(N + 2)
    gammaVector_Curr = np.zeros(N + 2)
    gammaVector_Prev = np.zeros(N + 2)
    kappa_f = np.zeros(N + 1)
    kappa_b = np.zeros(N + 1)
    ladderVector = np.zeros((nCoefficients, nIterations))
    kappaVector = np.zeros((nCoefficients, nIterations))
    posterioriErrorMatrix = np.zeros((N + 2, nIterations))

    # Initialize Parameters
    kappa_f = np.zeros(N + 1)
    kappa_b = np.zeros(N + 1)
    deltaVector = np.zeros(N + 1)

    gammaVector_Prev = np.ones(N + 2)

    xiMin_f_Prev = np.ones(N + 2) * Epsilon
    xiMin_b_Prev = np.ones(N + 2) * Epsilon
    xiMin_b_PPrev = np.ones(N + 2) * Epsilon

    error_b_Prev = np.zeros(N + 2)

    for it in range(nIterations):

        # Set Values for Section 0(Zero)
        gammaVector_Curr[0] = 1
        error_b_Curr[0] = input[it]
        error_f[0] = input[it]
        xiMin_f_Curr[0] = input[it]**2 + Lambda * xiMin_f_Prev[0]
        xiMin_b_Curr[0] = xiMin_f_Curr[0]
        posterioriErrorMatrix[0][it] = desired_signal[it]

        # Propagate the Order Update Equations
        for ot in range(N+1):
            # Delta Time Update
            deltaVector[ot] = Lambda * deltaVector[ot] + \
                (error_b_Prev[ot]*error_f[ot])/gammaVector_Prev[ot]

            # Order Update Equations
            kappa_f[ot] = deltaVector[ot]/xiMin_f_Curr[ot]
            kappa_b[ot] = deltaVector[ot]/xiMin_b_Curr[ot]

            # Gamma Time Update
            gammaVector_Curr[ot] = (1-abs(kappa_b[ot])**2)*gammaVector_Prev[ot]

            # Error Time Update
            error_f[ot+1] = error_f[ot] - kappa_f[ot]*error_b_Prev[ot]
            error_b_Curr[ot+1] = error_b_Prev[ot] - kappa_b[ot]*error_f[ot]

            # xiMin Time Update
            xiMin_f_Curr[ot+1] = xiMin_f_Curr[ot] - \
                kappa_f[ot]*xiMin_b_Prev[ot]
            xiMin_b_Curr[ot+1] = xiMin_b_Curr[ot] - \
                kappa_b[ot]*xiMin_f_Curr[ot]

            # Store Coefficients
            ladderVector[ot][it] = kappa_f[ot]
            kappaVector[ot][it] = kappa_b[ot]

            # Update previous values
            gammaVector_Prev[ot] = gammaVector_Curr[ot]
            error_b_Prev[ot] = error_b_Curr[ot]
            xiMin_f_Prev[ot] = xiMin_f_Curr[ot]
            xiMin_b_Prev[ot] = xiMin_b_Curr[ot]
            xiMin_b_PPrev[ot] = xiMin_b_Prev[ot]

            # Store errors
            posterioriErrorMatrix[ot][it] = error_b_Curr[ot]

        # Feedforward Filter
        ladderVector[:, it] = kappa_f
        kappaVector[:, it] = kappa_b
        # Update for next iteration
        xiMin_f_Prev = xiMin_f_Curr
        xiMin_b_Prev = xiMin_b_Curr
        error_b_Prev = error_b_Curr
        gammaVector_Prev = gammaVector_Curr
        xiMin_b_PPrev = xiMin_b_Prev

    return ladderVector, kappaVector, posterioriErrorMatrix
