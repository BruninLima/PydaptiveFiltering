#  RLS.rls.py
#
#      Implements the Fast Transversal RLS algorithm for REAL valued data.
#      (Algorithm 8.1 - book: Adaptive Filtering: Algorithms and Practical
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


def Fast_RLS(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, Lambda: float, N: int, Epsilon: float, verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the Fast Transversal RLS algorithm for REAL valued data. 
        (Algorithm 8.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = Fast_RLS(Filter, desired_signal, input_signal, Delta, Lambda, verbose)

    Inputs
    -------
        filter  : Adaptive Filter                                      filter object
        desired : Desired signal                                       numpy array (row vector)
        input   : Input signal to feed filter                          numpy array (row vector)
        Lambda  : Forgetting factor                                    float
        N       : predictor Order                                      int
        Epsilon : Initialization of xiMin_backward and xiMin_forward   float 
        verbose : Verbose boolean                                      bool

    Outputs
    -------
        dictionary:
            outputs      : Store the estimated output of each iteration.        numpy array (column vector)
            errors       : Store the error for each iteration.                  numpy array (column vector)
            coefficients : Store the estimated coefficients for each iteration  numpy array (column vector)
            outputs_posteriori : Store the a posteriori estimated output of each iteration. (column vector)
            errors_posteriori  : Store the a posteriori error for each iteration.           (column vector)

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
    nIterations = input_signal.size
    nCoefficients = Filter.filter_order + 1

    # Pre Allocations
    coefficientVector = np.zeros((nCoefficients, nIterations + 1))
    errorVector = np.zeros((nIterations,))
    outputVector = np.zeros((nIterations,))
    errorPriori = np.zeros((nIterations,))
    errorPosteriori = np.zeros((nIterations,))
    xiMin_f_Curr = 0
    xiMin_f_Prev = Epsilon
    xiMin_b = Epsilon
    gamma_Np1 = 0
    gamma_N = 1
    w_f = np.zeros(nCoefficients,)
    w_b = np.zeros(nCoefficients,)
    error_f = 0
    error_f_line = 0
    error_b = 0
    error_b_line = 0
    phiHatN = np.zeros(nCoefficients, 1)
    phiHatNPlus1 = np.zeros(nCoefficients+1,)

    regressor = np.zeros(Filter.filter_order+1, dtype=input_signal.dtype)

    # Main Loop
    for it in range(nIterations):

        regressor = np.concatenate(([input_signal[it]], regressor))[
            :Filter.filter_order+1]

        error_f_line = regressor.T.conj() @ (-1*w_f)
        error_f = error_f_line * gamma_N
        xiMin_f_Curr = Lambda * xiMin_f_Prev + error_f_line * error_f
        phiHatNPlus1 = np.concatenate((np.array([0]), phiHatN)) + 1/(
            Lambda * xiMin_f_Prev) * error_f_line * np.concatenate((np.array([1]), -1 * w_f))
        w_f = w_f + phiHatN*error_f
        gamma_Np1 = (Lambda * xiMin_f_Prev * gamma_N) / xiMin_f_Curr
        error_b_line = Lambda * xiMin_b * phiHatNPlus1[-1]
        gamma_N = 1 / (1 / (gamma_Np1) - phiHatNPlus1[-1]*error_b_line)

        error_b = error_b_line * gamma_N
        xiMin_b = Lambda * xiMin_b + error_b*error_b_line
        w_b = w_b + phiHatN * error_b

        # Joint Process Estimation

        errorPriori[it] = desired_signal[it] - \
            regressor.T.conj() @ coefficientVector[:, it]
        errorPosteriori[it] = errorPriori[it] * gamma_N
        coefficientVector[:, it + 1] = coefficientVector[:,
                                                         it] + phiHatN * errorPosteriori[it]

        Filter.coefficients = coefficientVector[:, it + 1]
        Filter.coefficients_history = coefficientVector[:, it + 1]

        # K - 1 Info
        xiMin_f_Prev = xiMin_f_Curr

    if verbose == True:
        print(" ")
        print('Total runtime {:.03} ms'.format((time() - tic)*1000))

    return errorPosteriori, errorPriori, coefficientVector

#   EOF
