#  BlindFilters.bilinear_RLS.py
#
#      Implements the Bilinear RLS algorithm for REAL valued data.
#      (Algorithm 11.3 - book: Adaptive Filtering: Algorithms and Practical
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


def bilinear_RLS(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, step: float = 1e-2, verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the Bilinear RLS algorithm for REAL valued data.
        (Algorithm 11.3 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = bilinear_RLS(Filter, desired_signal, input_signal, L, gamma, step, verbose)

    Inputs
    -------
        filter  : Adaptive Filter                       filter object
        desired : Desired signal to feed filter         numpy array (row vector)
        input   : Input signal to feed filter           numpy array (row vector)
        step    : Convergence (relaxation) factor.      float
        verbose : Verbose boolean                       bool

    Outputs
    -------
        dictionary:
            outputs      : Store the estimated output of each iteration.        numpy array (collumn vector)
            errors       : Store the error for each iteration.                  numpy array (collumn vector)
            coefficients : Store the estimated coefficients for each iteration  numpy array (collumn vector)
            adaptedFilter: Store the adapted filter object                      filter object

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

    # Initialization Procedure
    tic = time()
    nIterations = input_signal.size
    nCoefficients = Filter.filter_order + 1

    # Pre Allocations
    coefficientVector = np.zeros((nCoefficients, nIterations + 1))
    errorVector = np.zeros((nIterations,))
    outputVector = np.zeros((nIterations,))

    # Initial Coefficients Values
    coefficientVector[:, 0] = Filter.coefficients

    # Main Loop
    for it in range(nIterations):

        regressor = np.concatenate(([input_signal[it]], regressor))[
            :Filter.filter_order+1]

        outputVector[it] = np.dot(coefficientVector[:, it], regressor)

        errorVector[it] = desired_signal[it] - outputVector[it]

        next_coefficients = coefficientVector[:, it] - step * 2 * \
            errorVector[it] * np.conj(outputVector[it]) * regressor

        coefficientVector[:, it + 1] = next_coefficients
        Filter.coefficients_history.append(next_coefficients)

    Filter.coefficients = next_coefficients

    if verbose == True:
        print(" ")
        print('Total runtime {:.03} ms'.format((time() - tic)*1000))

    return {'outputs': outputVector,
            'errors': errorVector,
            'coefficients': Filter.coefficients_history}

#   EOF
