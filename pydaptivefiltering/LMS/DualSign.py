#  LMS.DualSign.py
#
#      Implements the DualSign LMS algorithm for REAL valued data.
#      (Modified version of Algorithm 4.1 - book: Adaptive Filtering: Algorithms
#                                                       and Practical Implementation, Diniz)
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


def DualSign(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, rho: float, gamma: int, step: float = 1e-2,  verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the DualSign LMS algorithm for REAL valued data.
        (Modified version of Algorithm 4.1 - book: Adaptive Filtering: Algorithms
                                                and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = Dual_sign(Filter, desired_signal, input_signal, step, verbose)

    Inputs
    -------
        filter  : Adaptive Filter                                       filter object
        desired : Desired signal                                        numpy array (row vector)
        input   : Input signal to feed filter                           numpy array (row vector)
        rho     : Error modulus threshold                               float
        gamma   : Gain factor. Gamma is a power of two. (gamma > 1)     int
        step    : Convergence (relaxation) factor.                      float
        verbose : Verbose boolean                                       bool

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

    # Initialization
    tic = time()
    nIterations = desired_signal.size

    regressor = np.zeros(Filter.filter_order+1, dtype=input_signal.dtype)
    error_vector = np.array([])
    outputs_vector = np.array([])
    dualSignError = 0
    # Main Loop
    for it in range(nIterations):

        regressor = np.concatenate(([input_signal[it]], regressor))[
            :Filter.filter_order+1]

        coefficients = Filter.coefficients
        output_it = np.dot(coefficients.conj(), regressor)

        error_it = desired_signal[it] - output_it

        if abs(error_it) > rho:
            dualSignError = gamma*np.sign(error_it)
        else:
            dualSignError = np.sign(error_it)

        next_coefficients = coefficients + 2 * step * dualSignError * regressor

        error_vector = np.append(error_vector, error_it)
        outputs_vector = np.append(outputs_vector, output_it)

        Filter.coefficients = next_coefficients
        Filter.coefficients_history.append(next_coefficients)

    if verbose == True:
        print(" ")
        print('Total runtime {:.03} ms'.format((time() - tic)*1000))

    return {'outputs': outputs_vector,
            'errors': error_vector, 'coefficients': Filter.coefficients_history, 'adaptedFilter': Filter}

#   EOF
