#  LMS.Power2_Error.py
#
#      Implements the Power-of-Two Error LMS algorithm for REAL valued data.
#      (Modified version of Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical
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


def Power2_Error(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, bd: int, tau: float, step: float = 1e-2, verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the Power-of-Two Error LMS algorithm for REAL valued data.
        (Modified version of Algorithm 4.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = Power2_error(Filter, desired_signal,
                                    input_signal, tau, step, verbose)

    Inputs
    -------
        filter  : Adaptive Filter                       filter object
        desired : Desired signal                        numpy array (row vector)
        input   : Input signal to feed filter           numpy array (row vector)
        bd      : Word length (signal bit)              int 
        tau     : Gain Factor                           float
        step    : Convergence (relaxation) factor.      float
        verbose : Verbose boolean                       bool

    Outputs
    -------
        dictionary:
            outputs      : Store the estimated output of each iteration.        numpy array (collumn vector)
            errors       : Store the error for each iteration.                  numpy array (collumn vector)
            coefficients : Store the estimated coefficients for each iteration  numpy array (collumn vector)

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

    # Main Loop
    for it in range(nIterations):

        regressor = np.concatenate(([input_signal[it]], regressor))[
            :Filter.filter_order+1]

        coefficients = Filter.coefficients
        output_it = np.dot(coefficients.conj(), regressor)

        error_it = desired_signal[it] - output_it

        if (abs(error_it) >= 1):
            power2Error = np.sign(error_it)

        elif (abs(error_it < 2**(-bd+1))):
            power2Error = tau*np.sign(error_it)

        else:
            power2Error = (2**(np.floor(np.log2(abs(error_it))))
                           )*np.sign(error_it)

        next_coefficients = coefficients + 2*step*power2Error*regressor

        error_vector = np.append(error_vector, error_it)
        outputs_vector = np.append(outputs_vector, output_it)

        Filter.coefficients = coefficients
        Filter.coefficients_history.append(next_coefficients)

        if verbose == True:
            print(" ")
            print('Total runtime {:.03} ms'.format((time() - tic)*1000))

    # Output
    return {'outputs': outputs_vector,
            'errors': error_vector, 'coefficients': Filter.coefficients_history, 'adaptedFilter': Filter}
