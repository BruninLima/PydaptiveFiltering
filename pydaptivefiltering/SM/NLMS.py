#  SM.NLMS.py
#
#      Implements the Set-membership Normalized LMS algorithm for COMPLEX valued data.
#      (Algorithm 6.1 - book: Adaptive Filtering: Algorithms and Practical
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


def NLMS(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, gamma_bar: float, gamma:float, verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the Set-membership Normalized LMS algorithm for COMPLEX valued data.
        (Algorithm 6.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = SM.NLMS(Filter, desired_signal, input_signal, verbose)

    Inputs
    -------
        filter  : Adaptive Filter                       filter object
        desired : Desired signal                        numpy array (row vector)
        input   : Input signal to feed filter           numpy array (row vector)
        gamma_bar : Upper bound for the error modulus.
        gamma_barVector       : Upper bound vector for the error modulus. (COLUMN vector)
        gamma                 : Regularization factor.
        memoryLength          : Reuse data factor.
        verbose : Verbose boolean                       bool

    Outputs
    -------
        dictionary:
            outputs      : Store the estimated output of each iteration.        numpy array (collumn vector)
            errors       : Store the error for each iteration.                  numpy array (collumn vector)
            coefficients : Store the estimated coefficients for each iteration. numpy array (collumn vector)

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


    Comments:
            Set-membership filtering implies that the (adaptive) filter coefficients are only
        updated if the magnitude of the error is greater than S.gamma_bar. In practice, we
        choose S.gamma_bar as a function of the noise variance (sigma_n2). A commom choice
        is S.gamma_bar = sqrt(5 * sigma_n2).

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
    nUpdates = 0

    # Main Loop

    prefixedInput = np.concatenate(
        (np.zeros(Filter.filter_order), input_signal))

    for it in range(nIterations):

        regressor = prefixedInput[it+(Filter.filter_order):-1]

        coefficients = Filter.coefficients

        output_it = np.dot(coefficients.conj(), regressor)

        error_it = desired_signal[it] - output_it

        if abs(error_it) > gamma_bar:

            nUpdates += 1

            mu = 1 - (gamma_bar/abs(error_it))

        else:

            mu = 0
            
        error_vector = np.append(error_vector, error_it)
        outputs_vector = np.append(outputs_vector, output_it)
        
        coefficients = coefficients + mu / (gamma + regressor.T.conj()*regressor)*(error_it.conj()*regressor)

        Filter.coefficients = coefficients

    if verbose == True:
        print(" ")
        print('Total runtime {:.03} ms'.format((time() - tic)*1000))

    return {'outputs': outputs_vector,
            'errors': error_vector, 'coefficients': Filter.coefficients_history}, nUpdates

#   EOF
