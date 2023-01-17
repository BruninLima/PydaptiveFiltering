#  RLS_Alt.py
#
#      Implements the Alternative RLS algorithm for COMPLEX valued data.
#      RLS_Alt differs from RLS in the number of computations. The RLS_Alt
#      uses an auxiliar variable (psi) in order to reduce the computational burden.
#      (Algorithm 5.4 - book: Adaptive Filtering: Algorithms and Practical
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


def RLS_Alt(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, Delta: float, Lambda: float, verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the Alternative RLS algorithm for COMPLEX valued data. 
        (Algorithm 5.4 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = RLS(Filter, desired_signal, input_signal, Delta, Lambda, verbose)

    Inputs
    -------
        filter  : Adaptive Filter                       filter object
        desired : Desired signal                        numpy array (row vector)
        input   : Input signal to feed filter           numpy array (row vector)
        Delta   :
        Lambda  :
        verbose : Verbose boolean                       bool

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
    nIterations = desired_signal.size

    regressor = np.zeros(Filter.filter_order+1, dtype=input_signal.dtype)
    error_vector = np.array([])
    outputs_vector = np.array([])
    outputs_posteriori = np.array([])
    errors_posteriori = np.array([])

    S_d = Delta*np.eye(Filter.filter_order+1)

    # Main Loop
    for it in range(nIterations):

        regressor = np.concatenate(([input_signal[it]], regressor))[
            :Filter.filter_order+1]

        coefficients = Filter.coefficients
        #   a priori estimated output
        output_it = np.dot(coefficients.conj(), regressor)
        outputs_vector = np.append(outputs_vector, output_it)

        #   a priori error
        error_it = desired_signal[it] - output_it
        error_vector = np.append(error_vector, error_it)

        psi = S_d * regressor

        S_d = (1/Lambda)*(S_d - (psi*psi.conj()) /
                          (Lambda + psi.conj()*regressor))

        next_coefficients = coefficients + error_it.conj() * S_d @ regressor

        #   a posteriori estimated output
        outputs_posteriori = np.append(
            outputs_posteriori, np.dot(next_coefficients.conj(), regressor))
        #   a posteriori estiamted error
        errors_posteriori = np.append(
            errors_posteriori, desired_signal[it] - outputs_posteriori[-1])

        #   Adapt Filter
        Filter.coefficients = next_coefficients
        Filter.coefficients_history.append(next_coefficients)

    if verbose == True:
        print(" ")
        print('Total runtime {:.03} ms'.format((time() - tic)*1000))

    return {'outputs': outputs_vector,
            'errors': error_vector, 'coefficients': Filter.coefficients_history}

#   EOF
