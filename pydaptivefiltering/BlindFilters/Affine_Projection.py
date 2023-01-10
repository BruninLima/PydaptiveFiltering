#  BlindFilters.affine_projection.py
#
#      Implements the Complex Affine-Projection Constant-Modulus algorithm for COMPLEX valued data.
#      (Algorithm 13.4 - book: Adaptive Filtering: Algorithms and Practical
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


def affine_projection(Filter, input_signal: np.ndarray, gamma: float, L: int, step: float = 1e-2, verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the Complex Affine-Projection Constant-Modulus algorithm for COMPLEX valued data.
        (Algorithm 13.4 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = affine_projection(Filter, desired_signal, input_signal, L, gamma, step, verbose)

    Inputs
    -------
        filter  : Adaptive Filter                       filter object
        input   : Input signal to feed filter           numpy array (row vector)
        gamma   : Regularization factor.                float
        L       : Number of iterations to store.        int
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

    # Initialization
    tic = time()
    nIterations = input_signal.size

    regressor = np.zeros((L+1, Filter.filter_order+1),
                         dtype=input_signal.dtype)
    error_vector = np.array([])
    outputs_vector = np.array([])

    # Main Loop
    for it in range(nIterations):

        regressor[1:] = regressor[:-1]
        regressor[0] = input_signal[it:(Filter.filter_order)+it+1]

        coefficients = Filter.coefficients

        output_it = np.dot(regressor, coefficients.conj())

        desiredLevelConj = np.sign(output_it)

        error_it = (desiredLevelConj[it: L + 1] - output_it).conj()

        aux_matrix = np.linalg.inv(
            regressor @ regressor.T + gamma * np.eye(L+1))
        next_coefficients = coefficients + step * \
            (regressor.T @ aux_matrix @ error_it)

        error_vector = np.append(error_vector, error_it)
        outputs_vector = np.append(outputs_vector, output_it)

        Filter.coefficients = next_coefficients
        Filter.coefficients_history.append(next_coefficients)

    if verbose == True:
        print(" ")
        print('Total runtime {:.03} ms'.format((time() - tic)*1000))

    return {'outputs': outputs_vector,
            'errors': error_vector,
            'coefficients': Filter.coefficients_history}

#   EOF
