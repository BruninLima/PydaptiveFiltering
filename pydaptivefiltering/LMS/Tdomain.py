#  LMS.Tdomain.py
#
#      Implements the Transform-Domain LMS algorithm for COMPLEX valued data.
#      (Algorithm 4.4 - book: Adaptive Filtering: Algorithms and Practical
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
from scipy.fftpack import dct


def Tdomain(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, gamma: float, alpha: float, initialPower: float, T, step: float = 1e-2, verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the Transform-Domain LMS algorithm for COMPLEX valued data.
        (Algorithm 4.4 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = Tdomain(Filter, desired_signal, input_signal, step, verbose)

    Inputs
    -------
        filter  : Adaptive Filter                       filter object
        desired : Desired signal                        numpy array (row vector)
        input   : Input signal to feed filter           numpy array (row vector)
        gamma   : Regularization factor.                float
                    (small positive constant to avoid singularity)
        alpha   : Used to estimate eigenvalues of Ru    float
                    (0 << alpha << 0.1)
        initialPower: Initial Power.                    float
        T       : Transform applied to the signal       unitary MATRIX (filterOrder+1 x filterOrder+1)
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
    coefficientsVectorT = np.array([])

    # Main Loop
    for it in range(nIterations):

        regressor = np.concatenate(([input_signal[it]], regressor))[
            :Filter.filter_order+1]

        regressorT = T @ regressor

        powerVector = alpha*(regressorT * regressorT.conj()
                             ) + (1 - alpha)*(powerVector)

        coefficients = Filter.coefficients

        output_it = np.dot(coefficients.conj(), regressorT)

        error_it = desired_signal[it] - output_it

        next_coefficients = coefficients + step * error_it.conj() * regressorT / \
            (gamma + powerVector)

        error_vector = np.append(error_vector, error_it)
        outputs_vector = np.append(outputs_vector, output_it)
        coefficientsVectorT = np.append(coefficientsVectorT, next_coefficients)

        Filter.coefficients = T.conj() @ next_coefficients
        Filter.coefficients_history.append(next_coefficients)

    if verbose == True:
        print(" ")
        print('Total runtime {:.03} ms'.format((time() - tic)*1000))

    return {'outputs': outputs_vector,
            'errors': error_vector, 'coefficients': Filter.coefficients_history, 'coefficientsT': coefficientsVectorT, 'adaptedFilter': Filter}

#   EOF
