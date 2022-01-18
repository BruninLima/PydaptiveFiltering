#  RLS.py
#
#      Implements the QR-RLS algorithm for REAL valued data.
#      (Algorithm 9.1 - book: Adaptive Filtering: Algorithms and Practical
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


def RLS(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, Lambda: float, verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the QR-RLS algorithm for REAL valued data.
        (Algorithm 9.1 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = RLS(Filter, desired_signal, input_signal, Lambda, verbose)

    Inputs
    -------
        filter  : Adaptive Filter                                      filter object
        desired : Desired signal                                       numpy array (row vector)
        input   : Input signal to feed filter                          numpy array (row vector)
        Lambda  : Forgetting factor                                    float
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
    nIterations = desired_signal.size
    nCoefficients = Filter.filter_order + 1
    regressor = np.zeros(nCoefficients, dtype=input_signal.dtype)
    error_vector = np.array([])
    outputs_vector = np.array([])
    outputs_posteriori = np.array([])
    errors_posteriori = np.array([])

    # Scalar Values
    gamma = 0
    cosTetaI = 0
    sinTetaI = 0
    cI = 0
    dLine = 0

    # Backsubstituition Procedure
    coefficients = np.zeros(nCoefficients, nCoefficients)

    coefficients[0][0] += (desired_signal[0]/input_signal[0])

    for kt in range(nCoefficients):
        coefficients[1, kt] += (desired_signal[0]/input_signal[0])
        for ct in range(1, kt):
            coefficients[ct, kt] += desired_signal[ct] / \
                input_signal[1] - (input[1:ct]*coefficients[ct-1::1, kt])

    # Build Initial Matrices

    ULineMatrix = np.zeros(nCoefficients, nCoefficients)
    for it in range(nCoefficients):
        # Uline
        ULineMatrix[it, 1] = (lambda**(it/2))*input_signal[nCoefficients-it]
        # dLine_q2
        # TODO

    # Main Loop
    for it in range(nIterations):

        regressor = np.concatenate(([input_signal[it]], regressor))[
            :Filter.filter_order+1]

        #   a priori estimated output
        output_it = np.dot(coefficients.T.conj(), regressor)
        outputs_vector = np.append(outputs_vector, output_it)

        #   a priori error
        error_it = desired_signal[it] - output_it
        error_vector = np.append(error_vector, error_it)

        S_d = (1/Lambda)*(S_d - (S_d * regressor * regressor.conj()*S_d) /
                          (Lambda + regressor.T.conj()*S_d*regressor))
        p_d = Lambda*p_d + desired_signal[it].conj()*regressor

        next_coefficients = S_d@p_d

        #   a posteriori estimated output
        outputs_posteriori = np.append(
            outputs_posteriori, np.dot(next_coefficients.conj(), regressor))
        #   a posteriori estimated error
        errors_posteriori = np.append(
            errors_posteriori, desired_signal[it] - outputs_posteriori[-1])

        #   Adapt Filter
        Filter.coefficients = next_coefficients
        Filter.coefficients_history.append(next_coefficients)

    if verbose == True:
        print(" ")
        print('Total runtime {:.03} ms'.format((time() - tic)*1000))

    return {'outputs': outputs_vector,
            'errors': error_vector,
            'coefficients': Filter.coefficients_history,
            'outputs_posteriori': outputs_posteriori,
            'errors_posteriori': errors_posteriori}

#   EOF
