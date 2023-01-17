#  ErrorEquation.py
#
#      Implements the Error Equation RLS algorithm for REAL valued data.
#      (Algorithm 10.3 - book: Adaptive Filtering: Algorithms and Practical
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


def ErrorEquation(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, Lambda: float, M: int, N: int, delta: float, verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the Error Equation RLS algorithm for REAL valued data.
        (Algorithm 10.3 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = ErrorEquation(Filter, desired_signal, input_signal, Lambda, verbose)

    Inputs
    -------
        filter  : Adaptive Filter                                      filter object
        desired : Desired signal                                       numpy array (row vector)
        input   : Input signal to feed filter                          numpy array (row vector)
        Lambda  : Forgetting factor                                    float
        M       : Adaptive Filter numerator order                      int
        N       : Adaptive Filter denominator order                    int
        delta   : Regularization factor                                float
        verbose : Verbose boolean                                      bool

    Outputs
    -------
        dictionary:
            outputs      : Store the estimated output of each iteration.        numpy array (column vector)
            errors       : Store the error for each iteration.                  numpy array (column vector)
            coefficients : Store the estimated coefficients for each iteration  numpy array (column vector)
            thetaVector  : Store the estimated theta for each iteration         numpy array (column vector)
            errorVector_e: Store the auxiliary error for each iteration         numpy array (column vector)

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
    nCoefficients = M + N + 1

    regressor = np.zeros(nCoefficients, dtype=input_signal.dtype)
    regressor_e = np.zeros(nCoefficients, dtype=input_signal.dtype)
    errorVector = np.array([])
    errorVector_e = np.array([])
    outputsVector = np.array([])
    outputsVector_e = np.array([])
    thetaVector = np.array([])
    S_d = np.zeros(nCoefficients, nCoefficients)

    # Initial State Weight Vector
    S_d = np.linalg.inv(delta*np.eye(nCoefficients))

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
        ULineMatrix[it, 1] = (Lambda**(it/2))*input_signal[nCoefficients-it]
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
