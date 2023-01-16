#  RLS.rls.py
#
#      Implements the RLS algorithm for COMPLEX valued data.
#      (Algorithm 5.3 - book: Adaptive Filtering: Algorithms and Practical
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


def RLS(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, Delta: float, Lambda: float, verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the RLS algorithm for COMPLEX valued data. 
        (Algorithm 5.3 - book: Adaptive Filtering: Algorithms and Practical Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = RLS(Filter, desired_signal, input_signal, Delta, Lambda, verbose)

    Inputs
    -------
        filter  : Adaptive Filter                                      filter object
        desired : Desired signal                                       numpy array (row vector)
        input   : Input signal to feed filter                          numpy array (row vector)
        Delta   : The matrix delta*eye is the initial value of the
                inverse of the deterministic autocorrelation matrix.   float
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
    nIterations = input_signal.size
    nCoefficients = Filter.filter_order + 1


    # Pre Allocations
    coefficientVector = np.zeros((nCoefficients, nIterations + 1))
    errorVector       = np.zeros((nIterations,))
    outputVector      = np.zeros((nIterations,))
    errorPosteriori   = np.zeros((nIterations,))
    outputPosteriori  = np.zeros((nIterations,))

    regressor = np.zeros(Filter.filter_order+1, dtype=input_signal.dtype)

    S_d = Delta*np.eye(Filter.filter_order+1)
    p_d = np.zeros(Filter.filter_order+1)

    # Initial Coefficients Values
    coefficientVector[:, 0] = S_d@p_d

    # Main Loop
    for it in range(nIterations):

        regressor = np.concatenate(([input_signal[it]], regressor))[
            :Filter.filter_order+1]

        #   a priori estimated output
        outputVector[it] = np.dot(coefficientVector[:, it].conj(), regressor)

        #   a priori error
        errorVector[it] = desired_signal[it] - outputVector[it]
        
        S_d = (1/Lambda)*(S_d - (S_d * regressor * regressor.conj()*S_d) /
                          (Lambda + regressor.T.conj()*S_d*regressor))
        p_d = Lambda*p_d + desired_signal[it].conj()*regressor

        next_coefficients = S_d@p_d

        #   a posteriori estimated output
        outputPosteriori[it] = np.dot(next_coefficients.conj(), regressor)
        #   a posteriori estiamted error
        errorPosteriori[it] = desired_signal[it] - outputPosteriori[it]

        #   Adapt Filter
        Filter.coefficients = next_coefficients
        Filter.coefficients_history.append(next_coefficients)

    if verbose == True:
        print(" ")
        print('Total runtime {:.03} ms'.format((time() - tic)*1000))

    return {'outputs': outputVector,
            'errors': errorVector,
            'coefficients': Filter.coefficients_history,
            'outputs_posteriori': outputPosteriori,
            'errors_posteriori': errorPosteriori}

#   EOF
