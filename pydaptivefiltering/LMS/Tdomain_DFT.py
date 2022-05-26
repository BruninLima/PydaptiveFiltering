#  LMS.Tdomain_DFT.py
#
#      Implements the Transform-Domain LMS algorithm, based on the Discrete
#      Fourier Transform (DFT), for COMPLEX valued data.
#      (Algorithm 4.4 - book: Adaptive Filtering: Algorithms and Practical
#                                                      Implementation, Diniz)
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
from scipy.fft import fft, ifft


def Tdomain_DFT(Filter, desired_signal: np.ndarray, input_signal: np.ndarray, step: float = 1e-2, verbose: bool = False) -> dict:
    """
    Description
    -----------
        Implements the Transform-Domain LMS algorithm, based on the Discrete
        Fourier Transform (DFT), for COMPLEX valued data.
        (Algorithm 4.4 - book: Adaptive Filtering: Algorithms and Practical
                                                        Implementation, Diniz)

    Syntax
    ------
    OutputDictionary = Tdomain_DFT(Filter, desired_signal, input_signal, step, verbose)

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
        step    : Convergence (relaxation) factor.      float
        verbose : Verbose boolean                       bool

    Outputs
    -------
        dictionary:
            outputs      : Store the estimated output of each iteration.        numpy array (collumn vector)
            errors       : Store the error for each iteration.                  numpy array (collumn vector)
            coefficients : Store the estimated coefficients for each iteration  numpy array (collumn vector)
                                in the ORIGINAL domain.
            coefficientsDFT: Store the estimated coefficients for each iteration numpy array (collumn vector)
                                in the TRANSFORM domain.

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
    coefficientVectorDFT = np.array([])

    coefficientVectorDFT.append(coefficientVectorDFT, fft(Filter.coefficients)/np.sqrt(Filter.filter_order + 1))
    powerVector = initialPower*np.ones((Filter.filter_order + 1))

    # Main Loop
    for it in range(nIterations):

        regressorDFT = fft(np.concatenate(([input_signal[it]], regressor))[
            :Filter.filter_order+1]) / np.sqrt(Filter.filter_order + 1)

        powerVector = alpha*(regressorDFT * regressorDFT.conj()
                             ) + (1 - alpha)*(powerVector)

        output_it = np.dot(coefficientVectorDFT[it].conj(), regressorDFT)

        error_it = desired_signal[it] - output_it

        next_coefficients = coefficientVectorDFT[-1] + step * \
            error_it.conj() * regressorDFT / (gamma + powerVector)

        coefficientVectorDFT = np.append(
            coefficientVectorDFT, next_coefficients)

        error_vector = np.append(error_vector, error_it)

        outputs_vector = np.append(outputs_vector, output_it)

        Filter.coefficients = ifft(next_coefficients)
        Filter.coefficients_history.append(next_coefficients)

    if verbose == True:
        print(" ")
        print('Total runtime {:.03} ms'.format((time() - tic)*1000))

    return {'outputs': outputs_vector,
            'errors': error_vector, 'coefficients': Filter.coefficients_history, 'coefficientsDFT': coefficientVectorDFT, 'adaptedFilter': Filter}

#   EOF
