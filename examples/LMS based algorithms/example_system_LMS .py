import numpy as np
import pydaptivefiltering as pdf


def LMS_example():
    """
    Types:
    ------
    None -> '(Filter, dictionary["outputs", "errors", "coefficients"])'

    Example for the LMS algorithm, SignData and SignError. 

    """
    # Parameters
    # Number of iterations
    K = 500
    H = np.array([0.32+0.21*1j, -0.3+0.7*1j, 0.5-0.8*1j, 0.2+0.5*1j]).T
    # Uknown System
    Wo = H
    # Noise Power
    sigman2 = 0.04
    # Number of coefficients of the adaptative filter
    N = 4
    # Convergence factor (step) (0 < μ < 1)
    step = 0.002
    # Tolerance: If |error| < tol, stops at the current run.
    tol = 0.0

    # Initializing
    W = np.ones(shape=(N, K+1))
    # Input at a certain iteration (tapped delay line)
    X = np.zeros(N)
    x = (np.random.randn(K) + np.random.randn(K)*1j)/np.sqrt(2)
    # complex noise
    n = np.sqrt(sigman2/2) * (np.random.randn(K) +
                              np.random.randn(K)*1j)
    d = []

    for i in range(K):
        # (tapped delay line)
        X = np.concatenate(([x[i]], X))[:N]
        d.append(np.dot(Wo.conj(), X))

    # desired signal
    d = np.array(d) + n

    # Istanciating Adaptive Filter
    Filter = pdf.AdaptiveFilter(W[:, 1])
    print(" Adapting with LMS \n")
    # Adapting with the LMS Algorithm
    Output = pdf.LMS.LMS(Filter, d, x, step, tolerance=tol)
    # Filter Reset
    Filter._reset()
    print(" Adaptign with SignData \n")
    # Adapting with the LMS - SignData Algorithm
    OutputSD = pdf.LMS.SignData(Filter, d, x, step, tolerance=tol)
    # Filter Reset
    Filter._reset()
    print(" Adapting with SignError \n")
    # Adapting with the LMS - SignError Algorithm
    OutputSE = pdf.LMS.SignError(Filter, d, x, step, tolerance=tol)

    Outputs = (Output, OutputSD, OutputSE)
    return (Filter, Outputs)


Filter, Result = LMS_example()
