import numpy as np
import pydaptivefiltering as pdf
import matplotlib.pyplot as plt


def LMS_example():
    """
    Types:
    ------
    None -> '(Filter, dictionary["outputs", "errors", "coefficients"])'

    Example for the LMS algorithm

    """
    # Parameters
    K = 50  # Number of iterations
    H = np.array([0.32+0.21*1j, -0.3+0.7*1j, 0.5-0.8*1j, 0.2+0.5*1j]).T
    Wo = H  # Uknown System
    sigman2 = 0.04  # Noise Power
    N = 4    # Number of coefficients of the adaptative filter
    step = 0.2  # Convergence factor (step) (0 < μ < 1)
    tol = 0.0  # Tolerance: If |error| < tol, stops at the current run.
    # Initializing
    W = np.ones(shape=(N, K+1))
    X = np.zeros(N)   # Input at a certain iteration (tapped delay line)
    x = (np.random.randn(K) + np.random.randn(K)*1j)/np.sqrt(2)
    n = np.sqrt(sigman2/2) * (np.random.randn(K) +
                              np.random.randn(K)*1j)  # complex noise
    d = []

    for i in range(K):
        X = np.concatenate(([x[i]], X))[:N]  # (tapped delay line)
        d.append(np.dot(Wo.conj(), X))

    d = np.array(d)  # + n

    Filter = pdf.AdaptiveFilter(W[:, 1])  # Istanciating Adaptive Filter
    # Adapting with the LMS Algorithm
    Output = Filter.adapt_LMS(d, x, step, tolerance=tol)

    return (Filter, Output)


Filter, Result = LMS_example()
