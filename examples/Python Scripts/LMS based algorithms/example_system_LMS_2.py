#################################################################################
#                        Example: System Identification                         #
#################################################################################
#                                                                               #
#  In this example we have a typical system identification scenario. We want    #
# to estimate the filter coefficients of an unknown system given by Wo. In      #
# order to accomplish this task we use an adaptive filter with the same         #
# number of coefficients, N, as the unknown system. The procedure is:           #
# 1)  Excitate both filters (the unknown and the adaptive) with the signal      #
#   x. In this case, x is chosen according to the 4-QAM constellation.          #
#   The variance of x is normalized to 1.                                       #
# 2)  Generate the desired signal, d = Wo' x + n, which is the output of the    #
#   unknown system considering some disturbance (noise) in the model. The       #
#   noise power is given by sigma_n2.                                           #
# 3)  Choose an adaptive filtering algorithm to govern the rules of coefficient #
#   updating.                                                                   #
#                                                                               #
#     Adaptive Algorithm used here: LMS                                         #
#                                                                               #
#################################################################################

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pydaptivefiltering as pdf


def LMS_example():
    # Number of realizations within the ensemble
    ensemble = 109
    # Number of iterations
    K = 500
    # Unknown system
    H = np.array([0.32 + 0.21j, -0.3 + 0.7j, 0.5 - 0.8j, 0.2 + 0.5j])
    Wo = H
    # Noise power
    sigma_n2 = 0.04
    # Number of coefficients of the adaptive filter
    N = 4
    # Convergence factor (step)  (0 < mu < 1)
    mu = 0.1

    # Initializing & Allocating memory
    W = np.ones((N, K+1, ensemble), dtype=H.dtype)
    Filter = pdf.AdaptiveFilter(W[:, 1, 0])
    MSE = np.zeros((K, ensemble))
    MSEmin = np.zeros((K, ensemble))

    for l in range(ensemble):
        X = np.zeros((N, 1))
        d = []
        # Creating the input signal (normalized)
        x = (np.random.randn(K) + np.random.randn(K)*1j)/np.sqrt(2)
        sigma_x2 = np.var(x)
        # Complex noise
        n = np.sqrt(sigma_n2 / 2) * (np.random.randn(K, 1) +
                                     1j * np.random.randn(K, 1))

        for k in range(K):
            # Tapped delay line
            X = np.vstack((x[k], X[:(N-1), :]))
            d.append(np.dot(Wo.conj(), X))

        d = np.array(d) + n
        print(" Adapting with LMS \n")
        Output = pdf.LMS.LMS(Filter, d, x, mu)
        Coefficients = np.array(Output['coefficients']).T
        errorSignal, W[:, :, l] = Output['errors'], Coefficients

        MSE[:, l] = MSE[:, l] + (np.abs(errorSignal)) ** 2
        MSEmin = MSEmin + (np.abs(n)) ** 2

    W_av = np.sum(W, axis=2) / ensemble
    MSE_av = np.sum(MSE, axis=1) / ensemble
    MSEmin_av = np.sum(MSEmin, axis=1) / ensemble

    # Plotting
    plt.figure(figsize=(16, 16))

    plt.subplot(221)
    plt.gca().set_title('Learning Curve for MSE [dB]')
    plt.plot(np.arange(1, K+1), MSE_av)
    plt.gca().semilogy(MSE)
    plt.grid()

    plt.subplot(222)
    plt.gca().set_title('Learning Curve for MSEmin [dB]')
    plt.gca().semilogy(MSEmin)
    plt.grid()

    plt.subplot(223)
    plt.gca().set_title('Evolution of the Coefficients (Real Part)')
    real_part = np.real(W_av[0, :])
    plt.gca().plot(real_part)
    plt.grid()

    plt.subplot(224)
    plt.gca().set_title('Evolution of the Coefficients (Imaginary Part)')
    imag_part = np.imag(W_av[0, :])
    plt.gca().plot(imag_part)
    plt.grid()

    plt.tight_layout(pad=4.0)
    plt.show()


if __name__ == "__main__":
    LMS_example()
