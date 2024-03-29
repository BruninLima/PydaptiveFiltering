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
#     Adaptive Algorithm used here: DFT-Transform-Domain LMS                    #
#                                                                               #
#################################################################################

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pydaptivefiltering as pdf


def Tdomain_DFT_example():
    """
    Types:
    ------
    None -> '(Filter, dictionary["outputs", "errors", "coefficients"])'

    Example for the DFT-Transform-Domain LMS algorithm. 

    """
    # Parameters
    K = 70                  # Number of iterations
    H = np.array([0.32+0.21*1j, -0.3+0.7*1j, 0.5-0.8*1j, 0.2+0.5*1j])
    Wo = H                  # Unknown System
    sigma_n2 = 0.04         # Noise Power
    N = 4                   # Number of coefficients of the adaptive filter
    gamma = 1e-8            # Small constant to avoid division by zero
    alpha = 0.097           # small positive constant   (0 < α <= 1)
    initialPower = 0        # Initial Power to normalize the input signal
    mu = 0.1                # Convergence factor (step) (0 < μ < 1)

    # Initializing
    W = np.ones(shape=(N, K+1))
    # Input at a certain iteration (tapped delay line)
    X = np.zeros(N)
    x = (np.random.randn(K) + np.random.randn(K)*1j)/np.sqrt(2)
    # complex noise
    n = np.sqrt(sigma_n2/2) * (np.random.randn(K) +
                               np.random.randn(K)*1j)
    d = []

    for k in range(K):

        # input signal (tapped delay line)
        X = np.concatenate(([x[k]], X))[:N]
        d.append(np.dot(Wo.conj(), X))

    # desired signal
    d = np.array(d) + n

    # Adaptive Filter
    Filter = pdf.AdaptiveFilter(W[:, 1])
    print(" Adapting with DCT-Transform-Domain LMS algorithm...")
    # Adaptive Algorithm
    Output = pdf.LMS.Tdomain_DFT(Filter, d, x, alpha, gamma, initialPower, mu)

    return (Filter, Output, n)


if __name__ == "__main__":
    # Running the model
    Filter, Output, ComplexNoise = Tdomain_DFT_example()

    # Plotting
    plt.figure(figsize=(16, 16))

    plt.subplot(221)
    plt.gca().set_title('Learning Curve for MSE [dB]')
    MSE = [abs(err)**2 for err in Output['errors']]
    plt.gca().semilogy(MSE)
    plt.grid()
    plt.subplot(222)
    plt.gca().set_title('Learning Curve for MSEmin [dB]')
    MSEmin = [abs(n)**2 for n in ComplexNoise]
    plt.gca().semilogy(MSEmin)
    plt.grid()
    plt.subplot(223)
    plt.gca().set_title('Evolution of the Coefficients (Real Part)')
    real_part = [coef.real for coef in Output['coefficients']]
    plt.gca().plot(real_part)
    plt.grid()
    plt.subplot(224)
    plt.gca().set_title('Evolution of the Coefficients (Imaginary Part)')
    imag_part = [coef.imag for coef in Output['coefficients']]
    plt.gca().plot(imag_part)
    plt.grid()

    plt.tight_layout(pad=4.0)
    plt.show()

# EOF
