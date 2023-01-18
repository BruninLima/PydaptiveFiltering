#################################################################################
#                        Example: Channel Equalization                          #
#################################################################################
#                                                                               #
#  In this example we have a typical channel equalization scenario. We want     #
# to estimate the transmitted sequence with 4-QAM symbols. In                   #
# order to accomplish this task we use an adaptive filter with N coefficients.  #
# The procedure is:                                                             #
# 1)  Apply the originally transmitted signal distorted by the channel plus     #
#   environment noise as the input signal to an adaptive filter.                #
#   In this case, the transmitted signal is a random sequence with 4-QAM        #
#   symbols and unit variance. The channel is a multipath complex-valued        #
#   channel whose impulse response is h = [1.1+j*0.5, 0.1-j*0.3, -0.2-j*0.1]^T  #
#   In addition, the environment noise is AWGN with zero mean and               #
#   variance 10^(-2.5).                                                         #
# 2)  Choose an adaptive filtering algorithm to govern the rules of coefficient #
#   updating.                                                                   #
#                                                                               #
#     Adaptive Algorithm used here: Affine_projectionCM                         #
#                                                                               #
#################################################################################

import numpy as np
import matplotlib.pyplot as plt
import pydaptivefiltering as pdf
from numpy.linalg import toeplitz

ensemble = 200
K = 10000
Ksim = 400
H = np.array([1.1 + 1j*0.5, 0.1-1j*0.3, -0.2-1j*0.1])
sigma_x2 = 1
sigma_n2 = 10**(-2.5)
N = 5
mu = 0.001
delay = 1
constellation = np.array([(2 + 2j), (-2 + 2j), (2 - 2j), (-2 - 2j)])/np.sqrt(2)
HMatrix = toeplitz([H[0]] + [0]*(N-1), np.concatenate((H, [0]*(N-1))))

# Finding the Wiener filter
Rx = sigma_x2*np.eye(N+len(H)-1)
Rn = sigma_n2*np.eye(N)
Ry = np.matmul(np.matmul(HMatrix, Rx), np.conj(HMatrix.T)) + Rn
RxDeltaY = np.concatenate((np.concatenate((np.zeros(delay), np.array(
    [sigma_x2]))), np.zeros(N+len(H)-2-delay)))*(np.conj(HMatrix.T))
Wiener = np.conj(np.matmul(np.linalg.inv(Ry), RxDeltaY)).T

# Initializing & allocating memory
W = np.tile(Wiener, (1, (K+1-delay), ensemble)) + (np.random.normal(0, 1, (N,
                                                                           (K+1-delay), ensemble)) + 1j*np.random.normal(0, 1, (N, (K+1-delay), ensemble)))/4
MSE = np.zeros((K-delay, ensemble))

# Computing
for l in range(ensemble):
    n = np.sqrt(sigma_n2)*np.random.normal(0, 1, (1, K)) + \
        1j*np.random.normal(0, 1, (1, K))
    s = np.random.choice(constellation, 1, K)
    x = np.convolve(s, H) + n

    S = {'step': mu, 'filterOrderNo': (N-1), 'initialCoefficients': W[:, 0, l]}

    y, e, W[:, :, l] = CMA(x[delay:], S)

    MSE[:, l] = MSE[:, l] + (np.abs(e[:, 0])**2)
