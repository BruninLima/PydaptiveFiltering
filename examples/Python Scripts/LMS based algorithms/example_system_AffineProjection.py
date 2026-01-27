#################################################################################
#                         Example: System Identification                        #
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
#     Adaptive Algorithm used here: Affine Projection                           #
#                                                                               #
#################################################################################

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pydaptivefiltering.lms import AffineProjection
from pydaptivefiltering.base import AdaptiveFilter

# Define a Plant class to simulate the unknown system (Reference System)
class Plant(AdaptiveFilter):
    def optimize(self, **kwargs): 
        """The Plant represents the static unknown system and does not update."""
        pass

def run_affine_projection_example():
    """
    Example for the Affine Projection (AP) algorithm.
    Algorithm 4.6 - Adaptive Filtering: Algorithms and Practical Implementation, Diniz.
    """
    # 1. Experiment Parameters
    n_samples = 800
    # Unknown complex coefficients (Wo)
    h_unknown = np.array([0.32+0.21*1j, -0.3+0.7*1j, 0.5-0.8*1j, 0.2+0.5*1j])
    filter_order = len(h_unknown) - 1
    sigma_n2 = 0.04  # Noise power
    mu = 0.1         # Step size
    L = 2            # Reuse data factor
    gamma = 1e-8     # Regularization factor (avoid division by zero)

    # 2. Signal Generation
    # Input x(k): Complex white noise
    x = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
    
    # Complex Additive White Gaussian Noise (AWGN)
    noise = np.sqrt(sigma_n2/2) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    
    # Generate Desired Signal d(k) = Wo^H * x(k) + n(k)
    plant = Plant(filter_order, w_init=h_unknown)
    d = plant.filter_signal(x) + noise

    # 3. Adaptive Filtering Execution
    ap_filter = AffineProjection(filter_order=filter_order, step=mu, L=L, gamma=gamma)
    
    print("-" * 60)
    print(f"Starting Affine Projection Adaptation (L={L})")
    print(f"Number of Samples: {n_samples} | Step size: {mu}")
    
    tic = time()
    results = ap_filter.optimize(input_signal=x, desired_signal=d, verbose=False)
    runtime = (time() - tic) * 1000
    
    print(f"Adaptation Finished in {runtime:.03f} ms")
    print("-" * 60)

    # 4. Graphical Visualization
    plt.figure(figsize=(14, 10))
    
    # --- Subplot 1: MSE Learning Curve ---
    plt.subplot(2, 2, 1)
    # Calculate Squared Error Magnitude
    mse = np.abs(results['errors'])**2
    plt.semilogy(mse, label='AP MSE')
    plt.axhline(y=sigma_n2, color='r', linestyle='--', label='Noise Floor')
    plt.title('Learning Curve: MSE [Magnitude]')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()

    # --- Subplot 2: Noise Power Analysis ---
    plt.subplot(2, 2, 2)
    noise_power = np.abs(noise)**2
    plt.semilogy(noise_power, color='orange', alpha=0.4, label='Complex Noise Power')
    plt.title('Instantaneous Noise Power')
    plt.xlabel('Samples')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()

    # --- Subplot 3: Coefficients Evolution (Real Part) ---
    plt.subplot(2, 2, 3)
    weights_hist = np.array(results['coefficients'])
    plt.plot(weights_hist.real)
    for val in h_unknown.real:
        plt.axhline(y=val, color='black', linestyle=':', linewidth=1, alpha=0.6)
    plt.title('Coefficients Evolution (Real Part)')
    plt.xlabel('Iterations')
    plt.grid(True, alpha=0.3)

    # --- Subplot 4: Coefficients Evolution (Imaginary Part) ---
    plt.subplot(2, 2, 4)
    plt.plot(weights_hist.imag)
    for val in h_unknown.imag:
        plt.axhline(y=val, color='black', linestyle=':', linewidth=1, alpha=0.6)
    plt.title('Coefficients Evolution (Imaginary Part)')
    plt.xlabel('Iterations')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_affine_projection_example()

# EOF