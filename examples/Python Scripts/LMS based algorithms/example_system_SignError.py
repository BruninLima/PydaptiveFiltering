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
#     Adaptive Algorithm used here: Sign-Error LMS                              #
#                                                                               #
#################################################################################

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pydaptivefiltering.lms import SignError
from pydaptivefiltering.base import AdaptiveFilter

# Define a Plant class to simulate the unknown system
class Plant(AdaptiveFilter):
    def optimize(self, **kwargs): 
        """The Plant represents the static unknown system and does not update."""
        pass

def run_sign_error_example():
    """
    Example for the Sign-Error LMS algorithm.
    Algorithm 4.1 - Adaptive Filtering: Algorithms and Practical Implementation, Diniz.
    The update uses the sign of the error: w(k+1) = w(k) + 2 * mu * sgn(e(k)) * x(k)
    """
    # 1. Experiment Parameters
    n_samples = 1200  # Sign algorithms typically need more samples to converge
    h_unknown = np.array([0.32, -0.3, 0.5, 0.2]) # Real coefficients for standard Sign-Error
    filter_order = len(h_unknown) - 1
    sigma_n2 = 0.01  
    mu = 0.005       # Lower step size for better stability in sign-based algorithms

    # 2. Signal Generation
    # Standard Sign-Error is often used with real signals
    x = np.random.randn(n_samples) 
    noise = np.sqrt(sigma_n2) * np.random.randn(n_samples)
    
    # Generate Desired Signal d(k)
    plant = Plant(filter_order, w_init=h_unknown)
    d = plant.filter_signal(x) + noise

    # 3. Adaptive Filtering Execution
    se_filter = SignError(filter_order=filter_order, step=mu)
    
    print("-" * 60)
    print("Starting Sign-Error LMS Adaptation")
    print(f"Samples: {n_samples} | Step size (mu): {mu}")
    
    tic = time()
    results = se_filter.optimize(input_signal=x, desired_signal=d, verbose=False)
    runtime = (time() - tic) * 1000
    
    print(f"Adaptation Finished in {runtime:.03f} ms")
    print("-" * 60)

    # 4. Graphical Visualization
    plt.figure(figsize=(14, 10))
    
    # --- Subplot 1: MSE Learning Curve (Smoothed) ---
    plt.subplot(2, 2, 1)
    mse = np.abs(results['errors'])**2
    # Moving average to clarify the convergence trend
    smoothed_mse = np.convolve(mse, np.ones(50)/50, mode='valid')
    plt.semilogy(mse, alpha=0.3, label='Instantaneous MSE')
    plt.semilogy(smoothed_mse, color='black', label='Smoothed MSE (w=50)')
    plt.axhline(y=sigma_n2, color='r', linestyle='--', label='Noise Floor')
    plt.title('Learning Curve: MSE')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()

    # --- Subplot 2: Coefficients Evolution ---
    plt.subplot(2, 2, 2)
    weights_hist = np.array(results['coefficients'])
    plt.plot(weights_hist)
    for val in h_unknown:
        plt.axhline(y=val, color='black', linestyle=':', linewidth=1, alpha=0.7)
    plt.title('Weight Tracks')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)

    # --- Subplot 3: Final Error Distribution ---
    plt.subplot(2, 1, 2)
    plt.plot(results['errors'], color='green', alpha=0.6)
    plt.title('Error Signal e(k)')
    plt.xlabel('Samples')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sign_error_example()

# EOF