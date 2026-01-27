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
#     Adaptive Algorithm used here: Sign-Data LMS                               #
#                                                                               #
#################################################################################

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pydaptivefiltering.LMS import SignData
from pydaptivefiltering.main import AdaptiveFilter

# Define a Plant class to simulate the unknown system
class Plant(AdaptiveFilter):
    def optimize(self, **kwargs): 
        """The Plant represents the static unknown system and does not update."""
        pass

def run_sign_data_example():
    """
    Example for the Sign-Data LMS algorithm.
    Useful for reduced computational complexity by using the sign of the input.
    """
    # 1. Experiment Parameters
    # Sign algorithms often require more samples to converge than standard LMS
    n_samples = 1000  
    h_unknown = np.array([0.32, -0.3, 0.5, 0.2]) # Real coefficients for standard Sign-Data
    filter_order = len(h_unknown) - 1
    sigma_n2 = 0.01  
    mu = 0.01        # Smaller step size is often needed for stability in Sign algorithms

    # 2. Signal Generation
    x = np.random.randn(n_samples) # Real input for standard Sign-Data
    noise = np.sqrt(sigma_n2) * np.random.randn(n_samples)
    
    # Generate Desired Signal d(k)
    plant = Plant(filter_order, w_init=h_unknown)
    d = plant.filter_signal(x) + noise

    # 3. Adaptive Filtering Execution
    # Standard Sign-Data update: w(k+1) = w(k) + 2 * mu * e(k) * sgn(x(k))
    sd_filter = SignData(filter_order=filter_order, step=mu)
    
    print("-" * 60)
    print(f"Starting Sign-Data LMS Adaptation")
    print(f"Number of Samples: {n_samples} | Step size: {mu}")
    
    tic = time()
    results = sd_filter.optimize(input_signal=x, desired_signal=d, verbose=False)
    runtime = (time() - tic) * 1000
    
    print(f"Adaptation Finished in {runtime:.03f} ms")
    print("-" * 60)

    # 4. Graphical Visualization
    plt.figure(figsize=(14, 10))
    
    # --- Subplot 1: MSE Learning Curve ---
    plt.subplot(2, 2, 1)
    mse = np.abs(results['errors'])**2
    plt.semilogy(mse, label='Sign-Data MSE', alpha=0.6)
    # Moving average to see convergence trend clearly
    plt.semilogy(np.convolve(mse, np.ones(50)/50, mode='valid'), label='Smoothed MSE', color='black')
    plt.axhline(y=sigma_n2, color='r', linestyle='--', label='Noise Floor')
    plt.title('Learning Curve: MSE')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()

    # --- Subplot 2: Error Signal ---
    plt.subplot(2, 2, 2)
    plt.plot(results['errors'], color='green', alpha=0.5)
    plt.title('Estimation Error e(k)')
    plt.xlabel('Samples')
    plt.grid(True, alpha=0.3)

    # --- Subplot 3: Coefficients Evolution ---
    plt.subplot(2, 1, 2)
    weights_hist = np.array(results['coefficients'])
    plt.plot(weights_hist)
    for val in h_unknown:
        plt.axhline(y=val, color='black', linestyle=':', linewidth=1.2)
    plt.title('Coefficients Evolution')
    plt.xlabel('Iterations')
    plt.ylabel('Weight Value')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sign_data_example()

# EOF