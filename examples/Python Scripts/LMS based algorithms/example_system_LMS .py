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
#     Adaptive Algorithm used here: LMS                                         #
#                                                                               #
#################################################################################

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pydaptivefiltering.LMS import LMS
from pydaptivefiltering.main import AdaptiveFilter

# Define a Plant class to simulate the unknown system (Reference System)
class Plant(AdaptiveFilter):
    def optimize(self, **kwargs): 
        """The Plant represents the static unknown system and does not update."""
        pass

def run_system_identification():
    """
    Example: System Identification using the LMS Class.
    Based on Algorithm 3.2 from Paulo S. R. Diniz's book.
    """
    # 1. Experiment Parameters
    n_samples = 600
    # Unknown complex coefficients (Wo)
    h_unknown = np.array([0.32+0.21*1j, -0.3+0.7*1j, 0.5-0.8*1j, 0.2+0.5*1j])
    filter_order = len(h_unknown) - 1
    sigma_n2 = 0.04  # Noise power (disturbance)
    mu = 0.1         # Step size (convergence factor)

    # 2. Signal Generation
    # Input x(k): Complex white noise normalized to variance 1
    x = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
    
    # Complex Additive White Gaussian Noise (AWGN)
    noise = np.sqrt(sigma_n2/2) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    
    # Generate Desired Signal d(k) = Wo^H * x(k) + n(k)
    plant = Plant(filter_order, w_init=h_unknown)
    d = plant.filter_signal(x) + noise

    # 3. Adaptive Filtering Execution
    lms_filter = LMS(filter_order=filter_order, step=mu)
    
    print("-" * 60)
    print(f"Starting LMS Adaptation (Filter Order: {filter_order})")
    print(f"Number of Samples: {n_samples} | Step size: {mu}")
    
    tic = time()
    results = lms_filter.optimize(input_signal=x, desired_signal=d, verbose=False)
    runtime = (time() - tic) * 1000
    
    print(f"Adaptation Finished in {runtime:.03f} ms")
    print("-" * 60)

    # 4. Numerical Comparison (Steady State analysis)
    y = results['outputs']
    print("\nSteady State Samples (Last 5):")
    print(f"{'Sample':<8} | {'Desired (d)':<30} | {'Filter Output (y)':<30}")
    for i in range(n_samples - 5, n_samples):
        d_str = f"{d[i].real:+.3f} {d[i].imag:+.3f}j"
        y_str = f"{y[i].real:+.3f} {y[i].imag:+.3f}j"
        print(f"{i:<8} | {d_str:<30} | {y_str:<30}")

    # 5. Graphical Visualization
    plt.figure(figsize=(14, 10))
    
    # --- Subplot 1: MSE Learning Curve ---
    plt.subplot(2, 2, 1)
    mse_db = 10 * np.log10(np.abs(results['errors'])**2 + 1e-12)
    plt.plot(mse_db, label='LMS MSE [dB]')
    plt.axhline(y=10*np.log10(sigma_n2), color='r', linestyle='--', label='Noise Floor')
    plt.title('Learning Curve: MSE [dB]')
    plt.xlabel('Iterations')
    plt.ylabel('dB')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # --- Subplot 2: Signal Magnitude Comparison ---
    plt.subplot(2, 2, 2)
    plt.plot(np.abs(d[-50:]), 'ro-', label='Desired (d)', alpha=0.5)
    plt.plot(np.abs(y[-50:]), 'b*-', label='Output (y)', alpha=0.8)
    plt.title('Signal Magnitude (Last 50 Samples)')
    plt.xlabel('Samples')
    plt.ylabel('|Magnitude|')
    plt.grid(True, alpha=0.3)
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
    run_system_identification()

# EOF