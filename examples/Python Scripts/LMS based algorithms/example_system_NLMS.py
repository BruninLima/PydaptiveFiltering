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
#     Adaptive Algorithm used here: NLMS                                        #
#                                                                               #
#################################################################################

# Imports
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pydaptivefiltering.LMS import NLMS

def run_nlms_system_identification():
    """
    Example: System Identification using the Normalized LMS (NLMS) Class.
    Based on Algorithm 4.3 from Paulo S. R. Diniz's book.
    """
    # 1. Experiment Parameters
    n_samples = 600
    h_unknown = np.array([0.32+0.21*1j, -0.3+0.7*1j, 0.5-0.8*1j, 0.2+0.5*1j])
    filter_order = len(h_unknown) - 1
    sigma_n2 = 0.04  # Noise Power
    mu = 0.1         # Step size (relaxation factor)
    gamma = 1e-5     # Regularization term

    # 2. Signal Generation
    x = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) / np.sqrt(2)
    noise = np.sqrt(sigma_n2/2) * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    
    # Generate Desired Signal d(k)
    from pydaptivefiltering.main import AdaptiveFilter
    class Plant(AdaptiveFilter):
        def optimize(self, **kwargs): pass
    
    plant = Plant(filter_order, w_init=h_unknown)
    d = plant.filter_signal(x) + noise

    # 3. NLMS Execution
    nlms_filter = NLMS(filter_order=filter_order, step=mu, gamma=gamma)
    
    print("-" * 50)
    print(f"Starting NLMS Adaptation (Order: {filter_order})...")
    
    tic = time()
    results = nlms_filter.optimize(input_signal=x, desired_signal=d)
    runtime = (time() - tic) * 1000
    
    print(f"Adaptation Finished in {runtime:.03f} ms")
    print("-" * 50)

    # 4. Numerical Comparison (Last 5 samples)
    y = results['outputs']
    print("\nSteady State Samples (Last 5):")
    print(f"{'Sample':<8} | {'Desired (d)':<25} | {'Filter Output (y)':<25}")
    for i in range(n_samples - 5, n_samples):
        print(f"{i:<8} | {str(np.round(d[i], 4)):<25} | {str(np.round(y[i], 4)):<25}")

    # 5. Graphical Visualization
    plt.figure(figsize=(14, 12))
    
    # Subplot 1: Signal Comparison
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(np.abs(d[-50:]), 'ro-', label='Desired signal (d)', alpha=0.6)
    ax1.plot(np.abs(y[-50:]), 'b*-', label='Filter output (y)', alpha=0.8)
    ax1.set_title('NLMS Signal Comparison (Last 50 Samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Learning Curve MSE [dB]
    ax2 = plt.subplot(3, 2, 3)
    mse_db = 10 * np.log10(np.abs(results['errors'])**2 + 1e-12)
    ax2.plot(mse_db, label='NLMS MSE')
    ax2.set_title('Learning Curve: MSE [dB]')
    ax2.set_ylabel('dB')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Noise Floor
    ax3 = plt.subplot(3, 2, 4)
    mse_min = 10 * np.log10(np.abs(noise)**2 + 1e-12)
    ax3.plot(mse_min, color='orange', alpha=0.4, label='Noise Floor')
    ax3.set_title('Minimum MSE (Noise Floor)')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Real Coefficients
    ax4 = plt.subplot(3, 2, 5)
    weights_hist = np.array(results['coefficients'])
    ax4.plot(weights_hist.real)
    for val in h_unknown.real:
        ax4.axhline(y=val, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('Coefficients (Real Part)')
    ax4.grid(True, alpha=0.3)

    # Subplot 5: Imaginary Coefficients
    ax5 = plt.subplot(3, 2, 6)
    ax5.plot(weights_hist.imag)
    for val in h_unknown.imag:
        ax5.axhline(y=val, color='black', linestyle='--', alpha=0.5)
    ax5.set_title('Coefficients (Imaginary Part)')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_nlms_system_identification()