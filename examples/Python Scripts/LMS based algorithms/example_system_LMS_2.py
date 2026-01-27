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
from pydaptivefiltering.LMS import LMS

def run_lms_ensemble_example():
    # --- Parameters ---
    ensemble = 100       # Number of realizations
    K = 500              # Number of iterations (samples)
    sigma_n2 = 0.04      # Noise power
    mu = 0.1             # Step size
    
    # Unknown system (Plant)
    Wo = np.array([0.32 + 0.21j, -0.3 + 0.7j, 0.5 - 0.8j, 0.2 + 0.5j])
    N = len(Wo)
    order = N - 1

    # Allocation for ensemble averages
    mse_ensemble = np.zeros((K, ensemble))
    w_history_ensemble = np.zeros((N, K, ensemble), dtype=complex)

    print(f"Starting simulation: {ensemble} realizations...")

    for l in range(ensemble):
        # 1. Generate Input Signal (Complex White Noise / 4-QAM like)
        x = (np.random.randn(K) + 1j * np.random.randn(K)) / np.sqrt(2)
        
        # 2. Generate Complex Noise
        n = np.sqrt(sigma_n2 / 2) * (np.random.randn(K) + 1j * np.random.randn(K))
        
        # 3. Generate Desired Signal: d(k) = Wo^H * x(k) + n(k)
        # Using convolve for plant simulation
        d = np.convolve(x, Wo, mode='full')[:K] + n
        
        # 4. Instantiate and Optimize
        # We initialize with ones as in the original example
        lms_filter = LMS(filter_order=order, step=mu, w_init=np.ones(N, dtype=complex))
        output = lms_filter.optimize(x, d)
        
        # Store results for this realization
        mse_ensemble[:, l] = np.abs(output['errors'])**2
        w_history_ensemble[:, :, l] = np.array(output['coefficients']).T[:, :K]

        if (l + 1) % 20 == 0:
            print(f" Realization {l + 1}/{ensemble} completed.")

    # --- Ensemble Averaging ---
    mse_av = np.mean(mse_ensemble, axis=1)
    w_av = np.mean(w_history_ensemble, axis=2)

    # --- Plotting ---
    plt.figure(figsize=(12, 10))

    # Subplot 1: Learning Curve (MSE)
    plt.subplot(2, 2, 1)
    plt.semilogy(mse_av, color='blue', linewidth=2, label='Ensemble Average')
    plt.axhline(y=sigma_n2, color='red', linestyle='--', label='Minimum MSE (Noise Floor)')
    plt.title('Learning Curve - MSE [dB]')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    # Subplot 2: Single Realization vs Average
    plt.subplot(2, 2, 2)
    plt.semilogy(mse_ensemble[:, 0], color='gray', alpha=0.3, label='Single Realization')
    plt.semilogy(mse_av, color='blue', label='Average')
    plt.title('Instantaneous vs Average MSE')
    plt.xlabel('Iterations')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    # Subplot 3: Coefficients (Real Part)
    plt.subplot(2, 2, 3)
    for i in range(N):
        plt.plot(w_av[i, :].real, label=f'w[{i}] real')
        plt.axhline(y=Wo[i].real, color='black', linestyle=':', alpha=0.5)
    plt.title('Evolution of Coefficients (Real Part)')
    plt.xlabel('Iterations')
    plt.grid(True, alpha=0.3)

    # Subplot 4: Coefficients (Imaginary Part)
    plt.subplot(2, 2, 4)
    for i in range(N):
        plt.plot(w_av[i, :].imag, label=f'w[{i}] imag')
        plt.axhline(y=Wo[i].imag, color='black', linestyle=':', alpha=0.5)
    plt.title('Evolution of Coefficients (Imaginary Part)')
    plt.xlabel('Iterations')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_lms_ensemble_example()