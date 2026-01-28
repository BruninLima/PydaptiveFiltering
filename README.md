# pydaptivefiltering

<!---<img src="exemplo-image.png" alt="exemplo imagem">--->

> **A high-performance Python package for Adaptive Filtering.** > Implementations are strictly based on the book *Adaptive Filtering: Algorithms and Practical Implementation* by **Paulo S. R. Diniz**.

## 🛠️ Description

This package provides a modern Python alternative to the **MATLAB Adaptive Filtering Toolbox**.

## 📌 Table of Contents
* [Algorithms & Progress](#-project-status--algorithms)
* [Installation](#install-instructions)
* [Usage Examples](#-examples-of-uses)
* [Neural Adaptation (MLP)](#-quick-example-neural-adaptive-filtering-mlp)

## 🚀 Project Status & Algorithms

The package currently covers the vast majority of classical and advanced linear adaptive algorithms, all validated via `pytest` with steady-state convergence scenarios.

### Current Implementation

The project is currently on its early stages (v0.9). The planned order of work for every kind of algorithm is: (Algorithm(1), Examples(2), Notebooks(3)). The following is the planned and current progress for each of the algorithms:

- [11/11] LMS based algorithms
- [2/2] RLS based algorithms 
- [5*/5] SetMembership Algorithms
- [4/4] Lattice-based RLS 
- [2/2] Fast Transversal RLS
- [1/1] QR
- [5/5] IIR Filters
- [6*/6] Nonlinear Filters
- [3/3] Subband Filters
- [4/4] BlindFilters
- [1/1] Kalman Filters

**simplified_puap:** The `set_membership.simplified_puap.py` implementation is currently under technical review and may not exhibit expected convergence in this version.
**complex_rbf:** The `nonlinear.complex_rbf.py` implementation is currently under technical review and may exhibit unexpected warnings in this version.

---
## 💻 Requirements

* **Python 3.10+** (Officially tested on 3.12.7)
* **NumPy** and **SciPy**


---
## Install Instructions

To install the package with pip:

```
pip install pydaptivefiltering
```

---
## ☕ Examples of uses

The package is designed to be intuitive for both research and production. You can easily compare linear and non-linear approaches for system identification or signal prediction.

A comprehensive collection of Jupyter Notebooks is available at `<Examples/Jupyter Notebooks/>`, covering:
* **System Identification:** Comparing LMS vs. RLS convergence.
* **Non-linear Modeling:** Using Volterra and Bilinear filters to model saturation and feedback.
* **Neural Adaptation:** Benchmarking MLP with Momentum against classical adaptive filters.

Basic usage pattern:
```python
import pydaptivefiltering as pdf
import numpy as np

# 1. Define your data (Input and Desired)
x = np.random.randn(5000)
d = my_system_output(x)

# 2. Instantiate the filter
filt = pdf.VolterraRLS(memory=3, forgetting_factor=0.99)

# 3. Optimize and Analyze
results = filt.optimize(x, d)
print(f"Final MSE: {np.mean(results['errors'][-100:]**2)}")
```



---
## ⚡ Quick Example: Neural Adaptive Filtering (MLP)

The package now supports a **Multilayer Perceptron (MLP)** designed for adaptive filtering tasks, featuring Backpropagation with Momentum and selectable activation functions (`tanh`, `sigmoid`).

```python
import numpy as np
import matplotlib.pyplot as plt
from pydaptivefiltering as pdf 

# --- 1. Generate Non-linear Data ---
# System: d(k) = x(k)^2 + 0.5*x(k-1)
N = 3000
x = np.random.uniform(-1, 1, N)
d = np.zeros(N)
for k in range(1, N):
    d[k] = (x[k]**2) + 0.5*x[k-1] + 0.01*np.random.randn()

# --- 2. Initialize the Adaptive MLP ---
# Configuration: 3 inputs [x(k), d(k-1), x(k-1)], 8 hidden neurons, Tanh activation
mlp = pdf.MultilayerPerceptron(
    n_neurons=8, 
    input_dim=3, 
    step=0.01, 
    momentum=0.9, 
    activation='tanh'
)

# --- 3. Run Optimization ---
res = mlp.optimize(x, d)

# --- 4. Visualize ---
plt.plot(10*np.log10(res['errors']**2), alpha=0.5, label='Squared Error (dB)')
plt.title(f"MLP Convergence (Final MSE: {np.mean(res['errors'][-500:]**2):.5f})")
plt.legend()
plt.show()
```
---

## 🧪 Technical Validation

The package uses rigorous unit testing to ensure the mathematical integrity of each structure. We currently maintain **70+ automated tests** covering:

1. **Numerical Stability:** Floating-point operations for complex structures (Lattice/QR).
2. **Convergence Accuracy:** Verification of steady-state MSE (Mean Square Error) against system noise levels.
3. **Tracking Performance:** Capability of algorithms to adapt to abrupt system changes (Step response).

To run tests locally:

```bash
pytest

```

---
## 📝 License

This project is under the license found at [LICENSE](LICENSE.md).


![GitHub repo size](https://img.shields.io/github/repo-size/BruninLima/PydaptiveFiltering?style=for-the-badge)
![GitHub language count](https://img.shields.io/github/languages/count/BruninLima/PydaptiveFiltering?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/BruninLima/PydaptiveFiltering?style=for-the-badge)
![Bitbucket open issues](https://img.shields.io/bitbucket/issues/BruninLima/PydaptiveFiltering?style=for-the-badge)
![Bitbucket open pull requests](https://img.shields.io/bitbucket/pr-raw/BruninLima/PydaptiveFiltering?style=for-the-badge)

---
## 📖 References

* **Diniz, P. S. R.** (2020). *Adaptive Filtering: Algorithms and Practical Implementation*. Springer.
* MATLAB Adaptive Filtering Toolbox.
