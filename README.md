# pydaptivefiltering

<!---<img src="exemplo-image.png" alt="exemplo imagem">--->

> **A high-performance Python package for Adaptive Filtering.** > Implementations are strictly based on the book *Adaptive Filtering: Algorithms and Practical Implementation* by **Paulo S. R. Diniz**.

## 🛠️ Description

This package provides a modern Python alternative to the **MATLAB Adaptive Filtering Toolbox**.

## 🚀 Project Status & Algorithms

The package currently covers the vast majority of classical and advanced linear adaptive algorithms, all validated via `pytest` with steady-state convergence scenarios.

### Current Implementation

The project is currently on its early stages (v0.7). The planned order of work for every kind of algorithm is: (Algorithm(1), Examples(2), Notebooks(3)). The following is the planned and current progress for each of the algorithms:

- [11/11] LMS based algorithms
- [2/2] RLS based algorithms 
- [5*/5] SetMembership Algorithms
- [4/4] Lattice-based RLS 
- [2*/2] Fast Transversal RLS
- [1/1] QR
- [5/5] IIR Filters
- [0/6] Nonlinear Filters
- [0/3] Subband Filters
- [0/4] BlindFilters
- [0/?] Kalman Filters
- 
**PUAP:** The `SetMembership_Simp_PUAP` implementation is currently under technical review and may not exhibit expected convergence in this version.

**StabFastRLS:** The `Fast_TransversalRLS.StabFastRLS` implementation is currently under technical review and may exhibit unexpected warnings in this version.

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
To install the package with conda:
```
conda install pydaptivefiltering
```

---
## ☕ Examples of uses

A good place to start is with the jupyter notebooks that can be found at the <Examples\Jupyter Notebooks\> folder
```
import pydaptivefiltering as pdf
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
## ☕ Quick Example (Lattice RLS)

```python
import numpy as np
from pydaptivefiltering.LatticeRLS.LRLS_pos import LatticeRLS

# Setup data
n_samples = 2000
x = np.random.randn(n_samples)
d = ... # Your desired signal

# Initialize Filter
model = LatticeRLS(filter_order=10, lambda_factor=0.99, epsilon=0.01)

# Optimization Process
res = model.optimize(x, d)

# Results
print(f"Final Estimation Error: {res['errors'][-1]}")

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
