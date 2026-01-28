# pydaptivefiltering

<p align="center">
  <strong>High-performance adaptive filtering in Python</strong><br>
  Implementations based on <em>Adaptive Filtering: Algorithms and Practical Implementation</em> (Paulo S. R. Diniz).
</p>

<p align="center">
  <a href="https://pypi.org/project/pydaptivefiltering/">
    <img src="https://img.shields.io/pypi/v/pydaptivefiltering.svg" alt="PyPI version">
  </a>
  <a href="https://BruninLima.github.io/pydaptivefiltering/index.html">
    <img src="https://img.shields.io/badge/docs-pdoc-blue" alt="Docs">
  </a>
  <a href="https://github.com/BruninLima/PydaptiveFiltering/blob/main/LICENSE.md">
    <img src="https://img.shields.io/github/license/BruninLima/PydaptiveFiltering.svg" alt="License">
  </a>
</p>


## 📌 Table of Contents
* [Algorithms & Progress](#-project-status--algorithms)
* [Installation](#install-instructions)
* [Usage Examples](#-examples-of-uses)
* [Neural Adaptation (MLP)](#-quick-example-neural-adaptive-filtering-mlp)

---

## Install

```bash
pip install pydaptivefiltering
```

### Requirements

- Python **3.10+** (tested on <!-- PYTHON_VERSION_TESTED e.g. 3.12 -->)
- NumPy, SciPy

---

## Quickstart (60 seconds)

All algorithms follow the same pattern:

1. Instantiate the filter/model
2. Run `optimize(input_signal, desired_signal)`
3. Inspect `outputs`, `errors`, `coefficients`, and optional `extra`

Returned dictionary keys:
- `outputs`: model output
- `errors`: error sequence (see `error_type`)
- `coefficients`: coefficient history (snapshots over time)
- `runtime_s`: runtime in seconds
- `error_type`: `"a_priori"`, `"a_posteriori"`, or `"output_error"`
- `extra`: optional internal states (when enabled)


### System identification (example with Volterra-RLS):

```python
import numpy as np
import pydaptivefiltering as pdf

rng = np.random.default_rng(0)

# Synthetic system: d[k] = 0.7 x[k] - 0.2 x[k-1] + noise
N = 5000
x = rng.standard_normal(N)
d = np.zeros(N)
for k in range(1, N):
    d[k] = 0.7*x[k] - 0.2*x[k-1] + 0.05*rng.standard_normal()

filt = pdf.VolterraRLS(memory=3, forgetting_factor=0.99, delta=1.0)
res = filt.optimize(x, d)

mse_tail = np.mean(res["errors"][-500:]**2)
print("Final tail MSE:", mse_tail)
print("Final coefficient vector length:", res["coefficients"][-1].size)
```

> Tip: `res["extra"]` may include additional trajectories when `return_internal_states=True`.

---

### Example: Neural Adaptive Filtering (MLP)

Nonlinear system:
\[
d(k) = x(k)^2 + 0.5\,x(k-1) + \eta(k)
\]

```python
import numpy as np
import matplotlib.pyplot as plt
import pydaptivefiltering as pdf

rng = np.random.default_rng(1)

N = 3000
x = rng.uniform(-1, 1, N)
d = np.zeros(N)
for k in range(1, N):
    d[k] = (x[k]**2) + 0.5*x[k-1] + 0.01*rng.standard_normal()

mlp = pdf.MultilayerPerceptron(
    n_neurons=8,
    input_dim=3,
    step_size=0.01,   # Keep consistent with the library API
    momentum=0.9,
    activation="tanh",
)

res = mlp.optimize(x, d)

plt.plot(10*np.log10(res["errors"]**2 + 1e-12), alpha=0.8)
plt.title(f"MLP Convergence (Final MSE: {np.mean(res['errors'][-500:]**2):.6f})")
plt.xlabel("Iteration")
plt.ylabel("Squared Error (dB)")
plt.show()
```

<!-- PLACEHOLDER: Put a convergence plot screenshot here -->
<p align="center">
  <img src="<!-- LINK_OR_PATH_TO_MLP_PLOT_IMAGE -->" alt="MLP convergence plot" width="750">
</p>

---

## Algorithms (overview)

> This is an overview. For the full list, check the documentation: <a href="https://BruninLima.github.io/pydaptivefiltering/index.html">Docs</a>
Algorithms categories are based on the chapters of the book <em>Adaptive Filtering: Algorithms and Practical Implementation</em> (Paulo S. R. Diniz).

| Module / Category | Examples (classes) | Data type |
|---|---|---|
| `lms/` (LMS family) | `LMS`, `NLMS`, `AP`, … | Real/Complex |
| `rls/` (RLS family) | `RLS`, `RLSAlt`, … | Complex |
| `fast_rls/` (Fast transversal RLS) | `FastRLS`, `StabFastRLS`, … | Real/Complex |
| `qr_decomposition/` (QR-RLS) | `QRRLS` | Real |
| `set_membership/` (Set-membership) | `SMNLMS`, `SMBNLMS`, `SMAffineProjection`, `SimplifiedSMAP` | Complex |
| `nonlinear/` (Nonlinear) | `VolterraLMS`, `VolterraRLS`, `RBF`, `MultilayerPerceptron`, … | Real |
| `subband/` (Subband / frequency-domain) | `OLSBLMS`, `DLCLLMS`, `CFDLMS` | Real |
| `lattice/` (Lattice-based RLS) | <!-- e.g., `LatticeRLS`, `...` --> | Real/Complex |
| `iir/` (IIR adaptive filters) | <!-- e.g., `IIRLMS`, `...` --> | Real/Complex |
| `blind/` (Blind filtering) | <!-- e.g., `CMA`, `...` --> | Complex |
| `kalman/` (Kalman filters) | `Kalman`, … | Real |
---

### Known limitations (this release)

- ⚠️ `set_membership.simplified_puap.py`: under technical review (convergence may differ from reference).
- ⚠️ `nonlinear.complex_rbf.py`: under technical review (convergence may differ from reference).

---

## Notebooks

- 🧪 Examples and notebooks: [Notebooks](examples/Jupyter%20Notebooks/)

---

## 📝 License

This project is under the license found at [LICENSE](LICENSE.md).


![GitHub repo size](https://img.shields.io/github/repo-size/BruninLima/PydaptiveFiltering?style=for-the-badge)
![GitHub language count](https://img.shields.io/github/languages/count/BruninLima/PydaptiveFiltering?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/BruninLima/PydaptiveFiltering?style=for-the-badge)
![Bitbucket open issues](https://img.shields.io/bitbucket/issues/BruninLima/PydaptiveFiltering?style=for-the-badge)
![Bitbucket open pull requests](https://img.shields.io/bitbucket/pr-raw/BruninLima/PydaptiveFiltering?style=for-the-badge)

## References

- Diniz, P. S. R. (2020). *Adaptive Filtering: Algorithms and Practical Implementation*. Springer.
- MATLAB Adaptive Filtering Toolbox (for comparison).

<!-- PLACEHOLDER: Footer image (institution / lab / sponsor) -->
<p align="center">
  <img src="<!-- LINK_OR_PATH_TO_FOOTER_IMAGE -->" alt="Footer image" width="650">
</p>
