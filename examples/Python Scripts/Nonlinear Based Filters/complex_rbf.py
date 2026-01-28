import numpy as np
import matplotlib.pyplot as plt

def generate_matlab_like_data(dim=500, Sx=1.0, Sn=1e-1, seed=0):
    rng = np.random.default_rng(seed)
    n = Sn * rng.standard_normal(dim)

    x = Sx * (rng.standard_normal(dim) + 1j * rng.standard_normal(dim))

    xl1 = np.zeros(dim, dtype=complex)
    xl2 = np.zeros(dim, dtype=complex)
    xl1[1:] = x[:-1]
    xl2[1:] = xl1[:-1]

    d = (-0.08*x - 0.15*xl1 + 0.14*xl2
         + 0.055*(x**2) + 0.3*x*xl2 - 0.16*(xl1**2) + 0.14*(xl2**2)
        ) + n

    return x, d

def run_crbf_matlab_like(dim=500, Nneur=10, ur=0.01, uw=0.01, us=0.01, seed=0):
    from pydaptivefiltering import ComplexRBF

    x, d = generate_matlab_like_data(dim=dim, Sx=1.0, Sn=1e-1, seed=seed)

    crbf = ComplexRBF(
        n_neurons=Nneur,
        input_dim=3,
        ur=ur, uw=uw, us=us,
        sigma_init=1.0,
        rng=np.random.default_rng(seed),
    )

    res = crbf.optimize(x, d, return_internal_states=True)
    e = np.asarray(res.errors).ravel()
    mse = np.abs(e)**2

    return res, mse

res, mse = run_crbf_matlab_like(seed=1)

plt.figure()
plt.plot(10*np.log10(mse + 1e-12))
plt.title("Learning curve (instantaneous |e|^2) [dB]")
plt.xlabel("k")
plt.ylabel("|e[k]|^2 [dB]")
plt.grid(True)
plt.show()

print("outputs shape:", np.asarray(res.outputs).shape)
print("errors shape:", np.asarray(res.errors).shape)
print("coeff history shape:", np.asarray(res.coefficients).shape)
print("extra keys:", None if res.extra is None else list(res.extra.keys()))
