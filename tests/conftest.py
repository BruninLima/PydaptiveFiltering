import pytest
import numpy as np
from scipy import signal

@pytest.fixture
def calculate_msd():
    def _calc(w_true, w_est):
        w_true = np.asarray(w_true)
        w_est = np.asarray(w_est)
        return float(np.mean(np.abs(w_true - w_est)**2))
    return _calc

@pytest.fixture
def correlated_data():
    np.random.seed(42)
    n_samples = 2000
    h_unknown = np.array([0.6, -0.3, 0.1, 0.05], dtype=complex)
    order = len(h_unknown) - 1
    white_noise = (np.random.randn(n_samples) + 1j*np.random.randn(n_samples))
    x = np.convolve(white_noise, [1, 0.8, 0.5], mode='full')[:n_samples]
    d = np.convolve(x, h_unknown, mode='full')[:n_samples]
    return x, d, h_unknown, order

@pytest.fixture
def system_data():
    np.random.seed(42)
    n_samples = 5000
    w_optimal = np.array([0.4, -0.2, 0.1], dtype=complex)
    order = len(w_optimal) - 1
    x = (np.random.randn(n_samples) + 1j*np.random.randn(n_samples)) / np.sqrt(2)
    d_ideal = signal.lfilter(w_optimal, 1, x)
    return {"x": x, "d_ideal": d_ideal, "w_optimal": w_optimal, "order": order, "n_samples": n_samples}

# Criei este apelido para resolver os erros de "lms_data not found"
@pytest.fixture
def lms_data(correlated_data):
    x, d, h_unknown, order = correlated_data
    return {"x": x, "d": d, "h_unknown": h_unknown, "order": order, "n_samples": len(x)}

@pytest.fixture
def rls_test_data():
    np.random.seed(42)
    n_samples = 1000
    w_optimal = np.array([0.5, -0.4, 0.2], dtype=complex)
    order = len(w_optimal) - 1
    u = (np.random.randn(n_samples) + 1j*np.random.randn(n_samples))
    x = np.zeros(n_samples, dtype=complex)
    for i in range(1, n_samples):
        x[i] = 0.9 * x[i-1] + u[i]
    d = np.convolve(x, w_optimal, mode='full')[:n_samples]
    return {"x": x, "d": d, "w_optimal": w_optimal, "order": order, "n_samples": n_samples}



# --- Novas Fixtures Não-Lineares ---

@pytest.fixture
def quadratic_system_data():
    """Gera dados de um sistema não-linear quadrático para Volterra."""
    np.random.seed(42)
    n_samples = 1500
    x = np.random.randn(n_samples)
    d = np.zeros(n_samples)
    # Modelo: d(k) = -0.76x(k) - 1.0x(k-1) + 0.5x(k)^2 + ruído
    for k in range(1, n_samples):
        d[k] = -0.76*x[k] - 1.0*x[k-1] + 0.5*(x[k]**2) + 0.01*np.random.randn()
    return {"x": x, "d": d, "n_samples": n_samples}

@pytest.fixture
def rbf_mapping_data():
    """Gera uma função seno para mapeamento não-linear (RBF/MLP)."""
    np.random.seed(42)
    n_samples = 1000
    x_axis = np.linspace(0, 4*np.pi, n_samples)
    input_sig = np.sin(x_axis)
    desired = input_sig**2 # Alvo não-linear
    return {"x": input_sig, "d": desired}


@pytest.fixture
def correlated_data_real():
    np.random.seed(42)
    n_samples = 2000
    h_unknown = np.array([0.6, -0.3, 0.1, 0.05], dtype=float)  # REAL
    order = len(h_unknown) - 1

    white_noise = np.random.randn(n_samples).astype(float)    # REAL
    x = np.convolve(white_noise, [1.0, 0.8, 0.5], mode='full')[:n_samples].astype(float)
    d = np.convolve(x, h_unknown, mode='full')[:n_samples].astype(float)

    return x, d, h_unknown, order


@pytest.fixture
def system_data_real():
    np.random.seed(42)
    n_samples = 5000
    w_optimal = np.array([0.4, -0.2, 0.1], dtype=float)  # REAL
    order = len(w_optimal) - 1

    x = np.random.randn(n_samples).astype(float)         # REAL
    d_ideal = signal.lfilter(w_optimal, 1, x).astype(float)

    return {"x": x, "d_ideal": d_ideal, "w_optimal": w_optimal, "order": order, "n_samples": n_samples}

@pytest.fixture
def rls_test_data_real():
    np.random.seed(42)
    n_samples = 1000
    w_optimal = np.array([0.5, -0.4, 0.2], dtype=float)  # REAL
    order = len(w_optimal) - 1

    u = np.random.randn(n_samples).astype(float)         # REAL
    x = np.zeros(n_samples, dtype=float)
    for i in range(1, n_samples):
        x[i] = 0.9 * x[i-1] + u[i]

    d = np.convolve(x, w_optimal, mode='full')[:n_samples].astype(float)

    return {"x": x, "d": d, "w_optimal": w_optimal, "order": order, "n_samples": n_samples}

@pytest.fixture
def lms_data_real(correlated_data_real):
    x, d, h_unknown, order = correlated_data_real
    return {"x": x, "d": d, "h_unknown": h_unknown, "order": order, "n_samples": len(x)}
