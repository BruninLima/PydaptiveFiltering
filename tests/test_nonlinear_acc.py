# tests/test_nonlinear_accuracy.py
import pytest
import numpy as np
from pydaptivefiltering.nonlinear import VolterraLMS, VolterraRLS, RBF

def test_volterra_lms_convergence(quadratic_system_data):
    data = quadratic_system_data
    # Usando o dicionário vindo da fixture do conftest
    mu = np.array([0.05, 0.05, 0.01, 0.01, 0.01]) 
    filt = VolterraLMS(memory=2, step=mu)
    res = filt.optimize(data['x'], data['d'])
    
    final_mse = np.mean(res['errors'][-100:]**2)
    assert final_mse < 1e-2

def test_volterra_rls_convergence(quadratic_system_data):
    data = quadratic_system_data
    # Usando o dicionário vindo da fixture do conftest
    filt = VolterraRLS(memory=2, forgetting_factor=0.99)
    res = filt.optimize(data['x'], data['d'])
    
    final_mse = np.mean(res['errors'][-100:]**2)
    assert final_mse < 1e-2

def test_rbf_mapping_accuracy(rbf_mapping_data):
    data = rbf_mapping_data
    filt = RBF(n_neurons=15, input_dim=1, uw=0.05)
    res = filt.optimize(data['x'], data['d'])
    
    final_error = np.mean(np.abs(res['errors'][-100:]))
    assert final_error < 0.1