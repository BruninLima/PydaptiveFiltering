import pytest
import numpy as np
from pydaptivefiltering.nonlinear import (
    VolterraLMS, VolterraRLS, RBF, 
    ComplexRBF, MultilayerPerceptron, 
    BilinearRLS )

@pytest.fixture
def dummy_data():
    n_samples = 100
    x = np.random.randn(n_samples)
    d = np.random.randn(n_samples)
    return x, d

def test_volterra_lms_sanity(dummy_data):
    x, d = dummy_data
    filt = VolterraLMS(memory=3, step=0.01)
    res = filt.optimize(x, d)
    
    coeffs_array = np.array(res['coefficients'])
    
    assert len(res['outputs']) == len(x)
    assert coeffs_array.shape == (len(x) + 1, 9) 

def test_volterra_rls_sanity(dummy_data):
    x, d = dummy_data
    filt = VolterraRLS(memory=2, forgetting_factor=0.99)
    res = filt.optimize(x, d)
    
    coeffs_array = np.array(res['coefficients'])
    
    assert len(res['errors']) == len(x)
    assert coeffs_array.shape[1] == 5 

def test_rbf_sanity(dummy_data):
    x, d = dummy_data
    filt = RBF(n_neurons=5, input_dim=3)
    res = filt.optimize(x, d)
    assert len(res['outputs']) == len(x)
    assert filt.w.shape == (5,)

def test_complex_rbf_sanity():
    n_samples = 50
    x = (np.random.randn(n_samples) + 1j*np.random.randn(n_samples))
    d = (np.random.randn(n_samples) + 1j*np.random.randn(n_samples))
    filt = ComplexRBF(n_neurons=4, input_dim=2)
    res = filt.optimize(x, d)
    assert np.iscomplexobj(res['outputs'])

def test_mlp_sanity(dummy_data):
    x, d = dummy_data
    filt = MultilayerPerceptron(n_neurons=10, input_dim=3)
    res = filt.optimize(x, d)
    assert 'outputs' in res
    assert len(res['outputs']) == len(x)

def test_bilinear_rls_sanity(dummy_data):
    x, d = dummy_data
    filt = BilinearRLS(forgetting_factor=0.98) 
    res = filt.optimize(x, d)
    
    coeffs_array = np.array(res['coefficients'])
    assert coeffs_array.shape[1] == 4