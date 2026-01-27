from pydaptivefiltering.RLS.RLS import RLS
from pydaptivefiltering.RLS.RLS_Alt import RLS_Alt

def test_rls_shapes(rls_test_data):
    """Verifica se as dimensões de saída do RLS Clássico estão corretas."""
    x, d = rls_test_data["x"], rls_test_data["d"]
    order = rls_test_data["order"]
    n_samples = len(x)
    
    model = RLS(filter_order=order, delta=0.1, lamb=0.98)
    res = model.optimize(x, d)
    
    assert len(res['outputs']) == n_samples
    assert len(res['coefficients']) == n_samples + 1
    assert model.w.shape == (order + 1,)

def test_rls_alt_shapes(rls_test_data):
    """Verifica se as dimensões de saída do RLS Alternativo estão corretas."""
    x, d = rls_test_data["x"], rls_test_data["d"]
    order = rls_test_data["order"]
    n_samples = len(x)
    
    model = RLS_Alt(filter_order=order, delta=0.1, lamb=0.98)
    res = model.optimize(x, d)
    
    assert len(res['outputs']) == n_samples
    assert len(res['outputs_posteriori']) == n_samples
    assert model.w.shape == (order + 1,)

def test_rls_inheritance(rls_test_data):
    """Garante que o regressor foi inicializado via classe base AdaptiveFilter."""
    order = rls_test_data["order"]
    
    model = RLS(filter_order=order, delta=0.01, lamb=1.0)
    
    assert hasattr(model, 'regressor')
    assert len(model.regressor) == order + 1