import numpy as np
from pydaptivefiltering.RLS.RLS import RLS
from pydaptivefiltering.RLS.RLS_Alt import RLS_Alt
from pydaptivefiltering.LMS.LMS import LMS

def test_rls_vs_lms_speed(rls_test_data, calculate_msd): # Adicionei calculate_msd aqui
    x, d = rls_test_data["x"], rls_test_data["d"]
    w_opt, order = rls_test_data["w_optimal"], rls_test_data["order"]

    model_lms = LMS(filter_order=order, step=0.01)
    model_lms.optimize(x, d)
    msd_lms = calculate_msd(w_opt, model_lms.w) # Agora funciona via injeção

    model_rls = RLS(filter_order=order, delta=0.01, lamb=0.99)
    model_rls.optimize(x, d)
    msd_rls = calculate_msd(w_opt, model_rls.w)

    assert msd_rls < msd_lms / 10

def test_rls_equivalence(rls_test_data):
    """RLS e RLS_Alt devem produzir resultados idênticos (equivalência matemática)."""
    x, d = rls_test_data["x"], rls_test_data["d"]
    order = rls_test_data["order"]

    model1 = RLS(order, delta=0.1, lamb=0.98)
    model2 = RLS_Alt(order, delta=0.1, lamb=0.98)
    
    model1.optimize(x, d)
    model2.optimize(x, d)
    
    # Comparamos os pesos finais com tolerância de precisão float
    np.testing.assert_allclose(
        model1.w, model2.w, rtol=1e-7, atol=1e-7,
        err_msg="RLS e RLS_Alt divergiram matematicamente."
    )

def test_perfect_identification(rls_test_data, calculate_msd):
    """Em ambiente sem ruído, o RLS deve identificar o sistema quase perfeitamente."""
    x, d = rls_test_data["x"], rls_test_data["d"]
    w_opt, order = rls_test_data["w_optimal"], rls_test_data["order"]

    # Lambda=1.0 para planta invariante no tempo e sem ruído
    model = RLS_Alt(order, delta=1e-6, lamb=1.0) 
    model.optimize(x, d)
    
    msd = calculate_msd(w_opt, model.w)
    assert msd < 1e-10