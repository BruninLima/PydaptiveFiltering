from pydaptivefiltering.SetMembership import SM_NLMS, SM_BNLMS, SM_AP, SM_Simp_AP

def test_compare_sm_performance(system_data, calculate_msd):
    """
    Benchmarking comparativo dos algoritmos Set-Membership.
    Utiliza dados do system_data (conftest.py).
    """
    x, d = system_data["x"], system_data["d_ideal"]
    w_true, order = system_data["w_optimal"], system_data["order"]
    n_samples = len(x)
    
    # Parâmetros específicos para Set-Membership
    gamma_val = 0.1
    L_val = 2
    gamma_vector = [0.0] * (L_val + 1)

    filters = {
        "SM-NLMS": SM_NLMS(filter_order=order, gamma_bar=gamma_val, gamma=1e-6),
        "SM-BNLMS": SM_BNLMS(filter_order=order, gamma_bar=gamma_val, gamma=1e-6),
        "SM-AP": SM_AP(filter_order=order, gamma_bar=gamma_val, gamma_bar_vector=gamma_vector, gamma=1e-3, L=L_val),
        "SM-Simp-AP": SM_Simp_AP(filter_order=order, gamma_bar=gamma_val, gamma=1e-3, L=L_val)
    }

    for name, filt in filters.items():
        res = filt.optimize(x, d)

        # Usando a função auxiliar do conftest para validar acurácia dos pesos
        msd = calculate_msd(w_true, filt.w)
        update_rate = (res['n_updates'] / n_samples) * 100

        # Validações de sanidade
        assert msd < 0.1, f"{name} falhou: MSD muito alto ({msd:.4f})"
        assert update_rate < 100.0, f"{name} atualizou em todas as iterações (não está economizando processamento)"