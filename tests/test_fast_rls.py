# tests/test_fast_rls.py

import pytest
import numpy as np
from pydaptivefiltering.Fast_Transversal_RLS import FastRLS, StabFastRLS

class TestFastTransversalRLS:
    
    @pytest.mark.parametrize("filter_class", [FastRLS, StabFastRLS])
    def test_convergence_optimal_system(self, filter_class, rls_test_data, calculate_msd):
        """
        Verifica se ambos os algoritmos convergem para o sistema ótimo 
        em um cenário de dados correlacionados.
        """
        data = rls_test_data
        
        # Inicialização do filtro (seguindo o padrão __init__ definido)
        f = filter_class(
            filter_order=data["order"],
            lamb=0.999,
            epsilon=0.01
        )
        
        # Execução da otimização
        res = f.optimize(data["x"], data["d"])
        
        # Os coeficientes finais estão no último índice da história
        w_final = res["coefficients"][-1]
        
        # Cálculo do MSD (Mean Square Deviation)
        msd = calculate_msd(data["w_optimal"], w_final)
        
        # Critério de convergência: MSD deve ser muito baixo para RLS (ex: < 1e-4)
        assert msd < 5e-4, f"O algoritmo {filter_class.__name__} não convergiu adequadamente. MSD: {msd}"

    @pytest.mark.parametrize("filter_class", [FastRLS, StabFastRLS])
    def test_output_shapes(self, filter_class, rls_test_data):
        """Verifica se as dimensões das saídas no dicionário estão corretas."""
        data = rls_test_data
        f = filter_class(filter_order=data["order"])
        res = f.optimize(data["x"], data["d"])
        
        n = data["n_samples"]
        m = data["order"] + 1
        
        assert len(res["outputs"]) == n
        assert len(res["errors"]) == n
        assert len(res["coefficients"]) == n + 1 # w_history deve ter n+1 entradas
        assert res["coefficients"][0].shape == (m,)
        
        # Verificação de campos extras do RLS
        assert "outputs_posteriori" in res
        assert "errors_posteriori" in res
        assert len(res["errors_posteriori"]) == n

    def test_stab_fast_rls_parameters(self, rls_test_data):
        """Verifica se o StabFastRLS aceita os parâmetros de estabilização (kappas)."""
        data = rls_test_data
        # Testando com valores de kappa diferentes do padrão
        f = StabFastRLS(
            filter_order=data["order"],
            kappa1=1.0,
            kappa2=2.0,
            kappa3=1.0
        )
        res = f.optimize(data["x"], data["d"])
        
        assert res["outputs"] is not None
        assert f.kappa1 == 1.0

    def test_complex_data_handling(self, system_data, calculate_msd):
        """Testa especificamente a capacidade de lidar com dados complexos da fixture system_data."""
        data = system_data
        f = FastRLS(filter_order=data["order"], lamb=0.98, epsilon=1.0)
        
        res = f.optimize(data["x"], data["d_ideal"])
        w_final = res["coefficients"][-1]
        
        msd = calculate_msd(data["w_optimal"], w_final)
        # RLS em 5000 amostras com ruído zero deve ter erro desprezível
        assert msd < 1e-10