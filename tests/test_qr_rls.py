import numpy as np
from pydaptivefiltering.QR.RLS import QR_RLS  

class TestQRRLS:
    """
    Classe de teste para o algoritmo QR-RLS.
    Testa convergência, sanidade de dimensões e comportamento com dados complexos.
    """

    def test_convergence_optimal_system(self, rls_test_data, calculate_msd):
        """
        Verifica se o QR-RLS converge para os coeficientes ótimos em um 
        cenário de dados correlacionados (onde o RLS brilha).
        """
        data = rls_test_data
        
        # Inicialização do filtro
        # QR-RLS costuma ser muito estável, então lamb=0.99 é seguro
        f = QR_RLS(
            filter_order=data["order"],
            lamb=0.99
        )

        # Execução
        res = f.optimize(data["x"], data["d"])
        
        w_final = res["coefficients"][-1]
        msd = calculate_msd(data["w_optimal"], w_final)

        # O RLS deve atingir uma precisão muito alta
        # Critério: MSD < 1e-6 para 1000 amostras
        assert msd < 1e-6, f"O QR-RLS não convergiu adequadamente. MSD: {msd}"

    def test_output_shapes(self, system_data):
        """
        Teste de sanidade: verifica se as dimensões dos vetores de saída 
        correspondem ao tamanho do sinal de entrada.
        """
        data = system_data
        f = QR_RLS(filter_order=data["order"])
        
        res = f.optimize(data["x"], data["d_ideal"])
        
        n = len(data["x"])
        assert len(res["outputs"]) == n
        assert len(res["errors"]) == n
        assert len(res["errors_posteriori"]) == n
        assert len(res["coefficients"]) == n or len(res["coefficients"]) == n + 1

    def test_complex_data_handling(self, correlated_data):
        """
        Verifica se o algoritmo lida corretamente com sinais complexos 
        sem gerar partes imaginárias espúrias ou erros de casting.
        """
        x, d, h_true, order = correlated_data
        
        f = QR_RLS(filter_order=order)
        res = f.optimize(x, d)
        
        # Verifica se o output é complexo
        assert np.iscomplexobj(res["outputs"])
        assert np.iscomplexobj(res["coefficients"][0])
        
        # Verifica se não há NaNs (comum em falhas de estabilidade do RLS)
        assert not np.isnan(res["outputs"]).any()
        assert not np.isnan(res["coefficients"][-1]).any()

    def test_initial_weight_definition(self, system_data):
        """
        Verifica se o filtro respeita os pesos iniciais fornecidos no construtor.
        """
        data = system_data
        w_init = np.array([1.0 + 1j, 0.5 - 0.5j, 0.0], dtype=complex)
        
        f = QR_RLS(filter_order=data["order"], w_init=w_init)
        
        # Antes de otimizar, o w deve ser igual ao w_init
        assert np.allclose(f.w, w_init)
        
        res = f.optimize(data["x"][:10], data["d_ideal"][:10])
        # O histórico deve começar próximo ao valor inicial
        assert np.allclose(res["coefficients"][0], w_init, atol=1e-1)

    def test_error_decrease(self, lms_data):
        """
        Verifica se a energia do erro diminui ao longo do tempo, 
        indicando que o aprendizado está ocorrendo.
        """
        data = lms_data
        f = QR_RLS(filter_order=data["order"], lamb=0.98)
        
        res = f.optimize(data["x"], data["d"])
        errors = np.abs(res["errors"])**2
        
        # Comparamos a média dos erros nas primeiras 50 amostras 
        # com a média das últimas 50 amostras.
        initial_error = np.mean(errors[:50])
        final_error = np.mean(errors[-50:])
        
        assert final_error < initial_error, "O erro deveria diminuir após o período de adaptação."

    def test_lambda_unity_stability(self, rls_test_data):
        """
        Testa o caso limite lambda = 1.0 (RLS padrão sem esquecimento).
        O QR-RLS deve ser estável nesse cenário.
        """
        data = rls_test_data
        f = QR_RLS(filter_order=data["order"], lamb=1.0)
        
        # Se não houver exceções e o resultado não for NaN, passou no teste de sanidade
        res = f.optimize(data["x"], data["d"])
        assert not np.isnan(res["coefficients"][-1]).any()