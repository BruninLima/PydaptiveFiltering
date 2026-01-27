# tests/test_qr_rls.py

import numpy as np
import pytest
from pydaptivefiltering import QRRLS


class TestQRRLS:
    """
    Testes para o QR-RLS REAL-only (Algorithm 9.1, Diniz).
    """

    def test_convergence_optimal_system_real(self, rls_test_data_real, calculate_msd):
        """
        Verifica convergência para os coeficientes ótimos em cenário correlacionado.
        """
        data = rls_test_data_real

        f = QRRLS(filter_order=data["order"], lamb=0.99)
        res = f.optimize(data["x"], data["d"])

        # w_final pode ser complex dtype internamente, mas deve ser REAL na prática
        w_final = np.asarray(res["coefficients"][-1])
        assert np.max(np.abs(np.imag(w_final))) < 1e-10
        w_final = np.real(w_final).astype(float)

        w_opt = np.asarray(data["w_optimal"], dtype=float)

        msd = calculate_msd(w_opt, w_final)

        # Critério forte (se estiver muito difícil, ajuste MSD/threshold)
        assert msd < 1e-6, f"O QR-RLS não convergiu adequadamente. MSD: {msd}"

    def test_output_shapes_real(self, system_data_real):
        """
        Sanidade: dimensões das saídas.
        """
        data = system_data_real
        f = QRRLS(filter_order=data["order"], lamb=0.99)

        res = f.optimize(data["x"], data["d_ideal"])

        n = len(data["x"])
        assert "outputs" in res
        assert "errors" in res
        assert "coefficients" in res

        y = np.asarray(res["outputs"])
        e = np.asarray(res["errors"])
        coeffs = res["coefficients"]

        assert y.shape == (n,)
        assert e.shape == (n,)
        assert len(coeffs) in (n, n + 1)

        # real-only => sem imaginário espúrio
        assert np.max(np.abs(np.imag(y))) < 1e-10
        assert np.max(np.abs(np.imag(e))) < 1e-10

    def test_complex_data_rejected(self, correlated_data):
        """
        Como agora o QR_RLS é REAL-only, dados complexos devem levantar TypeError.
        """
        x, d, _, order = correlated_data

        # Garantia do teste: correlated_data deve ser complexo
        assert np.iscomplexobj(x) or np.iscomplexobj(d)

        f = QRRLS(filter_order=order, lamb=0.99)

        with pytest.raises(TypeError):
            _ = f.optimize(x, d)

    def test_initial_weight_definition_real(self, system_data_real):
        """
        Pesos iniciais reais devem ser respeitados NO CONSTRUTOR.
        Observação: o QR_RLS (estilo Diniz/MATLAB) pode re-inicializar internamente
        durante optimize, então não exigimos que coefficients[0] == w_init.
        """
        data = system_data_real

        w_init = np.array([1.0, 0.5, 0.0], dtype=float)
        assert len(w_init) == data["order"] + 1

        f = QRRLS(filter_order=data["order"], lamb=0.99, w_init=w_init)

        # antes do optimize, deve respeitar w_init
        assert np.allclose(np.real(f.w), w_init)
        assert np.max(np.abs(np.imag(f.w))) < 1e-12

        res = f.optimize(data["x"][:10], data["d_ideal"][:10])

        # sanidade: histórico existe e tem vetores do tamanho certo
        coeffs = res["coefficients"]
        assert len(coeffs) in (10, 11)
        w0 = np.asarray(coeffs[0])
        assert w0.shape == (data["order"] + 1,)

        # e sempre real-only
        assert np.max(np.abs(np.imag(w0))) < 1e-10
        assert np.max(np.abs(np.imag(np.asarray(coeffs[-1])))) < 1e-10


    def test_error_decrease_real(self, lms_data_real):
        """
        Critério robusto: o erro não deve explodir e deve ficar pequeno no final.
        Evita falso negativo quando o erro já é ~0 (nível de precisão de máquina).
        """
        data = lms_data_real
        f = QRRLS(filter_order=data["order"], lamb=0.98)

        res = f.optimize(data["x"], data["d"])

        e = np.asarray(res["errors"])
        assert np.max(np.abs(np.imag(e))) < 1e-10
        e = np.real(e).astype(float)

        err_pow = e**2

        # janelas maiores para reduzir ruído numérico
        initial = float(np.mean(err_pow[:200]))
        final = float(np.mean(err_pow[-200:]))

        # 1) não explode
        assert np.isfinite(final)
        assert final < 1e3 * max(initial, 1e-15), (
            f"Erro parece ter explodido: initial={initial:.3e}, final={final:.3e}"
        )

    # 2) deve ser pequeno no final (ajuste se seu dataset tiver ruído forte)
        assert final < 1e-6, f"Erro final grande demais: {final:.3e}"

    def test_lambda_unity_stability_real(self, rls_test_data_real):
        """
        Testa lamb=1.0 (sem esquecimento). Deve ser estável e não gerar NaN.
        """
        data = rls_test_data_real
        f = QRRLS(filter_order=data["order"], lamb=1.0)

        res = f.optimize(data["x"], data["d"])

        w_last = np.asarray(res["coefficients"][-1])
        assert not np.isnan(np.real(w_last)).any()
        assert np.max(np.abs(np.imag(w_last))) < 1e-10
