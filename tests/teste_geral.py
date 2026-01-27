# tests/test_geral.py
import numpy as np
import pytest
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Type, List, Tuple

# ============================================================
# Importa os filtros do pacote (ajuste o nome do pacote se necessário)
# ============================================================

from pydaptivefiltering import (
    LMS, NLMS, AffineProjection, SignData, SignError, DualSign, LMS_Newton,
    Power2_Error, TDomain, TDomain_DCT, TDomain_DFT,
    RLS, RLS_Alt,
    SM_NLMS, SM_BNLMS, SM_AP, SM_Simp_AP, SM_Simp_PUAP,
    LatticeRLS, LatticeRLSErrorFeedback, LatticeRLS_Priori, NormalizedLatticeRLS,
    FastRLS, StabFastRLS,
    QR_RLS,
    ErrorEquation, GaussNewton, GaussNewton_GradientBased, RLS_IIR, Steiglitz_McBride,
    BilinearRLS, ComplexRBF, MultilayerPerceptron, RBF, VolterraLMS, VolterraRLS,
)

# Futuros (deixe comentado até existir no pacote; quando implementar, descomente)
# from pydaptivefiltering import CFD_LMS, DLCL_LMS, OLSB_LMS
# from pydaptivefiltering import Blind_Affine_Projection, CMA, Godard, Sato


# ============================================================
# Helpers de geração de dados
# ============================================================

def generate_fir_supervised_data(
    n_samples: int,
    order: int,
    noise_std: float = 0.01,
    complex_data: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Gera dataset supervisionado padrão:
        x ~ N(0,1), d = w_true^H x_k + ruido

    Convenção: x_k = [x[k], x[k-1], ..., x[k-order]] (mais recente primeiro)
    w_true tem comprimento order+1.
    """
    rng = np.random.default_rng(seed)

    if complex_data:
        x = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
        w_true = (rng.standard_normal(order + 1) + 1j * rng.standard_normal(order + 1)) / np.sqrt(2)
        noise = noise_std * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
    else:
        x = rng.standard_normal(n_samples).astype(float)
        w_true = rng.standard_normal(order + 1).astype(float)
        noise = noise_std * rng.standard_normal(n_samples).astype(float)

    x_pad = np.zeros(n_samples + order, dtype=x.dtype)
    x_pad[order:] = x

    d = np.zeros(n_samples, dtype=x.dtype)
    for k in range(n_samples):
        xk = x_pad[k: k + order + 1][::-1]
        # real: dot; complex: w^H x
        if complex_data:
            d[k] = np.vdot(w_true, xk)  # conj(w)^T x
        else:
            d[k] = float(np.dot(w_true, xk))

    d = d + noise

    return {"x": x, "d": d, "w_true": w_true, "order": order, "n_samples": n_samples, "complex": complex_data}


def tail_mse(e: np.ndarray, tail: int = 200) -> float:
    e = np.asarray(e)
    if e.size < tail:
        tail = e.size
    return float(np.mean(np.abs(e[-tail:]) ** 2))


def rel_w_error(w_est: np.ndarray, w_true: np.ndarray, eps: float = 1e-12) -> float:
    w_est = np.asarray(w_est)
    w_true = np.asarray(w_true)
    num = np.linalg.norm(w_est - w_true)
    den = np.linalg.norm(w_true) + eps
    return float(num / den)


def extract_last_w(coefficients: Any) -> np.ndarray:
    """
    Seu framework geralmente retorna list de w ao longo do tempo.
    Aceita list, np.ndarray (n_taps x steps), etc.
    """
    if isinstance(coefficients, list):
        return np.asarray(coefficients[-1])
    arr = np.asarray(coefficients)
    if arr.ndim == 2:
        return arr[:, -1]
    return arr


def has_key(res: Dict[str, Any], key: str) -> bool:
    return isinstance(res, dict) and (key in res)


# ============================================================
# Definição de perfis de teste
# ============================================================

@dataclass(frozen=True)
class FilterSpec:
    cls: Type
    name: str
    category: str  # "supervised_linear", "supervised_nonlinear", "iir", "blind", etc.
    supports_complex: bool = True
    supports_real: bool = True
    # thresholds padrão (ajuste por filtro quando necessário)
    mse_threshold_real: Optional[float] = None
    mse_threshold_complex: Optional[float] = None
    wrel_threshold_real: Optional[float] = None
    wrel_threshold_complex: Optional[float] = None
    # kwargs default para instanciar
    init_kwargs: Optional[Dict[str, Any]] = None


# ============================================================
# Registro central (adicione novos filtros aqui)
# ============================================================

FILTERS: List[FilterSpec] = [
    # -------- LMS family (supervised linear) --------
    FilterSpec(LMS, "LMS", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=5e-3, mse_threshold_complex=5e-3, wrel_threshold_real=0.25, wrel_threshold_complex=0.35,
               init_kwargs={"step": 1e-2}),
    FilterSpec(NLMS, "NLMS", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=5e-3, mse_threshold_complex=5e-3, wrel_threshold_real=0.25, wrel_threshold_complex=0.35,
               init_kwargs={}),
    FilterSpec(AffineProjection, "AffineProjection", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=5e-3, mse_threshold_complex=5e-3, wrel_threshold_real=0.30, wrel_threshold_complex=0.40,
               init_kwargs={}),
    FilterSpec(SignData, "SignData", "supervised_linear", supports_complex=False, supports_real=True,
               mse_threshold_real=2e-2, wrel_threshold_real=0.6, init_kwargs={}),
    FilterSpec(SignError, "SignError", "supervised_linear", supports_complex=False, supports_real=True,
               mse_threshold_real=2e-2, wrel_threshold_real=0.6, init_kwargs={}),
    FilterSpec(DualSign, "DualSign", "supervised_linear", supports_complex=False, supports_real=True,
               mse_threshold_real=2e-2, wrel_threshold_real=0.6, init_kwargs={}),
    FilterSpec(LMS_Newton, "LMS_Newton", "supervised_linear", supports_complex=False, supports_real=True,
               mse_threshold_real=1e-2, wrel_threshold_real=0.5, init_kwargs={}),
    FilterSpec(Power2_Error, "Power2_Error", "supervised_linear", supports_complex=False, supports_real=True,
               mse_threshold_real=2e-2, wrel_threshold_real=0.6, init_kwargs={}),

    # Transform-domain variants (dependem de params; aqui testamos sanidade+MSE relaxado)
    FilterSpec(TDomain, "TDomain", "supervised_linear", supports_complex=False, supports_real=True,
               mse_threshold_real=2e-2, wrel_threshold_real=0.8, init_kwargs={}),
    FilterSpec(TDomain_DCT, "TDomain_DCT", "supervised_linear", supports_complex=False, supports_real=True,
               mse_threshold_real=2e-2, wrel_threshold_real=0.8, init_kwargs={}),
    FilterSpec(TDomain_DFT, "TDomain_DFT", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=2e-2, mse_threshold_complex=2e-2, wrel_threshold_real=0.8, wrel_threshold_complex=0.9,
               init_kwargs={}),

    # -------- RLS family (supervised linear) --------
    FilterSpec(RLS, "RLS", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=2e-3, mse_threshold_complex=2e-3, wrel_threshold_real=0.15, wrel_threshold_complex=0.25,
               init_kwargs={"forgetting_factor": 0.995}),
    FilterSpec(RLS_Alt, "RLS_Alt", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=3e-3, mse_threshold_complex=3e-3, wrel_threshold_real=0.20, wrel_threshold_complex=0.30,
               init_kwargs={"forgetting_factor": 0.995}),

    # -------- Set-membership (supervised linear; thresholds mais relaxados) --------
    FilterSpec(SM_NLMS, "SM_NLMS", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=2e-2, mse_threshold_complex=2e-2, wrel_threshold_real=0.8, wrel_threshold_complex=0.9,
               init_kwargs={}),
    FilterSpec(SM_BNLMS, "SM_BNLMS", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=2e-2, mse_threshold_complex=2e-2, wrel_threshold_real=0.8, wrel_threshold_complex=0.9,
               init_kwargs={}),
    FilterSpec(SM_AP, "SM_AP", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=2e-2, mse_threshold_complex=2e-2, wrel_threshold_real=0.9, wrel_threshold_complex=0.9,
               init_kwargs={}),
    FilterSpec(SM_Simp_AP, "SM_Simp_AP", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=2e-2, mse_threshold_complex=2e-2, wrel_threshold_real=0.9, wrel_threshold_complex=0.9,
               init_kwargs={}),
    FilterSpec(SM_Simp_PUAP, "SM_Simp_PUAP", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=2e-2, mse_threshold_complex=2e-2, wrel_threshold_real=0.9, wrel_threshold_complex=0.9,
               init_kwargs={}),


    # -------- Lattice-based RLS (supervised linear; podem ser sensíveis) --------
    FilterSpec(LatticeRLS, "LatticeRLS", "supervised_linear", supports_complex=False, supports_real=True,
               mse_threshold_real=1e-2, wrel_threshold_real=0.6,
               init_kwargs={"forgetting_factor": 0.995}),
    FilterSpec(LatticeRLSErrorFeedback, "LatticeRLSErrorFeedback", "supervised_linear", supports_complex=False, supports_real=True,
               mse_threshold_real=1e-2, wrel_threshold_real=0.6,
               init_kwargs={"forgetting_factor": 0.995}),
    FilterSpec(LatticeRLS_Priori, "LatticeRLS_Priori", "supervised_linear", supports_complex=False, supports_real=True,
               mse_threshold_real=1e-2, wrel_threshold_real=0.6,
               init_kwargs={"forgetting_factor": 0.995}),
    FilterSpec(NormalizedLatticeRLS, "NormalizedLatticeRLS", "supervised_linear", supports_complex=False, supports_real=True,
               mse_threshold_real=1e-2, wrel_threshold_real=0.7,
               init_kwargs={"forgetting_factor": 0.995}),

    # -------- Fast transversal RLS --------
    FilterSpec(FastRLS, "FastRLS", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=2e-3, mse_threshold_complex=2e-3, wrel_threshold_real=0.20, wrel_threshold_complex=0.35,
               init_kwargs={"forgetting_factor": 0.995}),
    # StabFastRLS: sua versão atual é REAL-only
    FilterSpec(StabFastRLS, "StabFastRLS", "supervised_linear", supports_complex=False, supports_real=True,
               mse_threshold_real=2e-3, wrel_threshold_real=0.25,
               init_kwargs={"forgetting_factor": 0.995, "epsilon": 0.1}),

    # -------- QR-RLS --------
    FilterSpec(QR_RLS, "QR_RLS", "supervised_linear", supports_complex=True, supports_real=True,
               mse_threshold_real=3e-3, mse_threshold_complex=3e-3, wrel_threshold_real=0.25, wrel_threshold_complex=0.35,
               init_kwargs={"forgetting_factor": 0.995}),


    # -------- IIR / Nonlinear / Blind --------
    # Para esses, o "teste geral" deve ser sanidade/shape por padrão,
    # porque não dá pra exigir convergência linear para FIR w_true.
    FilterSpec(ErrorEquation, "ErrorEquation", "iir", supports_complex=True, supports_real=True, init_kwargs={}),
    FilterSpec(GaussNewton, "GaussNewton", "iir", supports_complex=True, supports_real=True, init_kwargs={}),
    FilterSpec(GaussNewton_GradientBased, "GaussNewton_GradientBased", "iir", supports_complex=True, supports_real=True, init_kwargs={}),
    FilterSpec(RLS_IIR, "RLS_IIR", "iir", supports_complex=True, supports_real=True, init_kwargs={}),
    FilterSpec(Steiglitz_McBride, "Steiglitz_McBride", "iir", supports_complex=True, supports_real=True, init_kwargs={}),


    FilterSpec(BilinearRLS, "BilinearRLS", "supervised_nonlinear", supports_complex=True, supports_real=True, init_kwargs={}),
    FilterSpec(ComplexRBF, "ComplexRBF", "supervised_nonlinear", supports_complex=True, supports_real=True, init_kwargs={}),
    FilterSpec(MultilayerPerceptron, "MultilayerPerceptron", "supervised_nonlinear", supports_complex=True, supports_real=True, init_kwargs={}),
    FilterSpec(RBF, "RBF", "supervised_nonlinear", supports_complex=False, supports_real=True, init_kwargs={}),
    FilterSpec(VolterraLMS, "VolterraLMS", "supervised_nonlinear", supports_complex=True, supports_real=True, init_kwargs={}),
    FilterSpec(VolterraRLS, "VolterraRLS", "supervised_nonlinear", supports_complex=True, supports_real=True, init_kwargs={}),
]


# ============================================================
# Funções para instanciar filtros de forma robusta
# ============================================================

def instantiate_filter(spec: FilterSpec, order: int):
    """
    Instancia o filtro tentando passar filter_order.
    Se a assinatura do __init__ variar, você pode adicionar regras aqui.
    """
    kwargs = dict(spec.init_kwargs or {})
    # a maioria segue (filter_order=...)
    try:
        return spec.cls(filter_order=order, **kwargs)
    except TypeError:
        # fallback: alguns podem usar "m" ou "order"
        try:
            return spec.cls(m=order, **kwargs)
        except TypeError:
            return spec.cls(order=order, **kwargs)


def run_optimize(filt, x, d):
    """
    Chama optimize de forma robusta (alguns aceitam verbose, outros kwargs específicos).
    """
    sig = inspect.signature(filt.optimize)
    params = sig.parameters

    if "verbose" in params:
        return filt.optimize(x, d, verbose=False)
    return filt.optimize(x, d)


# ============================================================
# Parametrização PyTest
# ============================================================

def specs_for(mode: str) -> List[FilterSpec]:
    if mode == "real":
        return [s for s in FILTERS if s.supports_real]
    if mode == "complex":
        return [s for s in FILTERS if s.supports_complex]
    raise ValueError(mode)


@pytest.mark.parametrize("spec", specs_for("real"), ids=lambda s: f"{s.name}[real]")
def test_all_filters_real_general(spec: FilterSpec):
    """
    Teste geral (REAL):
    - sempre testa sanidade (sem NaN/Inf, shapes)
    - para 'supervised_linear', testa MSE final e erro relativo de w
    """
    data = generate_fir_supervised_data(
        n_samples=1500,
        order=3,
        noise_std=0.01,
        complex_data=False,
        seed=2026,
    )
    x, d = data["x"], data["d"]
    w_true = data["w_true"]

    f = instantiate_filter(spec, order=data["order"])
    res = run_optimize(f, x, d)

    assert isinstance(res, dict), f"{spec.name}: optimize must return a dict"

    # outputs must exist for supervised filters (quase todos)
    if has_key(res, "outputs"):
        y = np.asarray(res["outputs"])
        assert y.shape == d.shape, f"{spec.name}: outputs shape mismatch"
        assert np.all(np.isfinite(np.real(y))), f"{spec.name}: outputs has NaN/Inf (real part)"
        assert np.all(np.isfinite(np.imag(y))), f"{spec.name}: outputs has NaN/Inf (imag part)"

    # coefficients history must exist for AdaptiveFilter family
    assert has_key(res, "coefficients"), f"{spec.name}: missing 'coefficients' history"

    # Se for linear supervisionado, mede acurácia
    if spec.category == "supervised_linear":
        # erros
        if has_key(res, "priori_errors"):
            e = np.asarray(res["priori_errors"])
        elif has_key(res, "errors"):
            e = np.asarray(res["errors"])
        else:
            # fallback: d - y
            y = np.asarray(res["outputs"])
            e = d - y

        assert e.shape == d.shape, f"{spec.name}: error vector shape mismatch"
        assert np.all(np.isfinite(np.real(e))), f"{spec.name}: error has NaN/Inf"

        mse = tail_mse(e, tail=200)

        # thresholds podem ser None => só sanidade
        if spec.mse_threshold_real is not None:
            assert mse < spec.mse_threshold_real, f"{spec.name}: MSE tail too high: {mse:.3e}"

        # erro relativo em w
        w_est = extract_last_w(res["coefficients"])
        # se algum filtro devolve complexo mas estamos em teste real: imag deve ser pequena
        if np.iscomplexobj(w_est):
            assert np.max(np.abs(np.imag(w_est))) < 1e-6, f"{spec.name}: complex w_est on real test"
            w_est = np.real(w_est)

        wrel = rel_w_error(w_est[: len(w_true)], w_true)
        if spec.wrel_threshold_real is not None:
            assert wrel < spec.wrel_threshold_real, f"{spec.name}: rel-w error too high: {wrel:.3f}"


@pytest.mark.parametrize("spec", specs_for("complex"), ids=lambda s: f"{s.name}[complex]")
def test_all_filters_complex_general(spec: FilterSpec):
    """
    Teste geral (COMPLEX):
    - sanidade + shapes
    - para 'supervised_linear', testa MSE final e erro relativo de w
    """
    data = generate_fir_supervised_data(
        n_samples=1500,
        order=3,
        noise_std=0.01,
        complex_data=True,
        seed=2026,
    )
    x, d = data["x"], data["d"]
    w_true = data["w_true"]

    f = instantiate_filter(spec, order=data["order"])
    res = run_optimize(f, x, d)

    assert isinstance(res, dict), f"{spec.name}: optimize must return a dict"

    if has_key(res, "outputs"):
        y = np.asarray(res["outputs"])
        assert y.shape == d.shape, f"{spec.name}: outputs shape mismatch"
        assert np.all(np.isfinite(np.real(y))), f"{spec.name}: outputs has NaN/Inf (real part)"
        assert np.all(np.isfinite(np.imag(y))), f"{spec.name}: outputs has NaN/Inf (imag part)"

    assert has_key(res, "coefficients"), f"{spec.name}: missing 'coefficients' history"

    if spec.category == "supervised_linear":
        if has_key(res, "priori_errors"):
            e = np.asarray(res["priori_errors"])
        elif has_key(res, "errors"):
            e = np.asarray(res["errors"])
        else:
            y = np.asarray(res["outputs"])
            e = d - y

        assert e.shape == d.shape, f"{spec.name}: error vector shape mismatch"
        assert np.all(np.isfinite(np.real(e))), f"{spec.name}: error has NaN/Inf (real part)"
        assert np.all(np.isfinite(np.imag(e))), f"{spec.name}: error has NaN/Inf (imag part)"

        mse = tail_mse(e, tail=200)
        if spec.mse_threshold_complex is not None:
            assert mse < spec.mse_threshold_complex, f"{spec.name}: MSE tail too high: {mse:.3e}"

        w_est = extract_last_w(res["coefficients"])
        wrel = rel_w_error(w_est[: len(w_true)], w_true)
        if spec.wrel_threshold_complex is not None:
            assert wrel < spec.wrel_threshold_complex, f"{spec.name}: rel-w error too high: {wrel:.3f}"
