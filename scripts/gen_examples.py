# gen_examples.py
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pydaptivefiltering as pdf


# --------------- config ---------------

ALGO_NAMES = [
    "LMS", "NLMS", "AffineProjection", "SignData", "SignError", "DualSign",
    "LMSNewton", "Power2ErrorLMS", "TDomainLMS", "TDomainDCT", "TDomainDFT",
    "RLS", "RLSAlt",
    "SMNLMS", "SMBNLMS", "SMAffineProjection", "SimplifiedSMPUAP", "SimplifiedSMAP",
    "LRLSPosteriori", "LRLSErrorFeedback", "LRLSPriori", "NormalizedLRLS",
    "FastRLS", "StabFastRLS",
    "QRRLS",
    "ErrorEquation", "GaussNewton", "GaussNewtonGradient", "RLSIIR", "SteiglitzMcBride",
    "BilinearRLS", "ComplexRBF", "MultilayerPerceptron", "RBF", "VolterraLMS", "VolterraRLS",
    "CFDLMS", "DLCLLMS", "OLSBLMS",
    "AffineProjectionCM", "CMA", "Godard", "Sato",
    "Kalman",
]

FAMILY_MAP = {
    # lms
    "LMS": "lms", "NLMS": "lms", "AffineProjection": "lms",
    "SignData": "lms", "SignError": "lms", "DualSign": "lms",
    "LMSNewton": "lms", "Power2ErrorLMS": "lms", "TDomainLMS": "lms",
    "TDomainDCT": "lms", "TDomainDFT": "lms",
    # rls
    "RLS": "rls", "RLSAlt": "rls",
    # set-membership
    "SMNLMS": "set_membership", "SMBNLMS": "set_membership",
    "SMAffineProjection": "set_membership", "SimplifiedSMPUAP": "set_membership",
    "SimplifiedSMAP": "set_membership",
    # lattice
    "LRLSPosteriori": "lattice", "LRLSErrorFeedback": "lattice",
    "LRLSPriori": "lattice", "NormalizedLRLS": "lattice",
    # fast-rls
    "FastRLS": "fast_rls", "StabFastRLS": "fast_rls",
    # qr
    "QRRLS": "qr_rls",
    # iir
    "ErrorEquation": "iir", "GaussNewton": "iir", "GaussNewtonGradient": "iir",
    "RLSIIR": "iir", "SteiglitzMcBride": "iir",
    # nonlinear
    "BilinearRLS": "nonlinear", "ComplexRBF": "nonlinear",
    "MultilayerPerceptron": "nonlinear", "RBF": "nonlinear",
    "VolterraLMS": "nonlinear", "VolterraRLS": "nonlinear",
    # subband
    "CFDLMS": "subband", "DLCLLMS": "subband", "OLSBLMS": "subband",
    # blind
    "AffineProjectionCM": "blind", "CMA": "blind", "Godard": "blind", "Sato": "blind",
    # kalman
    "Kalman": "kalman",
}

# Valores genéricos por nome de parâmetro (só entram se o __init__ aceitar)
PARAM_DEFAULTS: Dict[str, Any] = {
    "filter_order": 3,

    # IIR-like
    "zeros_order": 2,
    "poles_order": 2,

    # Steps comuns
    "step_size": 1e-2,
    "mu": 1e-2,
    "gamma": 1e-6,  # (levemente maior p/ estabilidade)

    # Transform-domain
    "alpha": 0.99,
    "initial_power": 1.0,

    # NLMS/RLS
    "epsilon": 1e-2,
    "delta": 10.0,
    "forgetting_factor": 0.99,
    "lambda_hat": 0.99,

    # MLP/RBF
    "n_neurons": 12,
    "input_dim": 4,

    # Set-membership
    "gamma_bar": 1.0,
    "gamma_bar_vector": 1.0,
    "L": 2,

    # Power2ErrorLMS
    "bd": 8,
    "tau": 0.25,

    # DualSign
    "rho": 0.5,

    # Subband
    "n_subbands": 64,
    "decimation": 32,
    "smoothing": 0.05,
}

# --------- Defaults por algoritmo (TUNADOS p/ evitar MSE explodindo) ---------
DEFAULT_KWARGS: Dict[str, Dict[str, Any]] = {
    "LMS": {"step_size": 0.05},
    "NLMS": {"step_size": 0.6, "epsilon": 1e-2},

    "AffineProjection": {"step_size": 0.01, "L": 2, "gamma": 1e-4},

    "RLS": {"forgetting_factor": 0.99, "epsilon": 1.0},
    "FastRLS": {"forgetting_factor": 0.99, "epsilon": 1.0},
    "StabFastRLS": {"forgetting_factor": 0.99, "epsilon": 1.0},

    # DualSign
    "DualSign": {"rho": 0.5, "gamma": 2.0, "step_size": 1e-2},

    # >>> Ajuste importante — LMSNewton precisa delta grande e mu pequeno
    "LMSNewton": {"forgetting_factor": 0.01, "step_size": 5e-2},

    "Power2ErrorLMS": {"bd": 8, "tau": 0.25, "step_size": 1e-2},

    # Transform-domain: mantém alpha alto, mas gamma e step mais conservadores
    "TDomainDCT": {"gamma": 1e-3, "alpha": 0.99, "initial_power": 1.0, "step_size": 5e-3},
    "TDomainDFT": {"gamma": 1e-3, "alpha": 0.99, "initial_power": 1.0, "step_size": 5e-3},
    "TDomainLMS": {"gamma": 1e-3, "alpha": 0.99, "initial_power": 1.0, "step_size": 5e-3},

    # IIR: Gauss-Newton é super sensível — use step bem menor + damping maior
    "SteiglitzMcBride": {"zeros_order": 2, "poles_order": 2, "step_size": 5e-4},
    "GaussNewton": {"zeros_order": 2, "poles_order": 2, "step_size": 2e-4, "gamma": 1e-2},
    "GaussNewtonGradient": {"zeros_order": 2, "poles_order": 2, "step_size": 5e-4, "gamma": 1e-3},
    "ErrorEquation": {"zeros_order": 2, "poles_order": 2, "step_size": 1e-3, "gamma": 1e-3},
    "RLSIIR": {"zeros_order": 2, "poles_order": 2, "forgetting_factor": 0.995, "epsilon": 1.0},

    # Set-membership: deixa mais conservador p/ MSE menor
    "SMAffineProjection": {"gamma_bar": 0.05, "gamma": 1e-4, "L": 2, "gamma_bar_vector": 1.0},
    "SimplifiedSMPUAP": {"gamma_bar": 1.0, "gamma": 1e-3, "L": 2},

    # Lattice
    "NormalizedLRLS": {"forgetting_factor": 0.995, "epsilon": 10.0},

    "ComplexRBF": {
        "n_neurons": 12,
        "input_dim": 4,
        "sigma_init": 3.0,
        "uw": 0.1,
        "ur": 1e-3,
        "us": 1e-3,
        "w_init": "np.zeros(12, dtype=np.complex128)",
    },

    # ---- BLIND defaults (Channel EQ) ----
    "AffineProjectionCM": {"step_size": 0.01, "gamma": 1e-4, "memory_length": 2},
    "CMA": {"step_size": 2e-4},
    "Godard": {"step_size": 2e-4},
    "Sato": {"step_size": 2e-4},

    "Kalman": {},
}

SUBBAND_EXAMPLE_DEFAULTS: Dict[str, Any] = {
    "n_subbands": 64,        # M
    "decimation": 32,        # L
    "filter_order": 3,       # Nw
    "step_size": 0.02,
    "gamma": 1e-1,
    "smoothing": 0.05,
}

CODE_VALUE_PREFIXES = ("(", "np.", "pdf.", "complex(", "float(")


@dataclass
class GlobalExampleDefaults:
    ensemble: int = 100
    K: int = 500
    sigma_n2: float = 0.04
    N: int = 4


# --------------- helpers ---------------

def camel_to_snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def get_algo_class(name: str):
    if not hasattr(pdf, name):
        return None
    return getattr(pdf, name)


def supports_complex(cls) -> bool:
    return bool(getattr(cls, "supports_complex", False))


def ensure_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def banner_system_id(algo_name: str) -> str:
    return f"""#################################################################################
#                        Example: System Identification                         #
#################################################################################
#                                                                               #
#  In this example we have a typical system identification scenario. We want    #
#  to estimate the filter coefficients of an unknown system given by Wo.        #
#                                                                               #
#     Adaptive Algorithm used here: {algo_name}                                 #
#                                                                               #
#################################################################################
"""


def banner_channel_eq(algo_name: str) -> str:
    return f"""#################################################################################
#                        Example: Channel Equalization                          #
#################################################################################
#                                                                               #
#  In this example we have a typical channel equalization scenario. We want     #
#  to estimate the transmitted sequence with 4-QAM symbols. In                  #
#  order to accomplish this task we use an adaptive filter with N coefficients. #
#  The procedure is:                                                            #
# 1) Apply the originally transmitted signal distorted by the channel plus      #
#    environment noise as the input signal to an adaptive filter.               #
#    The transmitted signal is a random sequence with 4-QAM symbols and unit    #
#    variance. The channel impulse response is:                                 #
#      h = [1.1+j*0.5, 0.1-j*0.3, -0.2-j*0.1]^T                                 #
#    Environment noise is AWGN with variance 10^(-2.5).                         #
# 2) Choose an adaptive filtering algorithm to govern coefficient updating.     #
#                                                                               #
#     Adaptive Algorithm used here: {algo_name}                                 #
#                                                                               #
#################################################################################
"""


def _format_kwarg_value(v: Any) -> str:
    if isinstance(v, str):
        if v.startswith(CODE_VALUE_PREFIXES):
            return v
        if v.isidentifier():
            return v
    return repr(v)


def _infer_filter_order_default(g: GlobalExampleDefaults) -> int:
    return int(g.N - 1)


def _render_instantiation(algo_name: str, algo_kwargs: Dict[str, Any]) -> str:
    kwargs_str = ", ".join([f"{k}={_format_kwarg_value(v)}" for k, v in algo_kwargs.items()])
    return f"flt = pdf.{algo_name}({kwargs_str})"


def _unitary_dft_matrix_code(n: int) -> str:
    return f"(np.fft.fft(np.eye({n}), axis=0) / np.sqrt({n}))"


def _render_pre_instantiation_code(algo_name: str, algo_kwargs: Dict[str, Any], g: GlobalExampleDefaults) -> str:
    if algo_name == "SimplifiedSMPUAP":
        m = int(algo_kwargs.get("filter_order", _infer_filter_order_default(g)))
        n_taps = m + 1
        return f"""
        up_selector = rng.integers(0, 2, size=({n_taps}, K), dtype=np.int8)
        up_selector[0, :] = 1
""".rstrip()

    if algo_name == "LMSNewton":
        m = int(algo_kwargs.get("filter_order", _infer_filter_order_default(g)))
        n_taps = m + 1
        delta = 0.1
        return f"""
        initial_inv_rx = (np.eye({n_taps}, dtype=complex) / {delta})
""".rstrip()

    return ""


# ----------------------------- subband generator -----------------------------

def _haar_qmf_filters_code() -> str:
    """
    Retorna código python para um filterbank 2-canais (Haar/QMF) com PR simples:
      h0 = [1, 1]/sqrt(2), h1 = [1, -1]/sqrt(2)
      f0 = h0, f1 = h1
    """
    return """
    # Simple 2-channel Haar/QMF analysis/synthesis banks (perfect reconstruction in ideal case)
    analysis_filters = np.array(
        [[1.0,  1.0],
         [1.0, -1.0]],
        dtype=float
    ) / np.sqrt(2.0)
    synthesis_filters = analysis_filters.copy()
""".rstrip()


def _subband_kwargs_and_precode(algo_name: str, cls) -> tuple[Dict[str, Any], str, int]:
    """
    Monta kwargs estáveis para CFDLMS/DLCLLMS/OLSBLMS e também retorna:
      - pre_code: trechos que precisam existir antes da instanciação (ex.: filterbanks)
      - L: decimation/block advance a ser passado pro harness
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    # defaults "gerais" para subband
    # (vamos ajustar dinamicamente conforme signature)
    base = dict(SUBBAND_EXAMPLE_DEFAULTS)

    pre_code = ""
    kwargs: Dict[str, Any] = {}

    # --- preencher por presença de parâmetros comuns ---
    # n_subbands / M
    if "n_subbands" in params:
        kwargs["n_subbands"] = int(base.get("n_subbands", 64))

    # filter_order
    if "filter_order" in params:
        kwargs["filter_order"] = int(base.get("filter_order", 3))

    # step_size
    if "step_size" in params:
        kwargs["step_size"] = float(base.get("step_size", 0.02))
    # alguns podem usar 'mu'
    if "mu" in params and "mu" not in kwargs:
        kwargs["mu"] = float(base.get("step_size", 0.02))

    # gamma
    if "gamma" in params:
        kwargs["gamma"] = float(base.get("gamma", 1e-1))

    # smoothing vs a
    if "smoothing" in params:
        kwargs["smoothing"] = float(base.get("smoothing", 0.05))
    if "a" in params:
        kwargs["a"] = float(base.get("a", base.get("smoothing", 0.05)))

    # decimation naming: decimation / decimation_factor / L
    if "decimation" in params:
        kwargs["decimation"] = int(base.get("decimation", 32))
    if "decimation_factor" in params:
        # OLSBLMS usa decimation_factor e default L=M (pode ficar pesado)
        # vamos forçar L=2 quando usarmos banco Haar 2-canais, senão usa base.
        kwargs["decimation_factor"] = int(base.get("decimation_factor", base.get("decimation", 32)))
    if "L" in params and "L" not in kwargs:
        kwargs["L"] = int(base.get("decimation", 32))

    # --- REQUIRED SPECIAL: OLSBLMS precisa analysis/synthesis_filters ---
    needs_analysis = ("analysis_filters" in params and params["analysis_filters"].default is inspect._empty)
    needs_synthesis = ("synthesis_filters" in params and params["synthesis_filters"].default is inspect._empty)

    if needs_analysis or needs_synthesis or ("analysis_filters" in params) or ("synthesis_filters" in params):
        # Para exemplos/CI: use um filterbank Haar 2-canais.
        # Isso exige M=2 e L=2 para fazer sentido.
        pre_code = _haar_qmf_filters_code()

        if "n_subbands" in params:
            kwargs["n_subbands"] = 2

        # força L=2 onde aplicável
        if "decimation" in params:
            kwargs["decimation"] = 1  # (CFDLMS não usa filterbank, mas se aparecer não faz mal)
        if "decimation_factor" in params:
            kwargs["decimation_factor"] = 2
        if "L" in params:
            kwargs["L"] = 2

        # conecta variáveis criadas no exemplo
        if "analysis_filters" in params:
            kwargs["analysis_filters"] = "analysis_filters"
        if "synthesis_filters" in params:
            kwargs["synthesis_filters"] = "synthesis_filters"

    # --- filtra se não tiver **kwargs ---
    if not has_var_kw:
        kwargs = {k: v for k, v in kwargs.items() if k in params}

    # --- garante requireds (sem default) ---
    required = []
    for name, p in params.items():
        if name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect._empty:
            required.append(name)

    for r in required:
        if r in kwargs:
            continue

        # preencher obrigatórios conhecidos
        if r == "n_subbands":
            kwargs[r] = int(base.get("n_subbands", 64))
        elif r == "filter_order":
            kwargs[r] = int(base.get("filter_order", 3))
        elif r in ("step_size", "mu"):
            kwargs[r] = float(base.get("step_size", 0.02))
        elif r == "gamma":
            kwargs[r] = float(base.get("gamma", 1e-1))
        elif r in ("smoothing", "a"):
            kwargs[r] = float(base.get("smoothing", 0.05))
        elif r in ("decimation", "decimation_factor", "L"):
            kwargs[r] = int(base.get("decimation", 32))
        elif r in ("analysis_filters", "synthesis_filters"):
            # se chegou aqui, precisa mesmo -> usa Haar
            if not pre_code:
                pre_code = _haar_qmf_filters_code()
            if r == "analysis_filters":
                kwargs[r] = "analysis_filters"
            else:
                kwargs[r] = "synthesis_filters"
            if "n_subbands" in params:
                kwargs["n_subbands"] = 2
            if "decimation_factor" in params:
                kwargs["decimation_factor"] = 2
            if "L" in params:
                kwargs["L"] = 2
        else:
            # fallback: tenta PARAM_DEFAULTS
            if r in PARAM_DEFAULTS:
                kwargs[r] = PARAM_DEFAULTS[r]
            else:
                raise RuntimeError(f"Subband init missing required param {r} for {algo_name}")

    if not has_var_kw:
        kwargs = {k: v for k, v in kwargs.items() if k in params}

    # --- define L pro harness ---
    # preferências por nome
    if "decimation" in kwargs:
        L_used = int(kwargs["decimation"])
    elif "decimation_factor" in kwargs:
        L_used = int(kwargs["decimation_factor"])
    elif "L" in kwargs:
        L_used = int(kwargs["L"])
    else:
        # fallback para o que for razoável
        L_used = 32

    return kwargs, pre_code, L_used

def render_subband_example(algo_name: str, g: GlobalExampleDefaults) -> str:
    cls = get_algo_class(algo_name)
    if cls is None:
        raise RuntimeError(f"Algorithm {algo_name} not found in pydaptivefiltering.")

    kwargs, pre_code, L = _subband_kwargs_and_precode(algo_name, cls)

    # defaults do harness (podem ser sobrescritos via env no script gerado)
    ensemble = 50
    K = 4096
    sigma_n2 = 1e-3

    kwargs_dump = ", ".join([f"{k}={_format_kwarg_value(v)}" for k, v in kwargs.items()])

    return f"""#################################################################################
#                        Example: System Identification                         #
#################################################################################
#                                                                               #
#  Subband algorithms can require filterbanks and/or block parameters.          #
#  We use the harness in pydaptivefiltering._utils.example_helper to handle:    #
#    - K multiple of L                                                          #
#    - ensemble averaging                                                       #
#                                                                               #
#     Adaptive Algorithm used here: {algo_name}                                 #
#                                                                               #
#################################################################################

from __future__ import annotations

import os
import numpy as np

import pydaptivefiltering as pdf
from pydaptivefiltering._utils.example_helper import (
    SubbandIDConfig,
    run_subband_system_id,
)
from pydaptivefiltering._utils.plotting import plot_learning_curve

def main(seed: int = 0, plot: bool = True):
    # ---------- CI/benchmark overrides ----------
    ensemble = int(os.getenv("PYDAF_ENSEMBLE", {ensemble}))
    K = int(os.getenv("PYDAF_K", {K}))
    sigma_n2 = float(os.getenv("PYDAF_SIGMA_N2", {sigma_n2}))

    # verbose only on first realization (optimize(verbose=True) if supported)
    verbose_first = bool(int(os.getenv("PYDAF_VERBOSE_FIRST", "1")))

    # L defined by algorithm builder, but allow override
    L = int(os.getenv("PYDAF_L", {int(L)}))

    Wo = np.array([0.32, -0.30, 0.50, 0.20], dtype=float)

    cfg = SubbandIDConfig(
        ensemble=ensemble,
        K=K,
        sigma_n2=sigma_n2,
        Wo=Wo,
    )

{pre_code if pre_code else ""}

    out = run_subband_system_id(
        make_filter=lambda: pdf.{algo_name}({kwargs_dump}),
        L=L,
        cfg=cfg,
        seed=seed,
        verbose_first=verbose_first,
    )

    if plot:
        plot_learning_curve(
            out["MSE_av"],
            out.get("MSEmin_av", None),
            title="{algo_name} learning curve (ensemble-averaged)",
        )

    # enrich output with consistent metadata for benchmarks
    if isinstance(out, dict):
        out = dict(out)
        out.update({{
            "algo": "{algo_name}",
            "family": "subband",
            "scenario": "system_id_subband",
            "seed": seed,
            "ensemble": ensemble,
            "K": K,
            "sigma_n2": sigma_n2,
            "L": L,
            "Wo": Wo,
        }})
    return out

if __name__ == "__main__":
    main(seed=0, plot=True)
"""


# --------------------- system identification generator ---------------------

def render_system_id_example(algo_name: str, family: str, g: GlobalExampleDefaults) -> str:
    cls = get_algo_class(algo_name)
    if cls is None:
        raise RuntimeError(f"Algorithm {algo_name} not found in pydaptivefiltering.")

    is_cx = supports_complex(cls)
    algo_kwargs = build_init_kwargs(algo_name, cls, g)
    if algo_kwargs is None:
        raise RuntimeError(
            f"Cannot autogenerate example for {algo_name}: missing required __init__ params."
        )

    dtype = "np.complex128" if is_cx else "float"
    gen_input = "generate_qam4_input" if is_cx else "generate_sign_input"

    if is_cx:
        wo_line = (
            "Wo = np.array([0.32 + 0.21j, -0.3 + 0.7j, 0.5 - 0.8j, 0.2 + 0.5j], "
            "dtype=np.complex128)"
        )
    else:
        wo_line = "Wo = np.array([0.32, -0.30, 0.50, 0.20], dtype=float)"

    pre_code = _render_pre_instantiation_code(algo_name, algo_kwargs, g)
    inst_line = _render_instantiation(algo_name, algo_kwargs)

    return f"""{banner_system_id(algo_name)}
from __future__ import annotations

from time import perf_counter
import os
import numpy as np

import pydaptivefiltering as pdf
from pydaptivefiltering._utils.example_helper import (
    {gen_input},
    build_desired_from_fir,
    pack_theta_from_result,
)
from pydaptivefiltering._utils.progress import (
    ProgressConfig,
    report_progress,
)
from pydaptivefiltering._utils.plotting import plot_system_id_single_figure

def main(seed: int = 0, plot: bool = True):
    rng_master = np.random.default_rng(seed)

    # ---------- CI/benchmark overrides ----------
    ensemble = int(os.getenv("PYDAF_ENSEMBLE", {g.ensemble}))
    K = int(os.getenv("PYDAF_K", {g.K}))
    sigma_n2 = float(os.getenv("PYDAF_SIGMA_N2", {g.sigma_n2}))
    N = int(os.getenv("PYDAF_N", {g.N}))

    # Progress control (default: off when plot=False)
    verbose_progress = bool(int(os.getenv("PYDAF_VERBOSE_PROGRESS", "0" if not plot else "1")))
    print_every = int(os.getenv("PYDAF_PRINT_EVERY", "10"))
    tail_window = int(os.getenv("PYDAF_TAIL_WINDOW", "50"))

    {wo_line}

    W = np.zeros((N, K + 1, ensemble), dtype={dtype})
    MSE = np.zeros((K, ensemble), dtype=float)
    MSE_aux = np.zeros((K, ensemble), dtype=float)
    MSEmin = np.zeros((K, ensemble), dtype=float)

    cfg = ProgressConfig(
        verbose_progress=verbose_progress,
        print_every=print_every,
        tail_window=tail_window,
    )

    t0 = perf_counter()

    for l in range(ensemble):
        t_real0 = perf_counter()
        rng = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))

        x = {gen_input}(rng, K)
        d, n = build_desired_from_fir(x, Wo, sigma_n2, rng)

{pre_code if pre_code else ""}

        {inst_line}
        res = flt.optimize(x.astype({dtype}), d.astype({dtype}), verbose=(l == 0))

        e = np.asarray(res.errors).ravel()
        MSE[:, l] = np.abs(e) ** 2
        MSE_aux[:, l] = MSE[:, l]
        MSEmin[:, l] = np.abs(n) ** 2

        W[:, :, l] = pack_theta_from_result(res=res, w_last=flt.w, n_coeffs=N, K=K)

        if cfg.verbose_progress:
            report_progress(
                algo_tag="{algo_name}",
                l=l, ensemble=ensemble, t0=t0, t_real0=t_real0,
                mse_col=MSE[:, l], cfg=cfg,
            )

    theta_av = np.mean(W, axis=2)
    MSE_av = np.mean(MSE, axis=1)
    MSE_aux_av = np.mean(MSE_aux, axis=1)
    MSEmin_av = np.mean(MSEmin, axis=1)

    total_s = float(perf_counter() - t0)
    print(f"[Example/{algo_name}] Total ensemble time: {{total_s:.2f}} s")

    if plot:
        plot_system_id_single_figure(
            MSE_av=MSE_av,
            MSEE_av=MSE_aux_av,
            MSEmin_av=MSEmin_av,
            theta_av=theta_av,
            poles_order=0,
            title_prefix="{algo_name}",
            show_complex_coeffs={str(is_cx)},
        )

    # ---------- Return rich metadata for benchmarks ----------
    return {{
        "algo": "{algo_name}",
        "family": "{family}",
        "scenario": "system_id",
        "supports_complex": {str(is_cx)},
        "dtype": "{dtype}",
        "seed": seed,
        "ensemble": ensemble,
        "K": K,
        "sigma_n2": sigma_n2,
        "N": N,
        "total_s": total_s,
        "Wo": Wo,
        "theta_av": theta_av,
        "MSE_av": MSE_av,
        "MSE_aux_av": MSE_aux_av,
        "MSEmin_av": MSEmin_av,
    }}

if __name__ == "__main__":
    main(seed=0, plot=True)
"""



# ---------------- BLIND (Channel EQ) ----------------

def build_init_kwargs_blind(algo_name: str, cls, *, N: int, mu: float, gamma: float, L: int) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    kwargs: Dict[str, Any] = dict(DEFAULT_KWARGS.get(algo_name, {}))

    if "step_size" in params:
        kwargs["step_size"] = float(mu)
    if "mu" in params:
        kwargs["mu"] = float(mu)

    if "gamma" in params:
        kwargs["gamma"] = float(gamma)

    if "filter_order" in params:
        kwargs["filter_order"] = int(N - 1)

    if "memory_length" in params:
        kwargs["memory_length"] = int(L)
    if "L" in params:
        kwargs["L"] = int(L)

    if "w_init" in params:
        kwargs["w_init"] = "w_init"

    if not has_var_kw:
        kwargs = {k: v for k, v in kwargs.items() if k in params}

    required = []
    for name, p in params.items():
        if name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect._empty:
            required.append(name)

    missing = [r for r in required if r not in kwargs]
    if missing:
        raise RuntimeError(f"Blind init missing required params for {algo_name}: {missing}")

    return kwargs


def render_channel_eq_example(algo_name: str) -> str:
    cls = get_algo_class(algo_name)
    if cls is None:
        raise RuntimeError(f"Algorithm {algo_name} not found in pydaptivefiltering.")
    if not supports_complex(cls):
        raise RuntimeError(f"Channel EQ template assumes complex support; {algo_name} is not complex?")

    # defaults (podem ser sobrescritos via env no script gerado)
    ensemble, K, Ksim = 200, 2000, 400
    N = 5
    delay = 1
    gamma = 1e-10
    L = 2

    if algo_name == "AffineProjectionCM":
        mu = 0.1
    else:
        mu = 2e-4

    algo_kwargs = build_init_kwargs_blind(cls=cls, algo_name=algo_name, N=N, mu=mu, gamma=gamma, L=L)
    inst_line = _render_instantiation(algo_name, algo_kwargs)

    return f"""{banner_channel_eq(algo_name)}
from __future__ import annotations

from time import perf_counter
import os
import numpy as np
import matplotlib.pyplot as plt

import pydaptivefiltering as pdf
from pydaptivefiltering._utils.comm import (
    qam4_constellation_unit_var,
    simulate_constellations,
)
from pydaptivefiltering._utils.channel import (
    wiener_equalizer,
    generate_channel_data,
)
from pydaptivefiltering._utils.progress import (
    ProgressConfigChannel,
    report_progressChannel,
)

def main(seed: int = 0, plot: bool = True):
    rng = np.random.default_rng(seed)

    # ---------- CI/benchmark overrides ----------
    ensemble = int(os.getenv("PYDAF_ENSEMBLE", {ensemble}))
    K = int(os.getenv("PYDAF_K", {K}))
    Ksim = int(os.getenv("PYDAF_KSIM", {Ksim}))

    # cenário / parâmetros do equalizador
    N = int(os.getenv("PYDAF_N", {N}))
    delay = int(os.getenv("PYDAF_DELAY", {delay}))
    gamma = float(os.getenv("PYDAF_GAMMA", {gamma}))
    L = int(os.getenv("PYDAF_L", {L}))

    # mu: default depende do algoritmo, mas pode sobrescrever por env
    _mu_default = {mu}
    mu = float(os.getenv("PYDAF_MU", _mu_default))

    # ruído e canal (também sobrescrevíveis se quiser)
    sigma_x2 = float(os.getenv("PYDAF_SIGMA_X2", "1.0"))
    sigma_n2 = float(os.getenv("PYDAF_SIGMA_N2", str(10 ** (-2.5))))

    # progress control (default: off when plot=False)
    verbose_progress = bool(int(os.getenv("PYDAF_VERBOSE_PROGRESS", "0" if not plot else "1")))
    print_every = int(os.getenv("PYDAF_PRINT_EVERY", "10"))
    tail_window = int(os.getenv("PYDAF_TAIL_WINDOW", "200"))
    optimize_verbose_first = bool(int(os.getenv("PYDAF_OPT_VERBOSE_FIRST", "0")))

    # fixed channel (can be overridden by env, if you want later)
    H = np.array(
        [1.1 + 1j * 0.5, 0.1 - 1j * 0.3, -0.2 - 1j * 0.1],
        dtype=np.complex128
    )

    constellation = qam4_constellation_unit_var()
    Wiener = wiener_equalizer(H, N, sigma_x2, sigma_n2, delay)

    # init weights: repeat Wiener + random perturbation
    W = np.repeat(Wiener, repeats=(K + 1 - delay), axis=1)
    W = np.repeat(W[:, :, None], repeats=ensemble, axis=2)
    W = W + (rng.normal(0.0, 1.0, size=W.shape) + 1j * rng.normal(0.0, 1.0, size=W.shape)) / 4.0

    K_eff = K - delay
    MSE = np.zeros((K_eff, ensemble), dtype=np.float64)

    cfg = ProgressConfigChannel(
        verbose_progress=verbose_progress,
        print_every=print_every,
        tail_window=tail_window,
        optimize_verbose_first=optimize_verbose_first,
    )

    t0 = perf_counter()
    for l in range(ensemble):
        t_real0 = perf_counter()

        # (NOTE) This uses the same rng for all l; if you want independent realizations,
        # you can spawn rng per realization similarly to system_id.
        _, x, _ = generate_channel_data(rng, K, H, sigma_n2, constellation)
        w_init = W[:, 0, l].copy()

        {inst_line}

        opt_verbose = bool(cfg.optimize_verbose_first and l == 0)
        res = flt.optimize(x[delay:], verbose=opt_verbose)

        e = np.asarray(res.errors).ravel()
        MSE[:, l] = (np.abs(e) ** 2)
        W[:, -1, l] = flt.w

        if cfg.verbose_progress:
            report_progressChannel(
                l=l,
                ensemble=ensemble,
                t0=t0,
                t_real0=t_real0,
                MSE_col=MSE[:, l],
                K_eff=K_eff,
                cfg=cfg,
            )

    total_s = float(perf_counter() - t0)
    print(f"[Example/{algo_name}] Total ensemble time: {{total_s:.2f}} s ({{total_s/max(1,ensemble):.3f}} s/realization)")

    W_av = np.mean(W, axis=2)
    MSE_av = np.mean(MSE, axis=1)

    equalizerInputMatrix, equalizerOutputVector, equalizerOutputVectorWiener = simulate_constellations(
        rng=rng,
        H=H,
        N=N,
        Ksim=Ksim,
        sigma_n2=sigma_n2,
        constellation=constellation,
        w_final=W_av[:, -1],
        w_wiener=Wiener,
    )

    if plot:
        theta = np.linspace(-np.pi, np.pi, 200)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        ax1, ax2, ax3, ax4 = axes.ravel()

        ax1.plot(np.cos(theta), np.sin(theta), linewidth=1)
        ax1.scatter(np.real(equalizerOutputVector), np.imag(equalizerOutputVector), s=10)
        ax1.scatter(np.real(constellation), np.imag(constellation), s=120, marker="o")
        ax1.set_title("Equalizer output ({algo_name})")
        ax1.grid(True); ax1.set_aspect("equal", adjustable="box"); ax1.set_xlim([-2,2]); ax1.set_ylim([-2,2])

        ax2.plot(np.cos(theta), np.sin(theta), linewidth=1)
        ax2.scatter(np.real(equalizerInputMatrix.ravel()), np.imag(equalizerInputMatrix.ravel()), s=10)
        ax2.scatter(np.real(constellation), np.imag(constellation), s=120, marker="o")
        ax2.set_title("Equalizer input")
        ax2.grid(True); ax2.set_aspect("equal", adjustable="box"); ax2.set_xlim([-2,2]); ax2.set_ylim([-2,2])

        ax3.plot(np.cos(theta), np.sin(theta), linewidth=1)
        ax3.scatter(np.real(equalizerOutputVectorWiener), np.imag(equalizerOutputVectorWiener), s=10)
        ax3.scatter(np.real(constellation), np.imag(constellation), s=120, marker="o")
        ax3.set_title("Equalizer output (Wiener)")
        ax3.grid(True); ax3.set_aspect("equal", adjustable="box"); ax3.set_xlim([-2,2]); ax3.set_ylim([-2,2])

        ax4.semilogy(np.arange(1, K_eff + 1), np.abs(MSE_av))
        ax4.set_title("Learning curve (ensemble-averaged)")
        ax4.grid(True)

        fig.tight_layout()
        plt.show()

    # ---------- Return rich metadata for benchmarks ----------
    return {{
        "algo": "{algo_name}",
        "family": "blind",
        "scenario": "channel_eq",
        "supports_complex": True,
        "seed": seed,
        "ensemble": ensemble,
        "K": K,
        "Ksim": Ksim,
        "K_eff": K_eff,
        "N": N,
        "delay": delay,
        "mu": mu,
        "gamma": gamma,
        "L": L,
        "sigma_x2": sigma_x2,
        "sigma_n2": sigma_n2,
        "H": H,
        "total_s": total_s,
        "Wiener": Wiener,
        "W_av": W_av,
        "MSE_av": MSE_av,
        # opcional: inclui outputs pra constelação
        "equalizer_output": equalizerOutputVector,
        "equalizer_output_wiener": equalizerOutputVectorWiener,
    }}

if __name__ == "__main__":
    main(seed=0, plot=True)
"""



def build_init_kwargs(algo_name: str, cls, g: GlobalExampleDefaults) -> Optional[Dict[str, Any]]:
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    kwargs: Dict[str, Any] = dict(DEFAULT_KWARGS.get(algo_name, {}))

    if "filter_order" in params and "filter_order" not in kwargs:
        kwargs["filter_order"] = _infer_filter_order_default(g)

    if algo_name == "LMSNewton" and "initial_inv_rx" in params and "initial_inv_rx" not in kwargs:
        m = int(kwargs.get("filter_order", _infer_filter_order_default(g)))
        n_taps = m + 1
        delta = 50.0
        kwargs["initial_inv_rx"] = f"(np.eye({n_taps}, dtype=complex) / {delta})"
        kwargs.setdefault("forgetting_factor", 0.995)
        kwargs.setdefault("step_size", 3e-3)

    if algo_name == "TDomainLMS" and "transform_matrix" in params and "transform_matrix" not in kwargs:
        m = int(kwargs.get("filter_order", _infer_filter_order_default(g)))
        n_taps = m + 1
        kwargs["transform_matrix"] = _unitary_dft_matrix_code(n_taps)
        if "assume_unitary" in params:
            kwargs.setdefault("assume_unitary", True)

    if algo_name == "SimplifiedSMPUAP" and "up_selector" in params and "up_selector" not in kwargs:
        kwargs["up_selector"] = "up_selector"

    if algo_name in ("LRLSPosteriori", "LRLSErrorFeedback", "NormalizedLRLS"):
        if "forgetting_factor" in params and "forgetting_factor" not in kwargs:
            kwargs["forgetting_factor"] = 0.995
        if "epsilon" in params and "epsilon" not in kwargs:
            kwargs["epsilon"] = 10.0
        if "delta" in params and "delta" not in kwargs:
            kwargs["delta"] = 10.0

    if "L" in kwargs and "gamma_bar_vector" in params:
        L = int(kwargs["L"])
        gb = float(kwargs.get("gamma_bar", 1.0))
        ggv = kwargs.get("gamma_bar_vector", None)
        if not isinstance(ggv, (list, tuple, np.ndarray)):
            kwargs["gamma_bar_vector"] = [gb] * (L + 1)
        else:
            if len(ggv) != (L + 1):
                kwargs["gamma_bar_vector"] = [gb] * (L + 1)

    if not has_var_kw:
        kwargs = {k: v for k, v in kwargs.items() if k in params}

    required = []
    for name, p in params.items():
        if name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect._empty:
            required.append(name)

    for r in required:
        if r in kwargs:
            continue
        if r == "filter_order":
            kwargs[r] = _infer_filter_order_default(g)
            continue
        if r == "transform_matrix":
            m = int(kwargs.get("filter_order", _infer_filter_order_default(g)))
            kwargs[r] = _unitary_dft_matrix_code(m + 1)
            continue
        if r in PARAM_DEFAULTS:
            if r == "input_dim":
                kwargs[r] = int(g.N)
            else:
                kwargs[r] = PARAM_DEFAULTS[r]

    if not has_var_kw:
        kwargs = {k: v for k, v in kwargs.items() if k in params}

    still_missing = [r for r in required if r not in kwargs]
    if still_missing:
        return None

    return kwargs


def _write_error_stub(out_path: Path, algo_name: str, err: Exception) -> None:
    out_path.write_text(
        f"""# Auto-example generation failed for {algo_name}
# Reason: {type(err).__name__}: {err}

def main(seed: int = 0, plot: bool = True):
    raise RuntimeError({repr(f"Auto-example generation failed for {algo_name}: {type(err).__name__}: {err}")})
""",
        encoding="utf-8",
    )


def main(out_root: str = "auto_examples", seed: int = 0) -> None:
    root = Path(out_root)
    g = GlobalExampleDefaults()

    for algo_name in ALGO_NAMES:
        family = FAMILY_MAP.get(algo_name)
        if family is None:
            continue

        out_dir = root / family
        ensure_folder(out_dir)

        snake = camel_to_snake(algo_name)

        if family == "blind":
            out_path = out_dir / f"example_channel_eq_{snake}.py"
            try:
                code = render_channel_eq_example(algo_name)
            except Exception as e:
                _write_error_stub(out_path, algo_name, e)
                print(f"[gen_examples] wrote ERROR stub {out_path} ({type(e).__name__}: {e})")
                continue

            out_path.write_text(code, encoding="utf-8")
            print(f"[gen_examples] wrote {out_path}")
            continue

        if family == "subband":
            out_path = out_dir / f"example_system_id_{snake}.py"
            try:
                code = render_subband_example(algo_name, g)
            except Exception as e:
                _write_error_stub(out_path, algo_name, e)
                print(f"[gen_examples] wrote ERROR stub {out_path} ({type(e).__name__}: {e})")
                continue

            out_path.write_text(code, encoding="utf-8")
            print(f"[gen_examples] wrote {out_path}")
            continue

        # -------- Kalman: você fará manualmente --------
        out_path = out_dir / f"example_system_id_{snake}.py"
        if family == "kalman":
            print(f"[gen_examples] skipped (manual) {algo_name}")
            _write_error_stub(out_path, algo_name, RuntimeError("manual example pending"))
            continue

        # -------- Default: system_id --------
        try:
            code = render_system_id_example(algo_name, family, g)
        except Exception as e:
            _write_error_stub(out_path, algo_name, e)
            print(f"[gen_examples] wrote ERROR stub {out_path} ({type(e).__name__}: {e})")
            continue

        out_path.write_text(code, encoding="utf-8")
        print(f"[gen_examples] wrote {out_path}")


if __name__ == "__main__":
    main()
