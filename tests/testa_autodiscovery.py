# tests/test_autodiscovery.py

from __future__ import annotations

import inspect
import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type, Optional

import numpy as np
import pytest

import pydaptivefiltering
from pydaptivefiltering.base import AdaptiveFilter


# ============================================================
# Helpers básicos
# ============================================================

def tail_mse(e: np.ndarray, tail: int = 500) -> float:
    e = np.asarray(e)
    tail = min(int(tail), int(e.size))
    if tail <= 0:
        return float("inf")
    return float(np.mean(np.abs(e[-tail:]) ** 2))


def rel_w_error(w_est: np.ndarray, w_true: np.ndarray, eps: float = 1e-12) -> float:
    w_est = np.asarray(w_est).reshape(-1)
    w_true = np.asarray(w_true).reshape(-1)
    num = np.linalg.norm(w_est - w_true)
    den = np.linalg.norm(w_true) + eps
    return float(num / den)


def extract_last_w(coefficients: Any) -> Any:
    """
    Retorna "últimos coeficientes" sem forçar para np.ndarray,
    pois alguns filtros (ex.: MLP) usam dict ou matriz 2D.
    """
    if coefficients is None:
        return None

    if isinstance(coefficients, list) and len(coefficients) > 0:
        return coefficients[-1]

    if isinstance(coefficients, dict):
        return coefficients

    try:
        arr = np.asarray(coefficients)
        if arr.dtype == object and arr.shape == ():
            return coefficients

        if arr.ndim == 2:
            if arr.shape[0] >= arr.shape[1]:
                return arr[-1, :]
            return arr[:, -1]
    except Exception:
        return coefficients

    return coefficients


def _coef_summary(coeffs: Any) -> str:
    """
    Resume coeficientes para debug sem quebrar (dict/list/array).
    """
    if coeffs is None:
        return "w=None"

    if isinstance(coeffs, dict):
        keys = list(coeffs.keys())
        sizes = []
        maxabs = []
        for k in keys:
            v = coeffs[k]
            if isinstance(v, list) and len(v) > 0:
                v_last = v[-1]
            else:
                v_last = v
            try:
                a = np.asarray(v_last)
                sizes.append(f"{k}:{a.size}")
                maxabs.append(np.max(np.abs(a)) if a.size > 0 else 0.0)
            except Exception:
                sizes.append(f"{k}:?")
                maxabs.append(0.0)
        return f"w(dict) keys={keys} sizes={sizes} max|w|~{(max(maxabs) if maxabs else 0.0):.3e}"

    try:
        a = np.asarray(coeffs).reshape(-1)
        if a.size == 0:
            return "w.size=0"
        return f"w.size={a.size} max|w|={np.max(np.abs(a)):.3e}"
    except Exception:
        return "w(unprintable)"


def run_optimize(filt: AdaptiveFilter, x: np.ndarray, d: np.ndarray) -> Dict[str, Any]:
    """
    Chama optimize respeitando a assinatura.

    - Supervisionado: optimize(x, d, ...)
    - Blind (ex.: Godard/CMA/Sato): optimize(x, ...)
    - verbose: passa verbose=False se existir
    """
    sig = inspect.signature(filt.optimize)
    params = list(sig.parameters.values())

    desired_names = {"desired_signal", "desired", "d", "reference", "target"}
    has_desired = any(p.name in desired_names for p in params)

    args = (x, d) if has_desired else (x,)

    kwargs: Dict[str, Any] = {}
    if "verbose" in sig.parameters:
        kwargs["verbose"] = False

    return filt.optimize(*args, **kwargs)


def instantiate_filter(cls: Type[AdaptiveFilter], order: int) -> AdaptiveFilter:
    """
    Instancia filtro tentando:
      0) hook opcional default_test_init_kwargs(order)
      1) nomes comuns de parâmetro de ordem: filter_order/m/order/memory
      2) IIR com M/N
      3) fallback posicional: cls(order, **extra_kwargs)

    Para alguns filtros não-lineares (ex.: MLP), passa kwargs úteis p/ convergir.
    """
    init_sig = inspect.signature(cls.__init__)
    init_params = init_sig.parameters
    param_names = [p for p in init_params.keys() if p != "self"]

    # 0) Hook opcional por classe (ideal para algoritmos com parâmetros obrigatórios)
    if hasattr(cls, "default_test_init_kwargs"):
        try:
            kw = cls.default_test_init_kwargs(order)  # type: ignore[attr-defined]
            if isinstance(kw, dict) and len(kw) > 0:
                return cls(**kw)  # type: ignore[arg-type]
        except Exception:
            pass

    extra_kwargs: Dict[str, Any] = {}

    # Overrides específicos
    if cls.__name__ == "MultilayerPerceptron":
        if "learning_rate" in init_params:
            extra_kwargs["learning_rate"] = 0.01
        if "mu" in init_params:
            extra_kwargs["mu"] = 0.01
        if "epochs" in init_params:
            extra_kwargs["epochs"] = 200
        if "n_epochs" in init_params:
            extra_kwargs["n_epochs"] = 200
        if "hidden_neurons" in init_params:
            extra_kwargs["hidden_neurons"] = 16
        if "n_hidden" in init_params:
            extra_kwargs["n_hidden"] = 16
        if "seed" in init_params:
            extra_kwargs["seed"] = 2026

    # 1) parâmetros comuns de ordem/memória
    for name in ("filter_order", "m", "order", "memory"):
        if name in init_params:
            try:
                return cls(**{name: order}, **extra_kwargs)
            except TypeError:
                pass

    # 2) IIR (ex.: SteiglitzMcBride) pede M e N
    if "M" in init_params and "N" in init_params:
        try:
            return cls(M=order, N=order, **extra_kwargs)
        except TypeError:
            pass
        try:
            return cls(order, order, **extra_kwargs)
        except TypeError:
            pass

    # 3) fallback posicional
    try:
        return cls(order, **extra_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Could not instantiate {cls.__name__} with order={order}. __init__ params: {param_names}"
        ) from e


def normalize_result(res: Dict[str, Any], d: np.ndarray) -> Dict[str, Any]:
    """
    Normaliza resultados para schema canônico:
      - y (output) opcional
      - e (error) obrigatório
      - coefficients (histórico ou final)
      - w_final (coef. final)
      - meta (dict)

    Aceita retorno legacy com: outputs/errors/priori_errors/coefficients.
    """
    y = None
    if "y" in res and res["y"] is not None:
        y = np.asarray(res["y"])
    elif "outputs" in res and res["outputs"] is not None:
        y = np.asarray(res["outputs"])

    e = None
    if "e" in res and res["e"] is not None:
        e = np.asarray(res["e"])
    elif "priori_errors" in res and res["priori_errors"] is not None:
        e = np.asarray(res["priori_errors"])
    elif "errors" in res and res["errors"] is not None:
        e = np.asarray(res["errors"])
    elif y is not None:
        e = np.asarray(d) - y
    else:
        raise AssertionError("Missing error: expected e/errors/priori_errors or outputs.")

    coeffs = res.get("coefficients", None)
    w_final = res.get("w_final", None)
    if w_final is None:
        w_final = extract_last_w(coeffs)

    out = dict(res)
    out["y"] = y
    out["e"] = e
    out["coefficients"] = coeffs
    out["w_final"] = w_final
    out.setdefault("meta", {})
    return out


def assert_schema(df: "DiscoveredFilter", resN: Dict[str, Any], d: np.ndarray, complex_data: bool):
    assert isinstance(resN, dict), f"{df.qualname}: result must be dict"
    assert "coefficients" in resN, f"{df.qualname}: missing coefficients"
    assert "w_final" in resN and resN["w_final"] is not None, f"{df.qualname}: missing w_final"
    assert "e" in resN and resN["e"] is not None, f"{df.qualname}: missing error vector"

    e = np.asarray(resN["e"])
    assert e.ndim == 1, f"{df.qualname}: error must be 1D (got shape={e.shape})"

    # Algumas rotinas retornam comprimento diferente (ex.: domínio decimado). Não força == d.shape aqui.
    # Vamos checar comprimento no teste dependendo do modo.
    assert np.all(np.isfinite(np.real(e))), f"{df.qualname}: error has NaN/Inf (real)"

    if complex_data:
        assert np.all(np.isfinite(np.imag(e))), f"{df.qualname}: error has NaN/Inf (imag)"
    else:
        assert np.max(np.abs(np.imag(e))) < 1e-8, f"{df.qualname}: unexpected imag error in REAL test"

    if resN["y"] is not None:
        y = np.asarray(resN["y"])
        assert y.ndim == 1, f"{df.qualname}: output must be 1D (got shape={y.shape})"
        assert np.all(np.isfinite(np.real(y))), f"{df.qualname}: outputs has NaN/Inf (real)"
        if complex_data:
            assert np.all(np.isfinite(np.imag(y))), f"{df.qualname}: outputs has NaN/Inf (imag)"
        else:
            assert np.max(np.abs(np.imag(y))) < 1e-8, f"{df.qualname}: unexpected imag outputs in REAL test"


def _debug_blob(df: "DiscoveredFilter", resN: Dict[str, Any], coeffs_last: Any) -> str:
    parts = [f" | module={df.module}"]
    y = resN.get("y", None)
    if y is not None:
        yy = np.asarray(y)
        parts.append(f" | y.shape={yy.shape}")
        parts.append(f" | max|y|={np.max(np.abs(yy)):.3e}")
    ee = np.asarray(resN["e"])
    parts.append(f" | e.shape={ee.shape}")
    parts.append(f" | max|e|={np.max(np.abs(ee)):.3e}")
    parts.append(" | " + _coef_summary(coeffs_last))
    return "".join(parts)


# ============================================================
# Auto-discovery
# ============================================================

@dataclass(frozen=True)
class DiscoveredFilter:
    cls: Type[AdaptiveFilter]
    qualname: str
    module: str
    supports_complex: bool


def iter_package_modules(pkg) -> List[str]:
    modules: List[str] = []
    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        return modules
    for modinfo in pkgutil.walk_packages(pkg_path, pkg.__name__ + "."):
        modules.append(modinfo.name)
    return modules


def _all_subclasses(cls: Type) -> List[Type]:
    out: List[Type] = []
    stack = [cls]
    seen = set()
    while stack:
        c = stack.pop()
        for sc in c.__subclasses__():
            if sc in seen:
                continue
            seen.add(sc)
            out.append(sc)
            stack.append(sc)
    return out


def discover_adaptive_filters() -> List[DiscoveredFilter]:
    for modname in iter_package_modules(pydaptivefiltering):
        try:
            importlib.import_module(modname)
        except Exception:
            continue

    discovered: List[DiscoveredFilter] = []
    for c in _all_subclasses(AdaptiveFilter):
        if inspect.isabstract(c):
            continue
        supports_complex = getattr(c, "supports_complex", None)
        if supports_complex is None:
            continue
        discovered.append(
            DiscoveredFilter(
                cls=c,
                qualname=c.__name__,
                module=c.__module__,
                supports_complex=bool(supports_complex),
            )
        )

    uniq: Dict[Tuple[str, str], DiscoveredFilter] = {(d.module, d.qualname): d for d in discovered}
    return sorted(uniq.values(), key=lambda z: (z.module, z.qualname))


DISCOVERED = discover_adaptive_filters()


# ============================================================
# Datasets
# ============================================================

def generate_fir_supervised_data(
    n_samples: int,
    order: int,
    noise_std: float = 0.01,
    complex_data: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
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
        xk = x_pad[k : k + order + 1][::-1]
        if complex_data:
            d[k] = np.vdot(w_true, xk)
        else:
            d[k] = float(np.dot(w_true, xk))

    d = d + noise
    return {"x": x, "d": d, "w_true": w_true, "order": order}


def generate_ar_stable(
    n_samples: int,
    ar_order: int,
    noise_std: float = 0.01,
    complex_data: bool = False,
    seed: int = 0,
    clip: float = 5.0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    if complex_data:
        a = (0.05 * rng.standard_normal(ar_order) + 1j * 0.05 * rng.standard_normal(ar_order))
        u = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
        eps = noise_std * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
        x = np.zeros(n_samples, dtype=complex)
    else:
        a = 0.05 * rng.standard_normal(ar_order)
        u = rng.standard_normal(n_samples).astype(float)
        eps = noise_std * rng.standard_normal(n_samples).astype(float)
        x = np.zeros(n_samples, dtype=float)

    for k in range(n_samples):
        acc = u[k]
        for i in range(1, ar_order + 1):
            if k - i >= 0:
                acc = acc + a[i - 1] * x[k - i]
        if clip is not None:
            acc = np.clip(acc.real, -clip, clip) + (1j * np.clip(acc.imag, -clip, clip) if complex_data else 0.0)
        x[k] = acc

    d = x + eps
    return {"x": x, "d": d, "w_true": None, "order": ar_order}


def generate_square_data(
    n_samples: int,
    noise_std: float = 0.01,
    complex_data: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    if complex_data:
        x = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
        eps = noise_std * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
        d = (x ** 2) + eps
    else:
        x = rng.standard_normal(n_samples).astype(float)
        eps = noise_std * rng.standard_normal(n_samples).astype(float)
        d = (x ** 2) + eps
    return {"x": x, "d": d, "w_true": None, "order": 0}


def generate_bilinear_data(
    n_samples: int,
    noise_std: float = 0.01,
    seed: int = 0,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples).astype(float)
    eps = noise_std * rng.standard_normal(n_samples).astype(float)

    alpha, beta, gamma, eta = 0.7, 0.15, 0.05, -0.03

    d = np.zeros(n_samples, dtype=float)
    x_prev = 0.0
    d_prev = 0.0
    for k in range(n_samples):
        d[k] = alpha * x[k] + beta * d_prev + gamma * x[k] * d_prev + eta * x_prev * d_prev + eps[k]
        x_prev = x[k]
        d_prev = d[k]

    return {"x": x, "d": d, "w_true": None, "order": 4}


# ============================================================
# Task selection
# ============================================================

@dataclass(frozen=True)
class TaskSpec:
    name: str
    generator: str  # "fir" | "ar_id" | "square" | "bilinear"
    order_for_init: int
    tail: int
    mse_real: float
    mse_complex: float
    compare_w: bool
    wrel_real: float
    wrel_complex: float


def classify_task(df: DiscoveredFilter) -> TaskSpec:
    name = df.qualname.lower()
    mod = df.module.lower()

    # Blind: não supervisionado -> só schema (não checar MSE/wrel)
    if "blind" in mod:
        return TaskSpec(
            name="blind_schema_only",
            generator="fir",
            order_for_init=7,
            tail=500,
            mse_real=float("inf"),
            mse_complex=float("inf"),
            compare_w=False,
            wrel_real=1.0,
            wrel_complex=1.0,
        )

    # IIR: não comparar w_true do FIR; use tarefa mais estável
    if "iir" in mod:
        return TaskSpec(
            name="iir_ar_identity",
            generator="ar_id",
            order_for_init=4,
            tail=500,
            mse_real=5e-1,
            mse_complex=5e-1,
            compare_w=False,
            wrel_real=1.0,
            wrel_complex=1.0,
        )

    # Subband: por enquanto só schema (muitos retornam domínios decimados)
    if "subband" in mod:
        return TaskSpec(
            name="subband_schema_only",
            generator="fir",
            order_for_init=3,
            tail=500,
            mse_real=float("inf"),
            mse_complex=float("inf"),
            compare_w=False,
            wrel_real=1.0,
            wrel_complex=1.0,
        )

    # Bilinear
    if "bilinear" in name:
        return TaskSpec(
            name="bilinear",
            generator="bilinear",
            order_for_init=4,
            tail=500,
            mse_real=2e-2,
            mse_complex=2e-2,
            compare_w=False,
            wrel_real=1.0,
            wrel_complex=1.0,
        )

    # Lattice
    if "lattice" in name or "lattice" in mod:
        return TaskSpec(
            name="ar_identity",
            generator="ar_id",
            order_for_init=4,
            tail=500,
            mse_real=3e-1,
            mse_complex=3e-1,
            compare_w=False,
            wrel_real=1.0,
            wrel_complex=1.0,
        )

    # Volterra
    if "volterra" in name or "volterra" in mod:
        return TaskSpec(
            name="fir_only_mse",
            generator="fir",
            order_for_init=3,
            tail=500,
            mse_real=4e-1,
            mse_complex=4e-1,
            compare_w=False,
            wrel_real=1.0,
            wrel_complex=1.0,
        )

    # MLP/RBF => square
    if any(k in name for k in ("multilayerperceptron", "mlp", "rbf")) or "nonlinear" in mod:
        return TaskSpec(
            name="square",
            generator="square",
            order_for_init=1,
            tail=500,
            mse_real=5e-2,
            mse_complex=5e-2,
            compare_w=False,
            wrel_real=1.0,
            wrel_complex=1.0,
        )

    # Default = FIR + wrel
    is_rls_like = any(k in df.qualname for k in ("RLS", "FastRLS", "QR_RLS", "StabFastRLS"))
    if is_rls_like:
        return TaskSpec(
            name="fir",
            generator="fir",
            order_for_init=3,
            tail=500,
            mse_real=3e-3,
            mse_complex=3e-3,
            compare_w=True,
            wrel_real=0.25,
            wrel_complex=0.40,
        )

    return TaskSpec(
        name="fir",
        generator="fir",
        order_for_init=3,
        tail=500,
        mse_real=5e-3,
        mse_complex=5e-3,
        compare_w=True,
        wrel_real=0.35,
        wrel_complex=0.50,
    )


def build_dataset(task: TaskSpec, complex_data: bool) -> Dict[str, Any]:
    n_samples = 3000
    seed = 2026
    noise_std = 0.01

    if task.generator == "fir":
        return generate_fir_supervised_data(
            n_samples=n_samples,
            order=task.order_for_init,
            noise_std=noise_std,
            complex_data=complex_data,
            seed=seed,
        )
    if task.generator == "ar_id":
        return generate_ar_stable(
            n_samples=n_samples,
            ar_order=task.order_for_init,
            noise_std=noise_std,
            complex_data=complex_data,
            seed=seed,
            clip=5.0,
        )
    if task.generator == "square":
        return generate_square_data(
            n_samples=n_samples,
            noise_std=noise_std,
            complex_data=complex_data,
            seed=seed,
        )
    if task.generator == "bilinear":
        if complex_data:
            raise ValueError("bilinear task is real-only")
        return generate_bilinear_data(
            n_samples=n_samples,
            noise_std=noise_std,
            seed=seed,
        )

    raise ValueError(f"Unknown generator: {task.generator}")


# ============================================================
# Test scaffolding
# ============================================================

SKIP_INIT: set[str] = set()

# Algoritmos que ainda precisam de teste dedicado/estabilização:
XFAIL_NUMERICAL = {
    "GaussNewton",
}

# Algoritmos que retornam erro/output com comprimento diferente (domínio decimado/subamostrado)
XFAIL_SHAPE = {
    "CFDLMS",
}


def discovered_ids(d: DiscoveredFilter) -> str:
    return f"{d.qualname} ({d.module})"


# ============================================================
# REAL / COMPLEX tests
# ============================================================

@pytest.mark.parametrize("df", DISCOVERED, ids=discovered_ids)
def test_all_filters_real_autodiscovery(df: DiscoveredFilter):
    # Blind normalmente é complexo e não supervisionado
    if "blind" in df.module.lower():
        pytest.skip(f"{df.qualname}: blind algorithm not tested in REAL supervised suite")

    if df.qualname in SKIP_INIT:
        pytest.skip(f"{df.qualname}: requires extra __init__ args (listed in SKIP_INIT)")

    task = classify_task(df)

    if df.qualname in XFAIL_NUMERICAL:
        pytest.xfail(f"{df.qualname}: numerical instability on generic dataset (needs dedicated test)")
    if df.qualname in XFAIL_SHAPE:
        pytest.xfail(f"{df.qualname}: returns non-fullband length (needs API standardization)")

    data = build_dataset(task, complex_data=False)

    x = np.asarray(data["x"])
    d = np.asarray(data["d"])
    w_true = data.get("w_true", None)

    try:
        f = instantiate_filter(df.cls, order=task.order_for_init)
    except TypeError as e:
        pytest.skip(str(e))

    res = run_optimize(f, x, d)
    assert isinstance(res, dict), f"{df.qualname}: optimize must return dict"

    resN = normalize_result(res, d)
    assert_schema(df, resN, d, complex_data=False)

    e = np.asarray(resN["e"])
    # Real supervisionado: exigimos comprimento fullband
    assert e.shape == d.shape, f"{df.qualname}: error shape mismatch: {e.shape} vs {d.shape}"

    coeffs_last = resN["w_final"]
    dbg = _debug_blob(df, resN, coeffs_last)

    mse = tail_mse(e, tail=task.tail)
    if np.isfinite(task.mse_real):
        assert mse < task.mse_real, f"{df.qualname}: MSE tail too high: {mse:.3e} (task={task.name}){dbg}"

    if task.compare_w:
        assert w_true is not None, f"{df.qualname}: expected w_true for FIR task{dbg}"

        w_est = np.asarray(coeffs_last).reshape(-1)
        w_true_arr = np.asarray(w_true).reshape(-1)

        assert w_est.size == w_true_arr.size, (
            f"{df.qualname}: w size mismatch. got={w_est.size}, expected={w_true_arr.size} (task={task.name}){dbg}"
        )

        if np.iscomplexobj(w_est):
            assert np.max(np.abs(np.imag(w_est))) < 1e-6, f"{df.qualname}: complex w_est on real test{dbg}"
            w_est = np.real(w_est)

        wrel = rel_w_error(w_est, w_true_arr)
        assert wrel < task.wrel_real, f"{df.qualname}: rel-w error too high: {wrel:.3f} (task={task.name}){dbg}"


@pytest.mark.parametrize("df", DISCOVERED, ids=discovered_ids)
def test_all_filters_complex_autodiscovery(df: DiscoveredFilter):
    if not df.supports_complex:
        pytest.skip(f"{df.qualname}: supports_complex=False")

    if df.qualname in SKIP_INIT:
        pytest.skip(f"{df.qualname}: requires extra __init__ args (listed in SKIP_INIT)")

    task = classify_task(df)

    if df.qualname in XFAIL_NUMERICAL:
        pytest.xfail(f"{df.qualname}: numerical instability on generic dataset (needs dedicated test)")
    if df.qualname in XFAIL_SHAPE:
        pytest.xfail(f"{df.qualname}: returns non-fullband length (needs API standardization)")

    # bilinear é real-only
    if task.generator == "bilinear":
        pytest.skip(f"{df.qualname}: bilinear task is real-only in this suite")

    data = build_dataset(task, complex_data=True)

    x = np.asarray(data["x"])
    d = np.asarray(data["d"])
    w_true = data.get("w_true", None)

    try:
        f = instantiate_filter(df.cls, order=task.order_for_init)
    except TypeError as e:
        pytest.skip(str(e))

    res = run_optimize(f, x, d)
    assert isinstance(res, dict), f"{df.qualname}: optimize must return dict"

    resN = normalize_result(res, d)
    assert_schema(df, resN, d, complex_data=True)

    e = np.asarray(resN["e"])
    # Em complex supervisionado: também exigimos comprimento fullband,
    # exceto para blind (schema_only) — mas blind tem optimize(x) e mesmo assim retorna e do tamanho de x.
    assert e.shape == d.shape, f"{df.qualname}: error shape mismatch: {e.shape} vs {d.shape}"

    coeffs_last = resN["w_final"]
    dbg = _debug_blob(df, resN, coeffs_last)

    mse = tail_mse(e, tail=task.tail)
    if np.isfinite(task.mse_complex):
        assert mse < task.mse_complex, f"{df.qualname}: MSE tail too high: {mse:.3e} (task={task.name}){dbg}"

    if task.compare_w:
        assert w_true is not None, f"{df.qualname}: expected w_true for FIR task{dbg}"

        w_est = np.asarray(coeffs_last).reshape(-1)
        w_true_arr = np.asarray(w_true).reshape(-1)

        assert w_est.size == w_true_arr.size, (
            f"{df.qualname}: w size mismatch. got={w_est.size}, expected={w_true_arr.size} (task={task.name}){dbg}"
        )

        wrel = rel_w_error(w_est, w_true_arr)
        assert wrel < task.wrel_complex, f"{df.qualname}: rel-w error too high: {wrel:.3f} (task={task.name}){dbg}"
