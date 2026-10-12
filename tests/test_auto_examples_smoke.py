from __future__ import annotations
import os
import sys
from pathlib import Path
import importlib.util
import numpy as np

REPO = Path(__file__).resolve().parents[1]
GEN_SCRIPT = REPO / "scripts" / "gen_examples.py"

def load_module(path: Path):
    name = f"ex_{path.stem}_{abs(hash(str(path)))%10**8}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod  # needed by dataclasses defined in the module
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def test_auto_examples_smoke(tmp_path):
    # auto_examples/ is git-ignored, so regenerate it from the current code.
    EXAMPLES = tmp_path / "auto_examples"
    load_module(GEN_SCRIPT).main(out_root=str(EXAMPLES), seed=0)

    os.environ["PYDAF_ENSEMBLE"] = "3"
    os.environ["PYDAF_K"] = "200"
    os.environ["MPLBACKEND"] = "Agg"

    files = sorted(p for p in EXAMPLES.rglob("*.py") if p.is_file() and not p.name.startswith("_"))
    assert files, "No auto_examples found"

    for p in files:
        rel = p.relative_to(EXAMPLES).as_posix().lower()

        # SKIP kalman
        if rel.startswith("kalman/") or "kalman" in p.name.lower():
            continue

        mod = load_module(p)
        if not hasattr(mod, "main"):
            continue

        ret = mod.main(seed=0, plot=False)

        assert ret is not None
        if isinstance(ret, dict) and "MSE_av" in ret:
            mse = np.asarray(ret["MSE_av"]).ravel()
            assert mse.size > 0
            assert np.isfinite(mse).all()
