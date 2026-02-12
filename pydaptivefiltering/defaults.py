from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


_THIS_DIR = Path(__file__).resolve().parent
_DEFAULTS_PATH = _THIS_DIR / "_utils" / "default_params.json"
_ALLOWED_PROFILES = {"speed", "balanced", "accuracy"}


class DefaultsError(RuntimeError):
    """Raised when default parameter profile/algorithm cannot be resolved."""


def _load_defaults_file(path: Path = _DEFAULTS_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise DefaultsError(
            f"Default params file not found at: {path}. "
            "Generate it with scripts/choose_defaults.py and copy/link to "
            "pydaptivefiltering/_utils/default_params.json."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise DefaultsError(f"Could not parse defaults file '{path}': {e}") from e

    if not isinstance(data, dict) or "profiles" not in data:
        raise DefaultsError(
            f"Invalid defaults file schema in '{path}'. Expected top-level key 'profiles'."
        )
    return data


def available_profiles(path: Path | None = None) -> tuple[str, ...]:
    """
    Return profiles available in the defaults file (sorted).
    """
    data = _load_defaults_file(path or _DEFAULTS_PATH)
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        return tuple()
    return tuple(sorted(profiles.keys()))


def available_algorithms(
    profile: str = "balanced",
    path: Path | None = None,
) -> tuple[str, ...]:
    """
    Return algorithm names available for a given profile.
    """
    data = _load_defaults_file(path or _DEFAULTS_PATH)
    profiles = data.get("profiles", {})
    p = profiles.get(profile, {})
    if not isinstance(p, dict):
        return tuple()
    return tuple(sorted(p.keys()))


def get_default_params(
    algo: str,
    profile: str = "balanced",
    *,
    path: Path | None = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Get default parameter dict for an algorithm/profile.

    Parameters
    ----------
    algo : str
        Algorithm class name, e.g. "LMS", "RLS", "FastRLS".
    profile : str
        One of {"speed","balanced","accuracy"} by convention.
    path : Path | None
        Optional custom defaults file path.
    strict : bool
        If True, raises DefaultsError when algo/profile not found.
        If False, returns {} on missing entries.

    Returns
    -------
    dict
        Parameter dictionary ready to pass into filter constructor.
    """
    if not isinstance(algo, str) or not algo.strip():
        raise DefaultsError("Argument 'algo' must be a non-empty string.")
    algo = algo.strip()

    if not isinstance(profile, str) or not profile.strip():
        raise DefaultsError("Argument 'profile' must be a non-empty string.")
    profile = profile.strip()

    data = _load_defaults_file(path or _DEFAULTS_PATH)
    profiles = data.get("profiles", {})

    if profile not in profiles:
        if strict:
            raise DefaultsError(
                f"Profile '{profile}' not found. Available: {sorted(profiles.keys())}"
            )
        return {}

    by_algo = profiles.get(profile, {})
    if not isinstance(by_algo, dict):
        if strict:
            raise DefaultsError(f"Invalid profile map for '{profile}'.")
        return {}

    params = by_algo.get(algo)
    if params is None:
        if strict:
            raise DefaultsError(
                f"No defaults for algo='{algo}' in profile='{profile}'. "
                f"Available algos: {sorted(by_algo.keys())[:20]}"
                + (" ..." if len(by_algo) > 20 else "")
            )
        return {}

    if not isinstance(params, dict):
        if strict:
            raise DefaultsError(
                f"Invalid params payload for algo='{algo}', profile='{profile}'."
            )
        return {}

    return dict(params)


def resolve_params(
    algo: str,
    *,
    user_params: Mapping[str, Any] | None = None,
    profile: str = "balanced",
    path: Path | None = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Merge default params with user params (user params override defaults).

    Useful helper for scripts/examples.

    Returns
    -------
    dict
        merged params.
    """
    base = get_default_params(algo, profile=profile, path=path, strict=strict)
    merged = dict(base)
    if user_params:
        merged.update(dict(user_params))
    return merged


__all__ = [
    "DefaultsError",
    "available_profiles",
    "available_algorithms",
    "get_default_params",
    "resolve_params",
]
