"""
World-module registry.

A "world" is a small Python module that declares a detector/phantom geometry
and how the simulator should wire actors around it (the contract is
documented in ``gatedetex.worlds._template``).

Worlds can come from three places, searched in this order:

1. Any directory added at runtime via :func:`add_world_dir` (or passed
   explicitly to a loading function) — lets users keep private world
   modules outside the package.
2. The ``GATEDETEX_WORLDS_PATH`` environment variable — a ``:``-separated
   (``;``-separated on Windows) list of directories, same idea as ``PATH``.
3. The built-in worlds shipped with the package (``gatedetex/worlds/``):
   ``quartz_cal``, ``radi_cal_energy``, ``scintx_sipm_array``.

This replaces the old approach of mutating ``sys.path`` inside
``simulator.py`` / ``analyze.py`` / ``plot_3d.py``, which meant those three
scripts each had to agree on where world modules lived (they didn't --
one looked in ``world_modules/``, the others looked in ``worlds/``).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

_BUILTIN_DIR = Path(__file__).resolve().parent
_EXTRA_DIRS: list[Path] = []

_ENV_VAR = "GATEDETEX_WORLDS_PATH"

# Files in this package that are not worlds themselves.
_NON_WORLD_MODULES = {"__init__", "registry", "_template"}


def add_world_dir(path: str | Path) -> None:
    """Register an additional directory to search for world modules.

    Call this before loading a world by name, e.g.::

        import gatedetex
        gatedetex.worlds.add_world_dir("~/my_worlds")
        gatedetex.run_simulation(world="my_custom_world", ...)
    """
    p = Path(path).expanduser().resolve()
    if p not in _EXTRA_DIRS:
        _EXTRA_DIRS.insert(0, p)  # most-recently-added wins


def _search_dirs() -> list[Path]:
    dirs = list(_EXTRA_DIRS)
    env_val = os.environ.get(_ENV_VAR, "")
    if env_val:
        sep = ";" if os.name == "nt" else ":"
        dirs += [Path(d).expanduser().resolve() for d in env_val.split(sep) if d]
    dirs.append(_BUILTIN_DIR)
    return dirs


def list_worlds() -> list[str]:
    """Return the names of every world module currently discoverable."""
    names: list[str] = []
    seen: set[str] = set()
    for d in _search_dirs():
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.py")):
            name = f.stem
            if name in _NON_WORLD_MODULES or name.startswith("_"):
                continue
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def load_world(name: str) -> ModuleType:
    """Import and return the world module registered under ``name``.

    Raises FileNotFoundError with a helpful message (including the list of
    worlds it *did* find) if the name can't be resolved anywhere.
    """
    for d in _search_dirs():
        candidate = d / f"{name}.py"
        if not candidate.is_file():
            continue
        module_name = f"gatedetex_world_{name}"
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    available = ", ".join(list_worlds()) or "(none found)"
    raise FileNotFoundError(
        f"World module '{name}' not found.\n"
        f"Searched: {[str(d) for d in _search_dirs()]}\n"
        f"Available worlds: {available}\n"
        f"Add a custom directory with gatedetex.worlds.add_world_dir(...) "
        f"or the {_ENV_VAR} environment variable."
    )
