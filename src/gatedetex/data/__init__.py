"""Bundled Geant4/GATE material and surface-property data files.

The original script looked for these files next to simulator.py
(``script_dir / "Materials.xml"`` etc.) but they actually lived in a
``materials/`` subdirectory, so ``physics_manager.surface_properties_file``
etc. were silently never set. Here they're proper package data, located
robustly regardless of the current working directory or install location.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path


def data_dir() -> Path:
    """Return the directory containing the bundled GATE data files."""
    return Path(importlib.resources.files("gatedetex.data"))


def materials_xml() -> Path:
    return data_dir() / "Materials.xml"


def surface_properties_xml() -> Path:
    return data_dir() / "SurfaceProperties.xml"


def gate_materials_db() -> Path:
    return data_dir() / "GateMaterials.db"
