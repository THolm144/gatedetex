"""
worlds/iron_quartz_array.py
============================
Iron absorber block + NxN quartz crystal / SiPM scanner array.

New-contract exports
--------------------
CAPABILITIES, BEAM_CONFIG, PHANTOM_CM, TARGET_VOLUME_NAME,
DETECTOR_VOLUME_NAMES, analyze(), get_geometry_primitives()
"""

import math
from pathlib import Path
import opengate as gate
# Import the modern geometry repetition utility
from opengate.geometry.utility import get_grid_repetition

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_N    = 10     # NxN grid size
DEFAULT_X_CM = 5.0   # Iron absorber depth in cm

# Pixel geometry (fixed)
_PITCH_MM    = 5.0
_QUARTZ_Z_MM = 5.0
_SIPM_XY_MM  = 3.0
_SIPM_Z_MM   = 0.1

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED CONSTANTS (at default N / X_CM)
# ─────────────────────────────────────────────────────────────────────────────

def _derive(n=DEFAULT_N, x_cm=DEFAULT_X_CM):
    array_mm        = n * _PITCH_MM
    iron_mm         = x_cm * 10.0
    array_z_mm      = _QUARTZ_Z_MM + _SIPM_Z_MM
    total_z_mm      = iron_mm + array_z_mm
    return {
        "array_mm":   array_mm,
        "iron_mm":    iron_mm,
        "array_z_mm": array_z_mm,
        "total_z_mm": total_z_mm,
    }


_D = _derive()

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR CONTRACT
# ─────────────────────────────────────────────────────────────────────────────

CAPABILITIES = {
    "optical":          True,
    "dose":             True,
    "sipm_hits":        True,
    "optical_exits":    True,
    "calorimeter_mode": False,
}

BEAM_CONFIG = {
    "direction": [0, 0, 1],    # beam travels +Z through iron then quartz
    "target_cm": [0, 0, 0],
    "offset_cm": 2.0,
}

TARGET_VOLUME_NAME = "Dosimetry_Target"

PHANTOM_CM = [
    _D["array_mm"] / 10.0,
    _D["array_mm"] / 10.0,
    _D["total_z_mm"] / 10.0,
]

EXPECTED_DEDX = 1.0

ACTIVATE_CALORIMETER_SETTINGS = False
CALORIMETER_Z_RES_MM = 0.1

ACTIVE_Z_RANGES_MM = [
    [0.0, _D["total_z_mm"]]
]

DETECTOR_VOLUME_NAMES = ["SiPM_pixel"]


# ─────────────────────────────────────────────────────────────────────────────
# WORLD CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_world(sim, units, n=DEFAULT_N, x_cm=DEFAULT_X_CM,
                materials_db="Materials.xml"):
    """
    Build the unified Dosimetry_Target containing the iron absorber and
    the NxN quartz/SiPM scanner array.
    """
    # Update PHANTOM_CM and _D for any runtime n/x_cm values
    global PHANTOM_CM, ACTIVE_Z_RANGES_MM, _D
    d = _derive(n, x_cm)
    _D = d  # Overwrite global _D with the runtime-derived values
    
    PHANTOM_CM = [d["array_mm"] / 10.0,
                  d["array_mm"] / 10.0,
                  d["total_z_mm"] / 10.0]
    ACTIVE_Z_RANGES_MM = [[0.0, d["total_z_mm"]]]

    # ── Mother volume ──────────────────────────────────────────────────────
    target_z_center = d["total_z_mm"] / 2.0

    target = sim.add_volume("Box", TARGET_VOLUME_NAME)
    target.material    = "G4_AIR"
    target.size        = [d["array_mm"] * units.mm,
                          d["array_mm"] * units.mm,
                          d["total_z_mm"] * units.mm]
    target.translation = [0, 0, target_z_center * units.mm]
    target.color       = [1, 1, 1, 0.0]

    # ── Iron absorber ──────────────────────────────────────────────────────
    iron_local_z = -(d["array_z_mm"] / 2.0)

    iron = sim.add_volume("Box", "Iron_Shield")
    iron.mother      = TARGET_VOLUME_NAME
    iron.material    = "G4_Fe"
    iron.size        = [d["array_mm"] * units.mm,
                        d["array_mm"] * units.mm,
                        d["iron_mm"]  * units.mm]
    iron.translation = [0, 0, iron_local_z * units.mm]
    iron.color       = [0.5, 0.5, 0.5, 0.8]

    # ── Scanner mother ─────────────────────────────────────────────────────
    scanner_local_z = d["iron_mm"] / 2.0

    scanner = sim.add_volume("Box", "Scanner_Mother")
    scanner.mother      = TARGET_VOLUME_NAME
    scanner.material    = "G4_AIR"
    scanner.size        = [d["array_mm"]   * units.mm,
                           d["array_mm"]   * units.mm,
                           d["array_z_mm"] * units.mm]
    scanner.translation = [0, 0, scanner_local_z * units.mm]
    scanner.color       = [1, 1, 1, 0.0]

    # ── Single pixel container ─────────────────────────────────────────────
    pixel = sim.add_volume("Box", "Pixel_Container")
    pixel.mother   = "Scanner_Mother"
    pixel.material = "G4_AIR"
    pixel.size     = [_PITCH_MM      * units.mm,
                      _PITCH_MM      * units.mm,
                      d["array_z_mm"] * units.mm]

    # MODERN REPEATER LOGIC:
    pixel.translation = get_grid_repetition(
        [n, n, 1],
        [_PITCH_MM * units.mm, _PITCH_MM * units.mm, 0 * units.mm]
    )

    # ── Quartz crystal ─────────────────────────────────────────────────────
    quartz = sim.add_volume("Box", "Quartz_crystal")
    quartz.mother      = "Pixel_Container"
    quartz.material    = "G4_SILICON_DIOXIDE"
    quartz.size        = [_PITCH_MM    * units.mm,
                          _PITCH_MM    * units.mm,
                          _QUARTZ_Z_MM * units.mm]
    quartz_local_z = -d["array_z_mm"] / 2.0 + _QUARTZ_Z_MM / 2.0
    quartz.translation = [0, 0, quartz_local_z * units.mm]
    quartz.color       = [0.0, 0.0, 1.0, 0.5]

    # ── SiPM ──────────────────────────────────────────────────────────────
    sipm = sim.add_volume("Box", DETECTOR_VOLUME_NAMES[0])
    sipm.mother      = "Pixel_Container"
    sipm.material    = "G4_Si"
    sipm.size        = [_SIPM_XY_MM * units.mm,
                        _SIPM_XY_MM * units.mm,
                        _SIPM_Z_MM  * units.mm]
    sipm_local_z = quartz_local_z + _QUARTZ_Z_MM / 2.0 + _SIPM_Z_MM / 2.0
    sipm.translation = [0, 0, sipm_local_z * units.mm]
    sipm.color       = [1.0, 0.0, 0.0, 1.0]

    return sim


def add_optical_surfaces(sim, units):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HOOK
# ─────────────────────────────────────────────────────────────────────────────

def analyze(batch_dir, run_dirs, meta, utils):
    """
    Iron-quartz analysis:
    - SiPM hit counts (Cherenkov vs Scintillation)
    - Depth-dose profile across full target (iron + quartz)
    - Timing resolution
    """
    hits_files  = [p for d in run_dirs for p in sorted(d.glob("detector_hits*.root"))]
    exits_files = [d / "optical_exited.root" for d in run_dirs]

    hits       = utils.analyse_hits(hits_files)
    exits      = utils.analyse_exits(exits_files)
    timing_res = utils.extract_timing_resolution(hits_files) if hits_files else 0.0
    centers, edep = utils.load_dose_mhd(run_dirs, meta["phantom_cm"])

    extra_lines = []
    total_hits  = sum(hits.values())
    total_prim  = meta["total_primaries"]

    cher_per_prim = hits.get("Cerenkov", 0) / total_prim if total_prim else 0.0
    extra_lines += [
        f"  Cherenkov hits / primary : {cher_per_prim:.4f}",
        f"  Total SiPM hits          : {total_hits:,}",
    ]

    if centers is not None and edep is not None:
        iron_z_cm   = meta.get("phantom_cm", PHANTOM_CM)[2] - \
                      (_D["array_z_mm"] / 10.0)
        iron_mask   = centers < iron_z_cm
        quartz_mask = ~iron_mask
        iron_edep   = float(edep[iron_mask].sum())
        quartz_edep = float(edep[quartz_mask].sum())
        extra_lines += [
            f"  Dose in iron absorber    : {iron_edep:.4f} MeV",
            f"  Dose in quartz array     : {quartz_edep:.4f} MeV",
        ]

    return {
        "hits":          hits,
        "exits":         exits,
        "dose_centers":  centers,
        "dose_edep":     edep,
        "timing_res_ps": timing_res,
        "extra_lines":   extra_lines,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOT_3D HOOK
# ─────────────────────────────────────────────────────────────────────────────

def get_geometry_primitives() -> list[dict]:
    d = _derive()

    total_z_cm    = d["total_z_mm"]  / 10.0
    iron_z_cm     = d["iron_mm"]     / 10.0
    array_z_cm    = d["array_z_mm"]  / 10.0
    array_half_xy = d["array_mm"]    / 20.0
    pitch_cm      = _PITCH_MM        / 10.0

    prims = []

    iron_center_z = total_z_cm / 2.0 - iron_z_cm / 2.0 - array_z_cm / 2.0
    prims.append({
        "type":      "box",
        "center":    [0.0, 0.0, iron_center_z],
        "half":      [array_half_xy, array_half_xy, iron_z_cm / 2.0],
        "color":     "#7f8c8d",
        "label":     "Iron absorber",
        "alpha":     0.5,
        "linewidth": 1.0,
    })

    scanner_center_z = total_z_cm / 2.0 + iron_z_cm / 2.0 - array_z_cm / 2.0
    prims.append({
        "type":      "box",
        "center":    [0.0, 0.0, scanner_center_z],
        "half":      [array_half_xy, array_half_xy, array_z_cm / 2.0],
        "color":     "#00ffcc",
        "label":     "Quartz array",
        "alpha":     0.2,
        "linewidth": 0.6,
    })

    prims.append({
        "type":      "box",
        "center":    [0.0, 0.0, scanner_center_z],
        "half":      [pitch_cm / 2, pitch_cm / 2, _QUARTZ_Z_MM / 20.0],
        "color":     "#00cfff",
        "label":     "Quartz pixel",
        "alpha":     0.8,
        "linewidth": 1.2,
    })

    sipm_center_z = scanner_center_z + _QUARTZ_Z_MM / 20.0 + _SIPM_Z_MM / 20.0
    prims.append({
        "type":      "box",
        "center":    [0.0, 0.0, sipm_center_z],
        "half":      [_SIPM_XY_MM / 20.0, _SIPM_XY_MM / 20.0, _SIPM_Z_MM / 20.0],
        "color":     "#f1c40f",
        "label":     "SiPM",
        "alpha":     0.9,
        "linewidth": 1.5,
    })

    return prims