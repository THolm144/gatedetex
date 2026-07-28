"""
worlds/scintx_sipm_array.py
============================
10 cm × 10 cm × 6 mm ScintX scintillator slab.
Four 3 mm × 3 mm SiPMs on the +Y thin edge.
Electron beam enters the +Z face, traveling in the -Z direction.

New-contract exports
--------------------
CAPABILITIES, BEAM_CONFIG, PHANTOM_CM, TARGET_VOLUME_NAME,
DETECTOR_VOLUME_NAMES, analyze(), get_geometry_primitives()
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SLAB_X_CM  = 10.0
SLAB_Y_CM  = 10.0
SLAB_Z_CM  = 0.6      # 6 mm thickness

MATERIAL   = "ScintX"

# SiPM geometry
SIPM_X_CM  = 0.3      # 3 mm — spans X
SIPM_Y_CM  = 0.01     # 0.1 mm — thin face points into +Y edge
SIPM_Z_CM  = 0.3      # 3 mm — spans Z (covers half slab thickness)

N_SIPMS         = 4
ARRAY_SPAN_CM   = 2.0  # center-to-center span of array along X
COUPLING_GAP_CM = 0.0  # 0 = direct contact
WORLD_MARGIN_CM = 5.0
SIPM_MATERIAL   = "G4_Si"

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR CONTRACT
# ─────────────────────────────────────────────────────────────────────────────

CAPABILITIES = {
    "optical":          True,
    "dose":             True,
    "sipm_hits":        True,
    "optical_exits":    False,
    "calorimeter_mode": False,
}

# Beam enters the +Z face traveling in -Z direction
BEAM_CONFIG = {
    "direction": [0, 0, -1],                 # traveling in -Z
    "target_cm": [0, 0, SLAB_Z_CM / 2],     # aim at center of +Z face
    "offset_cm": 2.0,                        # source 2 cm outside the face
}

ACTIVATE_CALORIMETER_SETTINGS = False
CALORIMETER_Z_RES_MM = 0.1

ACTIVE_Z_RANGES_MM = [
    [-SLAB_Z_CM * 5.0, SLAB_Z_CM * 5.0]   # [-3.0, +3.0] mm
]

TARGET_VOLUME_NAME = "scintillator"

PHANTOM_CM = [SLAB_X_CM, SLAB_Y_CM, SLAB_Z_CM]

EXPECTED_DEDX = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

if N_SIPMS == 1:
    _SIPM_CENTERS_X = [0.0]
else:
    _SIPM_CENTERS_X = [
        -ARRAY_SPAN_CM / 2 + i * ARRAY_SPAN_CM / (N_SIPMS - 1)
        for i in range(N_SIPMS)
    ]

# [x, y, z] in cm — flush against +Y edge, spread in X, centered in Z
SIPM_POSITIONS_CM = [
    [
        x,
        SLAB_Y_CM / 2 + COUPLING_GAP_CM + SIPM_Y_CM / 2,
        0.0,
    ]
    for x in _SIPM_CENTERS_X
]

DETECTOR_VOLUME_NAMES = [f"sipm_{i}" for i in range(N_SIPMS)]


# ─────────────────────────────────────────────────────────────────────────────
# WORLD CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_world(sim, units):
    sim.world.size = [
        (SLAB_X_CM + 2 * WORLD_MARGIN_CM) * units.cm,
        (SLAB_Y_CM + 2 * WORLD_MARGIN_CM) * units.cm,
        (SLAB_Z_CM + 2 * WORLD_MARGIN_CM + 5.0) * units.cm,
    ]
    sim.world.material = "G4_AIR"

    # Scintillator slab — centered at origin
    # +Z face at z = +SLAB_Z_CM/2  ← beam entry
    # +Y face at y = +SLAB_Y_CM/2  ← SiPM edge
    scint = sim.add_volume("Box", TARGET_VOLUME_NAME)
    scint.size = [
        SLAB_X_CM * units.cm,
        SLAB_Y_CM * units.cm,
        SLAB_Z_CM * units.cm,
    ]
    scint.material    = MATERIAL
    scint.translation = [0, 0, 0]

    # SiPMs — 3mm(X) × 0.1mm(Y) × 3mm(Z), flush on +Y edge, spread in X
    for i, pos in enumerate(SIPM_POSITIONS_CM):
        sipm = sim.add_volume("Box", f"sipm_{i}")
        sipm.size = [
            SIPM_X_CM * units.cm,   # 3 mm in X
            SIPM_Y_CM * units.cm,   # 0.1 mm thin face into slab edge
            SIPM_Z_CM * units.cm,   # 3 mm in Z
        ]
        sipm.material    = SIPM_MATERIAL
        sipm.translation = [
            pos[0] * units.cm,
            pos[1] * units.cm,
            pos[2] * units.cm,
        ]

    return sim


def add_optical_surfaces(sim, units):
    sim.physics_manager.add_optical_surface(
        volume_from="scintillator",
        volume_to="world",
        g4_surface_name="Tyvek",
    )

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HOOK
# ─────────────────────────────────────────────────────────────────────────────

def analyze(batch_dir, run_dirs, meta, utils):
    """
    ScintX-specific analysis:
    - Optical hits per SiPM channel (Cerenkov vs Scintillation)
    - Aggregate exits from the slab surface
    - Dose profile across the 6 mm slab
    - Timing resolution per SiPM
    """
    hits_files  = [p for d in run_dirs for p in sorted(d.glob("detector_hits*.root"))]
    exits_files = [d / "optical_exited.root" for d in run_dirs]

    hits        = utils.analyse_hits(hits_files)
    exits       = utils.analyse_exits(exits_files)
    centers, edep = utils.load_dose_mhd(run_dirs, meta["phantom_cm"],
                                        filename_glob="edep*.mhd")
    timing_res  = utils.extract_timing_resolution(hits_files) if hits_files else 0.0

    # Per-channel hit breakdown
    extra_lines = _per_channel_summary(hits_files, utils)

    return {
        "hits":         hits,
        "exits":        exits,
        "dose_centers": centers,
        "dose_edep":    edep,
        "timing_res_ps": timing_res,
        "extra_lines":  extra_lines,
    }


def _per_channel_summary(hits_files, utils) -> list[str]:
    """Break down photon hits per SiPM index."""
    if not hits_files:
        return []

    import uproot

    lines = ["  SiPM channel breakdown:"]
    for path in sorted(hits_files):
        if not path.exists():
            continue
        idx = path.stem.split("_")[-1]
        try:
            with uproot.open(path) as f:
                key = utils.first_tree_key(f, "detector", "phasespace")
                if not key:
                    continue
                tree = f[key]
                n    = len(tree["TrackID"].array(library="np"))
                lines.append(f"    sipm_{idx}: {n:,} photon entries")
        except Exception as e:
            lines.append(f"    sipm_{idx}: ERROR — {e}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# PLOT_3D HOOK
# ─────────────────────────────────────────────────────────────────────────────

def get_geometry_primitives() -> list[dict]:
    prims = []

    # Scintillator slab
    prims.append({
        "type":      "box",
        "center":    [0.0, 0.0, 0.0],
        "half":      [SLAB_X_CM / 2, SLAB_Y_CM / 2, SLAB_Z_CM / 2],
        "color":     "#00ffcc",
        "label":     "ScintX slab",
        "alpha":     0.35,
        "linewidth": 1.0,
    })

    # SiPMs
    for i, pos in enumerate(SIPM_POSITIONS_CM):
        prims.append({
            "type":      "box",
            "center":    pos,
            "half":      [SIPM_X_CM / 2, SIPM_Y_CM / 2, SIPM_Z_CM / 2],
            "color":     "#f1c40f",
            "label":     "SiPM" if i == 0 else "",
            "alpha":     0.9,
            "linewidth": 1.5,
        })

    return prims