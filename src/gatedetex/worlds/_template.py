"""
Template for a gatedetex world module.

Copy this file (rename it, drop the leading underscore) into a directory
of your own and point gatedetex at it, either with::

    import gatedetex
    gatedetex.worlds.add_world_dir("/path/to/my_worlds")

or by setting the GATEDETEX_WORLDS_PATH environment variable to that
directory. gatedetex then finds it by filename, e.g. a file named
``my_detector.py`` is loaded as world ``"my_detector"``.

────────────────────────────────────────────────────────────────────────
Required contract
────────────────────────────────────────────────────────────────────────
A world module MUST define:

    build_world(sim, units) -> None
        Add volumes to `sim` (an opengate.Simulation). `units` is
        gate.g4_units, use it for every dimensioned quantity, e.g.
        box.size = [10 * units.mm, ...].

Everything else below is optional; sensible defaults are used if omitted.

Module-level constants (all optional):

    CAPABILITIES : dict
        Which actors the simulator should wire up by default, e.g.
        {"optical": True, "dose": True, "sipm_hits": True,
         "optical_exits": False}. Any key omitted falls back to
         gatedetex's global default. Users can still override any of
         these from the CLI/API at run time.

    BEAM_CONFIG : dict
        {"direction": [x, y, z], "target_cm": [x, y, z], "offset_cm": f}
        Describes where the beam source sits and which way it points.

    PHANTOM_CM : [x, y, z]
        Size of the dosimetry phantom in cm, used to size the DoseActor
        and (if provided) for plotting/reporting.

    TARGET_VOLUME_NAME : str
        Name of the volume the DoseActor / hit actors attach to.
        Defaults to "target".

    DETECTOR_VOLUME_NAMES : list[str]
        Names of volumes that should get a per-detector hit actor when
        the "sipm_hits" capability is enabled.

Optional hooks:

    add_optical_surfaces(sim, units) -> None
        Called after build_world() if optical physics is enabled.

    configure_dose_actor(dose_actor, units) -> None
        Called after the DoseActor is created, for world-specific tweaks.

    analyze(batch_dir, run_dirs, meta, utils) -> dict
        Called by `gatedetex-analyze` / gatedetex.analysis.analyze to do
        world-specific post-processing. `utils` is
        gatedetex.analysis.utils. Should return a dict that may include
        any of: "hits", "exits", "dose_centers", "dose_edep",
        "timing_res_ps", "extra_lines" (list[str] appended to the report).
        If omitted, a generic hits/exits/dose analysis is used instead.

    get_geometry_primitives() -> list[dict]
        Called by `gatedetex-plot3d` to draw a wireframe of the geometry.
        Each dict: {"type": "box"|"tube", "center": [x,y,z] cm,
        "color": "#rrggbb", "label": str, plus "half": [hx,hy,hz] for
        boxes or "rmax"/"height"/"rmin" for tubes}.
"""

CAPABILITIES = {
    "optical": False,
    "dose": True,
    "sipm_hits": False,
    "optical_exits": False,
}

BEAM_CONFIG = {
    "direction": [0, 0, 1],
    "target_cm": [0, 0, 0],
    "offset_cm": 2.0,
}

PHANTOM_CM = [5.0, 5.0, 5.0]
TARGET_VOLUME_NAME = "target"
DETECTOR_VOLUME_NAMES: list[str] = []


def build_world(sim, units):
    """Build a trivial 5x5x5 cm water cube as a starting point."""
    target = sim.add_volume("Box", TARGET_VOLUME_NAME)
    target.material = "G4_WATER"
    target.size = [d * units.cm for d in PHANTOM_CM]
    target.translation = [0, 0, PHANTOM_CM[2] / 2 * units.cm]
    target.color = [0.2, 0.5, 1.0, 0.3]
