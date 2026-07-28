"""
Core simulation engine for gatedetex.

This is a refactor of the original standalone ``simulator.py`` script into
an importable API (:func:`run_simulation`) plus a thin CLI wrapper in
``gatedetex.cli``. Behavioural changes from the original script:

  * World modules are resolved via ``gatedetex.worlds`` (built-ins +
    user-added directories) instead of a hardcoded ``world_modules/``
    folder next to the script.
  * The optical/surface-properties XML and the material database are
    bundled package data (``gatedetex.data``), fixing a bug in the
    original where the simulator looked for them next to
    ``simulator.py`` but they actually lived in ``materials/``.
  * Everything is available as plain Python: ``run_simulation(**kwargs)``
    returns a dict of output paths instead of only being runnable as
    ``python simulator.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import worlds as _worlds
from .data import materials_xml, surface_properties_xml, gate_materials_db

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CAPABILITIES = {
    "optical": False,
    "dose": True,
    "sipm_hits": False,
    "optical_exits": False,
    "track_optical": False,
}

DEFAULT_BEAM_CONFIG = {
    "direction": [0, 0, 1],  # +Z  (beam travels in +Z by default)
    "target_cm": [0, 0, 0],
    "offset_cm": 2.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """All the knobs the original CLI exposed, now usable directly from Python."""

    world: str = "quartz_cal"
    particle: str = "e-"
    energy_kev: float = 9000.0
    n: int = 10_000
    threads: int = 4
    beam_radius: float = 1.0
    output_dir: str | None = None
    physics_list: str = "G4EmStandardPhysics_option4"
    run_id: int = 0

    # capability overrides: "on" | "off" | "world" (defer to the world module)
    optical: str = "world"
    dose: str = "world"
    sipm_hits: str = "world"
    track_optical: str = "world"
    no_cerenkov: bool = False

    run_timing_us: list[float] = field(default_factory=lambda: [0.0, 5.0])
    progress_bar: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# WORLD / CAPABILITY RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def resolve_capabilities(world, cfg: SimulationConfig) -> dict:
    caps = {**DEFAULT_CAPABILITIES, **getattr(world, "CAPABILITIES", {})}
    override_map = {
        "optical": cfg.optical,
        "dose": cfg.dose,
        "sipm_hits": cfg.sipm_hits,
        "track_optical": cfg.track_optical,
    }
    for key, val in override_map.items():
        if val == "on":
            caps[key] = True
        elif val == "off":
            caps[key] = False
    return caps


def resolve_beam_config(world) -> dict:
    return {**DEFAULT_BEAM_CONFIG, **getattr(world, "BEAM_CONFIG", {})}


def resolve_output_dirs(cfg: SimulationConfig) -> tuple[Path, Path]:
    if cfg.output_dir:
        batch_dir = Path(cfg.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = Path.cwd() / "outputs" / cfg.world / f"{ts}_{int(cfg.energy_kev)}keV"
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir, batch_dir


# ─────────────────────────────────────────────────────────────────────────────
# ACTOR WIRING
# ─────────────────────────────────────────────────────────────────────────────

def wire_actors(sim, world, caps: dict, run_id: int, units) -> dict:
    registry: dict[str, Any] = {
        "optical_exited_actor": None,
        "optical_tracker_actor": None,
        "hit_actors": [],
        "dose_actor": None,
    }

    target_vol = getattr(world, "TARGET_VOLUME_NAME", "target")
    detector_volumes = getattr(world, "DETECTOR_VOLUME_NAMES", [])

    if caps.get("track_optical", False) and caps.get("optical", False):
        tracker = sim.add_actor("PhaseSpaceActor", f"optical_tracker_{run_id}")
        tracker.attached_to = "world"
        tracker.output_filename = f"optical_tracks_{run_id}.root"
        tracker.steps_to_store = "all"
        tracker.attributes = [
            "ParticleName", "KineticEnergy", "TrackCreatorProcess",
            "Position", "TrackID", "EventID", "GlobalTime",
        ]
        registry["optical_tracker_actor"] = tracker

    if caps.get("optical", False) and caps.get("optical_exits", False):
        exited = sim.add_actor("PhaseSpaceActor", f"optical_exited_{run_id}")
        exited.attached_to = target_vol
        exited.output_filename = f"optical_exited_{run_id}.root"
        exited.steps_to_store = "exiting"
        exited.attributes = [
            "ParticleName", "KineticEnergy", "TrackCreatorProcess",
            "Position", "TrackID", "EventID", "GlobalTime",
        ]
        registry["optical_exited_actor"] = exited

    if caps.get("sipm_hits", False) and detector_volumes:
        for idx, vol_name in enumerate(detector_volumes):
            if vol_name not in sim.volume_manager.volumes:
                continue
            hits = sim.add_actor("PhaseSpaceActor", f"detector_hits_{idx}_{run_id}")
            hits.attached_to = vol_name
            hits.authorize_repeated_volumes = True
            hits.output_filename = f"detector_hits_{idx}_{run_id}.root"
            hits.steps_to_store = "entering"
            hits.attributes = [
                "ParticleName", "KineticEnergy", "Position",
                "TrackCreatorProcess", "TrackID", "EventID", "GlobalTime",
            ]
            registry["hit_actors"].append(hits)

    if caps.get("dose", True):
        phantom_cm = world.PHANTOM_CM
        dose = _wire_standard_dose(sim, target_vol, phantom_cm, run_id, units)
        if hasattr(world, "configure_dose_actor"):
            world.configure_dose_actor(dose, units)
        registry["dose_actor"] = dose

    return registry


def _wire_standard_dose(sim, target_vol: str, phantom_cm: list, run_id: int, units):
    dose = sim.add_actor("DoseActor", f"dose_actor_{run_id}")
    dose.attached_to = target_vol
    dose.output_filename = f"edep_{run_id}.mhd"
    dose.size = [
        int(round(phantom_cm[0] * 10)),
        int(round(phantom_cm[1] * 10)),
        int(round(phantom_cm[2] * 10)),
    ]
    dose.spacing = [1.0 * units.mm] * 3
    dose.hit_type = "random"
    dose.edep.active = True
    dose.dose.active = False
    dose.edep_uncertainty.active = False
    return dose


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE
# ─────────────────────────────────────────────────────────────────────────────

def add_beam_source(sim, cfg: SimulationConfig, beam_cfg: dict, units):
    direction = np.array(beam_cfg["direction"], dtype=float)
    direction /= np.linalg.norm(direction)

    target_cm = np.array(beam_cfg["target_cm"], dtype=float)
    offset_cm = beam_cfg["offset_cm"]
    source_pos_cm = target_cm - direction * offset_cm

    source = sim.add_source("GenericSource", f"{cfg.particle}_beam")
    source.particle = cfg.particle
    source.energy.mono = cfg.energy_kev * units.keV
    source.position.type = "disc"
    source.position.radius = cfg.beam_radius * units.cm
    source.position.translation = [
        source_pos_cm[0] * units.cm,
        source_pos_cm[1] * units.cm,
        source_pos_cm[2] * units.cm,
    ]
    source.direction.type = "momentum"
    source.direction.momentum = direction.tolist()
    source.n = cfg.n  # OpenGATE distributes this target natively across threads
    return source


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS
# ─────────────────────────────────────────────────────────────────────────────

def configure_physics(sim, cfg: SimulationConfig, caps: dict):
    sim.physics_manager.physics_list_name = cfg.physics_list

    surface_file = surface_properties_xml()
    if surface_file.exists():
        sim.physics_manager.surface_properties_file = str(surface_file)

    if caps["optical"]:
        optical_file = materials_xml()
        sim.physics_manager.special_physics_constructors.G4OpticalPhysics = True
        if optical_file.exists():
            sim.physics_manager.optical_properties_file = str(optical_file)

        if cfg.no_cerenkov:
            sim.g4_commands_before_init.append("/process/optical/processActivation Cerenkov false")
        else:
            sim.g4_commands_before_init.append("/process/optical/processActivation Cerenkov true")


# ─────────────────────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────────────────────

def save_metadata(cfg: SimulationConfig, batch_dir: Path, run_dir: Path, world,
                   caps: dict, beam_cfg: dict, actor_registry: dict) -> Path:
    dose = actor_registry["dose_actor"]
    metadata = {
        "world": cfg.world,
        "particle": cfg.particle,
        "energy_kev": cfg.energy_kev,
        "n_primaries": cfg.n,
        "threads": cfg.threads,
        "beam_radius_cm": cfg.beam_radius,
        "physics_list": cfg.physics_list,
        "batch_dir": str(batch_dir),
        "output_dir": str(run_dir),
        "material": getattr(world, "MATERIAL", "unknown"),
        "phantom_cm": getattr(world, "PHANTOM_CM", None),
        "target_volume": getattr(world, "TARGET_VOLUME_NAME", "target"),
        "capabilities": caps,
        "beam_config": beam_cfg,
        "dose_size_vox": dose.size if dose else None,
    }
    path = run_dir / f"sim_metadata_{cfg.run_id}.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(config: SimulationConfig | None = None, **overrides) -> dict:
    """Run one gatedetex simulation.

    Usage::

        import gatedetex
        result = gatedetex.run_simulation(
            world="quartz_cal", particle="e-", energy_kev=9000, n=10_000,
        )
        print(result["metadata_path"])

    Any :class:`SimulationConfig` field can be passed as a keyword argument
    instead of constructing the config object yourself.

    Returns a dict with keys: "batch_dir", "run_dir", "metadata_path",
    "capabilities", "beam_config", "world_module".
    """
    import opengate as gate  # imported lazily: not needed just to list/inspect worlds

    if config is None:
        config = SimulationConfig(**overrides)
    elif overrides:
        config = SimulationConfig(**{**asdict(config), **overrides})

    batch_dir, run_dir = resolve_output_dirs(config)
    world = _worlds.load_world(config.world)
    caps = resolve_capabilities(world, config)
    beam_cfg = resolve_beam_config(world)

    sim = gate.Simulation()
    sim.random_seed = 1000 + config.run_id
    sim.output_dir = str(run_dir)

    stats = sim.add_actor("SimulationStatisticsActor", f"sim_stats_{config.run_id}")
    stats.output_filename = f"stats_{config.run_id}.json"
    stats.track_types_flag = True

    units = gate.g4_units
    world.build_world(sim, units)
    if hasattr(world, "add_optical_surfaces"):
        world.add_optical_surfaces(sim, units)

    actor_registry = wire_actors(sim, world, caps, config.run_id, units)

    add_beam_source(sim, config, beam_cfg, units)
    configure_physics(sim, config, caps)

    db_path = gate_materials_db()
    if db_path.exists():
        sim.volume_manager.add_material_database(str(db_path))

    sim.number_of_threads = config.threads
    sim.progress_bar = config.progress_bar

    if config.run_timing_us:
        lo, hi = config.run_timing_us
        sim.run_timing_intervals = [[lo * units.us, hi * units.us]]

    sim.run()
    metadata_path = save_metadata(config, batch_dir, run_dir, world, caps, beam_cfg, actor_registry)

    return {
        "batch_dir": batch_dir,
        "run_dir": run_dir,
        "metadata_path": metadata_path,
        "capabilities": caps,
        "beam_config": beam_cfg,
        "world_module": world,
    }
