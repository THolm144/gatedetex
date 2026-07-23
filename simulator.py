import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import opengate as gate


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CAPABILITIES = {
    "optical":          False,
    "dose":             True,
    "sipm_hits":        False,
    "optical_exits":    False,
    "track_optical":    False,
}

DEFAULT_BEAM_CONFIG = {
    "direction": [0, 0, 1],   # +Z  (beam travels in +Z by default)
    "target_cm": [0, 0, 0],
    "offset_cm": 2.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="OpenGATE modular simulator")
    p.add_argument("--world",        default="quartz_8x8")
    p.add_argument("--particle",     default="e-")                    # FLASH Default: Electrons
    p.add_argument("--energy-kev",   type=float, default=9000)        # FLASH Default: 9 MeV (FLEX-9)
    p.add_argument("--n",            type=int,   default=10_000)
    p.add_argument("--threads",      type=int,   default=16)          # Scaled for local task blocks
    p.add_argument("--beam-radius",  type=float, default=1.0)
    p.add_argument("--output-dir",   default=None)
    p.add_argument("--physics-list", default="G4EmStandardPhysics_option4")
    p.add_argument("--run-id",       type=int,   default=0)           # Set default to 0
    # ── capability overrides ──────────────────────────────────────────────────
    p.add_argument("--optical",       choices=["on", "off", "world"],  default="world")
    p.add_argument("--dose",          choices=["on", "off", "world"],  default="world")
    p.add_argument("--sipm-hits",     choices=["on", "off", "world"],  default="world")
    p.add_argument("--track-optical", choices=["on", "off", "world"],  default="world")
    p.add_argument("--no-cerenkov", action="store_true", default=False)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# WORLD LOADING / RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def load_world(world_name: str, script_dir: Path):
    sys.path.insert(0, str(script_dir / "worlds"))
    try:
        return importlib.import_module(world_name)
    except ModuleNotFoundError:
        print(f"ERROR: world module '{world_name}' not found in worlds/")
        sys.exit(1)


def resolve_capabilities(world, args) -> dict:
    caps = {**DEFAULT_CAPABILITIES, **getattr(world, "CAPABILITIES", {})}
    override_map = {
        "optical":       args.optical,
        "dose":          args.dose,
        "sipm_hits":     getattr(args, "sipm_hits", "world"),
        "track_optical": args.track_optical,
    }
    for key, val in override_map.items():
        if val == "on":
            caps[key] = True
        elif val == "off":
            caps[key] = False
    return caps


def resolve_beam_config(world) -> dict:
    return {**DEFAULT_BEAM_CONFIG, **getattr(world, "BEAM_CONFIG", {})}


def resolve_output_dirs(args, script_dir: Path) -> tuple[Path, Path]:
    if args.output_dir:
        batch_dir = Path(args.output_dir)
    else:
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = script_dir / "runs" / args.world / f"{int(args.energy_kev)}keV_{ts}"

    batch_dir.mkdir(parents=True, exist_ok=True)   # ← ADD THIS
    return batch_dir, batch_dir


# ─────────────────────────────────────────────────────────────────────────────
# ACTOR WIRING
# ─────────────────────────────────────────────────────────────────────────────

def wire_actors(sim, world, caps: dict, run_id: int, units) -> dict:
    registry = {
        "optical_exited_actor":  None,
        "optical_tracker_actor": None,
        "hit_actors":            [],
        "dose_actor":            None,
    }

    target_vol       = getattr(world, "TARGET_VOLUME_NAME",    "target")
    detector_volumes = getattr(world, "DETECTOR_VOLUME_NAMES", [])

    # ── Optical tracking (Append run_id to avoid data collision) ─────────
    if caps.get("track_optical", False) and caps.get("optical", False):
        tracker = sim.add_actor("PhaseSpaceActor", f"optical_tracker_{run_id}")
        tracker.attached_to     = "world"
        tracker.output_filename = f"optical_tracks_{run_id}.root"
        tracker.steps_to_store  = "all"
        tracker.attributes = [
            "ParticleName", "KineticEnergy", "TrackCreatorProcess",
            "Position", "TrackID", "EventID", "GlobalTime",
        ]
        registry["optical_tracker_actor"] = tracker

    # ── Optical exits ─────────────────────────────────────────────────────
    if caps.get("optical", False) and caps.get("optical_exits", False):
        exited = sim.add_actor("PhaseSpaceActor", f"optical_exited_{run_id}")
        exited.attached_to     = target_vol
        exited.output_filename = f"optical_exited_{run_id}.root"
        exited.steps_to_store  = "exiting"
        exited.attributes = [
            "ParticleName", "KineticEnergy", "TrackCreatorProcess",
            "Position", "TrackID", "EventID", "GlobalTime",
        ]
        registry["optical_exited_actor"] = exited

    # ── Per-detector hit actors ───────────────────────────────────────────
    if caps.get("sipm_hits", False) and detector_volumes:
        for idx, vol_name in enumerate(detector_volumes):
            if vol_name not in sim.volume_manager.volumes:
                continue
            hits = sim.add_actor("PhaseSpaceActor", f"detector_hits_{idx}_{run_id}")
            hits.attached_to                = vol_name
            hits.authorize_repeated_volumes = True
            hits.output_filename            = f"detector_hits_{idx}_{run_id}.root"
            hits.steps_to_store             = "entering"
            hits.attributes = [
                "ParticleName", "KineticEnergy", "Position",
                "TrackCreatorProcess", "TrackID", "EventID", "GlobalTime",
            ]
            registry["hit_actors"].append(hits)

    # ── Dose actor ────────────────────────────────────────────────────────
    if caps.get("dose", True):
        phantom_cm = world.PHANTOM_CM
        dose = _wire_standard_dose(sim, target_vol, phantom_cm, run_id, units)
        if hasattr(world, "configure_dose_actor"):
            world.configure_dose_actor(dose, units)
        registry["dose_actor"] = dose

    return registry
    

def _wire_standard_dose(sim, target_vol: str, phantom_cm: list, run_id: int, units):
    dose = sim.add_actor("DoseActor", f"dose_actor_{run_id}")
    dose.attached_to      = target_vol
    dose.output_filename  = f"edep_{run_id}.mhd"
    dose.size = [
        int(round(phantom_cm[0] * 10)),
        int(round(phantom_cm[1] * 10)),
        int(round(phantom_cm[2] * 10)),
    ]
    dose.spacing                = [1.0 * units.mm] * 3
    dose.hit_type               = "random"
    dose.edep.active            = True
    dose.dose.active            = False
    dose.edep_uncertainty.active = False
    return dose


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE
# ─────────────────────────────────────────────────────────────────────────────

def add_beam_source(sim, args, world, beam_cfg: dict, units):
    direction = np.array(beam_cfg["direction"], dtype=float)
    direction /= np.linalg.norm(direction)

    target_cm  = np.array(beam_cfg["target_cm"], dtype=float)
    offset_cm  = beam_cfg["offset_cm"]
    source_pos_cm = target_cm - direction * offset_cm

    source = sim.add_source("GenericSource", f"{args.particle}_beam")
    source.particle             = args.particle
    source.energy.mono          = args.energy_kev * units.keV
    source.position.type        = "disc"
    source.position.radius      = args.beam_radius * units.cm
    source.position.translation = [
        source_pos_cm[0] * units.cm,
        source_pos_cm[1] * units.cm,
        source_pos_cm[2] * units.cm,
    ]
    source.direction.type     = "momentum"
    source.direction.momentum = direction.tolist()

    

    source.n = args.n  # OpenGATE distributes this target natively across threads
    return source


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS
# ─────────────────────────────────────────────────────────────────────────────

def configure_physics(sim, args, script_dir: Path, world, caps: dict, target_vol: str, units):
    sim.physics_manager.physics_list_name = args.physics_list

    surface_file = script_dir / "SurfaceProperties.xml"
    if surface_file.exists():
        sim.physics_manager.surface_properties_file = str(surface_file)

    if caps["optical"]:
        optical_file = script_dir / "Materials.xml"
        sim.physics_manager.special_physics_constructors.G4OpticalPhysics = True
        if optical_file.exists():
            sim.physics_manager.optical_properties_file = str(optical_file)

        if args.no_cerenkov:
            sim.g4_commands_before_init.append("/process/optical/processActivation Cerenkov false")
        else:
            sim.g4_commands_before_init.append("/process/optical/processActivation Cerenkov true")


# ─────────────────────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────────────────────

def save_metadata(args, batch_dir: Path, run_dir: Path, world, caps: dict, beam_cfg: dict, actor_registry: dict):
    dose = actor_registry["dose_actor"]
    metadata = {
        "world":              args.world,
        "particle":           args.particle,
        "energy_kev":         args.energy_kev,
        "n_primaries":        args.n,
        "threads":            args.threads,
        "beam_radius_cm":     args.beam_radius,
        "physics_list":       args.physics_list,
        "batch_dir":          str(batch_dir),
        "output_dir":         str(run_dir),
        "material":           getattr(world, "MATERIAL",    "unknown"),
        "phantom_cm":         getattr(world, "PHANTOM_CM",  None),
        "target_volume":      getattr(world, "TARGET_VOLUME_NAME", "target"),
        "capabilities":       caps,
        "beam_config":        beam_cfg,
        "dose_size_vox":      dose.size if dose else None,
    }
    path = run_dir / f"sim_metadata_{args.run_id}.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    batch_dir, run_dir = resolve_output_dirs(args, script_dir)
    world              = load_world(args.world, script_dir)
    caps               = resolve_capabilities(world, args)
    beam_cfg           = resolve_beam_config(world)

    sim = gate.Simulation()
    sim.random_seed = 1000 + args.run_id
    sim.output_dir = str(run_dir)

    stats = sim.add_actor("SimulationStatisticsActor", f"sim_stats_{args.run_id}")
    stats.output_filename  = f"stats_{args.run_id}.json"
    stats.track_types_flag = True

    units = gate.g4_units
    world.build_world(sim, units)
    if hasattr(world, "add_optical_surfaces"):
        world.add_optical_surfaces(sim, units)

    target_vol   = getattr(world, "TARGET_VOLUME_NAME", "target")
    actor_registry = wire_actors(sim, world, caps, args.run_id, units)

    add_beam_source(sim, args, world, beam_cfg, units)
    configure_physics(sim, args, script_dir, world, caps, target_vol, units)

    db_path = script_dir / "GateMaterials.db"
    if db_path.exists():
        sim.volume_manager.add_material_database(str(db_path))

    sim.number_of_threads = args.threads
    sim.progress_bar      = False  # Keep clean inside massive batch arrays

   
    # Find this section at the bottom of your main() function in simulator.py:
    sim.number_of_threads = args.threads
    sim.progress_bar      = False 

    # --- ADD THIS LINE FOR FLASH TIMING ---
    sim.run_timing_intervals = [[0.0 * units.us, 5.0 * units.us]]
    # ────────────────────────────────────

    sim.run()
    save_metadata(args, batch_dir, run_dir, world, caps, beam_cfg, actor_registry)


if __name__ == "__main__":
    main()