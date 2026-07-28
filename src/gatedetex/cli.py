"""Console-script entry points for gatedetex."""

from __future__ import annotations

import argparse

from .simulator import SimulationConfig, run_simulation
from .worlds import list_worlds


# ─────────────────────────────────────────────────────────────────────────────
# gatedetex-sim
# ─────────────────────────────────────────────────────────────────────────────

def _sim_parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="gatedetex-sim",
        description="Run an OpenGATE modular detector simulation.",
    )
    p.add_argument("--world", default="quartz_cal",
                   help=f"World module to simulate. Built-in: {', '.join(list_worlds())}")
    p.add_argument("--particle", default="e-")
    p.add_argument("--energy-kev", type=float, default=9000)
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--beam-radius", type=float, default=1.0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--physics-list", default="G4EmStandardPhysics_option4")
    p.add_argument("--run-id", type=int, default=0)
    p.add_argument("--optical", choices=["on", "off", "world"], default="world")
    p.add_argument("--dose", choices=["on", "off", "world"], default="world")
    p.add_argument("--sipm-hits", choices=["on", "off", "world"], default="world")
    p.add_argument("--track-optical", choices=["on", "off", "world"], default="world")
    p.add_argument("--no-cerenkov", action="store_true", default=False)
    p.add_argument("--world-dir", action="append", default=[],
                   help="Extra directory to search for world modules "
                        "(can be passed multiple times).")
    p.add_argument("--list-worlds", action="store_true",
                   help="List discoverable world modules and exit.")
    p.add_argument("--progress-bar", action="store_true", default=False)
    return p.parse_args(argv)


def sim_main(argv=None):
    args = _sim_parse_args(argv)

    from . import worlds as _worlds
    for d in args.world_dir:
        _worlds.add_world_dir(d)

    if args.list_worlds:
        for name in list_worlds():
            print(name)
        return

    config = SimulationConfig(
        world=args.world, particle=args.particle, energy_kev=args.energy_kev,
        n=args.n, threads=args.threads, beam_radius=args.beam_radius,
        output_dir=args.output_dir, physics_list=args.physics_list,
        run_id=args.run_id, optical=args.optical, dose=args.dose,
        sipm_hits=args.sipm_hits, track_optical=args.track_optical,
        no_cerenkov=args.no_cerenkov, progress_bar=args.progress_bar,
    )
    result = run_simulation(config)
    print(f"\nDone. Metadata written to {result['metadata_path']}")


# ─────────────────────────────────────────────────────────────────────────────
# gatedetex-analyze
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="gatedetex-analyze",
        description="Analyze a batch of gatedetex simulation runs.",
    )
    p.add_argument("--batch-dir", default=None)
    p.add_argument("--world", default=None,
                   help="World name (scopes auto-discovery; auto-detected if omitted)")
    p.add_argument("--world-dir", action="append", default=[])
    return p.parse_args(argv)


def analyze_main(argv=None):
    args = _analyze_parse_args(argv)

    from . import worlds as _worlds
    for d in args.world_dir:
        _worlds.add_world_dir(d)

    from .analysis import analyze_batch
    analyze_batch(batch_dir=args.batch_dir, world=args.world)


# ─────────────────────────────────────────────────────────────────────────────
# gatedetex-plot3d
# ─────────────────────────────────────────────────────────────────────────────

def _plot3d_parse_args(argv=None):
    from .analysis.plot_3d import VIEW_PARAMS
    p = argparse.ArgumentParser(
        prog="gatedetex-plot3d",
        description="Render 3-D optical-photon visualizations for a batch (headless).",
    )
    p.add_argument("--batch-dir", default=None)
    p.add_argument("--world", default=None)
    p.add_argument("--max-tracks", type=int, default=300)
    p.add_argument("--max-optical-steps", type=int, default=5000)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--views", nargs="+", default=list(VIEW_PARAMS.keys()),
                   choices=list(VIEW_PARAMS.keys()))
    p.add_argument("--world-dir", action="append", default=[])
    return p.parse_args(argv)


def plot3d_main(argv=None):
    args = _plot3d_parse_args(argv)

    from . import worlds as _worlds
    for d in args.world_dir:
        _worlds.add_world_dir(d)

    from .analysis import render_batch
    render_batch(
        batch_dir=args.batch_dir, world=args.world, max_tracks=args.max_tracks,
        max_optical_steps=args.max_optical_steps, dpi=args.dpi, views=args.views,
    )
