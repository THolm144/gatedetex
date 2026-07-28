"""
gatedetex — modular OpenGATE/Geant4 detector simulations.

Quickstart
----------
    import gatedetex

    gatedetex.list_worlds()
    # -> ['quartz_cal', 'radi_cal_energy', 'scintx_sipm_array']

    result = gatedetex.run_simulation(
        world="quartz_cal", particle="e-", energy_kev=9000, n=10_000,
    )
    print(result["run_dir"])

    from gatedetex.analysis import analyze_batch
    analyze_batch(batch_dir=result["batch_dir"])

Command line
------------
    gatedetex-sim --world quartz_cal --particle e- --energy-kev 9000 --n 10000
    gatedetex-analyze --batch-dir outputs/quartz_cal/...
    gatedetex-plot3d --batch-dir outputs/quartz_cal/...

Custom worlds
-------------
    gatedetex.worlds.add_world_dir("~/my_worlds")
    gatedetex.run_simulation(world="my_detector", ...)
"""

from importlib.metadata import PackageNotFoundError, version

from . import worlds
from .simulator import SimulationConfig, run_simulation
from .worlds import list_worlds

try:
    __version__ = version("gatedetex")
except PackageNotFoundError:  # running from a source checkout, not installed
    __version__ = "0.0.0.dev0"

__all__ = [
    "run_simulation",
    "SimulationConfig",
    "list_worlds",
    "worlds",
    "__version__",
]
