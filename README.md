# gatedetex

A modular [OpenGATE](https://opengate.readthedocs.io/)/Geant4 simulation
harness for running detector-physics simulations (dose, optical photons,
SiPM hits, timing) against pluggable geometry modules ("worlds"), plus
a matching batch-analysis toolkit.

This is a packaged, installable version of the original `gatedetex`
scripts. The simulation and analysis logic is unchanged; what changed is
*how you get it onto your machine and run it*:

- `pip install`-able, with a proper dependency list instead of "figure out
  what to `pip install` from the traceback".
- World modules and material/optical-property data are bundled and
  discovered automatically — no more `world_modules/` vs `worlds/`
  folder mismatches, and no more silently-ignored optical-physics files
  (the original looked for `Materials.xml` next to `simulator.py`, but it
  actually lived in `materials/` — fixed here).
- Console commands (`gatedetex-sim`, `gatedetex-analyze`,
  `gatedetex-plot3d`) as well as a plain Python API
  (`gatedetex.run_simulation(...)`).
- Custom detector geometries can be dropped in from anywhere on disk
  without editing the package.

## Installation

gatedetex depends on [OpenGATE](https://opengate.readthedocs.io/), which
in turn requires a working Geant4 installation. The easiest route is a
conda/mamba environment:

```bash
conda create -n gatedetex python=3.11
conda activate gatedetex
pip install opengate   # follow opengate's own install docs if this needs extra setup
```

Then install gatedetex itself, from a local clone:

```bash
git clone https://github.com/THolm144/gatedetex.git
cd gatedetex
pip install .
# or, for local development (editable install):
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install "gatedetex[analysis]"   # adds uproot, needed to read PhaseSpaceActor .root output
pip install "gatedetex[advanced]"   # adds uproot, pandas, scipy for the advanced sweep scripts in examples/
```

> **Note on ROOT/PyROOT:** the advanced example scripts under
> `examples/advanced_sweep_scripts/` also import `ROOT`/`SimpleITK`
> directly. Those aren't pip-installable in general; install them via
> conda (`conda install -c conda-forge root simpleitk`) if you need those
> specific scripts. The core package and `gatedetex-analyze` /
> `gatedetex-plot3d` only need `uproot`, which is pure pip.

## Quickstart

### Command line

```bash
# list the worlds (detector geometries) available
gatedetex-sim --list-worlds

# run a simulation
gatedetex-sim --world quartz_cal --particle e- --energy-kev 9000 --n 10000 --threads 4

# analyze the batch it just produced (prints + writes batch_analysis.txt)
gatedetex-analyze --batch-dir outputs/quartz_cal/<timestamp>_9000keV

# render headless 3-D views of optical photon hits/tracks
gatedetex-plot3d --batch-dir outputs/quartz_cal/<timestamp>_9000keV
```

### Python

```python
import gatedetex

gatedetex.list_worlds()
# ['quartz_cal', 'radi_cal_energy', 'scintx_sipm_array']

result = gatedetex.run_simulation(
    world="quartz_cal", particle="e-", energy_kev=9000, n=10_000, threads=4,
)
print(result["run_dir"])

from gatedetex.analysis import analyze_batch
analyze_batch(batch_dir=result["batch_dir"])
```

## Built-in worlds

| World | Description |
|---|---|
| `quartz_cal` | Iron absorber + NxN quartz-crystal/SiPM scanner array |
| `radi_cal_energy` | RADiCAL Shashlik calorimeter (LYSO/tungsten sampling stack) |
| `scintx_sipm_array` | 10×10×0.6 cm ScintX scintillator slab with a 4-SiPM edge array |

## Writing a custom world

A world is just a Python module. Copy
[`src/gatedetex/worlds/_template.py`](src/gatedetex/worlds/_template.py)
somewhere, rename it (drop the leading underscore), fill in
`build_world(sim, units)`, and point gatedetex at the directory it's in:

```python
import gatedetex
gatedetex.worlds.add_world_dir("~/my_worlds")
gatedetex.run_simulation(world="my_detector", ...)
```

or from the CLI:

```bash
gatedetex-sim --world my_detector --world-dir ~/my_worlds
```

or via the `GATEDETEX_WORLDS_PATH` environment variable (`:`-separated,
like `PATH`). The template file documents the full contract (required
`build_world`, optional `CAPABILITIES`, `BEAM_CONFIG`, `analyze()`,
`get_geometry_primitives()`, etc).

## `examples/`

`examples/advanced_sweep_scripts/` holds the original multi-geometry sweep
and diagnostic scripts (timing-resolution sweeps, ToF reconstruction,
depth-profile analysis across geometry variants). They're kept as
standalone scripts rather than packaged, because they reference specific
geometry variants (e.g. `rc_hex`, `dsb1_*`, `luagce_*`) that aren't part of
this repository's built-in worlds — they're meant to be adapted to your
own sweep, not run as-is. They also pull in `pandas`, `scipy`, `ROOT`, and
`SimpleITK`; see the "advanced" extra above.

## Repository layout

```
src/gatedetex/
  simulator.py        core simulation engine (SimulationConfig, run_simulation)
  cli.py               gatedetex-sim / gatedetex-analyze / gatedetex-plot3d
  worlds/               built-in world modules + discovery registry
  analysis/             analyze_batch, render_batch, shared analysis utilities
  data/                 bundled Materials.xml / SurfaceProperties.xml / GateMaterials.db
examples/
  advanced_sweep_scripts/  original multi-geometry sweep/diagnostic scripts
```
