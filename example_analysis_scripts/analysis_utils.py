"""
analysis_utils.py
=================
Shared analysis utilities used by analyze.py and world-specific analyze hooks.

Provides:
  - ROOT PhaseSpace file parsing  (hits, exits, timing)
  - DoseActor .mhd loading and aggregation
  - Calorimeter batch file aggregation (.txt / .npy)
  - Standard report section builders
  - Plot helpers (depth-dose profile)

World modules import from here; they never re-implement these primitives.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# uproot and itk are imported lazily so that worlds that don't need them
# (e.g. dose-only fast runs) don't fail on import.


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

_SCINT_KEYS    = {"Scintillation", "scintillation", "G4Scintillation"}
_CERENKOV_KEYS = {"Cerenkov", "cerenkov", "G4Cerenkov", "Cherenkov", "cherenkov"}


def classify_process(name: str) -> str:
    if name in _SCINT_KEYS:    return "Scintillation"
    if name in _CERENKOV_KEYS: return "Cerenkov"
    return "Other"


def _classify_vectorised(raw_process: np.ndarray) -> np.ndarray:
    proc = raw_process.astype("U64")
    is_cerenkov = (
        (proc == "Cerenkov")   | (proc == "cerenkov") |
        (proc == "G4Cerenkov") | (proc == "Cherenkov") |
        (proc == "cherenkov")
    )
    is_scint = (
        (proc == "Scintillation") | (proc == "scintillation") |
        (proc == "G4Scintillation")
    )
    codes = np.full(len(proc), 2, dtype=np.int8)   # 2 = Other
    codes[is_scint]    = 1
    codes[is_cerenkov] = 0
    return codes


def _pack_uid(event_ids: np.ndarray, track_ids: np.ndarray) -> np.ndarray:
    return (event_ids.astype(np.int64) << 32) | (track_ids.astype(np.int64) & 0xFFFFFFFF)


# ─────────────────────────────────────────────────────────────────────────────
# ROOT FILE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def first_tree_key(f, *substrings) -> str | None:
    for s in substrings:
        hit = next((k for k in f.keys() if s in k.lower()), None)
        if hit:
            return hit
    return None


def count_unique_by_process(paths: list[Path], tree_substrings: tuple) -> dict[str, int]:
    """
    Return {process_name: unique_photon_count} from a list of ROOT PhaseSpace files.
    Uniqueness is defined by (EventID, TrackID) pairs.
    """
    import uproot

    totals = {0: 0, 1: 0, 2: 0}
    for path in (p for p in paths if p.exists()):
        with uproot.open(path) as f:
            key = first_tree_key(f, *tree_substrings)
            if not key:
                continue
            tree      = f[key]
            track_ids = tree["TrackID"].array(library="np")
            event_ids = tree["EventID"].array(library="np")

            if "TrackCreatorProcess" in tree.keys():
                raw_proc = tree["TrackCreatorProcess"].array(library="np")
                codes    = _classify_vectorised(raw_proc)
            else:
                codes = np.full(len(track_ids), 2, dtype=np.int8)

            uids = _pack_uid(event_ids, track_ids)
            for code in (0, 1, 2):
                mask = codes == code
                if mask.any():
                    totals[code] += len(np.unique(uids[mask]))

    return {"Cerenkov": totals[0], "Scintillation": totals[1], "Other": totals[2]}


def analyse_hits(paths: list[Path]) -> dict[str, int]:
    return count_unique_by_process(paths, ("detector", "phasespace"))


def analyse_exits(paths: list[Path]) -> dict[str, int]:
    return count_unique_by_process(paths, ("exited", "phasespace"))


def load_root_positions(root_paths: list[Path], max_points: int,
                        process_filter: str | None = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read optical-photon hit positions (mm → cm) from ROOT PhaseSpace files.
    Returns (x, y, z) arrays subsampled to max_points.
    """
    import uproot

    xs, ys, zs = [], [], []
    for path in root_paths:
        if not path.exists():
            continue
        with uproot.open(path) as f:
            key = next(
                (k for k in f.keys()
                 if any(s in k.lower() for s in ("phasespace", "hits", "exited"))),
                None,
            )
            if key is None:
                continue
            tree  = f[key]
            names = _ensure_str(tree["ParticleName"].array(library="np"))
            mask  = (names == "opticalphoton") | (names == "optical_photon")

            if process_filter and "TrackCreatorProcess" in tree.keys():
                proc  = _ensure_str(tree["TrackCreatorProcess"].array(library="np"))
                mask &= np.array([process_filter in p.lower() for p in proc])

            xs.append(tree["Position_X"].array(library="np")[mask] / 10.0)
            ys.append(tree["Position_Y"].array(library="np")[mask] / 10.0)
            zs.append(tree["Position_Z"].array(library="np")[mask] / 10.0)

    if not xs:
        return np.empty(0), np.empty(0), np.empty(0)

    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    if len(x) > max_points:
        rng  = np.random.default_rng(42)
        idxs = rng.choice(len(x), max_points, replace=False)
        x, y, z = x[idxs], y[idxs], z[idxs]
    return x, y, z


def _ensure_str(arr: np.ndarray) -> np.ndarray:
    return np.array([
        x.decode("utf-8").strip() if isinstance(x, bytes) else str(x).strip()
        for x in arr
    ])


# ─────────────────────────────────────────────────────────────────────────────
# TIMING RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_timing_resolution(hits_files: list[Path], threshold_photon: int =1 ) -> float:
    """
    Simulate a leading-edge trigger at the N-th arriving photon per event per channel.
    Returns the best (lowest sigma) timing resolution in picoseconds across channels.
    Returns 0.0 if GlobalTime branch is absent or insufficient statistics.
    """
    import uproot

    channel_resolutions = []
    for path in hits_files:
        if not path.exists():
            continue
        if "hole_air" in path.name:
            continue

        event_trigger_times = []
        with uproot.open(path) as f:
            key = first_tree_key(f, "detector", "phasespace")
            if not key:
                continue
            tree = f[key]
            if "GlobalTime" not in tree.keys():
                continue

            events = tree["EventID"].array(library="np")
            times  = tree["GlobalTime"].array(library="np")

            for ev in np.unique(events):
                ev_times = np.sort(times[events == ev])
                if len(ev_times) >= threshold_photon:
                    event_trigger_times.append(ev_times[threshold_photon - 1] * 1000.0)

        if len(event_trigger_times) >= 1:
            sigma = float(np.std(event_trigger_times))
            channel_resolutions.append(sigma)

    return float(np.min(channel_resolutions)) if channel_resolutions else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DOSE ACTOR (.mhd)
# ─────────────────────────────────────────────────────────────────────────────

def load_dose_mhd(run_dirs: list[Path], phantom_cm: list[float],
                  filename_glob: str = "edep*.mhd"
                  ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Accumulate DoseActor .mhd files across run directories.
    Returns (depth_centers_cm, depth_profile_MeV) or (None, None).
    """
    import itk

    accumulated = None
    spacing_mm  = None

    for rdir in run_dirs:
        candidates = list(rdir.glob(filename_glob))
        if not candidates:
            candidates = list(rdir.glob("edep.mhd"))
        if not candidates:
            continue
        mhd_path = candidates[0]

        img = itk.imread(str(mhd_path), itk.F)
        arr = itk.array_from_image(img).transpose(2, 1, 0)   # ITK ZYX → XYZ

        if accumulated is None:
            accumulated = arr.astype(np.float64)
            spacing_mm  = float(img.GetSpacing()[2])
        elif arr.shape == accumulated.shape:
            accumulated += arr.astype(np.float64)

    if accumulated is None:
        return None, None

    depth_profile = accumulated.sum(axis=(0, 1))
    nz    = depth_profile.shape[0]
    dz_cm = spacing_mm / 10.0
    centers_cm = np.linspace(dz_cm / 2, phantom_cm[2] - dz_cm / 2, nz)
    return centers_cm, depth_profile


def load_calorimeter_mhd(run_dirs: list[Path],
                         long_glob:  str = "run_Dose_edep.mhd",
                         trans_glob: str = "transverse_shower_max_edep.mhd",
                         ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Accumulate longitudinal and transverse calorimeter .mhd files.
    Returns (long_1d_array, trans_2d_array), either may be None.
    """
    import itk

    long_acc  = None
    trans_acc = None

    for rdir in run_dirs:
        long_candidates = list(rdir.glob(long_glob))
        if long_candidates:
            arr = itk.array_from_image(itk.imread(str(long_candidates[0]))).reshape(-1)
            long_acc = arr.astype(np.float64) if long_acc is None else long_acc + arr

        trans_candidates = list(rdir.glob(trans_glob))
        if trans_candidates:
            arr = itk.array_from_image(itk.imread(str(trans_candidates[0])))
            arr2d = arr.reshape(arr.shape[1], arr.shape[2]).astype(np.float64)
            trans_acc = arr2d if trans_acc is None else trans_acc + arr2d

    return long_acc, trans_acc


# ─────────────────────────────────────────────────────────────────────────────
# METADATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_batch_metadata(run_dirs: list[Path], world_name: str | None
                        ) -> dict:
    """
    Aggregate sim_metadata_*.json and stats_*.json across all run directories.
    Returns a unified metadata dict.
    """
    total_primaries = 0
    total_optical   = 0
    phantom_cm      = None
    w_name          = world_name
    expected_dedx   = 1.0
    active_z_ranges = None
    dose_spacing_mm = 0.1
    capabilities    = {}

    for rdir in run_dirs:
        run_primaries_from_events = 0
        run_primaries_from_meta   = 0

        for meta_path in sorted(rdir.glob("sim_metadata_*.json")):
            with open(meta_path) as f:
                meta = json.load(f)
            run_primaries_from_meta += meta.get("n_primaries", 0)
            if not phantom_cm:
                phantom_cm = meta.get("phantom_cm")
            if not w_name:
                w_name = meta.get("world", "unknown")
            expected_dedx   = meta.get("expected_dedx", 1.0)
            active_z_ranges = meta.get("active_z_ranges_mm", active_z_ranges)
            dose_spacing_mm = meta.get("dose_spacing_mm", dose_spacing_mm)
            if not capabilities:
                capabilities = meta.get("capabilities", {})

        for stats_path in sorted(rdir.glob("stats_*.json")):
            with open(stats_path) as f:
                stats = json.load(f)
            total_optical += _sum_optical_counts(stats)
            # events = actual primaries Geant4 ran (ground truth, unaffected
            # by the source.n-per-thread multiplier); prefer this over the
            # sim_metadata n_primaries field, which reflects the CLI arg.
            run_primaries_from_events += stats.get("events", {}).get("value", 0)

        total_primaries += (run_primaries_from_events
                            if run_primaries_from_events
                            else run_primaries_from_meta)

    if not phantom_cm:
        raise RuntimeError("Could not find phantom_cm in any sim_metadata_*.json")

    return {
        "world":            w_name,
        "phantom_cm":       phantom_cm,
        "total_primaries":  total_primaries,
        "total_optical":    total_optical,
        "expected_dedx":    expected_dedx,
        "active_z_ranges":  active_z_ranges,
        "dose_spacing_mm":  dose_spacing_mm,
        "capabilities":     capabilities,
    }


def _sum_optical_counts(d) -> int:
    total = 0
    if isinstance(d, dict):
        for k, v in d.items():
            if "optical" in k.lower() and isinstance(v, (int, float)):
                total += int(v)
            else:
                total += _sum_optical_counts(v)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# BATCH DIRECTORY DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def find_batch_dir(base: Path, world_hint: str | None, user_dir: str | None) -> Path:
    if user_dir:
        p = Path(user_dir)
        if not any(p.glob("run_*")):
            deeper = _newest_batches(p)
            if deeper:
                return deeper[0]
        return p

    runs_root = base / "runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"No runs/ directory found in {base}")

    if world_hint:
        world_dir = runs_root / world_hint
        if world_dir.exists():
            batches = _newest_batches(world_dir)
            if batches:
                return batches[0]

    all_batches = [b for wd in runs_root.iterdir()
                   if wd.is_dir() for b in _newest_batches(wd)]
    if not all_batches:
        raise FileNotFoundError(f"No batch directories under {runs_root}")
    all_batches.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return all_batches[0]


def _newest_batches(world_dir: Path) -> list[Path]:
    result = [d for d in world_dir.iterdir()
              if d.is_dir() and any(d.glob("run_*"))]
    result.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return result


def find_runs(batch_dir: Path) -> list[Path]:
    runs = sorted(
        [d for d in batch_dir.glob("run_*") if d.is_dir()],
        key=lambda d: int(d.name.split("_")[-1]),
    )
    if not runs:
        raise FileNotFoundError(f"No run_* folders in {batch_dir}")
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD PLOT: DEPTH-DOSE PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def plot_dose_profile(centers_cm: np.ndarray, edep_MeV: np.ndarray,
                      total_primaries: int, phantom_cm: list[float],
                      world_name: str, batch_dir: Path) -> Path:
    dz_cm     = centers_cm[1] - centers_cm[0]
    edep_norm = edep_MeV / total_primaries / dz_cm

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(centers_cm, edep_norm, color="firebrick", lw=2.5, label="dE/dx")
    ax.fill_between(centers_cm, edep_norm, alpha=0.15, color="firebrick")

    for x, label in [(0, " Entry"), (phantom_cm[2], " Exit")]:
        ax.axvline(x=x, color="black", linestyle="--", lw=1.5, alpha=0.7)
        ax.text(x + 0.02, np.max(edep_norm) * 0.05, label,
                rotation=90, va="bottom", fontsize=9, color="black", alpha=0.8)

    ax.set_xlabel("Depth (cm)")
    ax.set_ylabel("Deposited Energy (MeV / cm)")
    ax.set_title(
        f"Depth-Dose Profile — {world_name}\n"
        f"{phantom_cm[0]}×{phantom_cm[1]}×{phantom_cm[2]} cm  |  "
        f"{total_primaries:,} primaries"
    )
    ax.set_xlim(-0.2, phantom_cm[2] + 0.2)
    ax.set_ylim(0, np.max(edep_norm) * 1.3)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()

    out = batch_dir / "depth_dose_profile.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STANDARD REPORT SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

W = 66   # report width


def report_header(meta: dict, batch_dir: Path) -> list[str]:
    phantom_cm = meta["phantom_cm"]
    return [
        "═" * W,
        "  BATCH SIMULATION ANALYSIS REPORT",
        f"  World         : {meta['world']}",
        f"  Geometry      : {phantom_cm[0]} × {phantom_cm[1]} × {phantom_cm[2]} cm",
        f"  Batch Dir     : {batch_dir}",
        f"  Total Primaries: {meta['total_primaries']:,}",
        "═" * W,
    ]


def report_optical_section(hits: dict[str, int], exits: dict[str, int],
                            total_optical: int, total_primaries: int,
                            total_edep: float) -> list[str]:
    total_hits  = sum(hits.values())
    total_exits = sum(exits.values())
    lines = [
        "", "─" * W, "  OPTICAL PHOTON FLOW", "─" * W,
        f"  {'Process':<18} {'Exited':>12} {'Detected':>12}",
    ]
    for proc in ("Cerenkov", "Scintillation", "Other"):
        lines.append(f"  {proc:<18} {exits.get(proc,0):>12,} {hits.get(proc,0):>12,}")
    lines += [
        f"  {'─'*44}",
        f"  {'TOTAL':<18} {total_exits:>12,} {total_hits:>12,}",
        "",
        "─" * W, "  COLLECTION METRICS", "─" * W,
    ]
    if total_edep > 0 and total_optical > 0:
        lines.append(f"  Created photons / MeV   : {total_optical/total_edep:,.2f}")
    if total_optical > 0:
        lines.append(f"  Grid escape efficiency  : {100*total_exits/total_optical:.4f}%")
        lines.append(f"  System efficiency       : {100*total_hits/total_optical:.4f}%")
    if total_exits > 0:
        lines.append(f"  Detector capture effic. : {100*total_hits/total_exits:.4f}%")
    lines.append(f"  Hits / primary          : {total_hits/total_primaries if total_primaries else 0:.4f}")
    if total_edep > 0:
        lines.append(f"  Hits / MeV deposited    : {total_hits/total_edep:.2f}")
    return lines


def report_dose_section(centers_cm: np.ndarray | None,
                        edep: np.ndarray | None,
                        total_primaries: int) -> list[str]:
    if centers_cm is None:
        return ["", "─" * W, "  DOSE : no .mhd files found", "─" * W]
    total_edep    = float(edep.sum())
    edep_per_prim = total_edep / total_primaries if total_primaries else 0.0
    dz_cm = centers_cm[1] - centers_cm[0] if len(centers_cm) > 1 else 0.0
    return [
        "", "─" * W, "  DEPTH-DOSE SUMMARY", "─" * W,
        f"  Total deposited energy  : {total_edep:.4f} MeV",
        f"  Deposited / primary     : {edep_per_prim:.4f} MeV",
        f"  Voxel size (Z)          : {dz_cm:.4f} cm",
        "  → depth_dose_profile.png",
    ]


def report_footer() -> list[str]:
    return ["═" * W]