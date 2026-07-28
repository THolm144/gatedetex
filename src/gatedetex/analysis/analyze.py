"""
Unified batch post-processor.

Reads world type from sim_metadata.json, imports the world module via the
gatedetex worlds registry, and calls ``world.analyze()`` to get structured
results. This module owns report formatting; worlds own data-extraction
logic.

World ``analyze()`` contract
-----------------------------
    def analyze(batch_dir, run_dirs, meta, utils) -> dict
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import worlds as _worlds
from . import utils


# ─────────────────────────────────────────────────────────────────────────────
# REPORT ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_report(batch_dir: Path, meta: dict, results: dict) -> str:
    lines = utils.report_header(meta, batch_dir)

    hits = results.get("hits", {})
    exits = results.get("exits", {})
    total_primaries = meta["total_primaries"]
    total_optical = meta["total_optical"]

    dose_edep = results.get("dose_edep")
    total_edep = float(dose_edep.sum()) if dose_edep is not None else 0.0

    caps = meta.get("capabilities", {})

    if caps.get("optical", False) or sum(hits.values()) > 0:
        lines += utils.report_optical_section(
            hits, exits, total_optical, total_primaries, total_edep
        )

        timing_res = results.get("timing_res_ps", 0.0)
        lines += ["", "─" * utils.W, "  CALIBRATION CONSTANTS", "─" * utils.W]
        c_exp = hits.get("Cerenkov", 0) / total_primaries if total_primaries else 0
        scint_lce = (hits.get("Scintillation", 0) / total_optical
                     if total_optical > 0 else 0.0)
        edep_per_prim = total_edep / total_primaries if total_primaries else 0.0
        lines += [
            f"  E_dep / primary      : {edep_per_prim:.4f} MeV",
            f"  C_exp (Cer hits/prim): {c_exp:.4f}",
            f"  e_LCE (Scint hits/created): {scint_lce:.6f}",
            f"  Timing resolution    : "
            + (f"{timing_res:.2f} ps" if timing_res > 0 else "N/A"),
        ]

    if caps.get("dose", True):
        lines += utils.report_dose_section(
            results.get("dose_centers"), dose_edep, total_primaries,
        )

    extra = results.get("extra_lines", [])
    if extra:
        lines += ["", "─" * utils.W, "  WORLD-SPECIFIC RESULTS", "─" * utils.W]
        lines += extra

    plots = results.get("plots_saved", [])
    if plots:
        lines += ["", "─" * utils.W, "  PLOTS SAVED", "─" * utils.W]
        lines += [f"  → {p}" for p in plots]

    lines += utils.report_footer()
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK ANALYZE METHOD
# ─────────────────────────────────────────────────────────────────────────────

def _generic_analyze(batch_dir: Path, run_dirs: list, meta: dict) -> dict:
    """Fallback used when the world has no analyze() hook: hits + exits + dose."""
    hits_files = [p for d in run_dirs for p in sorted(d.glob("detector_hits*.root"))]
    exits_files = [d / "optical_exited.root" for d in run_dirs]

    hits = utils.analyse_hits(hits_files)
    exits = utils.analyse_exits(exits_files)
    centers, edep = utils.load_dose_mhd(run_dirs, meta["phantom_cm"])
    timing_res = utils.extract_timing_resolution(hits_files) if hits_files else 0.0

    return {
        "hits": hits,
        "exits": exits,
        "dose_centers": centers,
        "dose_edep": edep,
        "timing_res_ps": timing_res,
    }


def _fallback_metadata(run_dirs: list, world_hint: str | None) -> dict:
    """Best-effort metadata reconstruction when sim_metadata_*.json is unusable."""
    world_name = world_hint or "unknown"
    meta = {
        "world": world_name,
        "total_primaries": 0,
        "total_optical": 0,
        "phantom_cm": [10.0, 10.0, 10.0],
        "capabilities": {"optical": True, "dose": True},
    }

    accumulated_primaries = 0
    accumulated_optical = 0
    for r_dir in run_dirs:
        run_meta_file = r_dir / "sim_metadata.json"
        if not run_meta_file.exists():
            continue
        try:
            raw_m = json.loads(run_meta_file.read_text())
            accumulated_primaries += raw_m.get("n_primaries", raw_m.get("total_primaries", 0))
            accumulated_optical += raw_m.get("total_optical", raw_m.get("n_optical", 0))
        except (json.JSONDecodeError, OSError):
            continue

    meta["total_primaries"] = accumulated_primaries or (1000 * len(run_dirs))
    meta["total_optical"] = accumulated_optical or (50_000 * len(run_dirs))
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def analyze_batch(batch_dir: str | Path | None = None, world: str | None = None,
                   base_dir: str | Path | None = None) -> dict:
    """Analyze one batch of simulation runs and write ``batch_analysis.txt``.

    Returns a dict with keys "report" (str), "report_path" (Path), and
    "results" (the raw dict returned by the world's analyze() hook / the
    generic fallback).
    """
    base = Path(base_dir).resolve() if base_dir else Path.cwd()

    resolved_batch_dir = utils.find_batch_dir(base, world, str(batch_dir) if batch_dir else None)
    run_dirs = utils.find_runs(resolved_batch_dir)

    print(f"  Batch dir   : {resolved_batch_dir}")
    print(f"  Run count   : {len(run_dirs)}")

    try:
        meta = utils.load_batch_metadata(run_dirs, world)
    except RuntimeError as e:
        print(f"  [Warning] Metadata discovery failed: {e}")
        print("  [Warning] Reconstructing fallback metadata from stats files...")
        meta = _fallback_metadata(run_dirs, world)
        print(f"  [Recovered] Estimated total primaries: {meta['total_primaries']}")
        print(f"  [Recovered] Estimated total optical  : {meta['total_optical']}")

    world_name = meta["world"]
    try:
        world_module = _worlds.load_world(world_name)
        print(f"  World module loaded  : {world_name}")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        world_module = None

    if world_module and hasattr(world_module, "analyze"):
        print(f"  Dispatching to {world_name}.analyze() …")
        results = world_module.analyze(resolved_batch_dir, run_dirs, meta, utils)
    else:
        print("  No world analyze() hook — running generic analysis.")
        results = _generic_analyze(resolved_batch_dir, run_dirs, meta)

    centers = results.get("dose_centers")
    edep = results.get("dose_edep")
    if centers is not None and edep is not None:
        plot_path = utils.plot_dose_profile(
            centers, edep, meta["total_primaries"], meta["phantom_cm"],
            world_name, resolved_batch_dir,
        )
        results.setdefault("plots_saved", []).append(plot_path.name)

    report = build_report(resolved_batch_dir, meta, results)
    report_path = resolved_batch_dir / "batch_analysis.txt"
    report_path.write_text(report)
    print(f"\n{report}")
    print(f"\n  Report → {report_path}")

    return {"report": report, "report_path": report_path, "results": results}
