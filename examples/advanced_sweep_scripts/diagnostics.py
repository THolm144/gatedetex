#!/usr/bin/env python3
"""
Diagnostic script — run this in place of the unified analyzer to figure out
WHY all four modules are coming back empty. Prints counts at every stage
instead of silently producing empty plots.

Usage: python3 diagnose_sweep.py
(place next to radi_cal_energy/, radi_cal_triple/, rc_hex/, rc_hex_triple/)
"""
from pathlib import Path
import numpy as np
import uproot

_TYVEK_THICK_MM = 0.2032
_W_THICK_MM = 2.5
_N_LYSO = 29
_N_W = 28
_GT_LO_NS = 0.0
_GT_HI_NS = 50.0

_SQUARE_HOLE_OFFSET = 3.7032
SQUARE_CAP_XY = np.array([
    [ _SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET],
    [-_SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET],
    [-_SQUARE_HOLE_OFFSET,  _SQUARE_HOLE_OFFSET],
    [ _SQUARE_HOLE_OFFSET, -_SQUARE_HOLE_OFFSET],
])
HEX_CAP_R_MM = 3.5
HEX_CAP_XY = np.array([
    [HEX_CAP_R_MM * np.cos(np.pi/2 + i*(np.pi/3)), HEX_CAP_R_MM * np.sin(np.pi/2 + i*(np.pi/3))]
    for i in range(6)
])

def extract_numerical_energy(label):
    try:
        return float(''.join(c for c in label if c.isdigit() or c == '.'))
    except ValueError:
        return 0.0

def diagnose(batch_dir: Path, is_hex: bool, label: str):
    print(f"\n{'='*70}")
    print(f"  {label}  ({batch_dir})")
    print(f"{'='*70}")

    if not batch_dir.exists():
        print("  [!] Directory does not exist.")
        return

    run_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    print(f"  run_* subdirectories found : {len(run_dirs)}")

    hit_files = sorted(list(batch_dir.rglob("detector_hits_*.root")))
    print(f"  detector_hits_*.root files : {len(hit_files)}")
    if not hit_files:
        print("  [!] No hit files at all -- stopping here for this energy.")
        return

    # Check file sizes / entry counts on first few files
    total_entries = 0
    empty_files = 0
    for fpath in hit_files[:len(hit_files)]:
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk:
                    empty_files += 1
                    continue
                n = f[tk].num_entries
                total_entries += n
                if n == 0:
                    empty_files += 1
        except Exception as e:
            print(f"    [!] Could not open {fpath.name}: {e}")
            empty_files += 1
    print(f"  Total tree entries (all channels, all files) : {total_entries:,}")
    print(f"  Files with zero entries / unreadable          : {empty_files} / {len(hit_files)}")

    if total_entries == 0:
        print("  [!] Zero entries across all files -- simulation produced no hits for this energy.")
        return

    # Detect SiPM z-plane
    detected_z_sensor = None
    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk: continue
                z_arr = f[tk]["Position_Z"].array(library="np")
                if len(z_arr) > 0:
                    abs_z = np.abs(z_arr)
                    detected_z_sensor = float(np.median(abs_z[abs_z > (np.max(abs_z) - 5.0)]))
                    break
        except Exception:
            continue
    print(f"  Detected SiPM z-plane : {detected_z_sensor}")
    if detected_z_sensor is None:
        print("  [!] Could not detect z-plane -- stopping.")
        return

    lyso_thick = 1.5 if abs(detected_z_sensor - 91.65) < 3.0 else 4.5
    print(f"  Deduced LYSO thickness : {lyso_thick} mm")

    cap_xy_map = HEX_CAP_XY if is_hex else SQUARE_CAP_XY
    t_indices = {1, 3, 5} if is_hex else {0, 1}
    e_indices = {0, 2, 4} if is_hex else {2, 3}

    up_times_by_ev, dw_times_by_ev = {}, {}
    up_first, down_first = {}, {}
    total_optical = 0
    total_near_up = 0
    total_near_dw = 0
    total_t_type = 0
    total_e_type = 0

    for fpath in hit_files:
        try:
            with uproot.open(fpath) as f:
                tk = next((k for k in f.keys() if "detector_hits" in k.split(";")[0]), None)
                if not tk: continue
                tree = f[tk]
                if tree.num_entries == 0: continue
                x = tree["Position_X"].array(library="np")
                y = tree["Position_Y"].array(library="np")
                z = tree["Position_Z"].array(library="np")
                gt = tree["GlobalTime"].array(library="np")
                lt = tree["LocalTime"].array(library="np")
                ev = tree["EventID"].array(library="np")
                pn = tree["ParticleName"].array(library="np")
        except Exception:
            continue

        is_optical = (pn == b"opticalphoton") | (pn == "opticalphoton")
        total_optical += int(is_optical.sum())

        dx = x[:, np.newaxis] - cap_xy_map[:, 0]
        dy = y[:, np.newaxis] - cap_xy_map[:, 1]
        channels = np.argmin(np.hypot(dx, dy), axis=1)

        near_up = np.abs(z + detected_z_sensor) < 2.5
        near_dw = np.abs(z - detected_z_sensor) < 2.5
        total_near_up += int(near_up.sum())
        total_near_dw += int(near_dw.sum())

        is_t = np.isin(channels, list(t_indices))
        is_e = np.isin(channels, list(e_indices))
        total_t_type += int(is_t.sum())
        total_e_type += int(is_e.sum())

        is_prompt = (gt >= _GT_LO_NS) & (gt <= _GT_HI_NS)

        m_t_up = is_t & is_optical & near_up
        m_t_dw = is_t & is_optical & near_dw
        m_e_up = is_e & is_prompt & near_up
        m_e_dw = is_e & is_prompt & near_dw

        for e, t in zip(ev[m_t_up], lt[m_t_up] * 1000.0):
            up_times_by_ev.setdefault(int(e), []).append(t)
        for e, t in zip(ev[m_t_dw], lt[m_t_dw] * 1000.0):
            dw_times_by_ev.setdefault(int(e), []).append(t)
        for eid, ti in zip(ev[m_e_up], gt[m_e_up]):
            key = int(eid)
            if key not in up_first or ti < up_first[key]: up_first[key] = float(ti)
        for eid, ti in zip(ev[m_e_dw], gt[m_e_dw]):
            key = int(eid)
            if key not in down_first or ti < down_first[key]: down_first[key] = float(ti)

    print(f"  Total opticalphoton hits (all channels)   : {total_optical:,}")
    print(f"  Total hits within near_up window           : {total_near_up:,}")
    print(f"  Total hits within near_dw window            : {total_near_dw:,}")
    print(f"  Total hits on T-type channels (any window)  : {total_t_type:,}")
    print(f"  Total hits on E-type channels (any window)  : {total_e_type:,}")
    print(f"  Unique events w/ T-type UP hits   : {len(up_times_by_ev):,}")
    print(f"  Unique events w/ T-type DOWN hits : {len(dw_times_by_ev):,}")
    common_t = set(up_times_by_ev) & set(dw_times_by_ev)
    print(f"  T-type UP∩DOWN coincident events   : {len(common_t):,}   <-- feeds sigma_t_ps")
    print(f"  Unique events w/ E-type UP hits    : {len(up_first):,}")
    print(f"  Unique events w/ E-type DOWN hits  : {len(down_first):,}")
    common_e = set(up_first) & set(down_first)
    print(f"  E-type UP∩DOWN coincident events   : {len(common_e):,}   <-- feeds tof_profile")

def main():
    base_dir = Path(__file__).resolve().parent
    modules = ["radi_cal_energy", "radi_cal_triple", "rc_hex", "rc_hex_triple"]

    for mod in modules:
        mod_path = base_dir / mod / "runs" / mod
        print(f"\n{'#'*70}")
        print(f"# MODULE: {mod}")
        print(f"{'#'*70}")
        if not mod_path.exists():
            print(f"  [!] Path does not exist: {mod_path}")
            continue
        sweeps = sorted(list(mod_path.glob("sweep_*")), key=lambda p: p.name)
        if not sweeps:
            print(f"  [!] No sweep_* directories found under {mod_path}")
            continue
        print(f"  All sweeps found: {[s.name for s in sweeps]}")
        target_sweep = sweeps[-1]
        print(f"  -> Selected (latest): {target_sweep.name}")

        energy_dirs = sorted(list(target_sweep.glob("*GeV")), key=lambda p: extract_numerical_energy(p.name))
        print(f"  Energy dirs in that sweep: {[e.name for e in energy_dirs]}")
        is_hex = "hex" in mod

        for edir in energy_dirs:
            diagnose(edir, is_hex, f"{mod} / {edir.name}")

if __name__ == "__main__":
    main()