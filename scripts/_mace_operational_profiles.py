"""Prespecified, independently sampled B/C/D/F finite-speed drum maneuvers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from _mace_floor_study import fingerprint, write_csv, write_json

FAMILIES = ("B", "C", "D", "F")
PHASES = ("all", "motion", "early_settling", "late_hold", "initial_hold")
TIME = np.round(np.linspace(0., 200., 501), 8)
RECIPE = "BCDF-independent-parameter-draws-v1"
WIDE_RECIPE = "BCDF-bidirectional-plus-minus30-duration-draws-v1"


def draw_wide_commands(family, rng):
    """Independent broad stress inputs, 15-75 deg; durations keep moves feasible."""
    u = rng.uniform
    direction = int(rng.choice([-1, 1]))
    if family == "B":
        requested = [(u(24, 40), 45+direction*u(3, 30), u(30, 60))]
    elif family == "C":
        amplitude = u(3, 30)
        weights = u(.6, 1.4, 3)
        levels = 45+direction*amplitude*np.cumsum(weights/weights.sum())
        requested = list(zip([u(20, 28), u(76, 84), u(132, 140)], levels, u(24, 32, 3)))
    elif family == "D":
        level = 45+direction*u(3, 30)
        requested = [(u(20, 28), level, u(45, 60)), (u(120, 128), 45., u(45, 60))]
    elif family == "F":
        requested = [(u(20, 28), 45+direction*u(3, 30), u(28, 40)),
                     (u(76, 84), 45-direction*u(3, 30), u(40, 48)),
                     (u(140, 148), 45., u(28, 40))]
    else:
        raise ValueError(f"Unknown family: {family}")
    moves, previous, last_end = [], 45., 0.
    for start, target, duration in requested:
        start, target, duration = grid(start), round(float(target), 6), grid(duration)
        end = grid(start+duration)
        speed = (target-previous)/duration
        if start < last_end or end > 200 or not 15 <= target <= 75 or not .01 <= abs(speed) <= 1.5:
            raise ValueError("Broad recipe produced an invalid angle/rate/schedule")
        moves.append(dict(start_s=start, end_s=end, from_deg=previous,
                          to_deg=target, speed_deg_s=speed))
        previous, last_end = target, end
    return moves


def grid(value):
    return round(round(float(value)/.4)*.4, 8)


def draw_commands(family, rng):
    """Study choices around the previously proposed cases; no outcome selection."""
    u = rng.uniform
    if family == "B":
        requested = [(u(24, 56), 45+u(1, 3), u(.05, .15))]
    elif family == "C":
        levels = 45+np.cumsum(u(.5, 1.25, 3))
        requested = list(zip([u(24, 36), u(74, 86), u(130, 142)], levels, u(.08, .16, 3)))
    elif family == "D":
        level = 45+u(2, 3)
        requested = [(u(20, 32), level, (level-45)/u(40, 60)),
                     (u(116, 124), 45., (level-45)/u(40, 60))]
    elif family == "F":
        requested = list(zip([u(32, 48), u(84, 100), u(140, 152)],
                             [45+u(.25, .75), 45-u(.25, .75), 45.], u(.04, .12, 3)))
    else:
        raise ValueError(f"Unknown family: {family}")
    moves, previous, last_end = [], 45., 0.
    for start, target, rate in requested:
        start, target = grid(start), round(float(target), 6)
        duration = max(.4, grid(abs(target-previous)/rate))
        end = grid(start+duration)
        speed = (target-previous)/(end-start)
        if start < last_end or end > 200 or not .02 <= abs(speed) <= .3:
            raise ValueError("Recipe produced an overlapping or out-of-bounds maneuver")
        moves.append(dict(start_s=start, end_s=end, from_deg=previous,
                          to_deg=target, speed_deg_s=speed))
        previous, last_end = target, end
    return moves


def control(moves, time=TIME):
    t = np.asarray(time, dtype=np.float64)
    angle, velocity = np.full(t.shape, 45.), np.zeros(t.shape)
    for m in moves:
        active = t >= m["start_s"]
        angle[active] = m["from_deg"]+m["speed_deg_s"]*np.clip(
            t[active]-m["start_s"], 0, m["end_s"]-m["start_s"])
        angle[t >= m["end_s"]] = m["to_deg"]
        velocity[(t >= m["start_s"]) & (t < m["end_s"])] = m["speed_deg_s"]
    return angle, velocity


def phase_masks(moves, *, settling_seconds=20., time=TIME):
    """Left-closed/right-open motion/settling; new motion overrides settling."""
    t = np.asarray(time)
    motion, settling = np.zeros(t.shape, bool), np.zeros(t.shape, bool)
    for m in moves:
        motion |= (t >= m["start_s"]) & (t < m["end_s"])
        settling |= (t >= m["end_s"]) & (t < m["end_s"]+settling_seconds)
    settling &= ~motion
    initial = t < moves[0]["start_s"]
    late = ~(motion | settling | initial)
    return dict(all=np.ones(t.shape, bool), motion=motion, early_settling=settling,
                late_hold=late, initial_hold=initial)


def mat_table(moves):
    """Exact piecewise-linear position; repeated knots encode velocity jumps.

    Acceleration is zero on open segments; corner impulses are not modeled.
    The RABL reactor consumes angle, not the acceleration annotation column.
    """
    rows = [[0., 45., 0., 0.]]
    for m in moves:
        rows.extend([[m["start_s"], m["from_deg"], 0., 0.],
                     [m["start_s"], m["from_deg"], m["speed_deg_s"], 0.],
                     [m["end_s"], m["to_deg"], m["speed_deg_s"], 0.],
                     [m["end_s"], m["to_deg"], 0., 0.]])
    rows.append([200., moves[-1]["to_deg"], 0., 0.])
    return np.asarray(rows, dtype=np.float64)


def manifest_identity(profiles):
    return hashlib.sha256(json.dumps(profiles, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate(out: Path, *, calibration_count, test_count, seed, settling_seconds, recipe=RECIPE):
    if out.exists():
        raise FileExistsError(f"Choose a new output_dir: {out}")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
           for n in (calibration_count, test_count)):
        raise ValueError("Operational counts must be positive integers")
    if recipe not in (RECIPE, WIDE_RECIPE):
        raise ValueError("Unknown operational recipe")
    start_index = 30001 if recipe == WIDE_RECIPE else 10001
    if calibration_count+test_count > 100000-start_index:
        raise ValueError("Operational profile IDs must stay within five digits")
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    profiles, seen, index = [], set(), start_index
    out.mkdir(parents=True)
    for split, count, rng in zip(("cal", "test"), (calibration_count, test_count), rngs):
        folder = out/"profiles"/split
        folder.mkdir(parents=True)
        for i in range(count):
            family = str(rng.choice(FAMILIES))
            moves = draw_wide_commands(family, rng) if recipe == WIDE_RECIPE else draw_commands(family, rng)
            key = json.dumps(moves, sort_keys=True)
            if key in seen:
                raise ValueError("Duplicate operational profile; change the prespecified seed")
            seen.add(key)
            stem = f"drum_profile_{index:05d}"
            index += 1
            path = folder/(stem+".mat")
            savemat(path, {"profile": mat_table(moves)}, format="4")
            profiles.append(dict(id="results_"+stem, split=split, family=family, moves=moves,
                                 mat_path=str(path), mat_sha256=fingerprint(path)["sha256"]))
    manifest = dict(recipe=recipe, seed=seed, time_domain_seconds=[0, 200],
                    baseline_angle_degrees=45, settling_seconds=settling_seconds,
                    family_sampling="Independent uniform draws from B/C/D/F in both splits; no fixed quotas or outcome selection",
                    profile_count=dict(cal=calibration_count, test=test_count), profiles=profiles,
                    identity=manifest_identity(profiles),
                    phase_definition="Motion [start,end); settling [end,end+window) excluding motion; initial hold before first move; late hold is the remaining post-move time.",
                    actuation="Piecewise constant velocity; ideal instantaneous velocity changes, no finite acceleration limit. MAT acceleration column is zero between corners; corner impulses are not represented.")
    if recipe == WIDE_RECIPE:
        manifest.update(study_label="Broad operational drum study: +/-30 degrees around 45 degrees",
                        allowed_angle_degrees=[15, 75], direction_sampling="Independent equal-probability initial positive/negative direction",
                        excursion_amplitude_degrees=[3, 30], applied_speed_bounds_degrees_s=[.01, 1.5],
                        duration_sampling="Prespecified durations, not the local-study 0.3 degrees/s limit. This is a broader stress study, not validated operating commands.")
    write_json(out/"operational_manifest.json", manifest)
    summaries = []
    for p in profiles:
        masks = phase_masks(p["moves"], settling_seconds=settling_seconds)
        summaries.append(dict(profile=p["id"], split=p["split"], family=p["family"],
                              **{phase+"_points": int(mask.sum()) for phase, mask in masks.items()}))
    write_csv(out/"profile_phase_counts.csv", summaries)
    return manifest


def load_manifest(out):
    m = json.loads((out/"operational_manifest.json").read_text())
    if m["recipe"] not in (RECIPE, WIDE_RECIPE) or m["identity"] != manifest_identity(m["profiles"]):
        raise ValueError("Operational manifest identity/recipe mismatch")
    for p in m["profiles"]:
        if fingerprint(Path(p["mat_path"]))["sha256"] != p["mat_sha256"]:
            raise ValueError(f"Operational MAT changed: {p['id']}")
    return m


def plot_profiles(out, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, constrained_layout=True)
    for family, ax in zip(FAMILIES, axes.flat):
        selected = [p for p in manifest["profiles"] if p["split"] == "test" and p["family"] == family][:12]
        for p in selected:
            ax.plot(TIME, control(p["moves"])[0], alpha=.5, lw=1)
        ax.set(title=f"{family}: first {len(selected)} prespecified test variations", xlim=(0, 200),
               xlabel="Time (s)", ylabel="Applied drum angle (degrees)")
        ax.grid(alpha=.2)
    fig.suptitle("Operational inputs only; no performance-based profile selection")
    fig.savefig(out/"operational_profile_variations.png", dpi=180)
    plt.close(fig)
