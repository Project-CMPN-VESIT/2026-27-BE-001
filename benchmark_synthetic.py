"""
benchmark_synthetic.py — Statistically-powered comparison of the production
causal engine vs. the standalone Granger-VAR baseline over MANY synthetic
scenarios with KNOWN ground truth (unlike compare_baselines.py, which only
had n=1 real scene with no ground truth and n=1 synthetic case).

Two scenario classes, randomized per trial (different seed each time):

  "collision" (positive class) — a rear-end collision with a KNOWN correct
    cause pair (V_LEAD, V_FOLLOW) and a KNOWN irrelevant distractor
    (V_OTHER) in a separate lane, physics randomized (speeds, braking
    strength, reaction lag, gap, noise) around the values validated in
    test_causal.py's fixed scenario.

  "null" (negative class) — 2-4 mutually independent vehicles in separate
    lanes >20m apart (this project's causal.interaction_distance_m), each at
    a roughly constant noisy speed. By construction there is NO causal
    relationship between any of them. This tests the false-positive /
    false-alarm rate on genuinely benign traffic — the project's own
    documented claim ("on free-flow traffic, the correct result is no
    causal chain at all") is exactly what this class checks.

What is scored, honestly:

  Production engine, collision class:
    - self_detected: did _find_primary_event correctly resolve to the
      CONTACT/COLLISION between V_LEAD and V_FOLLOW (its own incident
      detection, something Granger alone cannot do at all)?
    - attribution_correct: GIVEN that, does root_cause.primary_cause
      implicate only {V_LEAD, V_FOLLOW} and never V_OTHER?
    - end_to_end_correct = self_detected AND attribution_correct.

  Production engine, null class:
    - false_alarm: did a primary_event fire at all, despite nothing
      collision-like happening anywhere in the scene?

  Granger-VAR (FDR-corrected), collision class:
    - Granger has no incident-detection step of its own, so — to make this
      a fair test of attribution rather than penalizing it for a capability
      it was never designed to have — it is handed the TRUE cause pair
      directly (ground truth, not the engine's guess) and asked: is there a
      significant edge between V_LEAD and V_FOLLOW, and is V_OTHER not the
      top (lowest p-value) competing explanation?

  Granger-VAR, null class:
    - false_alarm: did ANY FDR-significant edge appear anywhere in the
      scene, despite every vehicle being independent by construction?

Usage: python benchmark_synthetic.py [n_trials]   (default 1500, split evenly)
Prints progress every 100 trials (this takes a while — the production engine
runs its full 4-method statistical stack + rule-based reasoning per trial).
"""

import shutil
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # statsmodels deprecation noise; not the point of this run

from app.config import settings
from app.pipeline.causal import get_causal_engine, _load_event_df, _fps_from_df
from app.pipeline.causal_kinematics import enrich_kinematics
from app.pipeline.baseline_granger import run_granger_baseline, rank_granger_root_cause

EVENT_ID = "EVT_BENCHMARK_SYNTH"


def _rows_for(event_id: str, oid: str, t_arr: np.ndarray, x: float, y_arr: np.ndarray, v_arr: np.ndarray) -> List[Dict]:
    return [
        dict(Event_ID=event_id, Timestamp=round(t_arr[i], 2), Frame_ID=i, Object_ID=oid,
             Class="car", BBox_X1=0, BBox_Y1=0, BBox_X2=10, BBox_Y2=10,
             Pos_X_m=x, Pos_Y_m=float(y_arr[i]), Velocity_mps=float(v_arr[i]))
        for i in range(len(t_arr))
    ]


def _make_collision_scenario(rng: np.random.RandomState) -> Tuple[List[Dict], Dict]:
    T, dt = 90, 0.1
    t_arr = np.array([-4.0 + i * dt for i in range(T)])

    lead_v0 = rng.uniform(12.0, 19.0)
    lead_decel = rng.uniform(8.0, 12.0)
    reaction_lag = rng.uniform(0.4, 0.65)
    follow_decel = rng.uniform(lead_decel * 0.65, lead_decel * 0.85)
    gap0 = rng.uniform(7.5, 9.5)
    noise = rng.uniform(0.02, 0.05)

    lead_vmin = max(2.0, lead_v0 - lead_decel * 1.2)

    def v_lead(t: float) -> float:
        if t < 0.0:
            return lead_v0
        if t < 1.2:
            return lead_v0 - lead_decel * t
        return lead_vmin

    after_brake_speed = max(3.0, lead_v0 - follow_decel * 1.0)

    def v_follow(t: float) -> float:
        if t < reaction_lag:
            return lead_v0
        if t < reaction_lag + 1.0:
            return lead_v0 - follow_decel * (t - reaction_lag)
        if t < reaction_lag + 1.3:
            frac = (t - (reaction_lag + 1.0)) / 0.3
            return after_brake_speed - (after_brake_speed - after_brake_speed * 0.3) * frac
        return after_brake_speed * 0.3

    v_lead_arr = np.array([v_lead(t) for t in t_arr]) + rng.randn(T) * noise
    v_follow_arr = np.array([v_follow(t) for t in t_arr]) + rng.randn(T) * noise
    y_lead = -42.0 + np.cumsum(v_lead_arr) * dt
    y_follow = (-42.0 - gap0) + np.cumsum(v_follow_arr) * dt

    other_v0 = rng.uniform(8.0, 18.0)
    other_noise = rng.uniform(0.1, 0.4)
    v_other = other_v0 + rng.randn(T) * other_noise
    y_other = -54.0 + np.cumsum(v_other) * dt

    rows = (
        _rows_for(EVENT_ID, "V_LEAD", t_arr, 1.75, y_lead, v_lead_arr)
        + _rows_for(EVENT_ID, "V_FOLLOW", t_arr, 1.75, y_follow, v_follow_arr)
        + _rows_for(EVENT_ID, "V_OTHER", t_arr, 25.0, y_other, v_other)
    )
    # Ground truth severity is derived from what the physics actually produced
    # (same-lane => distance ~= |y_lead - y_follow|), not assumed from the
    # random parameters — a random draw can land on a near-miss instead of
    # real contact, and treating that as a mislabeled "collision" would
    # corrupt the benchmark rather than reflect an engine failure.
    min_gap = float(np.min(np.abs(y_lead - y_follow)))
    if min_gap <= 3.0:
        severity = "contact"
    elif min_gap <= 6.0:
        severity = "near_collision"
    else:
        severity = "no_contact"
    ground_truth = {
        "type": "collision", "cause_pair": {"V_LEAD", "V_FOLLOW"}, "distractor": "V_OTHER",
        "min_gap_m": round(min_gap, 3), "severity": severity,
    }
    return rows, ground_truth


def _make_null_scenario(rng: np.random.RandomState) -> Tuple[List[Dict], Dict]:
    """
    Exactly 2 mutually independent vehicles, deliberately not "2-4" — the
    projection's reliable-region gate (settings.projection.max_abs_lateral_m
    = 25.0, max_depth_m = +-100.0; anything beyond is silently NaN'd out
    inside enrich_kinematics) leaves very little safe lateral room to fit
    more than two lanes >causal.interaction_distance_m (20m) apart from each
    other. -15m / +15m gives 30m separation, safely inside +-25m; the depth
    window and speed range are kept modest enough that 90 frames of travel
    stays well inside +-100m too. An earlier version of this generator used
    4 far-apart lanes (25/48/71m) without checking this bound — 3 of 4
    vehicles were silently NaN'd out, which would have corrupted every null
    trial (only ever 2 valid entities to test) had it gone unnoticed.
    """
    T, dt = 90, 0.1
    t_arr = np.array([-4.0 + i * dt for i in range(T)])

    rows: List[Dict] = []
    for k, x in enumerate((-15.0, 15.0)):
        v0 = rng.uniform(6.0, 13.0)
        noise = rng.uniform(0.02, 0.05)
        v_arr = v0 + rng.randn(T) * noise
        y0 = -70.0 + rng.uniform(-5.0, 5.0)
        y_arr = y0 + np.cumsum(v_arr) * dt
        rows.extend(_rows_for(EVENT_ID, f"V_{k}", t_arr, x, y_arr, v_arr))

    return rows, {"type": "null", "cause_pair": None, "distractor": None}


def _write_csv(rows: List[Dict]) -> None:
    d = settings.paths.dataset_dir / EVENT_ID
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / f"{EVENT_ID}_causal_data.csv", index=False)


def _run_trial(seed: int, scenario_type: str) -> Dict:
    rng = np.random.RandomState(seed)
    if scenario_type == "collision":
        rows, gt = _make_collision_scenario(rng)
    else:
        rows, gt = _make_null_scenario(rng)
    _write_csv(rows)

    result = {"seed": seed, "type": scenario_type}
    try:
        df_raw = _load_event_df(EVENT_ID)
        fps = _fps_from_df(df_raw)
        entity_series = enrich_kinematics(df_raw, smooth_window=settings.causal.smooth_window)

        t0 = time.perf_counter()
        engine_res = get_causal_engine().analyze_event(EVENT_ID)
        result["engine_time"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        granger_res = run_granger_baseline(entity_series, fps=fps)
        result["granger_time"] = time.perf_counter() - t0

        primary = engine_res.get("primary_event") or {}
        primary_type = primary.get("event_type")
        primary_ids = set(primary.get("object_ids") or [])
        root_cause = engine_res.get("root_cause") or {}
        pc = root_cause.get("primary_cause") or {}
        alts = root_cause.get("alternative_causes") or []
        all_cause_ids = set(pc.get("object_ids") or [])
        for a in alts:
            all_cause_ids |= set(a.get("object_ids") or [])

        if scenario_type == "collision":
            cause_pair = gt["cause_pair"]
            distractor = gt["distractor"]
            result["severity"] = gt["severity"]
            result["min_gap_m"] = gt["min_gap_m"]

            # Ground truth for "what type of primary event should fire" comes
            # from the actually-simulated physics (gt["severity"]), not from
            # having intended a collision — a random draw can land on a
            # near-miss instead, and that's not an engine failure.
            expected_types = {
                "contact": ("COLLISION", "CONTACT"),
                "near_collision": ("NEAR_COLLISION",),
                "no_contact": (),
            }[gt["severity"]]
            self_detected = bool(expected_types) and primary_type in expected_types and primary_ids == cause_pair
            attribution_correct = bool(all_cause_ids) and all_cause_ids <= cause_pair and distractor not in all_cause_ids
            result["engine_self_detected"] = self_detected
            result["engine_attribution_correct"] = attribution_correct
            result["engine_end_to_end_correct"] = self_detected and attribution_correct

            granger_root = rank_granger_root_cause(granger_res, list(cause_pair))
            gpc = granger_root.get("primary_cause")
            granger_correct = bool(gpc) and {gpc["source"], gpc["target"]} == cause_pair and distractor not in (gpc["source"], gpc["target"])
            result["granger_attribution_correct"] = granger_correct
        else:
            result["engine_false_alarm"] = primary_type is not None
            result["granger_false_alarm"] = len(granger_res.get("edges", [])) > 0

    except Exception as exc:
        result["error"] = str(exc)
    finally:
        shutil.rmtree(settings.paths.dataset_dir / EVENT_ID, ignore_errors=True)

    return result


def main() -> int:
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    n_collision = n_total // 2
    n_null = n_total - n_collision

    trials = [(1_000_000 + i, "collision") for i in range(n_collision)] + \
             [(2_000_000 + i, "null") for i in range(n_null)]

    results: List[Dict] = []
    t_start = time.perf_counter()
    for i, (seed, scenario_type) in enumerate(trials):
        results.append(_run_trial(seed, scenario_type))
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (len(trials) - (i + 1)) / rate
            print(f"  progress: {i + 1}/{len(trials)}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    errors = [r for r in results if "error" in r]
    collision = [r for r in results if r["type"] == "collision" and "error" not in r]
    null = [r for r in results if r["type"] == "null" and "error" not in r]

    print(f"\n{'=' * 72}\nBENCHMARK RESULTS  (n={len(results)}, {len(collision)} collision / {len(null)} null, {len(errors)} errors)\n{'=' * 72}")

    if errors:
        print(f"\n  {len(errors)} trials errored (showing first 3): ")
        for e in errors[:3]:
            print(f"    seed={e['seed']} type={e['type']}: {e['error']}")

    if collision:
        n = len(collision)
        by_sev = {"contact": 0, "near_collision": 0, "no_contact": 0}
        for r in collision:
            by_sev[r["severity"]] += 1
        print(f"\n--- COLLISION class (n={n}) — true cause=V_LEAD/V_FOLLOW, distractor=V_OTHER ---")
        print(f"  Actual simulated severity (ground truth, derived from physics, not intent):")
        print(f"    contact (<=3m)         : {by_sev['contact']}/{n}  ({100*by_sev['contact']/n:.1f}%)")
        print(f"    near_collision (3-6m)  : {by_sev['near_collision']}/{n}  ({100*by_sev['near_collision']/n:.1f}%)")
        print(f"    no_contact (>6m)       : {by_sev['no_contact']}/{n}  ({100*by_sev['no_contact']/n:.1f}%)  <- excluded from detection scoring below (nothing to detect)")

        scoreable = [r for r in collision if r["severity"] != "no_contact"]
        ns = len(scoreable) or 1
        self_det = sum(r["engine_self_detected"] for r in scoreable)
        attrib = sum(r["engine_attribution_correct"] for r in scoreable)
        e2e = sum(r["engine_end_to_end_correct"] for r in scoreable)
        granger_ok = sum(r["granger_attribution_correct"] for r in scoreable)
        print(f"\n  Scored against the {ns} trials where a real incident actually occurred:")
        print(f"  Engine  self-detected the correct incident      : {self_det}/{ns}  ({100*self_det/ns:.1f}%)")
        print(f"  Engine  attribution correct (given detection)   : {attrib}/{ns}  ({100*attrib/ns:.1f}%)")
        print(f"  Engine  END-TO-END correct                      : {e2e}/{ns}  ({100*e2e/ns:.1f}%)")
        print(f"  Granger attribution correct (true pair given)   : {granger_ok}/{ns}  ({100*granger_ok/ns:.1f}%)")
        print(f"  Engine  mean runtime  : {np.mean([r['engine_time'] for r in collision]):.3f}s")
        print(f"  Granger mean runtime  : {np.mean([r['granger_time'] for r in collision]):.3f}s")

    if null:
        n = len(null)
        engine_fa = sum(r["engine_false_alarm"] for r in null)
        granger_fa = sum(r["granger_false_alarm"] for r in null)
        print(f"\n--- NULL class (n={n}) — mutually independent vehicles, no true causal link ---")
        print(f"  Engine  false-alarm rate  (hallucinated an incident) : {engine_fa}/{n}  ({100*engine_fa/n:.1f}%)")
        print(f"  Granger false-alarm rate  (any FDR-significant edge) : {granger_fa}/{n}  ({100*granger_fa/n:.1f}%)")
        print(f"  Engine  mean runtime  : {np.mean([r['engine_time'] for r in null]):.3f}s")
        print(f"  Granger mean runtime  : {np.mean([r['granger_time'] for r in null]):.3f}s")

    total_time = time.perf_counter() - t_start
    print(f"\nTotal wall-clock time: {total_time:.0f}s ({total_time/60:.1f} min) for {len(results)} trials")
    return 0


if __name__ == "__main__":
    sys.exit(main())
