"""
Sweeps false-positive rate vs. number of independent entities N, for both
the Granger-VAR baseline and the production engine, on genuinely independent
("null") synthetic scenes. Every point is measured, not interpolated.

Granger-VAR: no spatial constraint (operates on speed series only), so it is
swept up to N=44 directly by generating N independent white-noise-plus-
constant speed series.

Production engine: DOES require spatially non-interacting entities (its
pairwise-feature step only looks at real BEV positions), and the reliable-
projection-region bounds (+-25m lateral, +-100m depth) physically cap how
many mutually->20m-separated ("causal.interaction_distance_m") entities fit
in the scene at once. Verified empirically below (assert zero pairwise
interactions before trusting any trial) rather than assumed -- exactly the
class of bug (silent NaN'ing / accidental interaction) that bit the earlier
benchmark_synthetic.py null-scenario generator. This caps the engine sweep
at N=12; going further would require entities closer than the geometry
allows to stay independent, which is a real physical limit, not a
convenience cutoff.
"""

import warnings
warnings.filterwarnings("ignore")

import shutil
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from app.config import settings
from app.pipeline.causal import get_causal_engine, _load_event_df, _fps_from_df
from app.pipeline.causal_kinematics import enrich_kinematics, compute_pairwise_features
from app.pipeline.baseline_granger import run_granger_baseline

EVENT_ID = "EVT_SWEEP_FPR"
T, DT = 90, 0.1
T_ARR = np.array([-4.0 + i * DT for i in range(T)])


# ── Granger-VAR sweep: pure speed series, no spatial component needed ───────

def _independent_speed_series(n: int, rng: np.random.RandomState) -> Dict[str, pd.DataFrame]:
    series = {}
    for k in range(n):
        v0 = rng.uniform(8.0, 15.0)
        noise = rng.uniform(0.03, 0.08)
        v = v0 + rng.randn(T) * noise
        series[f"V_{k}"] = pd.DataFrame({"frame_id": np.arange(T), "speed": v})
    return series


def granger_sweep(n_values: List[int]) -> List[Dict]:
    """
    Tracks TWO metrics per N, both needed to reconcile this sweep with the
    already-reported N=2 numbers elsewhere in the paper:
      - scene-level "any hit": fraction of independently-generated N-entity
        scenes containing >=1 flagged pair (this is what "false-alarm rate"
        already means in Table III at N=2 -- generalizing it to N>2 is the
        headline story: raw testing's chance of *some* spurious pair in the
        scene climbs toward 100% by the union bound as pair count grows).
      - per-pair rate: pooled fraction of individual directional tests
        flagged, averaged over many independent draws (the textbook Type-I
        error rate -- ~alpha under raw testing when every test is genuinely
        independent; included to show it does NOT depend on N, unlike the
        scene-level rate, and to explain why the N=2 scene-level number
        exceeds the naive per-pair rate: the two directions of a single
        pair share the same finite sample and are therefore correlated).
    Rep count scales down at high N (where a single scene already contains
    hundreds/thousands of pairs) so total compute stays roughly constant
    across the sweep instead of blowing up at N=44.
    """
    results = []
    for n in n_values:
        pairs_per_scene = max(n * (n - 1), 1)
        reps = int(min(40, max(6, round(1500 / pairs_per_scene))))
        raw_hits, corr_hits, total_pairs = 0, 0, 0
        raw_scene_hits, corr_scene_hits = 0, 0
        t0 = time.perf_counter()
        for r in range(reps):
            rng = np.random.RandomState(9_000_000 + n * 1000 + r)
            series = _independent_speed_series(n, rng)
            res = run_granger_baseline(series, fps=10.0)
            n_pairs = res.get("n_pairs_tested", 0)
            n_raw = len(res.get("edges_uncorrected", []))
            n_corr = len(res.get("edges", []))
            total_pairs += n_pairs
            raw_hits += n_raw
            corr_hits += n_corr
            raw_scene_hits += int(n_raw > 0)
            corr_scene_hits += int(n_corr > 0)
        results.append({
            "n": n, "reps": reps, "total_pairs": total_pairs,
            "per_pair_raw": raw_hits / max(total_pairs, 1),
            "per_pair_corrected": corr_hits / max(total_pairs, 1),
            "scene_raw": raw_scene_hits / reps,
            "scene_corrected": corr_scene_hits / reps,
        })
        print(f"  Granger  N={n:>3}  reps={reps:>2}  pairs={total_pairs:>5}  "
              f"scene_raw={100*raw_scene_hits/reps:.1f}%  scene_fdr={100*corr_scene_hits/reps:.1f}%  "
              f"per_pair_raw={100*raw_hits/max(total_pairs,1):.1f}%  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
    return results


# ── Production engine sweep: genuinely independent spatial scene ───────────

LANES = (-24.0, 1.0, 24.0)     # 3 lanes: pairwise lateral distances 25/23/48m, all safely >20m apart
                                # and all within the +-25m reliable-region bound
LANE_GAP_M = 28.0               # longitudinal spacing between same-lane vehicles
LANE_Y0 = -95.0                 # depth budget: with speed range below, last slot's travel stays inside +-100m


def _rows_for(oid: str, x: float, y: np.ndarray, v: np.ndarray) -> List[Dict]:
    return [
        dict(Event_ID=EVENT_ID, Timestamp=round(T_ARR[i], 2), Frame_ID=i, Object_ID=oid,
             Class="car", BBox_X1=0, BBox_Y1=0, BBox_X2=10, BBox_Y2=10,
             Pos_X_m=x, Pos_Y_m=float(y[i]), Velocity_mps=float(v[i]))
        for i in range(T)
    ]


def _make_independent_scene(n: int, rng: np.random.RandomState):
    """
    Same lane => same x => distance is purely |dy|. The first version of this
    generator budgeted only a 25m gap against a ~9m worst-case drift, leaving
    a ~16m worst-case residual gap -- BELOW causal.interaction_distance_m
    (20m), so same-lane vehicles could legitimately interact. That was a real
    bug in the margin arithmetic (compared the drift to zero instead of to
    the 20m threshold), and the geometry-verification step below caught it
    before it silently corrupted results. Fixed by tightening the speed
    range (so worst-case relative drift over the 9s clip is ~2.7m) and
    widening the gap to 28m, leaving a >20m residual margin.
    """
    rows = []
    n_lanes = len(LANES)
    for k in range(n):
        lane_x = LANES[k % n_lanes]
        slot = k // n_lanes
        v0 = rng.uniform(9.35, 9.65)         # narrow range -> worst-case relative drift ~2.7m over 9s
        noise = rng.uniform(0.02, 0.05)
        v = v0 + rng.randn(T) * noise
        y0 = LANE_Y0 + slot * LANE_GAP_M
        y = y0 + np.cumsum(v) * DT
        rows.extend(_rows_for(f"V_{k}", lane_x, y, v))
    return rows


def _write_csv(rows):
    d = settings.paths.dataset_dir / EVENT_ID
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / f"{EVENT_ID}_causal_data.csv", index=False)


def _verify_independent(n: int, seed: int) -> bool:
    """Confirm zero pairwise interaction actually occurs -- do not trust geometry by assumption."""
    rng = np.random.RandomState(seed)
    rows = _make_independent_scene(n, rng)
    _write_csv(rows)
    try:
        df_raw = _load_event_df(EVENT_ID)
        es = enrich_kinematics(df_raw, smooth_window=settings.causal.smooth_window)
        if len(es) != n:
            print(f"    ! only {len(es)}/{n} entities survived enrichment (bounds clamp) for N={n}")
            return False
        ps = compute_pairwise_features(es)
        if len(ps) != 0:
            print(f"    ! {len(ps)} unexpected pairwise interactions at N={n}: {list(ps.keys())}")
            return False
        return True
    finally:
        shutil.rmtree(settings.paths.dataset_dir / EVENT_ID, ignore_errors=True)


def engine_sweep(n_values: List[int], reps: int = 15) -> List[Dict]:
    results = []
    for n in n_values:
        # Verify geometry is actually safe at this N before trusting any trial.
        ok = all(_verify_independent(n, seed=8_000_000 + n * 100 + i) for i in range(3))
        if not ok:
            print(f"  Engine   N={n:>3}  SKIPPED -- geometry verification failed")
            continue

        false_alarms = 0
        t0 = time.perf_counter()
        for r in range(reps):
            rng = np.random.RandomState(8_500_000 + n * 1000 + r)
            rows = _make_independent_scene(n, rng)
            _write_csv(rows)
            try:
                res = get_causal_engine().analyze_event(EVENT_ID)
                if res.get("primary_event") is not None:
                    false_alarms += 1
            finally:
                shutil.rmtree(settings.paths.dataset_dir / EVENT_ID, ignore_errors=True)
        elapsed = time.perf_counter() - t0
        results.append({"n": n, "false_alarm_rate": false_alarms / reps, "mean_time": elapsed / reps})
        print(f"  Engine   N={n:>3}  false_alarm={100*false_alarms/reps:.1f}%  "
              f"mean_time={elapsed/reps:.3f}s  ({elapsed:.1f}s total)", flush=True)
    return results


if __name__ == "__main__":
    print("=== Granger-VAR sweep ===")
    granger_results = granger_sweep([2, 4, 8, 12, 16, 24, 32, 44])

    print("\n=== Production engine sweep (geometry-verified) ===")
    engine_results = engine_sweep([2, 4, 6, 8, 10, 12], reps=15)

    print("\n=== RESULTS (paste into figure script) ===")
    print("granger_n =", [r["n"] for r in granger_results])
    print("granger_scene_raw =", [round(r["scene_raw"], 4) for r in granger_results])
    print("granger_scene_corrected =", [round(r["scene_corrected"], 4) for r in granger_results])
    print("granger_per_pair_raw =", [round(r["per_pair_raw"], 4) for r in granger_results])
    print("granger_per_pair_corrected =", [round(r["per_pair_corrected"], 4) for r in granger_results])
    print("engine_n =", [r["n"] for r in engine_results])
    print("engine_false_alarm =", [round(r["false_alarm_rate"], 4) for r in engine_results])
