"""
compare_baselines.py — Head-to-head comparison of the production causal engine
(app/pipeline/causal.py: PCMCI+/LiNGAM/GES/CausalForest + rule-based event
chain + root-cause ranking + SES narrative) against the standalone classical
Granger-causality baseline (app/pipeline/baseline_granger.py).

IMPORTANT — what this script can and cannot honestly tell you:

  - There is no labeled dataset of 1,500 videos in this repo (or anywhere in
    this project) with ground-truth causal-root annotations. This script does
    NOT produce a precision/recall/F1 accuracy benchmark, because that would
    require ground truth that does not exist. Anyone asking "which is more
    accurate on real-world traffic video at scale" cannot get a real answer
    without first building or sourcing such a labeled dataset.

  - What CAN be measured honestly without ground truth:
      1. Wall-clock runtime (a real, method-agnostic cost comparison).
      2. Whether the two methods agree on which entities are causally
         implicated in whatever the production engine calls the primary event.
      3. On the ONE case where ground truth genuinely exists in this repo —
         the synthetic rear-end collision built in test_causal.py, where the
         correct cause (V_LEAD's braking) and an irrelevant distractor
         (V_OTHER) are known by construction — whether each method gets the
         right answer and correctly ignores the distractor.

  Run: python compare_baselines.py
"""

import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from app.config import settings
from app.pipeline.causal import get_causal_engine, _load_event_df, _fps_from_df
from app.pipeline.causal_kinematics import enrich_kinematics, compute_pairwise_features
from app.pipeline.baseline_granger import run_granger_baseline, rank_granger_root_cause

import test_causal as synth  # reuses the same synthetic collision + distractor scenario


def _run_both(event_id: str, label: str) -> dict:
    df_raw = _load_event_df(event_id)
    if df_raw is None or df_raw.empty:
        print(f"[{label}] no causal CSV found — skipping")
        return {}

    fps = _fps_from_df(df_raw)
    entity_series = enrich_kinematics(df_raw, smooth_window=settings.causal.smooth_window)
    pair_series = compute_pairwise_features(entity_series)

    t0 = time.perf_counter()
    engine_result = get_causal_engine().analyze_event(event_id)
    engine_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    granger_result = run_granger_baseline(entity_series, fps=fps)
    granger_time = time.perf_counter() - t0

    primary = engine_result.get("primary_event") or {}
    primary_ids = primary.get("object_ids") or []
    granger_root = rank_granger_root_cause(granger_result, primary_ids) if primary_ids else {"primary_cause": None, "alternative_causes": []}

    engine_root = engine_result.get("root_cause") or {}
    engine_cause_ids = set((engine_root.get("primary_cause") or {}).get("object_ids") or [])
    granger_cause_ids = set()
    if granger_root.get("primary_cause"):
        granger_cause_ids = {granger_root["primary_cause"]["source"], granger_root["primary_cause"]["target"]}

    print(f"\n{'=' * 70}\n{label}  (n_entities={len(entity_series)}, n_timesteps~{len(df_raw)})\n{'=' * 70}")
    print(f"  Production engine runtime : {engine_time:.2f}s")
    print(f"  Granger baseline runtime  : {granger_time:.2f}s")
    print(f"  Production engine status  : {engine_result.get('status')}")
    print(f"  Primary event             : {primary.get('event_type')} involving {primary_ids}")
    print(f"  Engine primary cause      : {(engine_root.get('primary_cause') or {}).get('label')}")
    n_pairs = granger_result.get('n_pairs_tested', len(granger_result.get('all_tests', [])))
    n_uncorr = len(granger_result.get('edges_uncorrected', []))
    n_corr = len(granger_result.get('edges', []))
    print(f"  Granger pairs tested            : {n_pairs}")
    print(f"  Granger 'significant' (raw p<.05): {n_uncorr}  ({100*n_uncorr/max(n_pairs,1):.0f}% of pairs — expect ~5% by chance alone)")
    print(f"  Granger significant (FDR-corrected): {n_corr}")
    if granger_root.get("primary_cause"):
        gc = granger_root["primary_cause"]
        print(f"  Granger primary cause     : {gc['label']} (p={gc['p_value']}, lag={gc['lag_seconds']}s)")
    else:
        print(f"  Granger primary cause     : none found pointing into the primary event's entities")
    agree = bool(engine_cause_ids) and bool(granger_cause_ids) and not engine_cause_ids.isdisjoint(granger_cause_ids)
    print(f"  Agreement on implicated entities : {agree}  (engine={engine_cause_ids or '{}'}, granger={granger_cause_ids or '{}'})")

    return {
        "engine_time": engine_time, "granger_time": granger_time,
        "engine_cause_ids": engine_cause_ids, "granger_cause_ids": granger_cause_ids,
        "primary_ids": set(primary_ids),
    }


def _run_synthetic() -> dict:
    synth._build_synthetic_csv()
    try:
        return _run_both(synth.EVENT_ID, "SYNTHETIC (ground truth known: cause=V_LEAD, distractor=V_OTHER)")
    finally:
        shutil.rmtree(settings.paths.dataset_dir / synth.EVENT_ID, ignore_errors=True)


def main() -> int:
    results = {}

    synth_res = _run_synthetic()
    if synth_res:
        results["synthetic"] = synth_res
        correct = {"V_LEAD", "V_FOLLOW"}
        engine_correct = bool(synth_res["engine_cause_ids"]) and synth_res["engine_cause_ids"] <= correct
        granger_correct = bool(synth_res["granger_cause_ids"]) and synth_res["granger_cause_ids"] <= correct
        granger_no_distractor = "V_OTHER" not in synth_res["granger_cause_ids"]
        print(f"\n  --- ground-truth check (synthetic only) ---")
        print(f"  Engine identified the correct cause (V_LEAD/V_FOLLOW), no distractor : {engine_correct}")
        print(f"  Granger identified the correct cause (V_LEAD/V_FOLLOW)               : {granger_correct}")
        print(f"  Granger avoided the V_OTHER distractor                                : {granger_no_distractor}")

    for event_dir in sorted((settings.paths.dataset_dir).glob("EVT_*")):
        event_id = event_dir.name
        if (event_dir / f"{event_id}_causal_data.csv").exists():
            res = _run_both(event_id, f"REAL EVENT {event_id} (no ground truth — agreement/runtime only)")
            if res:
                results[event_id] = res

    print(f"\n{'=' * 70}\nSUMMARY — {len(results)} case(s) run. See header comment for what this")
    print("script can and cannot honestly claim (no accuracy benchmark without labels).")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
