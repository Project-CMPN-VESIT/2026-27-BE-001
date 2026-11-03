"""
Regression test for the Track 2 causal engine (multi-method statistical engine +
rule-based event-chain reasoner + ranked root-cause selection).

Builds a synthetic rear-end collision with KNOWN ground truth, using only
per-vehicle velocity profiles (never hand-authored positions/gaps) so every
kinematic quantity — including the impact signature — falls out physically:
  - V_LEAD cruises at 15 m/s then brakes hard (~-10 m/s^2) starting at t=0s.
  - V_FOLLOW keeps cruising for a 0.6s reaction lag, then brakes later and
    more gently than V_LEAD, so the gap closes from 9.5m to ~1.6m (contact);
    it then brakes sharply to a near-stop right at/after impact (a real
    relative-deceleration spike = impact signature, and — because it starts
    at the moment of contact, not before — correctly excluded as a *cause*
    of the collision, only as its consequence).
  - V_OTHER is an unrelated distractor in a different lane, laterally more than
    interaction_distance_m away at all times, so it never becomes a candidate
    pair and should never appear anywhere in the reasoning about this incident.

Asserts:
  - the primary event correctly resolves to the CONTACT/COLLISION between
    V_LEAD and V_FOLLOW (not the distractor);
  - root_cause.primary_cause is populated, above the confidence floor, and
    attributes only to V_LEAD/V_FOLLOW;
  - the distractor V_OTHER never appears in primary_cause or any
    alternative_cause — the actual "did we identify the right accident" check;
  - any alternative_causes are genuinely distinct from the primary cause
    (different event type or different responsible entities), proving the
    engine ranks multiple reasons instead of picking one and discarding the
    rest.

Run: python test_causal.py
"""
import shutil
import sys

import numpy as np
import pandas as pd

from app.config import settings
from app.pipeline.causal import get_causal_engine

EVENT_ID = "EVT_SYNTHCAUSAL_TEST"


def _v_lead(t: float) -> float:
    if t < 0.0:
        return 15.0
    if t < 1.2:
        return 15.0 - 10.0 * t          # ~-10 m/s^2, reaches 3.0 m/s at t=1.2s
    return 3.0


def _v_follow(t: float) -> float:
    if t < 0.6:
        return 15.0                      # 0.6s reaction lag — no response yet
    if t < 1.6:
        return 15.0 - 9.0 * (t - 0.6)    # weaker, later braking than V_LEAD
    if t < 1.9:
        return max(2.0, 6.0 - (6.0 - 2.0) * (t - 1.6) / 0.3)  # sharp post-impact stop
    return 2.0


def _build_synthetic_csv() -> None:
    np.random.seed(1)
    T, dt = 90, 0.1
    G0 = 9.5  # initial LEAD-FOLLOW gap (m); closes to ~1.6m at the tightest point
    t_arr = np.array([-4.0 + i * dt for i in range(T)])

    v_lead = np.array([_v_lead(t) for t in t_arr]) + np.random.randn(T) * 0.03
    v_follow = np.array([_v_follow(t) for t in t_arr]) + np.random.randn(T) * 0.03

    # Offsets keep world_y within the projection's reliable depth region
    # (settings.projection: +/-100m) — at highway speed over a 9s clip the
    # raw cumulative travel (~80-110m) would otherwise drift past that bound
    # and get silently clamped to NaN mid-clip.
    y_lead = -42.0 + np.cumsum(v_lead) * dt
    y_follow = (-42.0 - G0) + np.cumsum(v_follow) * dt

    # V_OTHER: separate lane, far enough laterally to never be a candidate pair.
    v_other = 12.0 + np.random.randn(T) * 0.3
    y_other = -54.0 + np.cumsum(v_other) * dt

    rows = []
    for i in range(T):
        for oid, x, y, v in [
            ("V_LEAD", 1.75, y_lead[i], v_lead[i]),
            ("V_FOLLOW", 1.75, y_follow[i], v_follow[i]),
            ("V_OTHER", 25.0, y_other[i], v_other[i]),
        ]:
            rows.append(dict(
                Event_ID=EVENT_ID, Timestamp=round(t_arr[i], 2), Frame_ID=i,
                Object_ID=oid, Class="car", BBox_X1=0, BBox_Y1=0, BBox_X2=10, BBox_Y2=10,
                Pos_X_m=x, Pos_Y_m=y, Velocity_mps=v,
            ))
    d = settings.paths.dataset_dir / EVENT_ID
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / f"{EVENT_ID}_causal_data.csv", index=False)


def main() -> int:
    _build_synthetic_csv()
    try:
        res = get_causal_engine().analyze_event(EVENT_ID)
    finally:
        shutil.rmtree(settings.paths.dataset_dir / EVENT_ID, ignore_errors=True)

    status_ok = res.get("status") == "ok"
    print("status:", res.get("status"))

    primary = res.get("primary_event") or {}
    primary_ok = (
        primary.get("event_type") in ("COLLISION", "CONTACT")
        and set(primary.get("object_ids") or []) == {"V_LEAD", "V_FOLLOW"}
    )
    print(f"\nprimary_event: {primary.get('event_type')} involving {primary.get('object_ids')} "
          f"at t={primary.get('event_onset', {}).get('timestamp')}s (confidence={primary.get('confidence')})")
    print(f"primary event correctly identifies the LEAD/FOLLOW collision : {primary_ok}")

    root_cause = res.get("root_cause") or {}
    pc = root_cause.get("primary_cause") or {}
    alts = root_cause.get("alternative_causes") or []

    floor = settings.causal.min_root_cause_confidence
    pc_ok = (
        bool(pc)
        and pc.get("confidence", 0.0) >= floor
        and set(pc.get("object_ids") or []) <= {"V_LEAD", "V_FOLLOW"}
    )
    print(f"\nroot_cause.primary_cause: {pc.get('label')} (confidence={pc.get('confidence')}, "
          f"corroborated_by={pc.get('corroborated_by')})")
    print(f"primary cause is populated, confident, and entity-correct       : {pc_ok}")

    no_distractor = "V_OTHER" not in (pc.get("object_ids") or [])
    for a in alts:
        no_distractor = no_distractor and "V_OTHER" not in (a.get("object_ids") or [])
    print(f"distractor V_OTHER never appears in primary/alternative causes  : {no_distractor}")

    def _sig(c):
        return (c.get("label", "").split(" involving ")[0], tuple(sorted(c.get("object_ids") or [])))

    pc_sig = _sig(pc) if pc else None
    alts_distinct = all(_sig(a) != pc_sig for a in alts) and len({_sig(a) for a in alts}) == len(alts)
    print(f"\nalternative_causes ({len(alts)}):")
    for a in alts:
        print(f"   {a.get('label')} (confidence={a.get('confidence')})")
    print(f"alternate causes are genuinely distinct from the primary cause   : {alts_distinct}")

    relationship_graph = res.get("relationship_graph") or {}
    graph_ok = bool(relationship_graph.get("nodes")) and bool(relationship_graph.get("edges"))
    print(f"\nrelationship_graph populated (nodes+edges)                       : {graph_ok}")

    ses = res.get("ses_narrative") or {}
    chain = ses.get("chain") or []
    kinds = [n.get("kind") for n in chain]
    # Must alternate state/event/state/... and start+end on a state.
    alternates = bool(chain) and kinds[0] == "state" and kinds[-1] == "state" and all(
        kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1)
    )
    primary_node = next((n for n in chain if n.get("kind") == "event" and n.get("role") == "primary"), None)
    cause_node = next((n for n in chain if n.get("kind") == "event" and n.get("role") == "cause"), None)
    ses_ok = (
        alternates
        and primary_node is not None
        and primary_node.get("event_type") == primary.get("event_type")
        and cause_node is not None
        and set(cause_node.get("object_ids") or []) <= {"V_LEAD", "V_FOLLOW"}
        and len(ses.get("branches") or []) == len(alts)
        and bool(ses.get("narrative_text"))
    )
    print(f"\nses_narrative: {len(chain)} chain nodes ({' → '.join(kinds)}), {len(ses.get('branches') or [])} branches")
    print(f"SES chain alternates state/event, matches primary+cause, keeps alternates as branches : {ses_ok}")

    passed = status_ok and primary_ok and pc_ok and no_distractor and alts_distinct and graph_ok and ses_ok
    print("\nRESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
