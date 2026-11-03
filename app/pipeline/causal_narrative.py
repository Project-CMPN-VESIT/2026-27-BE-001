"""
causal_narrative.py — State-Event-State (SES) narrative layer.

Turns the causal engine's already-discovered outputs — the multi-method
statistical consensus (PCMCI+/LiNGAM/GES/CausalForest) and the corroborated
rule-based event chain — into a single linear causal story:

    S0 (lead-up) --[cause event]--> S1 (transition) --[primary event]--> S2 (aftermath)

Design intent
--------------
This is deliberately NOT a fifth causal-discovery method sitting alongside
PCMCI+/LiNGAM/GES/CausalForest. It does no independent inference and adds no
new evidence. It is a narration layer: it takes `primary_event` and
`root_cause` (already computed by `CausalEngine._find_primary_event` /
`_rank_root_causes` in causal.py, which already folds in statistical
corroboration) and re-tells them as a State→Event→State chain instead of a
flat ranked list. Same inputs always produce the same chain — no LLM, no
randomness — so it composes safely with the deterministic parts of the
engine and can be handed to Track 4 (synthesis.py) as structured evidence.

Alternative causes are NOT given their own parallel chains: a strict linear
narrative can't represent several competing hypotheses at once, so they are
kept as "branches" — divergent-hypothesis footnotes off the lead-up state —
rather than forcing an artificial second story.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config import settings
from app.pipeline.causal_events import TemporalEvent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Window summarization (deterministic — reads already-computed kinematics)
# ─────────────────────────────────────────────────────────────────────────────

def _trend_label(slope_mps2: float) -> str:
    thr = settings.causal.ses_trend_threshold_mps2
    if slope_mps2 <= -thr:
        return "decelerating"
    if slope_mps2 >= thr:
        return "accelerating"
    return "steady"


def _entity_window_stats(
    entity_series: Dict[str, pd.DataFrame],
    object_ids: List[str],
    t_start: float,
    t_end: float,
) -> Dict[str, Dict]:
    """Per-entity mean speed + speed trend within [t_start, t_end]."""
    stats: Dict[str, Dict] = {}
    for oid in object_ids:
        sdf = entity_series.get(oid)
        if sdf is None or "speed" not in sdf.columns or "timestamp" not in sdf.columns:
            continue
        window = sdf[(sdf["timestamp"] >= t_start) & (sdf["timestamp"] <= t_end)]
        speeds = window["speed"].dropna().to_numpy(dtype=float)
        if speeds.size == 0:
            continue
        ts = window.loc[window["speed"].notna(), "timestamp"].to_numpy(dtype=float)
        mean_kmh = float(np.mean(speeds)) * 3.6
        if speeds.size >= 2 and (ts[-1] - ts[0]) > 1e-6:
            slope = float((speeds[-1] - speeds[0]) / (ts[-1] - ts[0]))
        else:
            slope = 0.0
        stats[oid] = {
            "mean_speed_kmh": round(mean_kmh, 1),
            "speed_trend_mps2": round(slope, 2),
            "trend": _trend_label(slope),
        }
    return stats


def _pair_window_stats(
    pair_series: Dict[Tuple[str, str], pd.DataFrame],
    object_ids: List[str],
    t_start: float,
    t_end: float,
) -> Optional[Dict]:
    """Min distance + closing/opening for the tightest interacting pair in this
    window among object_ids. With >2 entities only the last (arbitrary) pair
    found is kept — this is a narrative summary, not exhaustive evidence."""
    ids_sorted = sorted(object_ids)
    best: Optional[Dict] = None
    for i in range(len(ids_sorted)):
        for j in range(i + 1, len(ids_sorted)):
            key = (ids_sorted[i], ids_sorted[j])
            pf = pair_series.get(key)
            if pf is None or "timestamp" not in pf.columns:
                continue
            window = pf[(pf["timestamp"] >= t_start) & (pf["timestamp"] <= t_end)]
            dist = window["distance"].dropna().to_numpy(dtype=float) if "distance" in window.columns else np.array([])
            if dist.size == 0:
                continue
            min_dist = float(np.min(dist))
            closing = None
            if "closing_speed" in window.columns:
                cs = window["closing_speed"].dropna()
                if not cs.empty:
                    closing = bool(cs.mean() > 0)
            candidate = {"object_ids": [key[0], key[1]], "min_distance_m": round(min_dist, 2), "closing": closing}
            if best is None or min_dist < best["min_distance_m"]:
                best = candidate
    return best


def _state_label(trends: Dict[str, Dict], pair: Optional[Dict]) -> str:
    values = {t["trend"] for t in trends.values()}
    if pair and pair.get("closing") and pair["min_distance_m"] < settings.causal.interaction_distance_m:
        return "Closing with braking response" if "decelerating" in values else "Closing distance"
    if not values:
        return "Unresolved"
    if values == {"decelerating"}:
        return "Deceleration"
    if values == {"accelerating"}:
        return "Acceleration"
    if "decelerating" in values and "accelerating" in values:
        return "Diverging speeds"
    return "Steady state"


def _state_summary_text(trends: Dict[str, Dict], pair: Optional[Dict]) -> str:
    if not trends:
        return "no kinematic data in this window"
    parts = [f"{oid} {s['trend']} at ~{s['mean_speed_kmh']:.0f} km/h" for oid, s in trends.items()]
    text = "; ".join(parts)
    if pair:
        a, b = pair["object_ids"]
        closing_txt = "closing" if pair["closing"] else ("opening" if pair["closing"] is False else "gap unclear")
        text += f"; gap {a}↔{b}: {pair['min_distance_m']:.1f} m ({closing_txt})"
    return text


def _make_state(
    state_id: str,
    stage: str,
    object_ids: List[str],
    t_start: float,
    t_end: float,
    entity_series: Dict[str, pd.DataFrame],
    pair_series: Dict[Tuple[str, str], pd.DataFrame],
) -> Dict:
    trends = _entity_window_stats(entity_series, object_ids, t_start, t_end)
    pair = _pair_window_stats(pair_series, object_ids, t_start, t_end)
    return {
        "kind": "state",
        "id": state_id,
        "stage": stage,
        "label": _state_label(trends, pair),
        "window_s": [round(t_start, 2), round(t_end, 2)],
        "object_ids": object_ids,
        "entities": trends,
        "pair": pair,
        "summary": _state_summary_text(trends, pair),
    }


def _make_event_node(
    evt: Optional[TemporalEvent],
    fallback_event_id: str,
    fallback_object_ids: List[str],
    fallback_event_type: str,
    timestamp: float,
    confidence: float,
    role: str,
    corroborated_by: List[str],
) -> Dict:
    event_type = evt.event_type if evt else fallback_event_type
    object_ids = evt.object_ids if evt else fallback_object_ids
    return {
        "kind": "event",
        "id": evt.event_id if evt else fallback_event_id,
        "role": role,  # "cause" | "primary"
        "event_type": event_type,
        "label": event_type.replace("_", " ").title(),
        "object_ids": object_ids,
        "timestamp": round(timestamp, 2),
        "confidence": round(confidence, 4),
        "corroborated_by": corroborated_by,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Narrative text (deterministic template, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _render_text(chain: List[Dict], branches: List[Dict]) -> str:
    parts = []
    for node in chain:
        if node["kind"] == "state":
            w = node["window_s"]
            parts.append(f"{node['label']} (t={w[0]}s to {w[1]}s): {node['summary']}")
        else:
            corro = f", corroborated by {', '.join(node['corroborated_by'])}" if node["corroborated_by"] else ""
            parts.append(
                f"{node['label']} involving {', '.join(node['object_ids'])} at t={node['timestamp']}s "
                f"(confidence {node['confidence']:.0%}{corro})"
            )
    text = " → ".join(parts)
    if branches:
        alt_txt = "; ".join(f"{b['label']} ({b['confidence']:.0%})" for b in branches)
        text += f". Other plausible precipitating factors considered but ranked lower: {alt_txt}."
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_ses_narrative(
    primary_evt: Optional[Dict],
    root_cause: Optional[Dict],
    all_events: List[TemporalEvent],
    onset_map: Dict[str, Dict],
    entity_series: Dict[str, pd.DataFrame],
    pair_series: Dict[Tuple[str, str], pd.DataFrame],
) -> Optional[Dict]:
    """
    Build the primary State-Event-State chain around `primary_evt`, using
    `root_cause` for the cause step. Returns None if there's no primary event
    (nothing happened worth narrating).
    """
    if not primary_evt:
        return None

    events_by_id = {e.event_id: e for e in all_events}
    primary_id = primary_evt["event_id"]
    primary_ts = primary_evt["event_onset"]["timestamp"]
    primary_end_ts = primary_evt["confirmation"]["timestamp"]
    primary_objects = primary_evt["object_ids"]

    lead_in = settings.causal.ses_lead_in_seconds
    aftermath = settings.causal.ses_aftermath_seconds

    cause = (root_cause or {}).get("primary_cause")
    cause_id = cause.get("cause_event_id") if cause else None
    has_distinct_cause = bool(cause_id and cause_id != primary_id and cause_id in events_by_id)

    chain: List[Dict] = []

    if has_distinct_cause:
        cause_evt = events_by_id[cause_id]
        cause_onset = onset_map.get(cause_id, {})
        cause_ts = cause_onset.get("onset_timestamp", cause_evt.start_timestamp)
        cause_end_ts = cause_onset.get("confirmation", {}).get(
            "confirmation_timestamp", cause_evt.end_timestamp
        )

        chain.append(_make_state(
            "S0", "lead_up", cause_evt.object_ids, cause_ts - lead_in, cause_ts,
            entity_series, pair_series,
        ))
        chain.append(_make_event_node(
            cause_evt, cause_id, cause_evt.object_ids, cause_evt.event_type,
            cause_ts, cause.get("confidence", cause_evt.confidence), "cause",
            cause.get("corroborated_by", []) or [],
        ))
        transition_objects = sorted(set(cause_evt.object_ids) | set(primary_objects))
        chain.append(_make_state(
            "S1", "transition", transition_objects, cause_end_ts, primary_ts,
            entity_series, pair_series,
        ))
    else:
        chain.append(_make_state(
            "S0", "lead_up", primary_objects, primary_ts - lead_in, primary_ts,
            entity_series, pair_series,
        ))

    chain.append(_make_event_node(
        events_by_id.get(primary_id), primary_id, primary_objects, primary_evt["event_type"],
        primary_ts, primary_evt["confidence"], "primary", [],
    ))
    chain.append(_make_state(
        "S2", "aftermath", primary_objects, primary_end_ts, primary_end_ts + aftermath,
        entity_series, pair_series,
    ))

    branches = [
        {
            "label": alt["label"],
            "cause_event_id": alt.get("cause_event_id"),
            "confidence": alt["confidence"],
            "mechanism": alt.get("mechanism"),
            "diverges_at": "S0",
        }
        for alt in (root_cause or {}).get("alternative_causes", []) or []
    ]

    return {
        "chain": chain,
        "branches": branches,
        "method": "deterministic_state_event_state",
        "narrative_text": _render_text(chain, branches),
    }
