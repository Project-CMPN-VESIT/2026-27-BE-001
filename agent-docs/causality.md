# Track 2: Causal Engine — How It Works

## Overview

The Causal Engine identifies **what happened** (the primary incident), **why** (a ranked, confidence-scored list of candidate causes — not just one), and **the story** (a deterministic State-Event-State narrative), by layering three things that build on each other rather than three independent opinions:

- A **multi-method statistical engine** (PCMCI+/LiNGAM/GES/CausalForest) over kinematic variables, merged into a consensus.
- A **rule-based event-chain reasoner** that extracts discrete events (braking, closing distance, contact/collision, …) and links temporally-ordered ones into a causal chain — boosted by statistical corroboration instead of reasoning about the CSV alone.
- A **SES (State-Event-State) narrative layer** that re-tells the ranked root cause as a linear causal story — not a fourth independent opinion, a deterministic narration of what the first two already found.

**Source:** `app/pipeline/causal.py`, `app/pipeline/causal_discovery.py`, `app/pipeline/causal_events.py`, `app/pipeline/causal_narrative.py`
**Routes:** `app/routes/causal_routes.py`

---

## Pipeline

1. **Load** the event's causal CSV (`{event_id}_causal_data.csv`) and enrich kinematics (`causal_kinematics.enrich_kinematics`).
2. **Build pairwise features** (distance, closing speed, relative speed/acceleration) for entities within `causal.interaction_distance_m`.
3. **Build the wide variable matrix** — also returns `var_to_entities`, mapping every variable column back to the object id(s) it describes (built once, at construction time; never reverse-parsed from the column name string later, which would be ambiguous for ids like `V_01`).
4. **Select ≤60 variables** (`_select_stat_variables`) prioritizing entities with the largest speed drop and the closest pairs.
5. **Run PCMCI+, LiNGAM, GES, CausalForest** and merge into `consensus_edges` (`build_consensus`): `final_confidence = 0.6×support_ratio + 0.4×avg_strength_norm`.
6. **Extract temporal events** per entity/pair (SUDDEN_BRAKING, CLOSING_DISTANCE, CONTACT/COLLISION, FALL, …) and **build the event causal chain** (`build_event_causal_chain`) — for every temporally-ordered event pair, accumulate confidence from shared entities, known event-type patterns, a lightweight Granger check, pairwise closing-speed evidence, **and now a statistical-corroboration bonus** (`causal.statistical_corroboration_bonus`) whenever `consensus_edges` independently found a link between the same two entities via `var_to_entities`.
7. **Localize event onset** (earliest physically-supported frame) and **identify the primary event** (`_find_primary_event`) — the highest-priority incident (COLLISION > CONTACT > NEAR_COLLISION > FALL > SUDDEN_BRAKING).
8. **Rank root causes** (`_rank_root_causes`) — walk the primary event's causal predecessors (direct + one hop back), dedupe so each survivor is a genuinely distinct reason (different event type or entities, never a restatement), keep the top `causal.max_alternative_causes` as alternates. Below `causal.min_root_cause_confidence`, reports "no corroborated precipitating cause" instead of guessing.
9. **Prune** the event graph to the depth-2 neighborhood of the primary event *and every ranked cause* (not just one), so alternate-cause evidence survives.
10. **Build the relationship graph** (`_build_relationship_graph`) — a small, ready-to-render "hypothesis graph" scoped to just the primary event and its ≤5 ranked causes (`root_cause.primary_cause` + `alternative_causes`) and the entities they involve — deliberately *not* the full depth-2-pruned event chain from step 9, which on a busy scene (many tracked entities) can still be hundreds of events and thousands of edges and renders as an unreadable hairball. Combines `entity_entity_statistical` (consensus edges resolved to real object ids, restricted to entities in the hypothesis set), `event_event_causal` (one edge per ranked cause straight into the primary event, using the same confidence shown in the root-cause panel), and `entity_event_involvement`.
11. **Build the SES narrative** (`causal_narrative.build_ses_narrative`) — re-tells `primary_event` + `root_cause.primary_cause` as a linear chain `S0 (lead-up) → [cause event] → S1 (transition) → [primary event] → S2 (aftermath)`. Each state's label/summary is derived deterministically from `entity_series`/`pair_series` in that time window (per-entity mean speed + speed-trend slope vs. `causal.ses_trend_threshold_mps2`, plus the tightest pair's min-distance/closing-or-opening). `alternative_causes` become `branches` — divergent-hypothesis footnotes off `S0` — instead of parallel chains, since one linear story can't represent several competing hypotheses at once. No LLM; same inputs always produce the same chain.
12. **Persist** to `dataset/{event_id}/causal_graph.json`.

Runs **automatically** at the end of Phase 2/3 for every event — batch pipeline (`_run_pipeline` in `app/routes/events.py`) and live feeds (`app/pipeline/monitor.py`) — non-fatally. `POST /api/causal/analyze/{event_id}` re-runs it manually.

---

## Key Design Decisions

- **Speed-primary by design.** Monocular-BEV acceleration is too noisy; variables are speeds and gaps.
- **The two causal subsystems corroborate, not just coexist.** Statistical consensus edges feed a confidence bonus into the rule-based event chain via `var_to_entities` — previously these were fully disconnected.
- **Ranked causes, never pick-one-and-discard.** `root_cause.primary_cause` + `alternative_causes` are all kept (and protected from pruning), each with its own confidence and `corroborated_by`.
- **SES narrates, it doesn't discover.** `causal_narrative.py` adds zero new evidence — it's a deterministic re-telling of `primary_event`/`root_cause` as a linear story, so it can never disagree with the ranking that produced it, and the same causal graph always narrates to the same chain.
- **Automatic, non-fatal.** Runs at the end of Phase 2 for every event; failure never fails the event — it stays "Extracted" and can be re-analyzed manually.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/causal/analyze/{event_id}` | Run the full engine on the event |
| `GET` | `/api/causal/{event_id}` | Fetch persisted causal graph |

---

## Output Schema (`causal_graph.json`, abridged)

```json
{
  "status": "ok",
  "event_id": "EVT_...",
  "primary_event": {
    "event_type": "COLLISION",
    "object_ids": ["V_01", "V_02"],
    "event_onset": {"frame": 100, "timestamp": 0.0, "reason": "contact_threshold_crossed"},
    "confidence": 0.9
  },
  "root_cause": {
    "primary_cause": {
      "label": "Closing Distance involving V_01, V_02",
      "cause_event_id": "EVT_PAIR_V_01_V_02_001",
      "object_ids": ["V_01", "V_02"],
      "mechanism": "Preceded the incident by 1.1s. Evidence: shared_entities:V_01,V_02; event_type_pattern:CLOSING_DISTANCE→COLLISION.",
      "confidence": 0.92,
      "corroborated_by": ["PCMCI+"],
      "lag_seconds": 1.1
    },
    "alternative_causes": [
      {"label": "Sudden Braking involving V_01", "confidence": 0.78, "corroborated_by": [], "...": "..."}
    ]
  },
  "relationship_graph": {
    "nodes": [{"id": "entity_V_01", "kind": "entity", "...": "..."}, {"id": "EVT_...", "kind": "event", "...": "..."}],
    "edges": [
      {"source": "entity_V_01", "target": "entity_V_02", "kind": "entity_entity_statistical", "relationship": "causes", "confidence": 0.81},
      {"source": "EVT_A", "target": "EVT_B", "kind": "event_event_causal", "relationship": "causes", "confidence": 0.9},
      {"source": "entity_V_01", "target": "EVT_A", "kind": "entity_event_involvement", "relationship": "involves", "confidence": 0.85}
    ]
  },
  "ses_narrative": {
    "chain": [
      {"kind": "state", "id": "S0", "stage": "lead_up", "label": "Closing distance",
       "window_s": [-4.1, -1.1], "entities": {"V_01": {"mean_speed_kmh": 28.0, "trend": "steady"}},
       "pair": {"object_ids": ["V_01", "V_02"], "min_distance_m": 9.4, "closing": true},
       "summary": "V_01 steady at ~28 km/h; gap V_01↔V_02: 9.4 m (closing)"},
      {"kind": "event", "id": "EVT_PAIR_V_01_V_02_001", "role": "cause", "event_type": "CLOSING_DISTANCE",
       "label": "Closing Distance", "object_ids": ["V_01", "V_02"], "timestamp": -1.1,
       "confidence": 0.92, "corroborated_by": ["PCMCI+"]},
      {"kind": "state", "id": "S1", "stage": "transition", "...": "..."},
      {"kind": "event", "id": "EVT_...", "role": "primary", "label": "Collision", "...": "..."},
      {"kind": "state", "id": "S2", "stage": "aftermath", "...": "..."}
    ],
    "branches": [
      {"label": "Sudden Braking involving V_01", "confidence": 0.78, "mechanism": "...", "diverges_at": "S0"}
    ],
    "method": "deterministic_state_event_state",
    "narrative_text": "Closing distance (t=-4.1s to -1.1s): ... → Closing Distance involving V_01, V_02 at t=-1.1s (confidence 92%, corroborated by PCMCI+) → ... → Collision involving V_01, V_02 at t=0.0s (confidence 90%) → ... . Other plausible precipitating factors considered but ranked lower: Sudden Braking involving V_01 (78%)."
  },
  "consensus_edges": ["..."],
  "temporal_events": ["..."],
  "explanation": "...",
  "confidence": 0.75,
  "note": "Multi-method causal engine (PCMCI+, LiNGAM, GES, CausalForest) narrated as a deterministic State-Event-State (SES) chain. Statistical results are ranked hypotheses, not causal proof."
}
```

---

## Caveats

- ~100-timestep clips are short for causal discovery; links are **ranked hypotheses**, not proof.
- On free-flow traffic, the correct result is *no* causal chain at all (only autoregression / an unattributed primary event).
- Statistical corroboration is a confidence **bonus**, not a requirement — the statistical engine only sees a capped subset of variables and can legitimately miss a real link that the rule-based reasoner still finds via direct kinematic evidence.
- The `ses_narrative` layer is a fixed 2-event/3-state shape (lead-up → cause → transition → primary → aftermath), not the old removed `scene.py`/`stage.py`/`rootcause.py`'s rigid N1–N5 template — it's derived from whatever `root_cause.primary_cause` already found rather than forcing a fixed number of stages onto every clip. It only narrates the *primary* causal path; the other ranked alternates appear as `branches`, not full parallel chains — a linear story can't represent several competing hypotheses cleanly.
- When `root_cause.primary_cause.cause_event_id` collapses to the primary event itself (the "no corroborated precipitating cause" case), the SES chain has no `cause` event — it's just `S0 → primary → S2`, three nodes instead of five.
