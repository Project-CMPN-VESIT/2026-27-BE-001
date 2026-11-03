# Causal Engine (Track 2) — High-Level Flow

## What It Does

Given a traffic incident clip, it identifies the primary incident and asks: **"Why did this happen — and what else might explain it?"** It combines a statistical causal-discovery ensemble (PCMCI+/LiNGAM/GES/CausalForest) with a rule-based event-chain reasoner, cross-corroborating the two, and outputs a **ranked** primary cause plus genuinely distinct alternative causes (never one answer with the alternatives thrown away).

---

## The Flow

```
┌─────────────────────────────────────────────────────────┐
│              INPUT: CAUSAL CSV                          │
│  Per-object kinematics from Phase 2 (perception):       │
│  Position (x,y), Velocity, Class, BBox per frame         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1: ENRICH KINEMATICS + PAIRWISE FEATURES           │
│  vx, vy, speed, ax, ay, acceleration, heading per entity  │
│  distance, closing_speed, relative_speed per close pair   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2: BUILD VARIABLES + var_to_entities                │
│  Wide frame x variable matrix, capped at 60 vars           │
│  var_to_entities maps every column back to its object id(s)│
│  at construction time (never reverse-parsed later)         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: STATISTICAL CAUSAL DISCOVERY                     │
│  PCMCI+, LiNGAM, GES, CausalForest, each independently     │
│  → merged into consensus_edges (method-agreement weighted) │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 4: EXTRACT TEMPORAL EVENTS                           │
│  Per entity: SUDDEN_BRAKING, SUDDEN_ACCELERATION, FALL, …   │
│  Per pair: CLOSING_DISTANCE, CONTACT, NEAR_COLLISION,        │
│            COLLISION (contact + impact-signature check)     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 5: BUILD EVENT CAUSAL CHAIN (rule-based + corroborated)│
│  For every temporally-ordered event pair, accumulate conf:  │
│   + shared entities, known event-type patterns, Granger check│
│   + STATISTICAL CORROBORATION: does a consensus_edge          │
│     (via var_to_entities) independently agree? (+bonus)      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 6: IDENTIFY PRIMARY EVENT (what happened)              │
│  Highest-priority incident: COLLISION > CONTACT >             │
│  NEAR_COLLISION > FALL > SUDDEN_BRAKING                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 7: RANK ROOT CAUSES (why it happened)                  │
│  Walk causal predecessors of the primary event (direct +      │
│  one hop back); dedupe so each survivor is a genuinely         │
│  distinct reason (different event type or entities);           │
│  keep top-N as primary + alternative_causes, each with its     │
│  own confidence + corroborated_by. Below the confidence         │
│  floor: "no corroborated precipitating cause" — not a guess.   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 8: PRUNE + BUILD RELATIONSHIP GRAPH                     │
│  Prune to depth-2 neighborhood of primary event AND EVERY      │
│  ranked cause (alternates' evidence is protected, not deleted).│
│  relationship_graph = entity_entity_statistical edges +        │
│  event_event_causal edges + entity_event_involvement edges     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 9: BUILD SES (STATE-EVENT-STATE) NARRATIVE               │
│  Deterministically re-tell primary_event + root_cause.         │
│  primary_cause as one linear chain:                             │
│    S0 (lead-up) → [cause event] → S1 (transition) →            │
│    [primary event] → S2 (aftermath)                             │
│  Each state's label/summary comes from entity_series/           │
│  pair_series in that window (speed trend, gap, closing/         │
│  opening) — no new evidence, no LLM. alternative_causes become  │
│  "branches" off S0, not parallel chains.                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         OUTPUT: causal_graph.json                        │
│  - primary_event (what happened)                          │
│  - root_cause: {primary_cause, alternative_causes}         │
│  - relationship_graph (nodes + 3 edge kinds)                │
│  - ses_narrative: {chain, branches, narrative_text}          │
│  - consensus_edges, temporal_events, explanation            │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **Speed-primary** (no raw acceleration as a headline variable) | Monocular-BEV acceleration is too noisy to trust |
| **`var_to_entities` built at construction, not reverse-parsed** | Object ids like `V_01` make string-splitting a column name like `V_01_V_02_distance` ambiguous |
| **Statistical corroboration is a bonus, not a gate** | The statistical engine only sees a capped ≤60-variable subset and can legitimately miss a real link the rule-based reasoner still finds |
| **Root causes are ranked, never picked-and-discarded** | A single fixed-priority pick threw away every alternative explanation; ranking + dedup surfaces genuinely distinct hypotheses instead |
| **Pruning protects every ranked cause, not just the primary event** | Otherwise alternate-cause evidence was deleted from the persisted graph before the frontend ever saw it |
| **SES narrates, never discovers** | `causal_narrative.py` adds zero new evidence — it's a deterministic re-telling of `root_cause`, so it can never disagree with the ranking that produced it |

---

## What This Replaces

Earlier revisions of this engine had a separate deterministic N1–N5 narrative/scene-role layer (`scene.py`/`stage.py`/`rootcause.py` — entities with roles like initiator/rider/pillion, five *fixed* staged windows, typed relations). That layer was removed along with the old per-target PCMCI+ engine and was not rebuilt as-is — the old schema doesn't match this engine's output.

A narrative layer was rebuilt fresh against the current schema instead: `app/pipeline/causal_narrative.py`'s SES (State-Event-State) chain. Unlike the old N1–N5 template, it isn't a fixed number of stages forced onto every clip — it's a 2-event/3-state shape (lead-up → cause → transition → primary → aftermath) derived from whatever `root_cause.primary_cause` already found, collapsing to a 3-node `S0 → primary → S2` chain when no distinct precipitating cause exists. It also doesn't attempt entity *roles* (initiator/rider/pillion) — it stays state/event/state, not a full incident-actor model.

---

## Caveats

- ~100-timestep clips are short for causal discovery; links are **ranked hypotheses**, not proof.
- On free-flow traffic, the correct result is *no* causal chain at all — an unattributed or absent root cause is a correct answer, not a failure.
- `test_causal.py` is the regression check: a synthetic rear-end collision with a known lead/follower mechanism and an unrelated distractor vehicle, asserting the primary event, ranked root cause, distractor exclusion, and the SES chain's shape are all correct.
- The SES layer only narrates the *primary* causal path; alternates surface as `branches`, not full parallel chains — a linear story can't represent several competing hypotheses cleanly.
