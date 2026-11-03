# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A video surveillance analysis platform. `context.md` is the authoritative design spec: the full system is a four-track "Traffic Intersection Causal Framework," and **all four tracks are now implemented** — Track 1 (data engineering), Track 2 (PCMCI+ causal engine), Track 3 (RAG search), and Track 4 (LLM situation-report synthesis). When touching pipeline behavior, cross-check `context.md` — the source often cites it (e.g. `Reference: context.md §4 Phase 1`), and its schemas/constants are treated as contract.

## Commands

Backend (from repo root; Windows PowerShell — activate `.venv\Scripts\activate`):
```bash
uvicorn app.main:app --reload --port 8000
```
Frontend (from `frontend/`):
```bash
npm run dev        # dev server on :3000
npm run build
npm run lint       # eslint (next lint config)
```

There is **no pytest suite**. The `test_*.py` and `debug_*.py` files at the repo root are standalone scripts run directly, and `tmp_*.py` are scratch scripts:
```bash
python test_rag.py            # verifies SigLIP + LanceDB init and text embedding
python test_causal.py         # regression test: engine identifies a known collision + ranks its correct root cause (PASS/FAIL, exit code)
python test_pipeline_api.py   # hits a running server on :8000 (health, pipeline 404, events)
python debug_siglip.py        # SigLIP loading diagnostics
python scripts/export_openvino.py # generate yolo11s_int8_openvino_model/ (INT8, default)
```

**Model / maintenance / demo scripts** (in `scripts/`):
```bash
python scripts/export_openvino.py         # build yolo11s_int8_openvino_model/ IR (INT8) — needed for the default OpenVINO backend
python scripts/dedup_lancedb.py           # one-time: collapse duplicate rows in the LanceDB entity_crops table
python scripts/ingest_clip.py "<video>" <trigger_sec>   # process a pre-curated clip window directly as an event (bypasses the 120s warmup)
python scripts/render_annotated.py <EVENT_ID>           # render an annotated demo mp4 (boxes + Object_IDs + speeds + event banner)
```

API docs at `http://localhost:8000/docs`. Dashboard at `http://localhost:3000`.

## Architecture

**Three-phase pipeline, run as a FastAPI background task.** A `POST /api/pipeline/run` with an absolute `video_path` inserts a `Processing` event row synchronously (so the frontend can poll it without a 404), then `_run_pipeline` in [app/routes/events.py](app/routes/events.py) executes the phases sequentially in the background:

1. **Phase 0 — Ingestion** ([app/pipeline/ingestion.py](app/pipeline/ingestion.py)): MOG2 background subtraction over the whole video. First 120s is a warmup that sets an adaptive foreground-ratio threshold; breaching it triggers a 10s event clip (4s pre-buffer from a rolling `deque` + 6s post-trigger), with a cooldown between events. The burst must be **confirmed by a rolling-median stop** (median motion over `stop_window_s` dropping to `stop_factor×` baseline) within `burst_reject_s`, else the burst is reverted — this rejects false starts; the confirmed **anchor** (burst frame) is exported at t=0 in the clip. Frames are JPEG-encoded in memory. **Only the first triggered event is processed** (prototype limitation). The trigger logic lives in a push-driven `FrameEventDetector` (fed one frame at a time) that `scan_for_events` drives for files and the live-feed `FeedMonitor` (see below) drives for streams.
2. **Phase 1 — Perception** ([app/pipeline/perception.py](app/pipeline/perception.py)): YOLOv11 + BoT-SORT tracking on every decoded frame, but records are kept only for downsampled target frames (source FPS → 10 FPS). BoT-SORT occasionally **merges a rider and the motorcycle into one track**; perception splits mixed-class tracks deterministically — the first class seen keeps the base ID, later classes get a `V_{id:02d}x{cls}` suffix (never mix kinematics of two objects under one ID). The inference backend is set by `yolo.backend` — it defaults to an **OpenVINO INT8 IR** (`yolo11s_int8_openvino_model/`, ~2× faster than PyTorch with best recall) and falls back to the `yolo11s.pt` PyTorch weights if the IR is absent; `device` is pinned (see gotchas). Bottom-center of each bbox is projected to bird's-eye-view meters via the homography matrix (computed dynamically on-the-fly using `src_pts` passed via API payload, or falling back to `config/homography.npy`). Velocity is computed per track. Returns a flat DataFrame + per-detection metadata + decoded frames.
3. **Phase 2 — Handoff** ([app/pipeline/handoff.py](app/pipeline/handoff.py)): drops ghost/fragment tracks observed in fewer than `interpolation.min_track_frames` frames (`_filter_short_tracks`, applied to both CSV and crops), interpolates occluded tracks (gaps ≤ `max_gap_frames` linearly interpolated, larger gaps NaN-padded to keep a uniform 100-frame grid), writes the causal CSV, encodes an archival `.mp4`, saves the single best crop per object, and updates the event row to `Extracted`.

**RAG (Track 3)** ([app/pipeline/rag.py](app/pipeline/rag.py)) is decoupled from the pipeline and lazy-initialized (weights load on first use, not import). `POST /api/rag/ingest/{event_id}` embeds that event's entity crops via SigLIP into LanceDB — ingest is **idempotent** (it deletes the event's existing rows first, so re-indexing replaces rather than accumulates). `POST /api/rag/search` embeds a text query and does vector search with **de-duplication** (collapses exact `(event,object)` repeats and near-identical embeddings). The pipeline does **not** auto-ingest into RAG — ingestion is a separate explicit call. SigLIP still runs on PyTorch (not OpenVINO).

**Causal Engine (Track 2)** ([app/pipeline/causal.py](app/pipeline/causal.py)) runs two subsystems that corroborate each other rather than reasoning in isolation:
- A **multi-method statistical engine** ([app/pipeline/causal_discovery.py](app/pipeline/causal_discovery.py)) — PCMCI+ (tigramite), LiNGAM, GES, CausalForest — over a capped set of speed/acceleration/gap variables (`_select_stat_variables`, ≤60 vars), merged into `consensus_edges` by `build_consensus` (`final_confidence = 0.6×support_ratio + 0.4×avg_strength_norm`).
- A **rule-based event-chain reasoner** ([app/pipeline/causal_events.py](app/pipeline/causal_events.py)) that extracts discrete `TemporalEvent`s (SUDDEN_BRAKING, CLOSING_DISTANCE, CONTACT/COLLISION, FALL, …) per entity/pair and links temporally-ordered event pairs into `event_edges`, whose confidence now gets a `causal.statistical_corroboration_bonus` boost whenever `var_to_entities` (built once, at variable-construction time — never reverse-parsed from column names) shows the statistical engine independently found a link between the same two entities.

`_find_primary_event` still identifies **what happened** (the highest-priority incident: COLLISION > CONTACT > NEAR_COLLISION > FALL > SUDDEN_BRAKING). `_rank_root_causes` then answers **why**: it walks that event's causal predecessors (direct + one hop back), dedupes so each survivor is a genuinely distinct reason (different event type or entities), and returns a `root_cause: {primary_cause, alternative_causes}` — up to `causal.max_alternative_causes` alternates, each with its own confidence, mechanism text, and `corroborated_by` (which statistical methods agreed). Below `causal.min_root_cause_confidence` it reports "no corroborated precipitating cause" rather than guessing. `_build_relationship_graph` emits a `relationship_graph` combining three edge kinds for the frontend: `entity_entity_statistical` (consensus edges resolved back to real object ids), `event_event_causal`, and `entity_event_involvement`.

A fourth layer, **SES (State-Event-State) narrative** ([app/pipeline/causal_narrative.py](app/pipeline/causal_narrative.py)), sits on top and re-tells `primary_event`/`root_cause` as a linear causal story instead of a flat ranked list: `S0 (lead-up) → [cause event] → S1 (transition) → [primary event] → S2 (aftermath)`, with each state's label/summary deterministically derived from `entity_series`/`pair_series` kinematics in that time window (speed trend, gap, closing/opening). It is **not a fifth discovery method** — it adds no new evidence, just narrates what the other four methods + the rule-based reasoner already found; same inputs always produce the same chain (no LLM). Alternative causes become `branches` (divergent-hypothesis footnotes off `S0`) rather than parallel chains. Output lives at `ses_narrative` in the result (`chain`, `branches`, `narrative_text`); Track 4 uses `narrative_text`'s ordering to structure the SitRep's Kinematic Timeline section.

It runs **automatically** at the end of Phase 2 for every event (batch pipeline in `app/routes/events.py` and live feeds in `app/pipeline/monitor.py`), non-fatally — a causal failure doesn't fail the event; `POST /api/causal/analyze/{event_id}` re-runs it manually. Persisted to `dataset/{event_id}/causal_graph.json`; `GET /api/causal/{event_id}` returns it. Caveat: ~100-timestep clips are short for causal discovery, so links are ranked hypotheses, and on benign free-flow traffic the correct result is *no* causal chain at all. `test_causal.py` proves it: identifies a synthetic collision correctly, ranks a genuine primary + alternate cause, excludes an unrelated distractor vehicle, shows statistical corroboration actually firing, and checks the SES chain alternates state/event/state around the correct primary+cause events.

**Synthesis (Track 4)** ([app/pipeline/synthesis.py](app/pipeline/synthesis.py)) turns an event's structured outputs into an LLM-written **situation report**. `POST /api/synthesis/{event_id}` distils event metadata + per-entity kinematics (from the causal CSV) + **SigLIP zero-shot colour attributes** + the Track 2 `primary_event`/`root_cause` into a **text-only evidence packet** (per context.md §1, *no raw imagery ever reaches the LLM*), derives **incident indicators** (a vehicle that decelerates sharply *and* whose track terminates mid-window = likely collision, plus the nearest entity at that instant), and calls an LLM to write the SitRep — instructed to use the given primary/alternative-cause confidence numbers rather than inventing its own. Provider-agnostic: any **OpenAI-compatible** `/chat/completions` endpoint via stdlib HTTP — defaults to **Google Gemini** (`gemini-flash-latest`, free tier), configurable for a local/edge server later. The API key is read from `$LLM_API_KEY` (or a repo-root `.env`, auto-loaded); without a key it still builds + persists the evidence packet. Output → `dataset/{event_id}/sitrep.md` (+ `sitrep.json`); `GET /api/synthesis/{event_id}` returns it. Speed spikes are clamped/flagged (`causal.max_plausible_speed_mps`) so implausible kinematics don't reach the model.

**Live feeds / camera ingestion Step 2** ([app/pipeline/monitor.py](app/pipeline/monitor.py)) is the streaming counterpart to batch `scan_for_events`. A `FeedMonitor` runs one **background thread per source** (managed by a `FeedManager` singleton) whose single read loop feeds every frame to **both** a `FrameEventDetector` (→ the full event pipeline, `process_event`+`finalize_event`) **and**, on sampled frames, a `ContinuousTracker`. The `ContinuousTracker` keeps the best crop per track and, when a track ends, indexes that vehicle into LanceDB via `RAG.index_vehicles` — this is the **hybrid** design: *every* vehicle is indexed (under a `FEED_{video_id}` bucket), not just event vehicles. Control via `GET /api/feeds`, `POST /api/feeds/{video_id}/start|stop`; monitors are stopped on app shutdown. Source type is auto-detected — a URL (`rtsp://`/`http://`/…) or webcam index is a **stream** (reconnect on drop), a local path is a **file** (stop at EOF). Caveats: each monitor loads its own YOLO model (memory scales with feed count); RTSP reconnect and event-during-monitoring are coded but only runtime-verified on a file source so far; no frontend UI yet.

**Three storage layers**, all rooted at repo root via `PathConfig`:
- `event_registry.db` — SQLite. `Master_Event_Log` (events) + `Video_Sources` (camera feeds). WAL mode. `init_db()` runs idempotent `ALTER TABLE` migrations and back-fills on every startup.
- `dataset/{Event_ID}/` — per-event `.mp4`, `{Event_ID}_causal_data.csv`, `entity_crops/{Event_ID}_{Object_ID}_crop.jpg`, and (after causal analysis) `causal_graph.json`.
- `dataset/feeds/{video_id}/` — best crops of continuously-indexed vehicles from live monitoring.
- `dataset/lancedb/` — LanceDB vector table `entity_crops` (event crops keyed by `event_id=EVT_…`, live-feed vehicles by `event_id=FEED_…`).

**Config** ([app/config.py](app/config.py)): a single frozen-dataclass `settings` singleton aggregating all tuning constants (video timing, MOG2 thresholds incl. the burst-confirmation knobs `burst_reject_s`/`stop_factor`, YOLO classes/backend/device, tracker params, `interpolation` incl. velocity smoothing, `projection` reliable-region bounds, `causal` multi-method + root-cause-ranking params, `synthesis` LLM/SitRep params, `feed` live-monitoring params, paths). Change constants here, not inline. Note the tracker params here are duplicated: `perception.py` regenerates `config/custom_tracker.yaml` from `settings.tracker` on every run, so the root-level and `config/` yaml files are overwritten artifacts, not source of truth. `SceneConfig`/`settings.scene` is unused dead config (a leftover from the deleted Gen-1 episode-staging module) — don't wire new code to it.

## Conventions and gotchas

- **Video paths must be absolute** on the host machine — the pipeline reads local files directly; there is no upload/storage of the source video.
- **Windows-first environment.** Paths, `ffmpeg` on PATH, and C++ build tools (for `lapx`/`sentencepiece`) are assumed. YOLO + SigLIP weights (~1GB) download on first run.
- **YOLO defaults to the OpenVINO backend.** Build the IR with `scripts/export_openvino.py` (`yolo11s_int8_openvino_model/` — a gitignored build artifact); without it the pipeline auto-falls-back to PyTorch. Inference `device` is pinned to `cpu` because the CUDA-built torch otherwise mis-detects a GPU and raises `Invalid device id`.
- The frontend `NEXT_PUBLIC_API_URL` defaults to `http://localhost:8000`; backend CORS is fully open (dev only).
- `frontend/src/lib/api.ts` declares privacy/audit endpoints (`/api/privacy/*`) that have **no backend implementation** — treat those client functions as unwired.
- IDs: events are `EVT_<12 hex>`, sources are `VID_<8 hex>`, tracked objects are `V_<nn>`.
- **Homography is dynamically computed.** The pipeline API accepts a `src_pts` payload (4 bounding points of a lane) to calculate the homography matrix on the fly for any camera/feed. If omitted, it falls back to a default `config/homography.npy`. Applying the wrong calibration yields garbage kinematics.
- **Track 4 needs an LLM key.** Set `$LLM_API_KEY` (or a repo-root `.env`, which the code auto-loads — it is gitignored). Defaults to Gemini's free OpenAI-compatible endpoint; override model/endpoint via `SynthesisConfig` or `$LLM_MODEL`/`$LLM_BASE_URL`. Thinking models need a high `max_tokens` or they truncate.

## Frontend caveat (important)

`frontend/AGENTS.md` warns: this is **Next.js 16** (App Router, React 19), which has breaking changes from earlier versions. Before writing frontend code, read the relevant guide under `frontend/node_modules/next/dist/docs/` rather than relying on prior Next.js knowledge.
