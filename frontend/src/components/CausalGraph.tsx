'use client';

import { useState, useRef, useMemo, useCallback } from 'react';
import { forceSimulation, forceLink, forceManyBody, forceCollide, forceCenter, SimulationNodeDatum } from 'd3-force';
import {
  FullCausalResult,
  RelationshipNode,
  RelationshipEdge,
  TemporalEvent,
} from '@/lib/api';
import styles from './CausalPanel.module.css';

// ── Colour helpers ────────────────────────────────────────────────────────────
const EVENT_COLOURS: Record<string, string> = {
  COLLISION: '#ef4444',
  CONTACT: '#f97316',
  NEAR_COLLISION: '#facc15',
  SUDDEN_BRAKING: '#3b82f6',
  SUDDEN_STOP: '#6366f1',
  SUDDEN_ACCELERATION: '#22c55e',
  CLOSING_DISTANCE: '#f59e0b',
  APPROACHING: '#a3a3a3',
  FALL: '#ec4899',
  TRAJECTORY_DEVIATION: '#8b5cf6',
  SWERVE: '#0ea5e9',
  DEFAULT: '#64748b',
};

function eventColour(type: string | null): string {
  return (type && EVENT_COLOURS[type]) || EVENT_COLOURS.DEFAULT;
}

const EVENT_NODE_COLOURS = [
  '#ef4444', '#f97316', '#facc15', '#22c55e', '#06b6d4',
  '#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6', '#a855f7',
];

function distinctEventColour(type: string | null, id: string): string {
  let hash = 0;
  for (let index = 0; index < id.length; index += 1) {
    hash = (hash * 31 + id.charCodeAt(index)) | 0;
  }
  const colourIndex = Math.abs(hash) % EVENT_NODE_COLOURS.length;
  const colour = EVENT_NODE_COLOURS[colourIndex];
  return colour === eventColour(type)
    ? EVENT_NODE_COLOURS[(colourIndex + 1) % EVENT_NODE_COLOURS.length]
    : colour;
}

function eventLabel(node: RelationshipNode): string {
  const type = (node.event_type || node.label).replace(/_/g, ' ');
  const objects = node.object_ids.length > 0 ? ` · ${node.object_ids.join(' / ')}` : '';
  const time = node.timestamp != null ? ` · t=${node.timestamp.toFixed(1)}s` : '';
  return `${type}${objects}${time}`;
}

function relColour(kind: string, rel: string): string {
  if (kind === 'entity_entity_statistical') return '#8b5cf6';
  if (kind === 'entity_event_involvement') return '#94a3b8';
  if (rel === 'causes') return '#ef4444';
  if (rel === 'supports') return '#f97316';
  return '#64748b';
}

function confOpacity(c: number): number {
  return 0.3 + Math.min(Math.max(c, 0), 1) * 0.6;
}

// ── Force-directed layout ─────────────────────────────────────────────────────

interface SimNode extends SimulationNodeDatum {
  id: string;
  kind: 'entity' | 'event';
  label: string;
  r: number;
  colour: string;
  node: RelationshipNode;
  event?: TemporalEvent;
}

interface SimLink {
  source: string | SimNode;
  target: string | SimNode;
  edge: RelationshipEdge;
}

const LINK_DISTANCE: Record<string, number> = {
  entity_entity_statistical: 170,
  event_event_causal: 140,
  entity_event_involvement: 100,
};

// Cap on how many statistical (entity↔entity) links are drawn — these are the
// biggest source of clutter since they scale with entity pairs, not with the
// actual incident narrative. Only the strongest survive; weaker ones are cut.
const MAX_STATISTICAL_EDGES = 6;
// Cap on how many "involved in" links a single entity draws — beyond this the
// fan-out of thin dashed lines from one node obscures the causal chain.
const MAX_INVOLVEMENT_PER_ENTITY = 4;
const MAX_GRAPH_NODES = 28;
const MAX_EVENT_NODES = 14;

function buildSimulation(
  result: FullCausalResult,
  width: number,
  height: number,
): { nodes: SimNode[]; links: SimLink[]; height: number } {
  const graph = result.relationship_graph;
  const eventsById = new Map((result.temporal_events || []).map((e) => [e.event_id, e]));

  if (!graph || graph.nodes.length === 0) {
    return { nodes: [], links: [], height };
  }

  const allEntityNodes = graph.nodes.filter((n) => n.kind === 'entity');
  const allEventNodes = graph.nodes.filter((n) => n.kind === 'event').sort(
    (a, b) => (a.timestamp ?? 0) - (b.timestamp ?? 0),
  );

  // Keep the graph centered on the incident narrative. Older persisted graphs
  // may contain the full event chain, which is technically valid but unreadable.
  const primaryId = result.primary_event?.event_id;
  const rankedCauseIds = [
    result.root_cause?.primary_cause?.cause_event_id,
    ...(result.root_cause?.alternative_causes ?? []).map((cause) => cause.cause_event_id),
  ].filter((id): id is string => Boolean(id));
  const priorityIds = new Set([primaryId, ...rankedCauseIds].filter(Boolean));
  const directlyConnected = graph.edges
    .filter((edge) => priorityIds.has(edge.source) || priorityIds.has(edge.target))
    .flatMap((edge) => [edge.source, edge.target]);
  directlyConnected.forEach((id) => priorityIds.add(id));

  const prioritizedEvents = allEventNodes.filter((node) => priorityIds.has(node.id));
  const remainingEvents = allEventNodes
    .filter((node) => !priorityIds.has(node.id))
    .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  const eventNodes = [...prioritizedEvents, ...remainingEvents].slice(0, MAX_EVENT_NODES);
  const involvedEntityIds = new Set(
    eventNodes.flatMap((node) => node.object_ids ?? []),
  );
  const entityNodes = allEntityNodes.filter((node) => involvedEntityIds.has(node.object_ids[0]));
  const selectedNodes = [...eventNodes, ...entityNodes].slice(0, MAX_GRAPH_NODES);
  const selectedNodeIds = new Set(selectedNodes.map((node) => node.id));
  const visibleEventNodes = selectedNodes.filter((node) => node.kind === 'event');
  const visibleEntityNodes = selectedNodes.filter((node) => node.kind === 'entity');

  // Scale the canvas up for graphs with a lot of nodes so the force layout
  // has room to spread out instead of cramming everything into a fixed box.
  const nodeCount = selectedNodes.length;
  const extraHeight = Math.max(0, nodeCount - 8) * 22;
  height = height + extraHeight;

  const cx = width / 2;
  const cy = height / 2;

  const simNodes: SimNode[] = [];

  // Seed positions only — the force simulation refines from here.
  visibleEntityNodes.forEach((n, i) => {
    const angle = (i / Math.max(visibleEntityNodes.length, 1)) * Math.PI * 2;
    const radius = Math.min(width, height) * 0.22;
    simNodes.push({
      id: n.id, kind: 'entity', label: n.label, r: 34, colour: '#2563eb', node: n,
      x: cx + Math.cos(angle) * radius, y: cy - radius + Math.sin(angle) * radius * 0.3,
    });
  });

  visibleEventNodes.forEach((n, i) => {
    const frac = (i + 1) / (visibleEventNodes.length + 1);
    simNodes.push({
      id: n.id, kind: 'event', label: eventLabel(n), r: 28, colour: distinctEventColour(n.event_type, n.id), node: n,
      event: eventsById.get(n.id),
      x: cx + (frac - 0.5) * width * 0.7, y: cy + height * 0.22,
    });
  });

  const nodeIds = selectedNodeIds;
  let edges = graph.edges.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));

  // Declutter: keep only the strongest statistical edges…
  const statistical = edges
    .filter((e) => e.kind === 'entity_entity_statistical')
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, MAX_STATISTICAL_EDGES);
  // …and cap how many involvement edges fan out of any one entity, keeping
  // the highest-confidence events for that entity.
  const involvementByEntity = new Map<string, RelationshipEdge[]>();
  for (const e of edges) {
    if (e.kind !== 'entity_event_involvement') continue;
    const list = involvementByEntity.get(e.source) ?? [];
    list.push(e);
    involvementByEntity.set(e.source, list);
  }
  const involvement = [...involvementByEntity.values()].flatMap((list) =>
    list.sort((a, b) => b.confidence - a.confidence).slice(0, MAX_INVOLVEMENT_PER_ENTITY),
  );
  const causal = edges.filter((e) => e.kind === 'event_event_causal');
  edges = [...causal, ...statistical, ...involvement];

  const simLinks: SimLink[] = edges.map((e) => ({ source: e.source, target: e.target, edge: e }));

  const sim = forceSimulation(simNodes)
    .force('link', forceLink<SimNode, SimLink>(simLinks)
      .id((d) => d.id)
      .distance((l) => LINK_DISTANCE[l.edge.kind] ?? 120)
      .strength((l) => 0.25 + l.edge.confidence * 0.5))
    .force('charge', forceManyBody().strength(-340))
    .force('collide', forceCollide<SimNode>().radius((d) => d.r + 22))
    .force('center', forceCenter(cx, cy))
    .stop();

  for (let i = 0; i < 300; i++) sim.tick();

  return { nodes: simNodes, links: simLinks, height };
}

// ── Arrow helper ──────────────────────────────────────────────────────────────
function arrowPath(from: SimNode, to: SimNode): string {
  const dx = (to.x ?? 0) - (from.x ?? 0);
  const dy = (to.y ?? 0) - (from.y ?? 0);
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 1) return '';
  const ux = dx / dist;
  const uy = dy / dist;
  const x1 = (from.x ?? 0) + ux * from.r;
  const y1 = (from.y ?? 0) + uy * from.r;
  const x2 = (to.x ?? 0) - ux * (to.r + 10);
  const y2 = (to.y ?? 0) - uy * (to.r + 10);
  const mx = (x1 + x2) / 2 - uy * 18;
  const my = (y1 + y2) / 2 + ux * 18;
  return `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`;
}

// ── Main component ────────────────────────────────────────────────────────────

interface Props {
  result: FullCausalResult;
  onSeekFrame?: (frame: number) => void;
  onHoverEvent?: (evt: TemporalEvent | null) => void;
}

export default function CausalGraphD3({ result, onSeekFrame, onHoverEvent }: Props) {
  const WIDTH = 900;
  const BASE_HEIGHT = 560;

  const { nodes, links, height: HEIGHT } = useMemo(
    () => buildSimulation(result, WIDTH, BASE_HEIGHT),
    [result],
  );

  // ── Zoom / Pan state ──────────────────────────────────────────────────────
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panRef = useRef({ startX: 0, startY: 0, panX: 0, panY: 0, active: false });
  const svgRef = useRef<SVGSVGElement>(null);

  const handleZoomSlider = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setZoom(Math.round(parseFloat(e.target.value) * 10) / 10);
  }, []);

  const handleReset = useCallback(() => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return;
    panRef.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y, active: true };
    setIsPanning(true);
    e.preventDefault();
  }, [pan.x, pan.y]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!panRef.current.active) return;
    const dx = e.clientX - panRef.current.startX;
    const dy = e.clientY - panRef.current.startY;
    setPan({ x: panRef.current.panX + dx, y: panRef.current.panY + dy });
  }, []);

  const onMouseUp = useCallback(() => {
    panRef.current.active = false;
    setIsPanning(false);
  }, []);

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom((z) => Math.round(Math.max(0.3, Math.min(3.0, z + delta)) * 10) / 10);
  }, []);

  if (nodes.length === 0) {
    return (
      <div className={styles.emptyState}>
        No relationship graph available — re-run causal analysis to generate one.
      </div>
    );
  }

  return (
    <div className={styles.graphWrap}>
      <div className={styles.graphToolbar}>
        <span className={styles.graphToolbarLabel}>Zoom</span>
        <input
          type="range"
          className={styles.graphZoomSlider}
          min="0.3"
          max="3.0"
          step="0.1"
          value={zoom}
          onChange={handleZoomSlider}
        />
        <span className={styles.graphZoomValue}>{Math.round(zoom * 100)}%</span>
        <button className={styles.graphResetBtn} onClick={handleReset}>Reset</button>
      </div>

      <div style={{ overflow: 'hidden', borderRadius: 'var(--radius-md)', cursor: isPanning ? 'grabbing' : 'grab' }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className={styles.graphSvg}
          style={{ width: '100%', height: `${Math.min(HEIGHT, 760)}px`, display: 'block' }}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
          onWheel={onWheel}
        >
          <defs>
            {(['causes', 'supports', 'precedes'] as const).map((rel) => (
              <marker
                key={rel}
                id={`arrow-${rel}`}
                viewBox="0 0 10 10"
                refX={9}
                refY={5}
                markerWidth={7}
                markerHeight={7}
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill={relColour('event_event_causal', rel)} />
              </marker>
            ))}
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`} style={{ transformOrigin: '0 0' }}>
            {/* Edges — involvement edges drawn first, underneath */}
            {[...links].sort((a, b) => (a.edge.kind === 'entity_event_involvement' ? -1 : 1) -
                                        (b.edge.kind === 'entity_event_involvement' ? -1 : 1))
              .map((l, i) => {
                const from = typeof l.source === 'object' ? l.source : nodes.find((n) => n.id === l.source);
                const to = typeof l.target === 'object' ? l.target : nodes.find((n) => n.id === l.target);
                if (!from || !to) return null;
                const d = arrowPath(from, to);
                if (!d) return null;
                const kind = l.edge.kind;
                const isInvolvement = kind === 'entity_event_involvement';
                const isCausal = kind === 'event_event_causal';
                const isStatistical = kind === 'entity_entity_statistical';
                const colour = relColour(kind, l.edge.relationship);
                // Labels are the biggest source of clutter, so only the core
                // causal chain (event→event) gets a permanent one — statistical
                // and involvement edges rely on colour/dash + the legend instead.
                return (
                  <g key={`edge-${i}`}>
                    <path
                      d={d}
                      fill="none"
                      stroke={colour}
                      strokeWidth={isInvolvement ? 1 : isStatistical ? 1.3 : l.edge.relationship === 'causes' ? 2.6 : 1.6}
                      strokeDasharray={
                        isInvolvement ? '2 4' : isStatistical ? '5 3' :
                        l.edge.relationship === 'precedes' ? '6 5' : undefined
                      }
                      opacity={isInvolvement ? 0.3 : isStatistical ? 0.55 : confOpacity(l.edge.confidence)}
                      markerEnd={
                        isCausal
                          ? `url(#arrow-${['causes', 'supports'].includes(l.edge.relationship) ? l.edge.relationship : 'precedes'})`
                          : undefined
                      }
                    />
                    {isCausal && (
                      <text
                        x={((from.x ?? 0) + (to.x ?? 0)) / 2 - ((to.y ?? 0) - (from.y ?? 0)) * 0.03}
                        y={((from.y ?? 0) + (to.y ?? 0)) / 2 + ((to.x ?? 0) - (from.x ?? 0)) * 0.03}
                        textAnchor="middle"
                        fill={colour}
                        fontSize={10.5}
                        opacity={0.85}
                      >
                        {l.edge.relationship} · {l.edge.confidence.toFixed(2)}
                      </text>
                    )}
                  </g>
                );
              })}

            {/* Nodes */}
            {nodes.map((n) => {
              const isCollision = n.node.event_type === 'COLLISION' || n.node.event_type === 'CONTACT';
              return (
                <g
                  key={n.id}
                  style={{ cursor: onSeekFrame ? 'pointer' : 'default' }}
                  onClick={() => {
                    if (onSeekFrame) {
                      const frame = n.node.frame ?? n.event?.start_frame;
                      if (frame != null) onSeekFrame(frame);
                    }
                  }}
                  onMouseEnter={() => n.kind === 'event' && onHoverEvent?.(n.event ?? null)}
                  onMouseLeave={() => onHoverEvent?.(null)}
                >
                  <circle
                    cx={n.x}
                    cy={n.y}
                    r={n.r}
                    fill={n.colour}
                    fillOpacity={n.kind === 'entity' ? 0.22 : 0.18}
                    stroke={n.colour}
                    strokeWidth={isCollision ? 3 : n.kind === 'entity' ? 2.2 : 1.8}
                    filter={isCollision ? 'url(#glow)' : undefined}
                  />
                  <text x={n.x} y={(n.y ?? 0) - 4} textAnchor="middle" fill={n.colour} fontSize={n.kind === 'entity' ? 12 : 11.5} fontWeight={600}>
                    {n.label}
                  </text>
                  {n.kind === 'event' && n.node.timestamp != null && (
                    <text x={n.x} y={(n.y ?? 0) + 12} textAnchor="middle" fill={n.colour} fontSize={10.5} opacity={0.8}>
                      t={n.node.timestamp.toFixed(1)}s · {(n.node.confidence * 100).toFixed(0)}%
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      <div className={styles.graphLegend}>
        <span style={{ color: '#2563eb' }}>● entity</span>
        <span style={{ color: '#ef4444' }}>● collision/contact</span>
        <span style={{ color: '#ef4444' }}>— causes</span>
        <span style={{ color: '#f97316' }}>— supports</span>
        <span style={{ color: '#64748b' }}>- - precedes (event↔event)</span>
        <span style={{ color: '#8b5cf6' }}>·· statistical link (entity↔entity)</span>
        <span style={{ color: '#94a3b8' }}>·· involved in (entity↔event)</span>
      </div>
    </div>
  );
}
