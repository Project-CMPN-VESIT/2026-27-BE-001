"""
baseline_granger.py — Standalone classical Granger-causality baseline.

This is NOT part of the production causal engine and is NOT imported by
causal.py. It exists purely so the multi-method engine (PCMCI+/LiNGAM/GES/
CausalForest + rule-based event chain + root-cause ranking, see causal.py)
can be compared against the textbook/industry-standard Granger-causality
method on the same input data — see compare_baselines.py at the repo root.

This is deliberately NOT the same thing as `_granger_check` in
causal_events.py. That helper is a lightweight Ridge-regression /
cross-validation approximation used as one piece of corroborating evidence
inside the rule-based event chain. This module is the real, standard
formulation: VAR-model-based Granger causality via statsmodels
(`grangercausalitytests`), with the stationarity handling and lag-order
selection a rigorous application of the method requires — so the comparison
is fair rather than a strawman.

Method, per ordered pair of entities (A, B), testing "does A Granger-cause B":
  1. Align A's and B's speed series on shared frames (inner join).
  2. Stationarity check via Augmented Dickey-Fuller; first-difference once if
     non-stationary (standard practice — Granger causality assumes
     stationarity, and raw speed is often trending/non-stationary over a
     short braking/collision clip).
  3. Select VAR lag order via AIC over the 2-variable system, capped the same
     way the rest of this codebase caps lags on short clips.
  4. Run grangercausalitytests(data[[B, A]], maxlag=selected_lag) — the
     statsmodels convention is columns [effect, cause] — and take the
     lag with the lowest ssr F-test p-value.
  5. Significant (p < alpha) => an edge A -> B.

Both directions (A->B and B->A) are tested independently, since Granger
causality is not symmetric — mirrors the directionality PCMCI+/LiNGAM/GES
already report in the production engine.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALPHA = 0.05          # significance threshold, standard convention
MIN_OBS = 30          # minimum shared frames to attempt the test at all
MAX_LAG_FRAC = 0.15   # lag cap as a fraction of series length, mirrors tau_max_frame_frac


def _stationary_or_differenced(series: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Return (series_to_use, was_differenced). ADF test; difference once if
    non-stationary. Never differences twice — a short clip doesn't have
    enough points left to test meaningfully after that."""
    from statsmodels.tsa.stattools import adfuller

    try:
        p_value = adfuller(series, autolag="AIC", result_object=False)[1]
    except Exception:
        return series, False
    if p_value < ALPHA:
        return series, False
    diffed = np.diff(series)
    return diffed, True


def _pairwise_granger(
    cause: np.ndarray,
    effect: np.ndarray,
) -> Dict:
    """
    Test whether `cause` Granger-causes `effect`. Returns
    {status, lag, p_value, f_stat, significant, differenced}.
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.tsa.api import VAR

    n = min(len(cause), len(effect))
    if n < MIN_OBS:
        return {"status": "insufficient", "reason": f"only {n} shared observations"}

    cause_s, cause_diffed = _stationary_or_differenced(cause[:n])
    effect_s, effect_diffed = _stationary_or_differenced(effect[:n])
    m = min(len(cause_s), len(effect_s))
    if m < MIN_OBS:
        return {"status": "insufficient", "reason": f"only {m} observations after differencing"}

    data = np.column_stack([effect_s[:m], cause_s[:m]])  # [effect, cause] — statsmodels convention

    max_lag = max(1, min(8, int(m * MAX_LAG_FRAC)))
    try:
        order_result = VAR(data).select_order(maxlags=max_lag)
        selected_lag = max(1, int(order_result.aic))
    except Exception:
        selected_lag = max_lag

    # Test ONLY at the single, a-priori AIC-selected lag. Scanning every lag
    # from 1..selected_lag and keeping whichever gives the smallest p-value
    # is itself an uncorrected multiple-comparisons search hidden inside a
    # single directional test — it inflates the false-positive rate on top
    # of (and independent from) the across-pairs FDR correction applied by
    # the caller, and was caught here precisely because it produced a ~40%
    # "significant" rate on synthetic null trials built from mutually
    # independent white-noise-plus-constant series, where the true rate
    # should be close to alpha.
    try:
        results = grangercausalitytests(data, maxlag=selected_lag)
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}

    best_lag = selected_lag
    best_f, best_p, _, _ = results[selected_lag][0]["ssr_ftest"]

    return {
        "status": "ok",
        "lag": best_lag,
        "p_value": round(float(best_p), 5),
        "f_stat": round(float(best_f), 4),
        "significant": bool(best_p < ALPHA),
        "differenced": bool(cause_diffed or effect_diffed),
    }


def run_granger_baseline(
    entity_series: Dict[str, pd.DataFrame],
    fps: float = 10.0,
    variable: str = "speed",
) -> Dict:
    """
    Run classical pairwise Granger causality over every ordered pair of
    entities' `variable` series (default: speed — matching this project's
    documented speed-primary design, since monocular-BEV acceleration is too
    noisy to trust as a headline variable; see causality.md).

    Testing every ordered pair independently at a flat alpha=0.05 inflates the
    false-positive rate badly as the entity count grows (e.g. 130 pairs tested
    at alpha=0.05 gives ~6-7 false positives expected under the null by chance
    alone — and empirically far more than that turn up "significant" on a busy
    real scene, which is the textbook multiple-comparisons problem). To keep
    this a fair, industry-standard baseline rather than a strawman, edges are
    corrected across ALL pairs tested for this event using Benjamini-Hochberg
    FDR correction (`statsmodels.stats.multitest.multipletests`, alpha=0.05)
    before being called "significant".

    Returns
    -------
    {
      "status": "ok" | "unavailable" | "insufficient",
      "method": "Granger-VAR",
      "n_entities_tested": int,
      "n_pairs_tested": int,
      "edges": [...],              # significant AFTER Benjamini-Hochberg FDR correction
      "edges_uncorrected": [...],  # significant at raw p<0.05, no correction — shown to make the multiple-comparisons effect visible
      "all_tests": [...],          # every pair tested, including non-significant
    }
    """
    try:
        import statsmodels  # noqa: F401
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return {"status": "unavailable", "reason": "statsmodels not installed", "edges": []}

    oids = [oid for oid, sdf in entity_series.items()
            if variable in sdf.columns and sdf[variable].notna().sum() >= MIN_OBS]
    if len(oids) < 2:
        return {"status": "insufficient", "reason": "fewer than 2 entities with enough data", "edges": []}

    series_by_oid = {}
    for oid in oids:
        sdf = entity_series[oid].sort_values("frame_id")
        s = sdf.set_index("frame_id")[variable]
        series_by_oid[oid] = s

    all_tests: List[Dict] = []

    for oid_a in oids:
        for oid_b in oids:
            if oid_a == oid_b:
                continue
            sa, sb = series_by_oid[oid_a], series_by_oid[oid_b]
            common = sa.index.intersection(sb.index)
            if len(common) < MIN_OBS:
                continue
            a_vals = sa.loc[common].to_numpy(dtype=float)
            b_vals = sb.loc[common].to_numpy(dtype=float)
            mask = np.isfinite(a_vals) & np.isfinite(b_vals)
            if mask.sum() < MIN_OBS:
                continue

            test = _pairwise_granger(a_vals[mask], b_vals[mask])
            all_tests.append({"source": oid_a, "target": oid_b, **test})

    ok_tests = [t for t in all_tests if t.get("status") == "ok"]
    if ok_tests:
        pvals = [t["p_value"] for t in ok_tests]
        corrected_sig, corrected_p, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
        for t, sig, cp in zip(ok_tests, corrected_sig, corrected_p):
            t["p_value_fdr"] = round(float(cp), 5)
            t["significant_fdr"] = bool(sig)

    def _to_edge(t: Dict) -> Dict:
        return {
            "source": t["source"], "target": t["target"],
            "lag_frames": t["lag"], "lag_seconds": round(t["lag"] / fps, 3),
            "p_value": t["p_value"], "p_value_fdr": t.get("p_value_fdr"),
            "f_stat": t["f_stat"], "relationship": "causes", "method": "Granger-VAR",
        }

    edges = [_to_edge(t) for t in ok_tests if t.get("significant_fdr")]
    edges_uncorrected = [_to_edge(t) for t in ok_tests if t.get("significant")]

    logger.info(
        "Granger baseline: %d/%d pairs raw-significant (p<%.2f), %d survive FDR correction, from %d entities",
        len(edges_uncorrected), len(ok_tests), ALPHA, len(edges), len(oids),
    )

    return {
        "status": "ok",
        "method": "Granger-VAR",
        "n_entities_tested": len(oids),
        "n_pairs_tested": len(ok_tests),
        "edges": edges,
        "edges_uncorrected": edges_uncorrected,
        "all_tests": all_tests,
    }


def rank_granger_root_cause(
    granger_result: Dict,
    primary_object_ids: List[str],
) -> Dict:
    """
    Given the baseline's edges and the object_ids of whatever the (external)
    "primary event" is, pick the Granger baseline's equivalent of a
    root-cause: the significant edge pointing INTO one of primary_object_ids
    with the lowest p-value, plus the rest ranked as alternates — so its
    output shape is directly comparable to CausalEngine's root_cause.

    A candidate source is NOT required to be outside primary_object_ids — the
    production engine's own root causes are frequently an earlier event
    between the SAME two entities (e.g. "V_LEAD/V_FOLLOW's earlier closing
    distance caused their own collision"), so excluding same-entity sources
    here would mechanically bias this comparison against Granger rather than
    reflect a genuine limitation of the method.

    Note the honest limitation this exposes: Granger causality alone has no
    concept of "the primary event" (a collision, a fall, ...) — it only sees
    continuous series. This function borrows the primary event from outside
    (from the production engine's `primary_event`) purely to make an
    apples-to-apples comparison possible; a Granger-only system deployed on
    its own would have no equivalent of Step 10 in causal.py at all.
    """
    incoming = [
        e for e in granger_result.get("edges", [])
        if e["target"] in primary_object_ids
    ]
    incoming.sort(key=lambda e: e["p_value"])
    if not incoming:
        return {"primary_cause": None, "alternative_causes": []}

    def _finalize(e: Dict) -> Dict:
        return {
            "label": f"{e['source']} Granger-causes {e['target']}",
            "source": e["source"],
            "target": e["target"],
            "p_value": e["p_value"],
            "f_stat": e["f_stat"],
            "lag_seconds": e["lag_seconds"],
        }

    return {
        "primary_cause": _finalize(incoming[0]),
        "alternative_causes": [_finalize(e) for e in incoming[1:]],
    }
