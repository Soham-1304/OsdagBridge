"""
parallel_bridge_optimizer.py
----------------------------
Parallel, multiprocessing weight-minimisation of a plate-girder bridge
superstructure, built on the generic engine in ``parallel_optimizer.py``.

Design vector
-------------
    x = [s, t_slab, D, bf, tf, tw]
        s       girder spacing                   (m)   [2.0, 4.0]
        t_slab  deck-slab thickness              (mm)
        D       girder overall depth             (mm)
        bf      flange width (symmetric I)       (mm)
        tf      flange thickness (symmetric I)   (mm)
        tw      web thickness                    (mm)

    Note: ``n`` (number of girders) is *derived* from ``s``:
        n = round(width / s),  then spacing is refined as  s = width / n
        so n is not an independent gene in the DE search space.

How it differs from the sequential ``bridge_optimizer.py``
---------------------------------------------------------
  * No module globals. All per-run context (span, width, densities, the base
    ``input_dict``) lives in a single picklable ``OptiConfig`` that is installed
    into every worker process **once** via the pool ``initializer`` - not shipped
    with every candidate, not read from globals that don't exist in a child.
  * The fitness function is a module-level, picklable callable and folds the
    feasibility check and the weight objective into one cross-process call,
    returning ``+inf`` for any infeasible / failing candidate.
  * Feasibility is evaluated through the **real, maintained** bridge pipeline
    (the same staged methods ``PlateGirderBridge.design()`` runs) up to the
    IRC 22:2015 DCR checks - CAD / deck / transverse stages are skipped because
    they are irrelevant to feasibility and expensive. CAD is generated once, in
    the parent process, for the winning design only.

Usage
-----
    from osdagbridge.core.optimizer.parallel_bridge_optimizer import optimize_parallel

    bridge = optimize_parallel(base_input_dict)   # a fully solved input_dict
    bridge.design()                               # full pipeline + CAD on the winner
"""

from __future__ import annotations

import copy
import gc
import math
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from osdagbridge.core.bridge_types.plate_girder.plategirderbridge import PlateGirderBridge
from osdagbridge.core.bridge_types.plate_girder.designer import SteelSection
from osdagbridge.core.utils.common import (
    KEY_SPAN,
    KEY_TS_OVERALL_WIDTH,
    KEY_TS_NO_OF_GIRDERS,
    KEY_TS_GIRDER_SPACING,
    KEY_TS_DECK_OVERHANG,
    KEY_TS_DECK_THICKNESS,
    KEY_DESIGN_MODE,
    KEY_MP_GIRDER_DEPTH,
    KEY_MP_GIRDER_WEB_DEPTH,
    KEY_MP_GIRDER_WEB_THICKNESS,
    KEY_MP_GIRDER_TOP_FLANGE_WIDTH,
    KEY_MP_GIRDER_TOP_FLANGE_THICKNESS,
    KEY_MP_GIRDER_BOTTOM_FLANGE_WIDTH,
    KEY_MP_GIRDER_BOTTOM_FLANGE_THICKNESS,
)

from .parallel_optimizer import ParallelOptimizer, OptimisationResult


# ------------------------------------------------------------------------------
#  Constants
# ------------------------------------------------------------------------------

# IS 2062 standard plate thickness list (mm) — tf and tw must come from here.
_STD_PLATES: List[int] = [
    6, 8, 10, 12, 14, 16, 18, 20, 22, 25,
    28, 32, 36, 40, 45, 50, 56, 63, 70, 80, 90, 100,
]

# Design mode that makes the pipeline treat girder dims as fixed numeric values
# (the candidate's), as opposed to "Optimized" which expects bounds.
_FIXED_DESIGN_MODE = "Custom"


# ------------------------------------------------------------------------------
#  Rounding helpers (snap continuous DE variables onto manufacturable values)
# ------------------------------------------------------------------------------

def _ceiling_plate(value_mm: float) -> float:
    """Smallest IS 2062 plate thickness >= value_mm (always structurally safe)."""
    for p in _STD_PLATES:
        if p >= value_mm:
            return float(p)
    return float(_STD_PLATES[-1])


def _ceil10(v: float) -> float:
    """Ceil to nearest 10 mm."""
    return float(math.ceil(v / 10.0) * 10.0)


def _round5(v: float) -> float:
    """Round to nearest 5 mm."""
    return float(round(v / 5.0) * 5.0)


def clamp(v: float, lo: float, hi: float) -> float:
    """Hard-clip v to [lo, hi]."""
    return max(lo, min(v, hi))


# ------------------------------------------------------------------------------
#  Per-run configuration (picklable; installed into every worker once)
# ------------------------------------------------------------------------------

@dataclass
class OptiConfig:
    """Everything a worker needs to evaluate a candidate. Must stay picklable."""
    base_input_dict : dict
    span_m          : float
    deck_width_m    : float
    steel_density   : float = 78.5   # kN/m^3
    concrete_density: float = 25.0   # kN/m^3

    @classmethod
    def from_input_dict(
        cls,
        base_input_dict : dict,
        steel_density   : float = 78.5,
        concrete_density: float = 25.0,
    ) -> "OptiConfig":
        return cls(
            base_input_dict  = copy.deepcopy(base_input_dict),
            span_m           = float(base_input_dict[KEY_SPAN]),
            deck_width_m     = float(base_input_dict[KEY_TS_OVERALL_WIDTH]),
            steel_density    = steel_density,
            concrete_density = concrete_density,
        )


# ------------------------------------------------------------------------------
#  Candidate normalisation: raw DE vector -> manufacturable, code-compliant dims
# ------------------------------------------------------------------------------

@dataclass
class Candidate:
    n       : int
    s       : float       # m
    overhang: float       # m
    t_slab  : float       # mm
    D       : float       # mm
    bf      : float       # mm
    tf      : float       # mm
    tw      : float       # mm
    dw      : float       # mm
    section : SteelSection


def normalize_candidate(x: np.ndarray, cfg: OptiConfig) -> Candidate:
    """
    Snap a raw DE vector onto a valid, manufacturable section + layout.

    ``s`` (girder spacing) is the primary layout variable, clamped to [2.0, 4.0] m.
    ``n`` (number of girders) is *derived* from ``s`` so that:
        n = round(width / s),  then  s = width / n  and  overhang = s / 2
    This ensures the bridge layout solver's convention is always satisfied:
        overall_width = n * spacing,  overhang = spacing / 2
    """
    s, t_slab, D, bf, tf, tw = (
        x[0], x[1], x[2], x[3], x[4], x[5]
    )

    span  = cfg.span_m
    width = cfg.deck_width_m

    # Layout ----------------------------------------------------------------
    # Spacing is the PRIMARY variable. Clamp to practical limits [2.0, 4.0] m,
    # then derive the integer girder count and re-compute the exact spacing.
    s        = clamp(s, 2.0, 4.0)
    n        = max(2, int(round(width / s)))
    s        = width / n          # exact spacing from integer n
    overhang = 0.5 * s
    t_slab   = _round5(clamp(t_slab, 150.0, 250.0))

    # Depth — respect span/depth ratio bounds, snap to 10 mm.
    D_lo = (span / 25.0) * 1000.0
    D_hi = (span / 15.0) * 1000.0
    D    = _ceil10(clamp(D, D_lo, D_hi))

    # Flange width — keep as a fraction of D so scaling stays proportional.
    bf_frac = clamp(bf / D, 0.20, 0.40)
    bf      = _round5(bf_frac * D)

    # Flange thickness — IS 2062 plate; shrink one step at a time if the web
    # depth collapses to <= 0.
    tf = _ceiling_plate(clamp(tf, 6.0, 100.0))
    dw = D - 2.0 * tf
    while dw <= 0.0 and tf > float(_STD_PLATES[0]):
        idx = _STD_PLATES.index(int(tf)) if int(tf) in _STD_PLATES else len(_STD_PLATES) - 1
        tf  = float(_STD_PLATES[max(0, idx - 1)])
        dw  = D - 2.0 * tf

    # Web thickness — IS 2062 plate; enforce slenderness minimum tw >= dw/250.
    min_tw = clamp(max(dw / 250.0, 6.0), 6.0, 40.0)
    tw     = max(_ceiling_plate(clamp(tw, 6.0, 40.0)), _ceiling_plate(min_tw))
    tw     = clamp(tw, 6.0, 40.0)

    section = SteelSection(D, bf, tf, bf, tf, tw)
    return Candidate(n, s, overhang, t_slab, D, bf, tf, tw, dw, section)


def candidate_weight_components(cand: Candidate, cfg: OptiConfig) -> dict:
    """
    Per-component superstructure weight breakdown (kN). Keeps the objective and
    the user-facing breakdown in one place so they can never disagree.

      steel_girders_kN : all N girders, A_steel x span x rho_steel
      deck_concrete_kN : deck slab volume x rho_concrete
      total_kN         : the optimisation objective
    """
    span  = cfg.span_m
    width = cfg.deck_width_m
    w_steel    = cand.n * cand.section.A_steel * span * cfg.steel_density / 1e6
    w_concrete = cand.t_slab * width * span * cfg.concrete_density / 1e3
    return {
        "steel_girders_kN": w_steel,
        "deck_concrete_kN": w_concrete,
        "total_kN":         w_steel + w_concrete,
    }


def candidate_weight(cand: Candidate, cfg: OptiConfig) -> float:
    """Proportional superstructure weight (kN): girder steel + deck concrete."""
    return candidate_weight_components(cand, cfg)["total_kN"]


def candidate_input_dict(cand: Candidate, cfg: OptiConfig) -> dict:
    """
    Build a full ``input_dict`` for this candidate: deep-copy the solved base
    dict, switch to fixed (Custom) mode, and overwrite the layout + girder
    section keys (girder dims in **mm**; the pipeline converts to metres).
    """
    d = copy.deepcopy(cfg.base_input_dict)

    d[KEY_DESIGN_MODE]        = _FIXED_DESIGN_MODE
    d[KEY_TS_NO_OF_GIRDERS]   = cand.n
    d[KEY_TS_GIRDER_SPACING]  = cand.s
    d[KEY_TS_DECK_OVERHANG]   = cand.overhang
    d[KEY_TS_DECK_THICKNESS]  = cand.t_slab

    # Girder dims, in mm. resolve_girder_value() resolves the un-suffixed base
    # key first, but per-girder G{i}.M1 keys are set too so every girder picks
    # up the same symmetric section regardless of how consumers index it.
    dims_mm = {
        KEY_MP_GIRDER_DEPTH:                   cand.D,
        KEY_MP_GIRDER_WEB_DEPTH:               cand.dw,
        KEY_MP_GIRDER_TOP_FLANGE_WIDTH:        cand.bf,
        KEY_MP_GIRDER_BOTTOM_FLANGE_WIDTH:     cand.bf,
        KEY_MP_GIRDER_TOP_FLANGE_THICKNESS:    cand.tf,
        KEY_MP_GIRDER_BOTTOM_FLANGE_THICKNESS: cand.tf,
        KEY_MP_GIRDER_WEB_THICKNESS:           cand.tw,
    }
    for base_key, val in dims_mm.items():
        d[base_key] = val
        for gi in range(cand.n):
            d[f"{base_key}.G{gi + 1}.M1"] = val

    return d


# ------------------------------------------------------------------------------
#  Feasibility via the real, maintained bridge pipeline (up to DCR only)
# ------------------------------------------------------------------------------

class _NullWriter:
    def write(self, _): pass
    def flush(self): pass


def reset_design(bridge: "Optional[PlateGirderBridge]") -> None:
    """
    Clear ONE evaluated design from memory.

    A single candidate produces a fully analysed ``PlateGirderBridge`` that holds
    large objects — the grillage model, the xarray result datasets, the IRC 22
    DCR engine, the deck-design output, the load-effect / deflection caches — and
    leaves this design's nodes / elements / loads in the OpenSees C++ global
    domain. None of that is needed once the design's details have been recorded.
    Across a 200-design run it would otherwise pile up, so we wipe it *per design*:
    design 1 is freed the instant its details are saved, then design 2 is built,
    evaluated, freed, and so on. Memory stays flat instead of growing with N.

      1. Wipe the OpenSees global domain (process-global C++ state shared by every
         design in this worker).
      2. Detach the bridge's heavy attributes so the Python objects — which form
         reference cycles via numpy / xarray and so survive plain refcounting —
         become collectable.
      3. Force a gc pass to reclaim them now, not at some later collection.

    Safe to call with ``bridge is None`` or on a partially built bridge: every
    step is best-effort.
    """
    # 1. OpenSees C++ global domain — shared across all designs in this process.
    try:
        import openseespy.opensees as ops
        ops.wipe()
    except Exception:
        pass

    # 2. Drop this design's heavy Python state.
    if bridge is not None:
        gm = getattr(bridge, "grillage_model", None)
        if gm is not None:
            # the ospgrillage / OpenSees model object and its dataset
            for attr in ("model", "_deduplicated_results", "_results"):
                try:
                    setattr(gm, attr, None)
                except Exception:
                    pass
        for attr in (
            "grillage_model", "result_data", "_results_with_envelope",
            "_dcr_engine", "design_results", "output_dict",
            "_load_effects_cache", "_deflections_cache",
            "_lc_summary", "_reaction_summary",
        ):
            try:
                setattr(bridge, attr, None)
            except Exception:
                pass

    # 3. Reclaim immediately so memory does not creep across the 200 designs.
    gc.collect()


def candidate_is_feasible(cand: Candidate, cfg: OptiConfig) -> bool:
    """
    Run the same staged pipeline ``PlateGirderBridge.design()`` runs, up to and
    including the IRC 22:2015 DCR checks, then report whether the controlling
    girder passes. Deck / transverse / CAD stages are intentionally skipped.

    The design's heavy state (grillage model, result datasets, OpenSees domain)
    is freed via ``reset_design`` in the ``finally`` the instant its DCR status
    has been read — so memory stays flat across the whole run.
    """
    bridge = PlateGirderBridge()
    try:
        bridge.set_input(candidate_input_dict(cand, cfg))

        # Pre-stage unit handling + stages 1..4G, mirroring design().
        bridge._resolve_optimized_bounds_to_mm()   # no-op in Custom mode
        bridge._convert_girder_dims_mm_to_m()
        bridge._validate_inputs()
        bridge._solve_bridge_layout()
        bridge._stage_grillage_setup()
        bridge.add_dead_loads()
        bridge.add_live_loads()
        bridge.add_wind_loads()
        bridge.add_temperature_load()
        bridge.add_seismic_loads()
        bridge._stage_load_combinations()
        dataset = bridge._reanalyze_with_dedup()
        dataset = bridge.create_envelope_load_case(dataset)

        # Stage 5: DCR checks. Populates bridge._dcr_engine.
        bridge._run_dcr_checks(dataset)

        engine = getattr(bridge, "_dcr_engine", None)
        if engine is None:
            return False
        return engine.overall_status() != "FAIL"
    finally:
        # This design's details have now been read, so clear it from memory
        # immediately — before the next candidate is built.
        reset_design(bridge)


# ------------------------------------------------------------------------------
#  Multiprocessing start method (cross-platform)
# ------------------------------------------------------------------------------

def default_start_method() -> str:
    """
    Best multiprocessing start method for the current OS.

    - Linux / macOS : "forkserver" — workers fork from a clean helper process,
      never inheriting the parent's non-fork-safe native state (Qt, OpenSees).
    - Windows        : "spawn" — the only method available; each worker is a
      fresh interpreter.

    This is chosen for the *current* process. It is safe to call from the
    isolated optimisation runner (a clean, Qt-free process), where "spawn"
    re-imports only the runner module — not the GUI app's ``__main__``.
    """
    import multiprocessing as _mp
    methods = _mp.get_all_start_methods()
    if "forkserver" in methods:
        return "forkserver"
    return "spawn"


# ------------------------------------------------------------------------------
#  Worker side: one global config per process, one picklable fitness function
# ------------------------------------------------------------------------------

_CFG: Optional[OptiConfig] = None
# Per-worker handle to the live core-monitor queue (a Manager().Queue passed in
# by the parent via _init_worker). None when no UI asked for a monitor.
_EVENT_Q = None


def _init_worker(cfg: OptiConfig, event_q=None) -> None:
    """Pool initializer: install the run config and silence pipeline output.

    ``event_q`` is the optional live core-monitor queue. Each worker pushes its
    start/done events onto it; the parent drains them and forwards to the UI.
    """
    global _CFG, _EVENT_Q
    _CFG = cfg
    _EVENT_Q = event_q
    # The design pipeline is chatty (stdout) and emits many UserWarnings
    # (no footpath/railing/median) for every candidate. Silence both so they
    # don't flood the parent's terminal during the parallel search.
    import warnings
    sys.stdout = _NullWriter()
    sys.stderr = _NullWriter()
    warnings.filterwarnings("ignore")


def _emit_core_event(ev: dict) -> None:
    """Best-effort push of a live 'which core is on which design' event."""
    q = _EVENT_Q
    if q is None:
        return
    try:
        q.put(ev)
    except Exception:
        pass


def _fitness(x: np.ndarray) -> float:
    """
    Picklable fitness used by every worker. Returns the candidate's weight if it
    is feasible, ``+inf`` otherwise (so infeasible candidates always lose greedy
    selection). Any exception in the heavy pipeline is treated as infeasible.

    Memory note: ``candidate_is_feasible`` already frees the design via
    ``reset_design`` the moment its details are read. The ``finally`` here is a
    backstop ``reset_design(None)`` — it wipes the OpenSees global domain and
    forces a gc pass even if the candidate blew up before a bridge was built — so
    memory stays flat across the whole run.
    """
    import os
    import time

    cfg = _CFG
    if cfg is None:
        raise RuntimeError("_fitness called before _init_worker installed OptiConfig")

    pid = os.getpid()
    t0  = time.time()
    try:
        cand = normalize_candidate(np.asarray(x, dtype=float), cfg)
        # Tell the live monitor this core has STARTED this design.
        _emit_core_event({
            "ev": "start", "pid": pid,
            "n": cand.n, "t_slab": cand.t_slab, "D": cand.D,
            "bf": cand.bf, "tf": cand.tf, "tw": cand.tw,
        })
        feasible = candidate_is_feasible(cand, cfg)
        weight   = candidate_weight(cand, cfg) if feasible else None
        _emit_core_event({
            "ev": "done", "pid": pid, "feasible": bool(feasible),
            "weight": weight, "secs": round(time.time() - t0, 1),
        })
        return weight if feasible else math.inf
    except Exception:
        _emit_core_event({
            "ev": "done", "pid": pid, "feasible": False,
            "weight": None, "secs": round(time.time() - t0, 1),
        })
        return math.inf
    finally:
        # Backstop: candidate_is_feasible already reset its bridge; this covers
        # the path where it never got that far.
        reset_design(None)


# ------------------------------------------------------------------------------
#  Bounds + warm-start (parent process only — closure over cfg is fine here)
# ------------------------------------------------------------------------------

def _make_bounds(cfg: OptiConfig):
    """Return a bounds_func(x) -> (n_dims, 2) array. Called only in the parent."""
    span  = cfg.span_m
    width = cfg.deck_width_m

    def bounds(x: np.ndarray) -> np.ndarray:
        D    = x[2]        # was x[3], now shifted because n is removed
        D_lo = span * 1000.0 / 25.0
        D_hi = span * 1000.0 / 15.0
        return np.array([
            [2.0,        4.0],                            # s   (m) — primary variable
            [150.0,      250.0],                          # t_slab (mm)
            [D_lo,       D_hi],                           # D   (mm)
            [0.20 * D,   0.40 * D],                       # bf  (mm)
            [6.0,        100.0],                          # tf  (mm)
            [6.0,        40.0],                           # tw  (mm)
        ])

    return bounds


def initial_guess(cfg: OptiConfig) -> np.ndarray:
    """DDCL empirical warm-start point seeded into population slot 0."""
    span  = cfg.span_m
    width = cfg.deck_width_m

    D   = _ceil10(span * 1000.0 / 18.0)     # typical plate-girder depth/span
    bf  = _round5(0.3 * D)
    tf  = _ceiling_plate(bf / 24.0)
    dw  = D - 2.0 * tf
    tw  = _ceiling_plate(max(dw / 200.0, 6.0))

    s      = clamp(width / 4.0, 2.0, 4.0)   # aim for ~3m spacing as default
    t_slab = 150.0
    return np.array([s, t_slab, D, bf, tf, tw], dtype=float)


# ------------------------------------------------------------------------------
#  Public entry point
# ------------------------------------------------------------------------------

def optimize_parallel(
    base_input_dict : dict,
    steel_density   : float          = 78.5,
    concrete_density: float          = 25.0,
    pop_size        : int            = 50,
    generations     : int            = 300,
    tol             : float          = 1e-1,
    seed            : int            = 42,
    max_workers     : Optional[int]  = None,
    build_cad       : bool           = False,
    on_candidate    : "Optional[callable]" = None,
    on_core_event   : "Optional[callable]" = None,
    start_method    : "Optional[str]" = None,
    max_tasks_per_child : Optional[int] = 1,
) -> PlateGirderBridge:
    """
    Optimise the plate-girder superstructure for minimum weight, evaluating the
    DE population in parallel across CPU cores.

    Parameters
    ----------
    base_input_dict : a fully solved bridge input_dict (post
                      ``solve_extend_basic_input_dict``) carrying span, overall
                      width, materials, loading, etc. Candidates override only
                      the layout + girder-section keys.
    build_cad       : if True, run the full ``design()`` (incl. CAD) on the
                      winning design before returning.
    on_candidate    : optional UI hook called once per evaluated candidate with
                      a single info dict:
                        {done, total, gen, feasible, weight_kN, components,
                         best_kN, vector}
                      ``done``/``total`` give the "x/50" progress (total ==
                      pop_size); ``components`` is the per-component weight
                      breakdown for feasible candidates (else None).

    Returns
    -------
    PlateGirderBridge built from the best feasible design vector. Raises
    ``RuntimeError`` if no feasible candidate was found.
    """
    cfg = OptiConfig.from_input_dict(base_input_dict, steel_density, concrete_density)
    if start_method is None:
        start_method = default_start_method()   # forkserver on POSIX, spawn on Windows

    # Live per-core monitor plumbing. Workers push start/done events onto a
    # Manager queue (picklable across any start method); a background thread here
    # drains it and forwards to on_core_event. Only set up when a UI asked for it.
    import threading
    _mgr      = None
    _event_q  = None
    _drainer  = None
    _stop_drn = threading.Event()
    if on_core_event is not None:
        import multiprocessing as _mp
        _mgr     = _mp.Manager()
        _event_q = _mgr.Queue()

        def _drain_events():
            while not _stop_drn.is_set():
                try:
                    ev = _event_q.get(timeout=0.2)
                except Exception:
                    continue
                if ev is None:
                    break
                try:
                    on_core_event(ev)
                except Exception:
                    pass
            # flush anything still queued at shutdown
            while True:
                try:
                    ev = _event_q.get_nowait()
                except Exception:
                    break
                if ev is None:
                    continue
                try:
                    on_core_event(ev)
                except Exception:
                    pass

        _drainer = threading.Thread(target=_drain_events, daemon=True)
        _drainer.start()

    # Translate the engine's low-level progress into a UI-friendly info dict,
    # tracking the running best so the loader can show it live. The bar is made
    # monotonic over the whole run (initial population + every generation) so it
    # fills to 100% instead of resetting each generation.
    best_so_far = {"kN": math.inf}
    total_batches = generations + 1   # batch 0 = initial population
    records: List[dict] = []          # one row per evaluated design, for the report tab

    def _engine_progress(done, total, gen, vector, fitness):
        feasible = math.isfinite(fitness)
        if feasible and fitness < best_so_far["kN"]:
            best_so_far["kN"] = fitness

        # Always recover the candidate's manufacturable dims + weight breakdown
        # so the report shows the actual design that was evaluated (feasible or
        # not). For infeasible candidates the weight is still meaningful.
        cand       = normalize_candidate(np.asarray(vector, dtype=float), cfg)
        components = candidate_weight_components(cand, cfg)

        record = {
            "index":      len(records) + 1,
            "gen":        gen,
            "round":      "Initial" if gen == 0 else f"Gen {gen}",
            "feasible":   feasible,
            "status":     "PASS" if feasible else "FAIL",
            # design values actually used (post-snapping)
            "n":          cand.n,
            "spacing_m":  round(cand.s, 3),
            "t_slab_mm":  cand.t_slab,
            "D_mm":       cand.D,
            "bf_mm":      cand.bf,
            "tf_mm":      cand.tf,
            "tw_mm":      cand.tw,
            "dw_mm":      cand.dw,
            # weights
            "steel_kN":   round(components["steel_girders_kN"], 1),
            "deck_kN":    round(components["deck_concrete_kN"], 1),
            "total_kN":   round(components["total_kN"], 1),
        }
        records.append(record)

        if on_candidate is None:
            return
        overall_done  = gen * total + done
        overall_total = total_batches * total
        on_candidate({
            "done":          done,
            "total":         total,
            "gen":           gen,
            "overall_done":  overall_done,
            "overall_total": overall_total,
            "feasible":      feasible,
            "weight_kN":     fitness if feasible else None,
            "components":    components,
            "best_kN":       best_so_far["kN"] if math.isfinite(best_so_far["kN"]) else None,
            "vector":        np.asarray(vector, dtype=float),
            "record":        record,
        })

    try:
        result: OptimisationResult = ParallelOptimizer.run(
            fitness_func    = _fitness,
            bounds_func     = _make_bounds(cfg),
            initial_guess   = initial_guess(cfg),
            tol             = tol,
            pop_size        = pop_size,
            generations     = generations,
            seed            = seed,
            max_workers     = max_workers,
            worker_init     = _init_worker,
            worker_initargs = (cfg, _event_q),
            progress_cb     = _engine_progress,
            # OS-aware: forkserver on POSIX, spawn on Windows (see default_start_method).
            # Intended to run inside the isolated `run_optimization` process, where
            # neither method touches the GUI app's __main__.
            start_method    = start_method,
            # Recycle each worker after one design so its memory (incl. C-level
            # OpenSees / ospgrillage state) is returned to the OS per design, not
            # only when the whole run ends.
            max_tasks_per_child = max_tasks_per_child,
        )
    finally:
        # Tear down the live-monitor drainer + manager.
        _stop_drn.set()
        if _event_q is not None:
            try: _event_q.put(None)
            except Exception: pass
        if _drainer is not None:
            _drainer.join(timeout=2.0)
        if _mgr is not None:
            try: _mgr.shutdown()
            except Exception: pass

    if not result.feasible:
        raise RuntimeError(
            "Optimisation found no feasible design — relax bounds, loosen tol, "
            "or check the base input_dict / loading."
        )

    print("Convergence achieved" if result.converged else "Convergence not achieved")
    print(f"Best superstructure weight: {result.best_fitness:.1f} kN")

    # Rebuild the winning bridge in the parent process from its design vector.
    cand   = normalize_candidate(result.best_vector, cfg)
    bridge = PlateGirderBridge()
    bridge.set_input(candidate_input_dict(cand, cfg))

    # Expose the full run for the report tab: every evaluated design (feasible or
    # not) with its values + weights, plus which one won.
    best_total = round(candidate_weight_components(cand, cfg)["total_kN"], 1)
    for r in records:
        r["is_best"] = (
            r["feasible"]
            and r["n"] == cand.n and r["D_mm"] == cand.D
            and r["bf_mm"] == cand.bf and r["tf_mm"] == cand.tf
            and r["tw_mm"] == cand.tw and r["t_slab_mm"] == cand.t_slab
            and abs(r["total_kN"] - best_total) < 1e-6
        )
    bridge.optimization_records = records
    bridge.optimization_summary = {
        "evaluated":   len(records),
        "feasible":    sum(1 for r in records if r["feasible"]),
        "infeasible":  sum(1 for r in records if not r["feasible"]),
        "best_total_kN": best_total,
        "converged":   result.converged,
    }

    if build_cad:
        bridge.design()

    return bridge
