"""collect_optimization_dataset.py
================================
Standalone dataset collector for the plate-girder bridge optimiser.

What it does
------------
Builds a full bridge ``input_dict`` exactly the way the GUI does
(``BASIC_INPUT_DICT`` + ``solve_extend_basic_input_dict``) and runs the
parallel Differential-Evolution weight optimisation (``optimize_parallel``)
for every combination of:

- span
- carriageway width
- median inclusion
- footpath option
- skew angle
- **girder steel grade** (E 350A / E 410A / E 450A)
- **deck concrete grade** (M40 / M50 / M60)

Every evaluated design is one row: the fixed inputs for that run plus
all DE-varying section variables and the resulting weights / feasibility.

The growing table is pickled (a pandas DataFrame) to
``ResourceFiles/optimization_dataset.pkl`` after every completed sweep
combination, so a crash never loses already-collected data. Load it back with
``pandas.read_pickle(...)``.

Run it
------
    cd OsdagBridge/src
    conda run -n osdag-env python collect_optimization_dataset.py

(Must be run from ``OsdagBridge/src`` with the ``osdag-env`` interpreter so the
``osdagbridge`` package and OpenSees / ospgrillage resolve.)
"""
from __future__ import annotations

import copy
import os
import time
from itertools import product
from typing import Iterable

import pandas as pd

from osdagbridge.core.bridge_types.plate_girder.defaults import (
    BASIC_INPUT_DICT,
    solve_extend_basic_input_dict,
)
from osdagbridge.core.optimizer.parallel_bridge_optimizer import optimize_parallel
from osdagbridge.core.utils.common import (
    KEY_CARRIAGEWAY_WIDTH,
    KEY_DECK_CONCRETE_GRADE_BASIC,
    KEY_FOOTPATH,
    KEY_GIRDER,
    KEY_INCLUDE_MEDIAN,
    KEY_SKEW_ANGLE,
    KEY_SPAN,
)

print("SCRIPT STARTED", flush=True)

# ------------------------------------------------------------------------------
# Sweep configuration
# ------------------------------------------------------------------------------

# Keep the original span sweep unless you want to tighten / expand it later.
SPAN_START = 20.0
SPAN_END = 25.0
SPAN_STEP = 0.5

# Carriageway sweep. 23.6 is not on a 0.25 grid, so the nearest aligned upper
# bound is 23.5. Change this to 23.75 if you want to include a value above 23.6.
CARRIAGEWAY_START = 4.25
CARRIAGEWAY_END = 10.0
CARRIAGEWAY_STEP = 0.25

# Fixed categorical layout inputs.
MEDIAN_OPTIONS = ["No", "Yes"]
FOOTPATH_OPTIONS = ["None", "Single Side", "Both Sides"]

# Skew sweep.
SKEW_START = -15
SKEW_END = 15
SKEW_STEP = 5

# Material grade sweep options (using same values as the GUI comboboxes).
# Steel grades from Steel_Grade_Properties DB table.
STEEL_GRADE_OPTIONS = ["E 350A", "E 410A", "E 450A"]
# Concrete grades from Concrete_Grade_Properties DB table.
CONCRETE_GRADE_OPTIONS = ["M40", "M50", "M60"]

# DE run size: 50 x (3 + 1) = 200 designs per optimisation call.
POP_SIZE = 50
GENERATIONS = 3

# Output location.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_THIS_DIR, "ResourceFiles")
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "optimization_dataset.pkl")


def _frange(start: float, end: float, step: float) -> list[float]:
    """Inclusive float range with rounded endpoints for grid-safe iteration."""
    count = int(round((end - start) / step))
    return [round(start + i * step, 10) for i in range(count + 1)]


def _int_range(start: int, end: int, step: int) -> list[int]:
    """Inclusive integer range."""
    return list(range(start, end + 1, step))


def build_base_input_dict(
    span_m: float,
    carriageway_width_m: float,
    include_median: str,
    footpath: str,
    skew_angle_deg: float,
    steel_grade: str = "E 350A",
    concrete_grade: str = "M40",
) -> dict:
    """
    Build a fully solved bridge ``input_dict`` for one sweep combination,
    mirroring what the GUI does before launching the optimiser:

      1. deep-copy the package default dict,
      2. fill in the required basic inputs,
      3. run ``solve_extend_basic_input_dict`` to compute layout and defaults
         in place.
    """
    d = copy.deepcopy(BASIC_INPUT_DICT)
    d[KEY_SPAN] = span_m
    d[KEY_CARRIAGEWAY_WIDTH] = carriageway_width_m
    d[KEY_SKEW_ANGLE] = skew_angle_deg
    d[KEY_INCLUDE_MEDIAN] = include_median
    d[KEY_FOOTPATH] = footpath

    # Set material grades (override the BASIC_INPUT_DICT defaults)
    d[KEY_GIRDER] = steel_grade
    d[KEY_DECK_CONCRETE_GRADE_BASIC] = concrete_grade

    solve_extend_basic_input_dict(d)  # in-place: layout + every default
    return d


def collect_for_case(
    span_m: float,
    carriageway_width_m: float,
    include_median: str,
    footpath: str,
    skew_angle_deg: float,
    steel_grade: str = "E 350A",
    concrete_grade: str = "M40",
) -> list[dict]:
    """
    Run one optimisation for a single sweep combination and return one row dict
    per evaluated design.

    Records are captured live via ``on_candidate`` so they survive even if the
    optimiser raises at the end (for example, if no feasible winner exists).
    """
    # --- detailed per-case input logging --------------------------------------
    print(
        "    [inputs] "
        f"span_m={span_m}, carriageway_width_m={carriageway_width_m}, "
        f"include_median={include_median!r}, footpath={footpath!r}, "
        f"skew_angle_deg={skew_angle_deg}, "
        f"steel_grade={steel_grade!r}, concrete_grade={concrete_grade!r}",
        flush=True,
    )

    t_build = time.time()
    base = build_base_input_dict(
        span_m=span_m,
        carriageway_width_m=carriageway_width_m,
        include_median=include_median,
        footpath=footpath,
        skew_angle_deg=skew_angle_deg,
        steel_grade=steel_grade,
        concrete_grade=concrete_grade,
    )
    build_dt = time.time() - t_build
    print(f"    [inputs] input_dict built in {build_dt:.2f}s", flush=True)

    rows: list[dict] = []
    _last_t = [time.time()]  # mutable holder so the closure can update it

    def _tag(record: dict) -> dict:
        r = dict(record)
        r["span_m"] = span_m
        r["carriageway_width_m"] = carriageway_width_m
        r["include_median"] = include_median
        r["footpath"] = footpath
        r["skew_angle_deg"] = skew_angle_deg
        r["steel_grade"] = steel_grade
        r["concrete_grade"] = concrete_grade
        r.pop("is_best", None)
        return r

    def _on_candidate(info: dict) -> None:
        rec = info.get("record")
        if rec is not None:
            now = time.time()
            design_dt = now - _last_t[0]
            _last_t[0] = now
            rows.append(_tag(rec))

            # Live per-design log so you can watch every evaluated candidate,
            # including how long that single design took to evaluate.
            def _g(key: str) -> str:
                v = rec.get(key)
                return "-" if v is None else f"{v}"

            print(
                f"      [design #{_g('index')}] "
                f"round={_g('round')} gen={_g('gen')} | "
                f"n={_g('n')} spacing={_g('spacing_m')}m t_slab={_g('t_slab_mm')}mm "
                f"D={_g('D_mm')}mm bf={_g('bf_mm')}mm tf={_g('tf_mm')}mm "
                f"tw={_g('tw_mm')}mm dw={_g('dw_mm')}mm | "
                f"steel={_g('steel_kN')}kN deck={_g('deck_kN')}kN total={_g('total_kN')}kN | "
                f"feasible={_g('feasible')} status={_g('status')} | "
                f"took {design_dt:.2f}s",
                flush=True,
            )

    best_records = None
    t_opt = time.time()
    try:
        bridge = optimize_parallel(
            base,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            build_cad=False,  # dataset only — no CAD
            on_candidate=_on_candidate,
        )
        best_records = getattr(bridge, "optimization_records", None)
    except RuntimeError as exc:
        # No feasible design for this combination — keep the rows captured live.
        print(
            f"    [warn] span={span_m}, width={carriageway_width_m}, "
            f"median={include_median}, footpath={footpath}, skew={skew_angle_deg}: {exc}",
            flush=True,
        )
    opt_dt = time.time() - t_opt
    print(f"    [inputs] optimise_parallel finished in {opt_dt:.2f}s", flush=True)

    # Prefer the optimiser's final records (guaranteed complete); fall back to
    # the live-captured rows.
    if best_records:
        rows = [_tag(r) for r in best_records]

    return rows


# Column order for the saved dataset: fixed inputs first, then the DE variables,
# then weights / status.
_COLUMNS = [
    "span_m",
    "carriageway_width_m",
    "include_median",
    "footpath",
    "skew_angle_deg",
    "steel_grade",
    "concrete_grade",
    "index",
    "round",
    "gen",
    "n",
    "spacing_m",
    "t_slab_mm",
    "D_mm",
    "bf_mm",
    "tf_mm",
    "tw_mm",
    "dw_mm",
    "steel_kN",
    "deck_kN",
    "total_kN",
    "feasible",
    "status",
]


def _save(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Keep known columns in a stable order; append any extras at the end.
    ordered = [c for c in _COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_pickle(OUTPUT_PKL)


def _sweep_cases() -> Iterable[tuple[float, float, str, str, int, str, str]]:
    spans = _frange(SPAN_START, SPAN_END, SPAN_STEP)
    carriageways = _frange(CARRIAGEWAY_START, CARRIAGEWAY_END, CARRIAGEWAY_STEP)
    skews = _int_range(SKEW_START, SKEW_END, SKEW_STEP)
    return product(spans, carriageways, MEDIAN_OPTIONS, FOOTPATH_OPTIONS, skews,
                   STEEL_GRADE_OPTIONS, CONCRETE_GRADE_OPTIONS)


def main() -> None:
    spans = _frange(SPAN_START, SPAN_END, SPAN_STEP)
    carriageways = _frange(CARRIAGEWAY_START, CARRIAGEWAY_END, CARRIAGEWAY_STEP)
    skews = _int_range(SKEW_START, SKEW_END, SKEW_STEP)

    total_cases = (
        len(spans)
        * len(carriageways)
        * len(MEDIAN_OPTIONS)
        * len(FOOTPATH_OPTIONS)
        * len(skews)
        * len(STEEL_GRADE_OPTIONS)
        * len(CONCRETE_GRADE_OPTIONS)
    )
    designs_per_case = POP_SIZE * (GENERATIONS + 1)

    print(
        "Collecting optimisation dataset with sweep grid:\n"
        f"  spans: {spans[0]} .. {spans[-1]} (step {SPAN_STEP})\n"
        f"  carriageway widths: {carriageways[0]} .. {carriageways[-1]} (step {CARRIAGEWAY_STEP})\n"
        f"  median: {MEDIAN_OPTIONS}\n"
        f"  footpath: {FOOTPATH_OPTIONS}\n"
        f"  skew angle: {skews[0]} .. {skews[-1]} (step {SKEW_STEP})\n"
        f"  girder steel grade: {STEEL_GRADE_OPTIONS}\n"
        f"  deck concrete grade: {CONCRETE_GRADE_OPTIONS}\n"
        f"  DE run: {POP_SIZE} x {GENERATIONS + 1} = {designs_per_case} designs per case\n"
        f"  total optimisation cases: {total_cases}\n"
        f"Output -> {OUTPUT_PKL}\n",
        flush=True,
    )

    all_rows: list[dict] = []
    t0 = time.time()

    for i, (span, width, median, footpath, skew, steel_grade, concrete_grade) in enumerate(_sweep_cases(), start=1):
        elapsed = time.time() - t0
        avg = elapsed / (i - 1) if i > 1 else 0.0
        eta = avg * (total_cases - (i - 1))
        print(
            f"[{i}/{total_cases}] span={span} m, width={width} m, median={median}, "
            f"footpath={footpath}, skew={skew} deg, "
            f"steel={steel_grade}, concrete={concrete_grade} "
            f"(elapsed {elapsed / 60:.1f} min, avg {avg:.1f}s/case, ETA {eta / 3600:.1f} h) ...",
            flush=True,
        )
        t_case = time.time()

        case_rows = collect_for_case(
            span_m=span,
            carriageway_width_m=width,
            include_median=median,
            footpath=footpath,
            skew_angle_deg=skew,
            steel_grade=steel_grade,
            concrete_grade=concrete_grade,
        )

        all_rows.extend(case_rows)
        _save(all_rows)  # checkpoint after every case

        print(
            f"    -> {len(case_rows)} designs collected ({len(all_rows)} total) "
            f"in {time.time() - t_case:.1f}s; saved to {os.path.relpath(OUTPUT_PKL, _THIS_DIR)}",
            flush=True,
        )

    print(
        f"\nDone. {len(all_rows)} rows across {total_cases} optimisation cases "
        f"in {time.time() - t0:.1f}s -> {OUTPUT_PKL}",
        flush=True,
    )


if __name__ == "__main__":
    main()
