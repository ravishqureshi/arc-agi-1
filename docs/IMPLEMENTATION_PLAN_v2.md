# ARC‑AGI Implementation Plan v2 — Fixed‑Point Closures (Submission‑First)

## 1) Goal & Gates
- **Goal:** produce a correct `predictions.json` for ARC‑AGI v1 (and align with ARC‑AGI‑2 constraints) using the **master operator** view.
- **Hard gates:**  
  1) One parameterization across **all train pairs** (observer = observed).  
  2) **Train exactness** only (no approximate fits).  
  3) Deterministic outputs and schema‑valid predictions.

## 2) Architecture (single runtime path)
- Treat each test grid as a **set‑valued grid** (10‑bit mask per cell).
- A **Closure** is a **monotone, shrinking, idempotent** map `apply(U) -> U'` that only **clears bits**.
- Build closures by **unifying parameters on train**, then iterate them to the **least fixed point**:
  - `U_{k+1} = (T_m ∘ ... ∘ T_1)(U_k)` until convergence.
  - If every cell is a singleton → decode grid = solution.
  - If some cells remain multi‑valued → **under‑constrained**; add another closure later.

**Code:**  
- Engine: `src/arc_solver/closure_engine.py`  
- Closures & unifiers: `src/arc_solver/closures.py`  
- Orchestration: `src/arc_solver/search.py::autobuild_closures`, `solve_with_closures`  
- CLI: `scripts/run_public.py` (writes predictions + receipts)

## 3) Minimal Receipts (per task)
- One JSONL line with: `{task, status, closures[{name,params}], fp{iters,cells_multi}, timing_ms, hashes{task_sha, closure_set_sha}}`.
- No dashboards, no heavy bills. Good enough for submission and triage.

## 4) Closure Families — Minimal High‑Leverage Set (order of implementation)
Implement in this order; stop when coverage stabilizes.

1. **KEEP_LARGEST_COMPONENT** — not‑largest cells narrow to `{bg}`.  
2. **OUTLINE_OBJECTS** — outline (outer/inner), scope (largest/all).  
3. **OPEN/CLOSE (k=1)** — morphology to close holes or remove spurs.  
4. **AXIS_PROJECTION_FILL** — extend along row/col to border.  
5. **SYMMETRY_COMPLETION** — union with v/h/diag/anti reflection.  
6. **MOD_PATTERN (p,q,anchor)** — general parity; congruence classes to colors.  
7. **DIAGONAL_REPEAT (dr,dc,k)** — tie color sets along diagonal chains.  
8. **TILING / TILING_ON_MASK** — motif on `ALL|NONZERO|BACKGROUND|NOT_LARGEST|PARITY|STRIPES`.  
9. **COPY_BY_DELTAS** — exact shifted‑mask equality (not centroid).

> Masks are **input‑only**; use `y` on train **only** to verify exact equality.

## 5) Unifier Pattern (applies to every family)
**Function:** `unify_<FAMILY>(train) -> list[Closure]`  
**Algorithm:**
1) Enumerate small candidate parameter sets (anchors, p/q, dr/dc/k, axis/scope, etc.).  
2) For each candidate, **build the closure** and call `verify_closures_on_train([candidate], train)`.  
3) If **every** train pair is exact, collect the candidate (usually only 1).  
4) Return the list of validated closures.

**Contract:** Same params must fit all train pairs; otherwise return `[]` and move on.

## 6) Closure.apply Contract (must‑hold properties)
- Intersect only (bitwise `AND`), **never add bits**.  
- Deterministic; no RNG, no wall‑clock, no IO.  
- ≤2 passes idempotence on typical `U` (practically stable).  
- All masks and geometry derived from **input `x`**; `y` used only in unifier verification.

## 7) Solve Path (per task)
1) Build closures via `autobuild_closures(train)` (fixed order above).  
2) For each test input `x`, run `run_fixed_point(closures, x)`.  
3) If singletons → decode and emit grid; else → deterministic fallback (lowest allowed color per cell) and mark `under_constrained`.  
4) Log the JSONL line (minimal receipts).

## 8) Tests & Acceptance
- For each family, add **2–3 tiny property tests**:
  - Train exactness on synthetic examples,
  - Shrinking property (`apply(U) ⊆ U`),
  - ≤2‑pass idempotence.
- Integration: `scripts/test_integration.py` must pass.
- CLI run: `scripts/run_public.py` produces predictions + receipts;  
  `scripts/determinism.sh` (jobs=1 vs jobs=N) must be byte‑identical;  
  `scripts/submission_validator.py` must pass.

## 9) Out‑of‑Scope (trimmed for submission)
- Beam search as runtime, fail clustering, auto‑nominators, mask boolean algebra library (beyond the listed templates), adjacency/merge/split/render families — add later only if needed.

## 10) Milestones (pragmatic)
- **M1**: Engine + KEEP_LARGEST + OUTLINE; CLI + validator + determinism pass.  
- **M2**: OPEN/CLOSE + AXIS_PROJECTION + SYMMETRY; coverage bump.  
- **M3**: MOD_PATTERN + DIAGONAL_REPEAT; periodic/diagonal tasks collapse.  
- **M4**: TILING(_ON_MASK) + COPY_BY_DELTAS; final baseline submission.

## Detailed Implementation Break down
Breakdown for **M1** only. It’s detailed like your B-items example, with exact file paths, function names, unifier scope, integration points, tiny tests, and acceptance. No step depends on future milestones. (Grounded in your v2 plan and earlier coverage plan.)  

---

### M1 — Fixed-Point Core + Two Closures + Submission Path ✅ COMPLETED

#### M1.1 — Fixed-Point Runtime Path is Primary ✅ COMPLETED

**What**
Lock the runtime to the master-operator fixed-point flow; no beam. Ensure we can build closures from train, run LFP on test, and log minimal receipts.

**Files**

* `scripts/run_public.py` (driver)
* `src/arc_solver/search.py` (`solve_with_closures`, `autobuild_closures`)
* `src/arc_solver/closure_engine.py` (`SetValuedGrid`, `Closure`, `run_fixed_point`, `verify_closures_on_train`)
* `src/arc_solver/utils.py` (`task_sha`, `closure_set_sha`, `log_receipt`)

**Do**

* Confirm `run_public.py` imports **`solve_with_closures`** and never calls beam.
* In `solve_with_closures`:

  * Build closures with `autobuild_closures(train)`; if none, mark `under_constrained`.
  * For each test input, call `run_fixed_point(closures, x_test)`; decode only if all cells are singletons; else use deterministic fallback (lowest allowed color).
  * Write one JSONL line per task with `{task, status, closures[{name,params}], fp{iters,cells_multi}, timing_ms, hashes{task_sha, closure_set_sha}}`. (Keep it minimal.) 

**Acceptance**

* Running one small dataset produces `runs/<date>/predictions.json` and `runs/<date>/receipts.jsonl` with those fields only.
* No references to beam in the driver path.

---

#### M1.2 — Closure #1: KEEP_LARGEST_COMPONENT ✅ COMPLETED

**Law (one-liner)**
All pixels not in the largest 4-connected component (ignoring `bg`) are forced to `{bg}`. 

**Files**

* `src/arc_solver/closures.py` (add closure class + unifier)
* `src/arc_solver/utils.py` (uses `components`)

**Add**

```python
# closures.py
class KEEP_LARGEST_COMPONENT_Closure(Closure):
    # name = "KEEP_LARGEST_COMPONENT"
    # params = {"bg": int}
    # apply(self, U): intersect-only; set non-largest pixels to {bg}

def unify_KEEP_LARGEST(train: list[tuple[np.ndarray,np.ndarray]]) -> list[Closure]:
    # Try bg in 0..9 or infer one; build candidate
    # verify_closures_on_train([candidate], train) → if exact on all pairs, return [candidate]; else []
```

**apply(U) – algorithm**

* From captured `x_input`, compute largest component mask `L` w.r.t. `bg`.
* Build per-cell bitmask that allows `{bg}` outside `L` and keeps current bits inside `L`.
* Return `U & mask` (intersect-only; deterministic; ≤2-pass stable).

**Unifier**

* Enumerate `bg ∈ {0..9}` (or infer from `x` background).
* For each, build the closure and require **train exactness** on every pair via `verify_closures_on_train`. Return the single validated closure (or `[]`).

**Integration**

* `src/arc_solver/search.py::autobuild_closures` → first line: `closures += unify_KEEP_LARGEST(train)`.

**Tiny tests**

* `tests/test_closures_minimal.py`:

  * Shrinking: `apply(U) ⊆ U`.
  * Idempotence ≤2 passes.
  * Train exactness on a 2-blob toy (largest kept).

**Acceptance**

* On toys, LFP yields singletons matching ground truth.
* Unifier returns at most one closure; returns `[]` if not exact.

---

#### M1.3 — Closure #2: OUTLINE_OBJECTS (outer, thickness=1) ✅ COMPLETED

**Law (one-liner)**
For chosen scope (largest | all), cells whose 4-neighbors include background are the **outline** and must take the object color; all non-outline object pixels clear. 

**Files**

* `src/arc_solver/closures.py` (add closure + unifier)

**Add**

```python
# closures.py
class OUTLINE_OBJECTS_Closure(Closure):
    # name = "OUTLINE_OBJECTS"
    # params = {"mode": "outer", "scope": "largest|all", "bg": int}
    # apply(self, U): build outline mask from x, intersect-only

def unify_OUTLINE_OBJECTS(train) -> list[Closure]:
    # candidates = (mode="outer", scope in {"largest","all"}, bg in 0..9 or inferred)
    # build closure; accept iff exact on all train pairs; return [candidate] or []
```

**apply(U) – algorithm**

* From `x_input`, compute components (bg) and a boolean `outline` mask:

  * For each chosen object set (largest or all), a pixel is outline if any of its 4-neighbors is background.
* Intersect `U` with:

  * For outline pixels: keep bits that include the object’s color.
  * For non-outline object pixels: clear all object-color bits (shrink).
  * Background remains intersected with current allowed set (no widening).

**Unifier**

* Enumerate `scope ∈ {"largest","all"}`, `bg` in {0..9} (or infer). `mode="outer"` fixed for M1 (no inner).
* Verify exactness on all train pairs.

**Integration**

* `src/arc_solver/search.py::autobuild_closures` → after KEEP_LARGEST: `closures += unify_OUTLINE_OBJECTS(train)`.
* Order aligns with plan (cheap → heavier). 

**Tiny tests**

* `tests/test_closures_minimal.py`:

  * A ring object: inner cleared, ring enforced.
  * Largest-only case: second component untouched except for background intersection.

**Acceptance**

* Train exactness holds for cases matching the law.
* LFP reaches singletons on minis; shrinking & 2-pass idempotence verified.

---

#### M1.4 — Registration & Solve Path Smoke ✅ COMPLETED

**What**
Wire the two unifiers into `autobuild_closures` in order and run a local smoke.

**Files**

* `src/arc_solver/search.py` (edit `autobuild_closures` only)

**Do**

```python
def autobuild_closures(train):
    closures = []
    closures += unify_KEEP_LARGEST(train)         # M1.2
    closures += unify_OUTLINE_OBJECTS(train)      # M1.3
    return closures
```

**Acceptance**

* `solve_with_closures` returns non-empty `closures` when the unifier fits, otherwise `[]`.
* No import or circular deps issues.

---

#### M1.5 — Submission Path: CLI + Determinism + Validator ✅ COMPLETED

**What**
Produce predictions, verify determinism and schema.

**Files**

* `scripts/run_public.py` (already reads/writes predictions & receipts)
* `scripts/determinism.sh`, `scripts/submission_validator.py` (use as-is)

**Do**

* Run:

  ```bash
  python scripts/run_public.py --arc_public_dir <PATH> --out runs/<DATE>
  bash scripts/determinism.sh <PATH>
  python scripts/submission_validator.py runs/<DATE>/predictions.json
  ```
* Ensure predictions are byte-identical across jobs=1 vs jobs=N; schema passes (ints 0–9, shapes match).

**Acceptance**

* Determinism OK, validator OK, receipts have minimal fields (closures, fp stats, hashes). 

---

#### M1.6 — Docs touch-ups (only if paths changed) ✅ COMPLETED

**What**
If you kept everything in current files, skip. If you folderized closures later, update anchors.

**Files**

* `docs/CONTEXT_INDEX.md` (only if you changed paths)
* `docs/IMPLEMENTATION_PLAN_v2.md` (optional: tick M1 items done)

**Acceptance**

* Context index accurately maps where unifiers/closures live so agents can navigate.

---

### M2 — OPEN/CLOSE + AXIS_PROJECTION + SYMMETRY
Breakdown for **M2** only. Same style as M1, not over-broken, with exact names, where to add, unifier scope, integration order, tiny tests, and acceptance. No step depends on future milestones.

#### M2.1 — Closure: OPEN/CLOSE (k=1) ✅ COMPLETED

**Law (one-liner)**

* **OPEN** = ERODE₁ then DILATE₁ on foreground objects (4-connected).
* **CLOSE** = DILATE₁ then ERODE₁ on foreground objects (4-connected).
  Resulting grid replaces object shapes; closure **narrows** U to those colors at those pixels (intersect-only).

**Files**

* `src/arc_solver/closures.py` (add closure + unifier)
* `tests/test_closures_minimal.py` (tiny tests)

**Add**

```python
# closures.py
class OPEN_CLOSE_Closure(Closure):
    # name = "OPEN_CLOSE"
    # params = {"mode": "open" | "close", "bg": int}
    # apply(self, U): compute morph(x_input, mode, k=1), intersect-only

def unify_OPEN_CLOSE(train) -> list[Closure]:
    # candidates: mode in {"open","close"}, bg in 0..9 (or inferred)
    # build closure for each candidate; accept iff exact on all train pairs
```

**apply(U) — algorithm**

* From `x_input` and `bg`, build binary masks per color or single FG mask (choose simplest that matches train).
* Compute morph result `y*` with k=1 using 4-neighbor structuring element.
* Build per-cell bitmasks from `y*` (one-hot allowed color at each pixel); return `U & mask` (intersect only).
* Deterministic; ≤2-pass idempotent in practice.

**Unifier**

* Enumerate `mode ∈ {"open","close"}` and `bg ∈ {0..9}` (or infer).
* For each candidate, verify **train exactness** on all pairs via fixed-point; return `[candidate]` or `[]`.

**Integration**

* In `src/arc_solver/search.py::autobuild_closures(train)` add after OUTLINE:
  `closures += unify_OPEN_CLOSE(train)`.

**Tiny tests**

* Shrinking: `apply(U) ⊆ U`.
* Idempotence ≤2 passes.
* Train exactness minis:

  * OPEN removes single-pixel spurs;
  * CLOSE fills single-pixel gaps in a ring.

**Acceptance**

* Train exactness holds where law applies; LFP singles on minis.

---

#### M2.2 — Closure: AXIS_PROJECTION_FILL

**Law (one-liner)**
From each object pixel, **extend along row or column to border** (no obstacles in M2), painting the object color along the ray(s). Scope can be `largest` or `all`.

**Files**

* `src/arc_solver/closures.py` (add closure + unifier)
* `tests/test_closures_minimal.py`

**Add**

```python
class AXIS_PROJECTION_Closure(Closure):
    # name = "AXIS_PROJECTION"
    # params = {"axis": "row" | "col", "scope": "largest|all", "mode": "to_border", "bg": int}
    # apply(self, U): derive projection mask from x_input; intersect-only

def unify_AXIS_PROJECTION(train) -> list[Closure]:
    # axis in {"row","col"}, scope in {"largest","all"}, mode fixed "to_border", bg inferred or in 0..9
    # accept candidate iff exact on all train pairs
```

**apply(U) — algorithm**

* From `x_input`, choose objects per `scope`; for each object pixel, paint rays along `axis` to image border; combine into a projection mask per color.
* Build per-cell allowed set from projection result; return `U & mask`.

**Unifier**

* Enumerate `(axis, scope, bg)`; choose the one that gives **train exactness** on all pairs; else `[]`.

**Integration**

* In `autobuild_closures(train)` add after OPEN/CLOSE:
  `closures += unify_AXIS_PROJECTION(train)`.

**Tiny tests**

* A vertical bar: row projection fills entire rows.
* A dot: col projection fills a column; largest vs all variants.

**Acceptance**

* Train exactness minis pass; shrinking & ≤2-pass idempotence hold.

---

#### M2.3 — Closure: SYMMETRY_COMPLETION

**Law (one-liner)**
Reflect content across an axis (v | h | main-diag | anti-diag) and **union** with original; result narrows U to the union colors at reflected pairs.

**Files**

* `src/arc_solver/closures.py` (add closure + unifier)
* `tests/test_closures_minimal.py`

**Add**

```python
class SYMMETRY_COMPLETION_Closure(Closure):
    # name = "SYMMETRY_COMPLETION"
    # params = {"axis": "v|h|diag|anti", "scope": "global|largest|per_object"}
    # apply(self, U): reflect x_input by axis/scope, union with x_input; intersect-only

def unify_SYMMETRY_COMPLETION(train) -> list[Closure]:
    # axis in {"v","h","diag","anti"}, scope in {"global","largest","per_object"}
    # accept iff exact on all train pairs
```

**apply(U) — algorithm**

* Compute `R(x_input)` by reflecting across `axis` with `scope`; union `x_input ∪ R(x_input)` to obtain `y*`.
* Intersect U with one-hot masks from `y*`.

**Unifier**

* Enumerate `(axis, scope)`; accept the single candidate that yields **train exactness** on every pair; else `[]`.

**Integration**

* In `autobuild_closures(train)` add after AXIS_PROJECTION:
  `closures += unify_SYMMETRY_COMPLETION(train)`.

**Tiny tests**

* Half-shape mirrored vertically to complete a full object (global).
* Largest-only reflection leaves small noise untouched.

**Acceptance**

* Minis converge to singletons; train exactness holds.

---

#### M2.4 — Registration & Smoke

**What**
Ensure the three unifiers are in order and solve path runs.

**Files**

* `src/arc_solver/search.py`

**Do**

```python
def autobuild_closures(train):
    closures = []
    closures += unify_KEEP_LARGEST(train)          # M1
    closures += unify_OUTLINE_OBJECTS(train)       # M1
    closures += unify_OPEN_CLOSE(train)            # M2.1
    closures += unify_AXIS_PROJECTION(train)       # M2.2
    closures += unify_SYMMETRY_COMPLETION(train)   # M2.3
    return closures
```

**Acceptance**

* `solve_with_closures` returns closures when any unifier fits; runs without touching beam.

---
