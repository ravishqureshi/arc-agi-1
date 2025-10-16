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

### M2 — OPEN/CLOSE + AXIS_PROJECTION + SYMMETRY ✅ COMPLETED
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

#### M2.2 — Closure: AXIS_PROJECTION_FILL ✅ COMPLETED

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

#### M2.3 — Closure: SYMMETRY_COMPLETION ✅ COMPLETED

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

#### M2.4 — Registration & Smoke ✅ COMPLETED

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
 ### Increase coverage Work order
 #### Work-Order

Milestone: A+B — Composition-Safe Unifiers + Per-Input Background (BG)  ✅ COMPLETED
Objective:
1) **A. Composition-Safe Unifier Gate** — change all existing unifiers to accept closures that are **solution-preserving and compatible** (not necessarily sufficient alone), then verify **train exactness on the collected set** at the end of `autobuild_closures`.  
2) **B. Per-Input BG** — remove global `bg` unification; closures infer **background per input** deterministically (border flood-fill) inside `apply`. This loosens incidental params without violating “observer = observed” (the method is unified; values are derived from x).

Scope (touch only what follows):
- Families already implemented: **KEEP_LARGEST_COMPONENT**, **OUTLINE_OBJECTS**, **OPEN_CLOSE**, **AXIS_PROJECTION**, **SYMMETRY_COMPLETION**.
- No new families in this WO.

---

## Files to Edit

- `src/arc_solver/closure_engine.py`
  - Add small helpers used by all unifiers:
    ```python
    def preserves_y(closure, train) -> bool: ...
    def compatible_to_y(closure, train) -> bool: ...
    # keeps: verify_closures_on_train(closures, train)
    ```
- `src/arc_solver/closures.py`
  - Update each unifier to use the **new gate** (see “Unifier Gate Changes”).
  - Update each closure to support **per-input BG**:
    - If `params` includes `"bg"` → allow `"bg": None` and infer at runtime inside `apply(x_input)` via border flood-fill.
- `src/arc_solver/search.py`
  - Change `autobuild_closures(train)` to:
    1) Collect all **candidate** closures that pass `preserves_y` **and** `compatible_to_y`.
    2) Run `verify_closures_on_train(collected, train)`.
    3) If it fails, **greedy back-off**: drop last added candidate(s) until it passes or return best passing subset (even empty).
- `tests/test_closures_minimal.py`
  - Add/adjust tiny property tests to reflect **per-input BG** and **composition** verification.

---

### A) Unifier Gate Changes (for all implemented families)
- **Replace “exact-alone” acceptance:**
  - ❌ Old: accept `candidate` iff `verify_closures_on_train([candidate], train)` is True.
- **With composition-safe gate:**
  - ✅ New: accept `candidate` iff:
    1) `preserves_y(candidate, train)` → for all pairs, `candidate.apply(singleton(y)) == singleton(y)`.
    2) `compatible_to_y(candidate, train)` → for all pairs, `candidate.apply(singleton(x))` **does not introduce colors** absent in `y` (and optionally is pointwise ≤ `singleton(y)`).
  - Return `[candidate]` if both pass; `[]` otherwise.
- **Final exactness at set level only**:
  - `autobuild_closures(train)` composes all accepted candidates and calls `verify_closures_on_train(collected, train)`; if it fails, back-off greedily.

### B) Per-Input BG (for closures that used a unified bg)
- In closure `params`, allow `"bg": None`.
- In `apply(self, U)`:
  - Compute BG **per input** via border flood-fill (deterministic).
  - Use this inferred BG for masks (largest, outline, morphology, projection, symmetry).
- In unifiers:
  - Do **not** unify a single `bg` across pairs; prefer `bg=None` and rely on per-input inference inside `apply`.

---

## Required Signatures / Pseudocode

**closure_engine.py**
```python
def preserves_y(closure, train) -> bool:
    for x, y in train:
        U = init_from_grid(y)     # singleton(y)
        U2 = closure.apply(U)     # must not contradict y
        if not U2.equals(U):      # exact equality
            return False
    return True

def compatible_to_y(closure, train) -> bool:
    for x, y in train:
        Ux = init_from_grid(x)
        Uy = init_from_grid(y)
        Ux2 = closure.apply(Ux)
        # Option A (strict color subset):
        if not Ux2.colors_subset_of(Uy):
            return False
        # (Optional) monotone-toward-y guard can be added if needed
    return True
search.py::autobuild_closures(train)

python
Copy code
def autobuild_closures(train):
    candidates = []
    candidates += unify_KEEP_LARGEST(train)
    candidates += unify_OUTLINE_OBJECTS(train)
    candidates += unify_OPEN_CLOSE(train)
    candidates += unify_AXIS_PROJECTION(train)
    candidates += unify_SYMMETRY_COMPLETION(train)

    # Compose-then-verify gate
    kept = []
    for c in candidates:
        kept.append(c)
        if not verify_closures_on_train(kept, train):
            kept.pop()  # greedy back-off

    return kept
closures.py — example param & bg inference

python
Copy code
class OUTLINE_OBJECTS_Closure(Closure):
    # params = {"mode":"outer", "scope":"largest|all", "bg": None}
    def apply(self, U):
        x = self.x_input
        bg = infer_bg_from_border(x)  # per-input
        outline_mask = compute_outline(x, bg, scope=self.params["scope"])
        # build per-cell allowed set; intersect-only
        return intersect_with_mask(U, outline_mask)
Data-Use Rule (training-only induction)
Allowed in unifiers/closures/tests: data/arc-agi_training_challenges.json (+ optional ..._training_solutions.json for local verify).

Forbidden in unifiers/closures/tests: any data/arc-agi_evaluation_* or data/arc-agi_test_*.

Submission runner may load evaluation/test to produce predictions, never to induce params.

Reviewers to Run
anti-hardcode-implementation-auditor-reviewer

math-closure-soundness-reviewer

submission-determinism-reviewer

(Each returns exactly one review file.)

Verify Commands
bash
Copy code
python scripts/run_public.py --arc_public_dir <FILL_THIS> --out runs/<DATE>
bash scripts/determinism.sh <FILL_THIS>
python scripts/submission_validator.py runs/<DATE>/predictions.json
Acceptance
All targeted unifiers now use the composition-safe gate (preserves_y & compatible_to_y), and autobuild_closures verifies exactness on the set (with greedy back-off).
Closures that used bg now support "bg": None and infer BG per input deterministically inside apply.
Tiny tests updated and pass (shrinking; ≤2-pass idempotence; minimal minis).
Determinism and schema validator pass on the specified ARC public dir.
Receipts JSONL unchanged in format; coverage shows an uptick (more tasks solvable with composition).

Here’s a tight, dependency-free breakdown for **M3**. It assumes you’ve finished **A+B** (composition-safe unifiers + per-input BG). No stubs, no future deps.

---

### M3 — MOD_PATTERN + DIAGONAL_REPEAT

#### M3.1 — Closure: MOD_PATTERN (periodic mask, (p,q,anchor)) ✅ COMPLETED

**Law (one-liner)**
Cells partitioned by congruence classes `((r−ar) mod p, (c−ac) mod q)` must take fixed color sets (derived from input only), producing the periodic pattern.

**Files**

* `src/arc_solver/closures.py` (add closure + unifier)
* `tests/test_closures_minimal.py` (tiny tests)

**Add**

```python
# closures.py
class MOD_PATTERN_Closure(Closure):
    # name = "MOD_PATTERN"
    # params = {"p": int, "q": int, "anchor": (int,int), "class_map": Dict[(int,int), Set[int]]}
    # apply(self, U): build periodic class masks from x_input & params; intersect-only

def unify_MOD_PATTERN(train) -> list[Closure]:
    # Anchors to try: {(0,0), bbox_corners(x), quadrant_origins(x)}
    # Infer (p,q) by periodicity in x relative to y for each pair (composition-safe checks)
    # Build class_map from input-derived colors; accept iff preserves_y && compatible_to_y on all pairs
```

**apply(U) — algorithm (intersect-only)**

* From `x_input` and `(p,q,anchor)`, compute class index per cell; map each class to allowed colors (from input’s role/usage as indicated by train).
* Build per-cell allowed set by class; return `U & mask`. Deterministic; ≤2-pass idempotent.

**Unifier (composition-safe)**

* Enumerate anchors; estimate `(p,q)` (small periods only, e.g., 2–6) by scanning residual structure.
* Build `class_map` from **input** (e.g., dominant/unique color per class) so `T(y)==y` on all pairs; reject if not.
* Return `[candidate]` when **preserves_y && compatible_to_y** on all pairs; else `[]`.

**Integration**

* In `src/arc_solver/search.py::autobuild_closures(train)` add **after SYMMETRY_COMPLETION**:
  `closures += unify_MOD_PATTERN(train)`

**Tiny tests**

* 2×3 periodic dots with anchor (0,0).
* Parity special case `(p,q)=(2,2)` acts as expected.
* Shrinking + ≤2-pass idempotence + train mini exactness (with composition).

**Acceptance**

* Candidate passes preserves_y/compatible checks; set of closures reaches train exactness; LFP singletons on minis.

---

#### Work-Order for Gate-Fix ✅ COMPLETED

Milestone: Gate-Fix — Composition Compatibility + Unifier Collection 

Objective:
Fix the composition gate so valid closures aren’t rejected early, and let unifiers **collect** all compatible candidates (composition decides). This should immediately raise coverage.

Scope:
- Replace current `compatible_to_y` with a **non-destructive** check on known-correct pixels.
- Update existing unifiers to **collect** all candidates that pass `preserves_y` + new `compatible_to_y` (do not early-return on first hit).
- No new closure families in this WO. Fixed-point only (no beam). Training-only induction.

Files to edit:
- `src/arc_solver/closure_engine.py`
  - Replace `compatible_to_y` with:
    ```python
    def compatible_to_y(closure, train):
        for x, y in train:
            if x.shape != y.shape:
                return False
            Ux  = init_from_grid(x)
            Ux2 = closure.apply(Ux, x)
            H, W = x.shape
            for r in range(H):
                for c in range(W):
                    if int(x[r,c]) == int(y[r,c]):
                        if not Ux2.allows(r, c, int(y[r,c])):  # do not destroy known-correct pixels
                            return False
        return True
    ```
- `src/arc_solver/closures.py`
  - For each implemented unifier (`unify_KEEP_LARGEST`, `unify_OUTLINE_OBJECTS`, `unify_OPEN_CLOSE`, `unify_AXIS_PROJECTION`, `unify_SYMMETRY_COMPLETION`, `unify_MOD_PATTERN`):
    - **Collect** valid candidates instead of returning the first:
      ```python
      valid = []
      for params in enumerate_candidates(...):
          cand = FAMILY_Closure(name, params)
          if preserves_y(cand, train) and compatible_to_y(cand, train):
              valid.append(cand)
      return valid  # may be 0..N; composition will verify the set
      ```
- `src/arc_solver/search.py`
  - Keep existing greedy composition verify:
    ```python
    kept = []
    for c in candidates:
        kept.append(c)
        if not verify_closures_on_train(kept, train):
            kept.pop()
    return kept
    ```

Tests to add/update:
- `tests/test_closures_minimal.py`
  - Add one tiny test that `compatible_to_y` **retains** y-colors where x==y.
  - Ensure unifiers can return multiple candidates and that composition still passes train exactness on a small synthetic case.

Data-use rule (training-only induction):
- Allowed: `data/arc-agi_training_challenges.json` (+ optional training solutions).
- **Forbidden** inside unifiers/closures/tests: any `..._evaluation_*` or `..._test_*`.

Reviewers to run:
- anti-hardcode-implementation-auditor-reviewer
- math-closure-soundness-reviewer
- submission-determinism-reviewer

Verify commands:
```bash
python scripts/run_public.py --arc_public_dir <FILL_THIS> --out runs/<DATE>
bash scripts/determinism.sh <FILL_THIS>
python scripts/submission_validator.py runs/<DATE>/predictions.json


**Work Order: NX‑UNSTICK (Mask‑Induction + Color Closures)** ✅ COMPLETED

**Goal:** Convert empty closure sets into composable ones and unlock palette‑changing tasks.
**Scope:** Update inducers (mask‑driven), add `COLOR_PERM`, add `RECOLOR_ON_MASK`, refit parameterization to strategy form where needed.
**Inputs:** Repo at HEAD; training dataset `data/arc-agi_training_challenges.json` + `..._solutions.json`.
**Guardrail:** Use **train** only for induction; test only for applying unified closures; no eval/test leakage.

**Tasks:**

1. **Refactor all inducers to mask‑driven checks**

   * For each existing family (KEEP_LARGEST, OUTLINE, OPEN/CLOSE, AXIS_PROJECTION, SYMMETRY, MOD_PATTERN):

     * Change acceptance from `y == T(x)` to:
       “`patch(x, T, M)` equals `y` **exactly** and `T` proposes **no changes** outside `M`”, where `M=(x≠y)`.”
   * Add unit tests: each inducer **accepts** a partial mask fit and **rejects** any edit outside `M`.

2. **Implement COLOR_PERM closure**

   * **Inducer:** Deduce a permutation π per pair; unify exactly across all pairs; ensure bijective on used set.
   * **Closure:** `U[p] ∩= {π(x(p))}`.

3. **Implement RECOLOR_ON_MASK(template, target_strategy)**

   * **Templates** to support now: `NONZERO`, `BACKGROUND`, `NOT_LARGEST`, `PARITY(anchor in {0,1}²)`, `STRIPES(axis in {row,col}, k in {2,3})`, `BORDER_RING(1)`.
   * **Strategies** to support now: `CONST(c)`, `MODE_ON_MASK`, `DOMINANT_BORDER`.
   * **Inducer:** Pick the (template, strategy, params) that makes `template(x) == M` and target equals y on M, unified across pairs.

4. **Parameter → strategy normalization**

   * For SYMMETRY and `bg` usage, replace constant params with functions: `axis="auto_best"`; `bg="auto_mode"`. Unify on the **function name**.
   * Update receipts to log the **strategy** and the **per‑pair realized value** for audit.

5. **Run & verify**

   * Re‑run training suite; ensure `receipts.jsonl` shows most tasks with ≥1 closure.
   * Track: (# tasks with closures), (# with at least one color closure), (# solved).
   * If coverage is still ~6/1000, dump top 20 unsolved with largest singleton gains and inspect missing families.

**Reviewers to run (with the same Context Pack):**

* `anti-hardcode-implementation-auditor-reviewer` — only to ensure no test/eval leakage and no “approximate residual” acceptance on train.
* `math-closure-soundness-reviewer` — verify the new laws: mask‑local fit, permutation bijectivity, and strategy unification.
* `submission-determinism-reviewer` — quick determinism pass (no new nondeterminism introduced).

**Acceptance:**

* Median closures per task on train ≥ 1.
* At least **one color closure** present in ≥ 25% of tasks.
* Coverage increases materially from 6/1000 (expect double‑digit tasks without adding any further families).


### Milestone: Canvas-Law + Shape-Aware Gates ✅ COMPLETED

Objective:
Enable shape-changing tasks by (1) inferring the output canvas and (2) removing same-shape guards in unifier gates. Then let closures operate on the **output-shaped** set-valued grid.

Scope:
1) Add CANVAS_SIZE unifier:
   - Infer (H_out, W_out) from train pairs (multipliers, periods, or bbox tiling).
   - Expose `canvas = {"H": H_out, "W": W_out, "mapper": φ}` where φ maps output coords to input coords when needed.

2) Gate fixes (shape-aware):
   - In `preserves_y(closure, train)`: build singleton(y)-shaped U and check `closure.apply(U, x)` equals U for each pair (no shape equality pre-check).
   - In `compatible_to_y(...)`: for cells where **y is already fixed** (same-law cells), ensure the candidate doesn’t **remove** that y-color; do not enforce global palette subset. No same-shape guard.

3) Output-shaped fixed point:
   - In `solve_with_closures`: after CANVAS_SIZE, initialize `U = TOP(H_out, W_out)`.
   - Pass `canvas` (H_out, W_out, φ) to closures so they can write on the larger grid.

4) Retarget two families to output canvas:
   - **TILING / TILING_ON_MASK**: write motif over the **output lattice**; use input-only masks if required.
   - **MOD_PATTERN**: evaluate classes on the output lattice `((r - ar) mod p, (c - ac) mod q)`.

Files to edit:
- `src/arc_solver/search.py`:
  - Call `unify_CANVAS_SIZE(train)` first; keep greedy set-level verify.
- `src/arc_solver/closure_engine.py`:
  - Remove same-shape early returns in `preserves_y`/`compatible_to_y`; allow y-shaped evaluation.
  - Add optional `canvas` argument plumbed to closures.
- `src/arc_solver/closures.py`:
  - Implement `unify_CANVAS_SIZE` (return a tiny `CanvasClosure` that only sets output shape for the engine).
  - Update TILING/MOD_PATTERN closures to write on `canvas.H,canvas.W` using `φ` where needed.

Tests:
- Tiny train minis where output is a 3× expansion of 2×2; MOD(2,3) over output lattice; TILING_ON_MASK on larger canvas.
- Assert shrinking + ≤2-pass idempotence on the output-shaped U.
- Verify train exactness with composition; predictions deterministic & schema-valid.

Data-use rule:
- Induction uses training files only; no eval/test in unifiers or tests.

Reviewers to run:
- anti-hardcode-implementation-auditor-reviewer
- math-closure-soundness-reviewer
- submission-determinism-reviewer

Verify commands:
```bash
python scripts/run_public.py --dataset=data/arc-agi_training_challenges.json --output=runs/<DATE>
bash scripts/determinism.sh data/arc-agi_training_challenges.json
python scripts/submission_validator.py runs/<DATE>/predictions.json

Acceptance:
Receipts show most tasks now have ≥1 closure (not []).
A non-trivial subset of shape-changing training tasks become solvable (e.g., tiling/periodic expansions like 00576224, 007bbfb7).
Coverage increases beyond 10/1000 on the same run; format unchanged.


### Milestone: Canvas Greedy Verify Fix + Param Normalization ✅ COMPLETED

Objective:
Make CANVAS_SIZE usable in composition and normalize its params, so shape-changing tasks can compose with other closures.

Scope:
1) Tag meta closures and keep them during greedy add/back-off.
2) Use **consistency verify** during incremental add; require **full exactness** only at the end.
3) Normalize CANVAS params to multipliers/mapper (no absolute H,W).

Files to edit:
- `src/arc_solver/closure_engine.py`
  - In `class Closure`, add: `is_meta: bool = False`.
  - In `CANVAS_SIZE_Closure`, set `is_meta = True`.
  - (No change to apply() logic; it may be identity.)
- `src/arc_solver/closures.py`
  - Normalize CANVAS params:
    - Replace absolute output `(H_out,W_out)` with `{'mode':'tile','kh':int,'kw':int}` or equivalent mapper φ.
    - Unifier must unify `(kh,kw)` (or φ) across all train pairs.
- `src/arc_solver/search.py`
  - Update `autobuild_closures(train)` greedy loop:
    ```python
    kept_meta, kept_nonmeta = [], []
    for c in candidates:
        if c.is_meta:
            kept_meta.append(c)
            continue  # don't verify on meta-only
        # try adding non-meta
        trial = kept_meta + kept_nonmeta + [c]
        if verify_consistent_on_train(trial, train):  # << consistency, not full exactness
            kept_nonmeta.append(c)
        # else skip c (do not drop meta or prior nonmeta that were consistent)
    # After loop: require full exactness on the combined set
    final = kept_meta + kept_nonmeta
    if not verify_closures_on_train(final, train):   # full exactness
        # greedy back-off on last non-meta until passes or empty
        while kept_nonmeta and not verify_closures_on_train(kept_meta + kept_nonmeta, train):
            kept_nonmeta.pop()
        final = kept_meta + kept_nonmeta
    return final
    ```
  - Add `verify_consistent_on_train(closures, train)`:
    - Apply to singleton(x) per pair, ensure **no contradictions** with y (e.g., do not clear y’s color where x==y; do not write outside mapper bounds). Do **not** require full singletons.

Tests to add/update:
- A shape-growing mini (2×2 → 6×6) where CANVAS_SIZE + TILING compose; assert:
  - `kept` contains CANVAS_SIZE even when alone initially.
  - Incremental verify passes on consistency.
  - Final verify requires full exactness and passes with the second (constraining) closure.
- A non-shape mini still passes.

Data-use rule:
- Training-only induction inside unifiers/closures/tests; no eval/test reads.

Reviewers to run:
- anti-hardcode-implementation-auditor-reviewer
- math-closure-soundness-reviewer
- submission-determinism-reviewer

Verify commands:
```bash
python scripts/run_public.py --dataset=data/arc-agi_training_challenges.json --output=runs/<DATE>
bash scripts/determinism.sh data/arc-agi_training_challenges.json
python scripts/submission_validator.py runs/<DATE>/predictions.json

Acceptance:
CANVAS_SIZE remains in closures for tasks needing shape growth (no longer dropped).
Composition with at least one constraining closure now achieves train exactness on shape-changing minis.
Coverage does not regress; expect >10/1000 once TILING/MOD over output lattice participate.

#### M3.2 — Closure: DIAGONAL_REPEAT (shift chain along Δ=(dr,dc), k steps)

**Law (one-liner)**
Repeat a template object/motif by shifting along a fixed diagonal step `Δ=(dr,dc)` for `k` steps; tie admissible colors along that chain.

**Files**

* `src/arc_solver/closures.py` (add closure + unifier)
* `tests/test_closures_minimal.py`

**Add**

```python
class DIAGONAL_REPEAT_Closure(Closure):
    # name = "DIAGONAL_REPEAT"
    # params = {"dr": int, "dc": int, "k": int, "template": TemplateSpec}
    # apply(self, U): construct chains by shifting template along Δ; intersect-only

def unify_DIAGONAL_REPEAT(train) -> list[Closure]:
    # Detect Δ by matching shifted masks between input & (compatible) target structure per pair
    # Infer k (small; e.g., 1–5) and a simple template (mask+color) from input only
    # Accept iff preserves_y && compatible_to_y across all pairs
```

**apply(U) — algorithm (intersect-only)**

* From `x_input` derive a **template mask** (e.g., smallest per-color object or motif region).
* Generate positions `{template + t·Δ | t=0..k}` clipped to bounds; union their allowed colors; intersect U with that union.

**Unifier (composition-safe)**

* For each pair, compute candidate `Δ` by aligning a source mask to a target mask (exact shift).
* Intersect Δ candidates across pairs; try small `k`.
* Build template from input only; require **preserves_y && compatible_to_y** across pairs.
* Return `[candidate]` or `[]`.

**Integration**

* In `autobuild_closures(train)` add **after MOD_PATTERN**:
  `closures += unify_DIAGONAL_REPEAT(train)`

**Tiny tests**

* A 2-pixel diagonal repeated thrice `Δ=(1,1)`;
* Anti diagonal `Δ=(1,−1)`;
* Shrinking + ≤2-pass idempotence + train mini exactness (with composition).

**Acceptance**

* Candidate passes preserves/compatible; full set attains train exactness; minis converge.

---

#### M3.3 — Registration & Smoke

**Do**

```python
def autobuild_closures(train):
    closures = []
    closures += unify_KEEP_LARGEST(train)           # M1
    closures += unify_OUTLINE_OBJECTS(train)        # M1
    closures += unify_OPEN_CLOSE(train)             # M2.1
    closures += unify_AXIS_PROJECTION(train)        # M2.2
    closures += unify_SYMMETRY_COMPLETION(train)    # M2.3
    closures += unify_MOD_PATTERN(train)            # M3.1
    closures += unify_DIAGONAL_REPEAT(train)        # M3.2
    # Composition-safe verify already in place from A+B:
    #   greedy keep if verify_closures_on_train(kept, train) passes
    return kept_or_verified_list
```

**Acceptance**

* `solve_with_closures` builds MOD_PATTERN and DIAGONAL_REPEAT when compatible; composition reaches train exactness where applicable; deterministic predictions and schema validator remain OK; receipts unchanged in shape.

---

### M4 — TILING(_ON_MASK) + COPY_BY_DELTAS (baseline push)

#### M4.1 — Closure: TILING(_ON_MASK) on the **output canvas** ✅ COMPLETED

**Law (one-liner)**
Fill the **output lattice** with a motif `m(h×w)` either globally or **on an input-only mask**; narrow `U` to the motif’s color at the tiled coordinates.

**Files**

* `src/arc_solver/closures.py` (closure + unifier)
* `tests/test_closures_minimal.py`

**Add**

```python
class TILING_Closure(Closure):
    # name = "TILING" or "TILING_ON_MASK"
    # params = {"motif": np.ndarray[h,w], "mode": "global|mask", "mask_template": Optional[str], "anchor": (ar,ac)}
    # apply(self, U, x): tile motif across the **output** canvas (use canvas mapper if present);
    #                    if mode="mask", restrict writes to mask_template(x); intersect-only.

def unify_TILING(train) -> list[Closure]:
    # 1) Extract candidate motif m from y (validate with x→M residual) but **derive** its colors from x where possible.
    # 2) Anchors: {(0,0), bbox corners, quadrant origins}; choose a single anchor across pairs.
    # 3) mode candidates:
    #    - "global"
    #    - "mask" with mask_template in {ALL, NONZERO, BACKGROUND, NOT_LARGEST, PARITY, STRIPES}
    # 4) Composition-safe gate: preserves_y && mask-exact on M (changes only where x≠y), no edits outside M.
    # 5) Return all candidates that pass; composition decides.
```

**Notes**

* Operate on **output-sized U** (use canvas φ if defined).
* Mask templates are **input-only**; y used only to verify.

**Tiny tests**

* Global tiling on 6×6 from a 2×2 motif (anchor (0,0)).
* Masked tiling: fill only BACKGROUND or NOT_LARGEST.

**Acceptance**

* Shrinking + ≤2-pass idempotence; composition reaches train exactness on minis.

---

#### M4.2 — Closure: COPY_BY_DELTAS (shifted-mask equality)

**Law (one-liner)**
Copy a **template mask** (object/motif) from input to a set of shifted locations `Δ = {δ₁,…,δ_k}` on the output canvas; intersect `U` with the template’s color at those destinations.

**Files**

* `src/arc_solver/closures.py`
* `tests/test_closures_minimal.py`

**Add**

```python
class COPY_BY_DELTAS_Closure(Closure):
    # name = "COPY_BY_DELTAS"
    # params = {"template": TemplateSpec, "deltas": List[Tuple[dr,dc]]}
    # apply(self, U, x): for each δ in deltas, place template colors at shifted coords; intersect-only.

def unify_COPY_BY_DELTAS(train) -> list[Closure]:
    # 1) For each pair, detect a small template from x (e.g., smallest per-color object or residual-bbox).
    # 2) For each destination object in y, require exact **shifted-mask equality**:
    #       shift(template, δ) == target_mask
    # 3) Intersect Δ-sets across pairs → common Δ
    # 4) Composition-safe gate: preserves_y && patch-exact on M only; collect candidate if passes.
```

**Notes**

* Works on **output** canvas; use canvas φ if needed (shape growth).
* Use **shifted-mask equality** (not centroids).

**Tiny tests**

* Copy a 2×2 block across a grid with Δ={(0,3),(3,0)}.
* Per-color template variant: copy only red template.

**Acceptance**

* Shrinking + ≤2-pass idempotence; composition attains train exactness on minis.

---

#### M4.3 — Registration & smoke

**Files**

* `src/arc_solver/search.py`

**Do**

```python
def autobuild_closures(train):
    kept = []
    # M1
    kept += unify_KEEP_LARGEST(train)
    kept += unify_OUTLINE_OBJECTS(train)
    # M2
    kept += unify_OPEN_CLOSE(train)
    kept += unify_AXIS_PROJECTION(train)
    kept += unify_SYMMETRY_COMPLETION(train)
    # M3
    kept += unify_MOD_PATTERN(train)
    # M4
    kept += unify_TILING(train)          # after MOD_PATTERN
    kept += unify_COPY_BY_DELTAS(train)  # after TILING

    # keep your composition logic: meta kept; incremental = consistency; final = full exactness
    return greedy_compose_with_verify(kept, train)
```

**Acceptance**

* Builds candidates for TILING and COPY_BY_DELTAS; composition stays deterministic; receipts unchanged in shape.

---

#### M4.4 — Submission smoke (baseline)

* Run training or evaluation split with your usual commands; ensure determinism, schema OK.
* Track: tasks with at least one **color closure** and at least one **canvas-aware** closure; solved count should move beyond low double digits.
