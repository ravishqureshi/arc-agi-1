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

