# Context Index — Repository Navigation (Submission Path)

**Last Updated:** 2025‑10‑15  
**Paradigm:** Master Operator / Fixed‑Point Closures (no beam at runtime)

> This index is the **single source of truth** for where to look and where to edit.
> If you add or move files, update this document in the same PR.

---

## 0) Sources of Truth (Docs to read first)

1. `docs/core/arc_agi_master_operator.md` — master operator & fixed‑point closures (set‑valued grid; closures are monotone, shrinking, idempotent).
2. `docs/IMPLEMENTATION_PLAN_v2.md` — current plan (what to implement next and how).
3. `docs/core/universe-intelligence.md` and `docs/core/universe-intelligence-v2.md` — background rationale and invariants (use as secondary anchors).
4. `docs/SUBMISSION_REQUIREMENTS.md` — determinism, schema, and CLI duties.
5. `docs/arc-agi-kaggle-docs.md` — Kaggle constraints (no internet, open‑source, time/budget).

---

## 1) Primary Runtime (Submission Path)

**Entry point (CLI):**  
- `scripts/run_public.py`  
  - Imports from `arc_solver`  
  - Calls **`solve_with_closures`** (fixed‑point solver)  
  - Writes `runs/<date>/predictions.json` and `runs/<date>/receipts.jsonl`

**Core engine & API surface (use these):**
- `src/arc_solver/closure_engine.py`
  - `SetValuedGrid` (10‑color bitmask cells)
  - `Closure` base class (`apply(U)` must only *clear bits*)
  - `run_fixed_point(closures, x_input)` — iterate closures until convergence
  - `verify_closures_on_train(closures, train)` — train exactness
- `src/arc_solver/closures.py`
  - **Closure classes** (e.g., `KEEP_LARGEST_COMPONENT_Closure`)
  - **Unifiers** per family (e.g., `unify_KEEP_LARGEST(train) -> [Closure, ...]`)
  - Add new closure families and their unifiers here
- `src/arc_solver/search.py`
  - `autobuild_closures(train)` — calls unifiers in a fixed order
  - `solve_with_closures(inst)` — orchestrates unification + fixed point (used by CLI)

**Types & helpers:**
- `src/arc_solver/types.py` — `Grid` alias, `Obj` dataclass for components, etc.
- `src/arc_solver/utils.py` — `G` (list→ndarray), `equal`, `components`, hashing (`task_sha`, `closure_set_sha`), and `log_receipt`.

**Submission scaffold:**
- `scripts/submission_validator.py` — schema/range validation for `predictions.json`
- `scripts/determinism.sh` — reproducibility (jobs=1 vs jobs=N, byte‑identical check)
- `scripts/test_integration.py` — small integration tests

**Datasets:**
- `data/arc-agi_training_challenges.json` — 1000 tasks (use for development/testing)
- `data/arc-agi_training_solutions.json` — 1000 solutions (ground truth for training)
- `data/arc-agi_evaluation_challenges.json` — 120 tasks (public evaluation set)
- `data/arc-agi_evaluation_solutions.json` — 120 solutions (public eval ground truth)
- `data/arc-agi_test_challenges.json` — test set (no solutions; for final submission)

---

## 2) Legacy / Reference (do not edit for submission)

- `src/ui_clean.py` — **legacy runner** (calls `solve_with_beam`) → *not used by CLI.*
- `src/arc_solver/operators.py` — legacy imperative operators (pre‑closure era).
- `src/arc_solver/inducers.py` — legacy inducers returning operators for beam.
- `src/arc_solver/search.py` also contains **beam code** (`beam_search`, `solve_with_beam`) → *unused by submission path.*

Keep these as reference only. Do not wire new features here.

---

## 3) How to Add a New Closure Family (One‑Page Protocol)

**You will only touch three places:**
1. **Implement unifier** in `src/arc_solver/closures.py`
   - Name: `unify_<FAMILY>(train) -> list[Closure]`
   - Must **unify one param set across all train pairs** and prove **train exactness**:
     - Build candidate closure(s)
     - `verify_closures_on_train([candidate], train)` must succeed
     - Return a list of validated `Closure` instances (often length 1)
2. **Implement closure** in `src/arc_solver/closures.py`
   - Class `<FAMILY>_Closure(Closure)` with fields `name`, `params`
   - `apply(self, U)` must be **monotone & shrinking** (only clear bits), deterministic
   - Derive any masks or geometry from **input `x` only** (no peeking at `y`)
3. **Register in order** in `src/arc_solver/search.py::autobuild_closures(train)`
   - Append `closures += unify_<FAMILY>(train)` at the appropriate point (cheap → heavier)

**Acceptance checks (must pass before PR merges):**
- Train exactness (all pairs) via `verify_closures_on_train(closures, train)`
- `apply(U)` shrinking property holds on small synthetic tests
- Two‑pass idempotence (re‑applying closure does not change `U`)
- CLI runs and validator passes; determinism holds

---

## 4) Minimal Receipts (per task)

- File: `runs/<date>/receipts.jsonl` — one line per task:
  ```json
  {
    "task": "<file>.json",
    "status": "solved|under_constrained|failed",
    "closures": [{"name":"...", "params":{...}}, ...],
    "fp": {"iters": N, "cells_multi": M},
    "timing_ms": {"unify": ms, "fixed_point": ms, "total": ms},
    "hashes": {"task_sha": "...", "closure_set_sha": "..."}
  }
  ```

---

## 5) Parametric Discipline Reference (Where to Look)

**For parametric patterns (exhaustive search, keyword-only params, fail-loud):**
- `src/arc_solver/closures.py` — see `unify_KEEP_LARGEST` (lines 89-112) and `KEEP_LARGEST_COMPONENT_Closure` (lines 39-86)
- `src/arc_solver/utils.py` — see `components(g, *, bg: int)` (line 53), `bbox_nonzero(g, *, bg: int)` (line 40)
- `src/arc_solver/search.py` — see bg extraction pattern (lines 300-310, 322-329)

**B1 is the gold standard** — all new closures follow this pattern.

---
