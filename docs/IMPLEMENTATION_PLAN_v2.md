# ARC‑AGI Coverage Plan — Universe‑Intelligence (A → B → D)

---

## Goal & non‑negotiables

* **Goal:** increase solved tasks rapidly on ARC public (and v2 where applicable) and produce a valid `predictions.json` for submission.
* **Non‑negotiables:**

  1. **Determinism** (same inputs → same outputs).
  2. **Receipts per task** (minimal, submission‑safe): **train residual == 0** gate, list of **closures** with params, and stable **hashes**.
  3. **Verifying unifiers:** every closure we add must include a **unifier** that **fits the same parameters across *all* train pairs** and **proves train exactness** before it is admitted.

> **Master idea (applies throughout B):** Replace “program search” with **fixed‑point closures**. Treat the test output as a **set‑valued grid**; apply **monotone, idempotent, shrinking closures** induced from train, iterate to the **least fixed point** (lfp). If all cells become singletons → output is proven; else it’s **under‑constrained** and D proposes the next closure to add.

**Repository context:** Work inside `ui_clean.py` (and `scripts/` as needed). If names differ, adapt accordingly.

---

## A) Observability & Coverage Harness (keep it lean but complete)

**Why now:** This is a small logging layer + a tiny runner and cache. It lets you see where to spend time in B, and makes D automatic. No dashboards.

### A1. Receipts & telemetry JSONL (per task) ✅ **MOSTLY COMPLETE**

**Status:** ✅ Implemented in `src/arc_solver/utils.py` (log_receipt function) and `solve_with_closures()` returns metadata with fp stats. ⚠️ Missing: closure_set_sha hash (TODO).

**Add:** a function `log_receipt(record: dict)` writing to `runs/YYYY-MM-DD/receipts.jsonl`.

**Schema (single line per task):**

```json
{
  "task": "0a1d4ef8.json",
  "status": "solved|under_constrained|failed",
  "closures": [
    {"name": "KEEP_LARGEST_COMPONENT", "params": {"bg": 0}},
    {"name": "OUTLINE_OBJECTS", "params": {"t":1,"mode":"outer","scope":"largest"}},
    {"name": "MOD_PATTERN", "params": {"p":2,"q":3,"anchor":[0,0],"class_map":{"(0,1)":"blue"}}}
  ],
  "fp": {"iters": 5, "cells_multi": 0},
  "timing_ms": {"build_closures": 13, "fixed_point": 97, "total": 122},
  "hashes": {"task_sha": "...", "closure_set_sha": "..."},
  "invariants": {
    "palette_delta": {"preserved": true, "delta": {"1": 0, "2": +4}},
    "component_delta": {"largest_kept": true, "count_delta": 0}
  }
}
```

**Hashing (deterministic):**

* `task_sha`: SHA‑256 over canonical JSON of **all train pairs**:

  ```python
  payload = {"train":[{"in":x.tolist(), "out":y.tolist()} for (x,y) in train_pairs]}
  task_sha = sha256(json.dumps(payload, sort_keys=True).encode())
  ```
* `closure_set_sha`: SHA‑256 over the **ordered** list of closures `{name, params}`.

**Hook points:**

* After each task is solved or declared under‑constrained/failed in the public runner.
* Inside the solver just after lfp convergence (to capture `fp.iters` and `fp.cells_multi`).

**Deliverables:**

* `runs/<date>/receipts.jsonl` with one line per task.
* Helper: `ensure_dir("runs/<date>")`.

---

## B) **Closure & Unifier Expansion** (primary lever for coverage)

**Principle:** Every capability is a **Closure + Unifier**.

* **Closure:** `apply(U) -> U'` is **monotone & shrinking** (only clears bits). Deterministic.
* **Unifier:** learns a **single param set** from **all train pairs** and **proves train exactness** (apply closures to train singleton grids → exact `y`).
* **Solve path (fixed‑point):** build closures → run lfp → if every cell is a singleton, decode; else **under‑constrained** (hand to D).

> **B0. Master operator / fixed‑point engine** ✅ **COMPLETE**
>
> **Implementation:** `src/arc_solver/closure_engine.py` (245 lines)
>
> * ✅ Set‑valued grid with 10‑bit masks (SetValuedGrid class)
> * ✅ `Closure` base class: `name`, `params`, `apply(U, x_input)`
> * ✅ `run_fixed_point(closures, x_input)` iterates until convergence; returns `(U*, stats)`
> * ✅ **Train verification**: `verify_closures_on_train()` checks all pairs reach singleton(y)
> * ✅ Helpers: init_top, init_from_grid, color_to_mask, set_to_mask
> * ✅ Integrated into `solve_with_closures()` in search.py

Below are the closure families to implement **in order**. For each, implement **(i) Unifier**, **(ii) Closure.apply()**, **(iii) Tests**, **(iv) Integrate**.

---

### B1) KEEP_LARGEST_COMPONENT ✅ **COMPLETE**

**Implementation:** `src/arc_solver/closures.py` (107 lines)

**Closure:** `KEEP_LARGEST_COMPONENT_Closure(bg)` — cells not in largest 4‑conn component intersect with `{bg}` only.

**Unifier:** `unify_KEEP_LARGEST(train, bg=0)`

* ✅ For every train `(x,y)`, compute largest component of `x` (ignore `bg`)
* ✅ Build closure, run fixed-point, verify exact match on ALL pairs
* ✅ Returns `[closure]` if verified, else `[]`

**Status:**
* ✅ Unifier implemented and tested
* ✅ Closure.apply() is monotone & shrinking
* ✅ Integrated into `autobuild_closures()` in search.py
* ✅ Unit tested on synthetic examples

---

### B2) OUTLINE_OBJECTS

**Closure:** `T_outline(th=1, mode="outer|inner", scope="largest|all")`.

**Unifier:**

* For each `(x,y)`, compute outline candidates; if a single `(mode, scope)` matches **all** pairs exactly → params.

**Integration:** After B1.

---

### B3) AXIS_PROJECTION_FILL

**Closure:** `T_project(axis="row|col", mode="to_border", scope="largest|all")`.

**Unifier:**

* Try `(axis, scope)` pairs; accept if `project(x)` equals `y` for **all** pairs.

**Integration:** After B2.

---

### B4) SYMMETRY_COMPLETION 

**Closure:** `T_reflect_union(axis="v|h|diag|anti", scope="global|largest|per_object")`.

**Unifier:**

* Require `y == x ∪ reflect_axis(x)` with one `(axis,scope)` across all pairs.

**Integration:** After B3.

---

### B5) LINE_FROM_EXTREMA ⚠️ DEFERRED

**Closure:** `T_line(strategy="furthest_pair|bbox_corners|centroids", color_role="object|dominant")`.

**Unifier:**

* Choose one strategy and role that makes `x + line == y` across all pairs.

**Integration:** After B4.

---

### B6) RECOLOR_BY_ROLE ⚠️ DEFERRED

**Closure:** `T_recolor(role="background|largest|rarest", to="dominant_border|mode_inside|fixed")`.

**Unifier:**

* Verify palette deltas match role mapping identically across all pairs.

**Integration:** After B5.

---

### B7) MORPHOLOGY (complete the set at small k)

**Closures:** `T_dilate(k=1)`, `T_erode(k=1)`, **`T_open()`**, **`T_close()`** (k=1 pipeline), and **`T_fill_outline()`** (optional if needed).

**Unifier:**

* Confirm `y == morph(x)` across pairs (start with k=1 only).
* Add **`T_open`/`T_close`** early—they close noisy outlines and holes common in public.

**Integration:** After B6.

---

### B8) QUADRANT_MAPPING ⚠️ DEFERRED

**Closure:** `T_map_quadrants(rule)` — copy/transform one quadrant into others.

**Unifier:**

* Split x into 2×2; compare quadrant signatures; encode minimal `rule` that matches **all** pairs.

**Integration:** After B7.

---

### B9) MOD_PATTERN (general parity)

**Closure:** `T_mod(p, q, anchor, class_map)` — generalizes parity to any `(p,q)`.

**Unifier:**

* Try anchors from set `{(0,0), bbox corners, quadrant origins}`.
* Infer `(p,q)` and `class_map` from y; accept if same `(p,q,anchor,class_map)` fits **all** pairs.

**Integration:** High priority (very common); place after B6–B7.

---

### B10) DIAGONAL_REPEAT

**Closure:** `T_diag_repeat(delta=(dr,dc), k, motif_spec)` — tie colors along diagonal chains.

**Unifier:**

* Detect diagonal displacement mapping from x→y; unify `(dr,dc,k)`; shape‑check shifted masks; accept exact match on all pairs.

**Integration:** High priority; next after B9.

---

### B11) QUADRANT‑ON‑MASK & REFLECT‑ON‑MASK

**Closures:**

* `T_quad_on_mask(source_quad, mask_template)`
* `T_reflect_on_mask(axis, mask_template)`

**Mask templates (input‑only):** `ALL, NONZERO, BACKGROUND, NOT_LARGEST, PARITY(p,q,anchor), STRIPES(axis,k), BORDER_RING(k)`.

**Unifier:**

* Set residual `M=(x≠y)`; compose 1–2 predicate masks (see B12) to **exactly** match `M` across pairs; accept if copy/reflect on that mask yields `y` for all pairs.

**Integration:** After B10.

---

### B12) MASK BOOLEAN ALGEBRA (internal mini‑library)

**Primitives (used by unifiers/closures, not exposed as standalone grid ops):**

* `MASK_NOT(A)`, `MASK_AND(A,B)`, `MASK_OR(A,B)`, `MASK_XOR(A,B)`.
* Predicate masks derived **from x only**: `BACKGROUND`, `NONZERO`, `NOT_LARGEST`, `PARITY(p,q,anchor)`, `STRIPES(axis,k)`, `BORDER_RING(k)`.

**Use:** to match `(x≠y)` exactly and drive masked closures (B11, B13).

**Integration:** Implement once and reuse in B9–B11–B13.

---

### B13) STENCIL / NEGATIVE‑SPACE

**Closures:**

* `T_apply_stencil(stencil_rect, color)` — fill within a rectangle or inferred stencil.
* `T_negative_space_in_bbox(rank)` — invert inside object bbox of given rank.

**Unifier:** infer stencil/bbox rank from y; verify across pairs.

**Integration:** Add if coverage stalls on such patterns.

---

### B14) ADJACENCY‑DRIVEN RECOLORS

**Closures:**

* `T_recolor_if_touches(color_a, to=color_b)`
* `T_border_touch_recolor(to=color)`

**Unifier:** build adjacency graph in x; verify affected set equals y exactly across pairs.

**Integration:** Add if fail clusters indicate adjacency patterns.

---

### B15) OBJECT ARITHMETIC

**Closures:**

* `T_match_counts_global(k)` — duplicate smallest until counts == k.
* `T_match_counts_by_color({c:k_c})`.

**Unifier:** infer k / `{k_c}`; verify exact counts across pairs.

**Integration:** Add if needed late.

---

### B16) MERGE / SPLIT

**Closures:**

* `T_merge_nearby(d)` — fuse components within L1≤d.
* `T_split_by_line(line_spec)` — cut objects on a line.

**Unifier:** infer `d` or `line_spec`; accept only if exact match across pairs.

**Integration:** Late, if needed.

---

### B17) RENDER FROM SUMMARY (rare)

**Closures:**

* `T_render_bar(height, color)`
* `T_render_grid_of_objects(rows, cols, color_order)`.

**Unifier:** detect y encodes x’s histogram; verify across pairs.

**Integration:** Only if a fail cluster clearly calls for it.

---

## D) Fail‑Driven Autobuilder (keep B fed continuously)

**Purpose:** When fixed‑point marks a task **under‑constrained**, use the residual to nominate **one closure** to add.

**Workflow:**

1. Read `receipts.jsonl` and `fail_clusters.json` after a run.
2. For each under‑constrained task, compute a **residual fingerprint** from the lfp (which cells remain multi‑valued).
3. Fire **nominators** that propose a closure + params to collapse the residual.

### D1. Δ‑Pattern nominator → `T_copy_by_deltas`

* Match objects (same color/shape); compute displacements; unify a Δ set; propose `T_copy_by_deltas(template_mask, Δset)`.

### D2. Modulo nominator → `T_mod`

* If residual is periodic (rows/cols or diagonal), infer `(p,q,anchor)`; propose `T_mod(...)`.

### D3. Mask nominator → masked variants (B11/B13/B14)

* Compose predicate masks with boolean algebra to match residual; propose `T_reflect_on_mask` / `T_quad_on_mask` / `T_recolor_by_mask`.

### D4. Morphology nominator → `T_open/T_close`

* Boundary‑heavy residual → propose opening/closing.

**Deliverable:**
`runs/<date>/d_suggestions.json` — list of `{task, signature, proposed: {closure_family, params}, rationale}` for manual acceptance.

**Acceptance:**

* For top 2 fail clusters, D proposes at least one closure that (once implemented) solves representative examples.

---

## Wiring & Order of Execution

1. **Implement A (A1–A3)**

   * JSONL receipts with `closures`, `fp.iters`, `fp.cells_multi`.
   * Coverage runner (`scripts/coverage.py`) and failure buckets.
   * Micro‑cache for invariants/closure helpers.

2. **Implement B closures (in order):**
   B1 KEEP_LARGEST_COMPONENT → B2 OUTLINE_OBJECTS → B3 AXIS_PROJECTION_FILL →
   B4 SYMMETRY_COMPLETION → B5 LINE_FROM_EXTREMA → B6 RECOLOR_BY_ROLE →
   **B7 OPEN/CLOSE (k=1)** → **B9 MOD_PATTERN** → **B10 DIAGONAL_REPEAT** →
   B8 QUADRANT_MAPPING → B11 QUADRANT‑ON‑MASK / REFLECT‑ON‑MASK →
   B12 MASK BOOLEAN ALGEBRA (internal) → B13–B17 as needed.

   * After each **two** families, run coverage, check `receipts.jsonl`, pick next.

3. **Enable D (continuous):**

   * After each public run, produce `fail_clusters.json`, then `d_suggestions.json` via nominators (D1–D4).
   * Approve and implement the highest‑value closure; re‑run.

---

## Acceptance & Milestones

* **Milestone 1 (A complete):**
  `receipts.jsonl` (with closures + fp fields) and `fail_clusters.json` produced; micro‑cache active.

* **Milestone 2 (Core B live):**
  B1–B7 implemented; **coverage up materially** on public.

* **Milestone 3 (Modulo/Diagonal):**
  B9 and B10 implemented; large bump on periodic/diagonal tasks.

* **Milestone 4 (Masked variants + algebra):**
  B11 and B12 reduce many residuals to zero; D nominators start landing consistently.

* **Definition of Done (per closure family):**

  1. **Unifier** returns a **single** param set that fits **all** train pairs.
  2. **Closure.apply()** is monotone & shrinking; preserves `singleton(y)` on train.
  3. Integrated into the **ordered** closure list.
  4. Tiny unit tests + at least 2 public tasks primarily solved by this family.

---

## Notes for Claude Code

* **No partial acceptance:** a unifier either fits **all** train pairs and proves train exactness or returns `None`.
* **Input‑only masks:** all mask logic derives from x (train inputs); y is used only to **verify**.
* **Receipts minimal:** JSONL only; no dashboards or edit‑bill narratives needed for submission.
* **Deterministic fallback:** if lfp leaves multi‑valued cells, emit a grid by picking the **lowest allowed color per cell**; mark `status="under_constrained"` so we can improve later.

---

**Use this plan verbatim.** Implement A first, then the B families in the listed order, and keep D running to propose the next closure when a task remains under‑constrained.
