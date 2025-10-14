# ARC‑AGI Coverage Plan — Universe‑Intelligence (A → B → D)

## Goal & non‑negotiables

* **Goal:** increase solved tasks rapidly on ARC public (and v2 where applicable).
* **Non‑negotiables:**

  1. **Determinism** (same inputs → same outputs).
  2. **Receipts** per task (proof artifacts): *train residual == 0* checks, palette/shape deltas, op sequence, and hashes.
  3. **Verifying inducers:** every operator we add must include an inducer that **unifies parameters across *all* train pairs** and **proves residual==0** before the operator is admitted into search.

**Repository context:** Work inside `ui_clean.py` (and `scripts/` as needed). If names differ in our codebase, adapt the function names accordingly.

---

## A) Observability & Coverage Harness (keep it lean but complete)

**Why now:** We add operators (B) fastest when we can *see* where they will pay off. This is a **20–40 line** logging layer + a small runner. No dashboards.

### A1. Receipts & telemetry JSONL (per task)

**Add:** a function `log_receipt(record: dict)` writing to `runs/YYYY-MM-DD/receipts.jsonl`.

**Schema (single line per task):**

```json
{
  "task": "0a1d4ef8.json",
  "status": "solved|failed",
  "program": ["KEEP_LARGEST_COMPONENT(bg=0)", "OUTLINE_OBJECTS(t=1,mode='outer')"],
  "total_residual": 0,
  "residuals_per_pair": [0, 0, 0],
  "palette_invariants": {"preserved": true, "delta": {"1": 0, "2": +4}},
  "component_invariants": {"largest_kept": true, "count_delta": 0},
  "beam": {"expanded": 143, "pruned": 412, "depth": 4},
  "timing_ms": {"build_ops": 13, "beam": 97, "total": 122},
  "hashes": {"task_sha": "...", "program_sha": "..."}
}
```

**Hook points:**

* After each task is solved or declared failed in your public runner (e.g., `run_public(...)` or equivalent).
* Inside the solver (e.g., `solve_task(...)` / `solve_with_beam(...)`) record beam stats and invariants (palette, component count, bbox deltas). Keep this lightweight.

**Deliverables:**

* `runs/<date>/receipts.jsonl` with one line per task.
* Minor helper: `ensure_dir("runs/<date>")`.

---

### A2. Coverage runner + failure buckets

**Add script:** `scripts/coverage.py`.

**Responsibilities:**

* Invoke the public solver over the dataset directory (same entrypoint you already use).
* Read `receipts.jsonl`.
* **Cluster failures** into coarse buckets to guide operator work.

**Failure signature (tuple) per failed task:**

* `palette_delta_type`: `"preserve" | "single-color-add" | "recolor" | "unknown"`.
* `component_count_delta`: `-1 | 0 | +1 | "varies"`.
* `symmetry_flags`: `{v,h,diag,anti}` booleans detected on input vs. output deltas.
* `bbox_delta_kind`: `"crop-to-largest" | "expand" | "none" | "unknown"`.
* `motif_present`: `true|false` (detected by small repeating pattern).
* `mask_hint`: `"background" | "nonzero" | "parity" | "border" | "stripe" | "unknown"`.

**Outputs:**

* `runs/<date>/fail_clusters.json` — array of `{signature, count, examples:[task_ids...]}` ordered by count descending.

**Acceptance:**

* Running `python scripts/coverage.py --arc_public_dir <path>` produces `receipts.jsonl` and `fail_clusters.json`.
* Top 3 clusters obvious at a glance (which operator family to add next).

---

### A3. Micro‑cache

* Add an in‑memory memo to avoid recomputing residuals for the **same prefix program** on **the same train set** during beam expansion:

  * Key: `task_sha + program_prefix_sha`.
  * Value: `residuals_per_pair`.
* Keep the cache local to a single task run (no persistence needed).

**Acceptance:**
Beam expansions drop (log counts before/after via `beam.expanded`).

---

## B) Operator & Inducer Expansion (primary lever for coverage)

**Principle:** Every operator is added with a **verifying inducer** that:

1. Extracts parameters from **all** train pairs,
2. Produces the **same** param set for all pairs,
3. **Proves** `residual == 0` across all train pairs (else reject),
4. Emits a concrete callable that the beam can use.

**Integration point:** `autobuild_operators(train_pairs)` (or equivalent) should return an **ordered list** of ready‑to‑use operators (cheap → expensive).

> Order matters. Put cheap, conservative edits first.

Below are operator families to implement **in order**. For each, implement **(i) Operator**, **(ii) Inducer**, **(iii) Tests**, **(iv) Integrate**.

---

### B1) KEEP_LARGEST_COMPONENT

**Operator:**

```python
def KEEP_LARGEST_COMPONENT(bg: int = 0) -> Callable[[Grid], Grid]:
    ...
```

* Keep only the largest 4‑connected component (by pixel count), zero elsewhere.

**Inducer (sketch):**

* For each train pair `(x,y)`, compute objects on `x` (ignoring `bg`).
* `y` should equal `x` but with all non‑largest components removed (allowing bbox crop or not — choose whichever **unifies across pairs**).
* If true for *all* pairs, emit `Operator("KEEP_LARGEST_COMPONENT", {"bg": bg}, f)`.

**Integration:** Insert after your existing crop/keep rules.

**Tests:**

* Positive: inputs with multiple blobs → output only largest blob.
* Negative: pairs where output recolors or transforms (reject induction).

---

### B2) OUTLINE_OBJECTS

**Operator:**

```python
def OUTLINE_OBJECTS(thickness: int = 1, mode: str = "outer", scope: str = "largest|all") -> Callable[[Grid], Grid]:
    ...
```

* Outline via 4‑neighborhood; `mode="outer"` paints boundary cells that touch background.

**Inducer:**

* For each train `(x,y)`, compute `outline(x)` under a small set of candidates:

  * `thickness ∈ {1}`, `mode ∈ {"outer","inner"}`, `scope ∈ {"largest","all"}`.
* Find the candidate whose `outline(x)` equals `y` for **all pairs**. Emit operator.

**Integration:** After B1.

---

### B3) AXIS_PROJECTION_FILL

**Operator:**

```python
def PROJECT_ALONG_AXIS(axis: str = "row|col", mode: str = "to_border|to_obstacle", scope: str = "largest|all") -> Callable[[Grid], Grid]:
    ...
```

* Extend object pixels along the chosen axis until image border or first obstacle (non‑bg).

**Inducer:**

* For each candidate `(axis, mode, scope)`, apply projection to `x` and compare to `y` across all pairs.
* If one candidate matches all, emit operator.

**Integration:** After B2.

---

### B4) SYMMETRY_COMPLETION

**Operator:**

```python
def REFLECT_COMPLETE(axis: str = "v|h|diag|anti", scope: str = "global|largest|per_object") -> Callable[[Grid], Grid]:
    ...
```

* Reflect selected content and union with the original.

**Inducer:**

* For each `axis`, check if `y == x ∪ reflect_axis(x)` (with `scope` variants).
* Emit only if one axis and scope unify across all pairs.

**Integration:** After B3.

---

### B5) LINE_FROM_EXTREMA

**Operator:**

```python
def DRAW_LINE(strategy: str = "furthest_pair|bbox_corners|centroids", color: str = "object|dominant") -> Callable[[Grid], Grid]:
    ...
```

* Draw straight line between chosen points; Bresenham or simple DDA on grid.

**Inducer:**

* Enumerate candidate strategies; verify `x + line == y` across pairs.

**Integration:** After B4.

---

### B6) RECOLOR_BY_ROLE

**Operator:**

```python
def RECOLOR_BY_ROLE(role: str = "background|largest|rarest", to: str = "dominant_border|mode_inside|fixed") -> Callable[[Grid], Grid]:
    ...
```

* Map colors by detected role to a unified target color.

**Inducer:**

* Check if palette delta matches role mapping identically across pairs (e.g., background→dominant_border color).

**Integration:** After B5.

---

### B7) MORPHOLOGY (small k)

**Operator:**

```python
def DILATE(k: int = 1) -> Callable[[Grid], Grid]: ...
def ERODE(k: int = 1) -> Callable[[Grid], Grid]: ...
```

**Inducer:**

* For k in {1,2} (start with 1):

  * If `DILATE(x,k) == y` across pairs → emit. Same for `ERODE`.

**Integration:** After B6.

---

### B8) QUADRANT_MAPPING

**Operator:**

```python
def MAP_QUADRANTS(rule: str) -> Callable[[Grid], Grid]:
    # rule encodes copy/transform from source quadrant to targets
```

**Inducer:**

* Split grid into 2×2 quadrants; check if `y` is obtained by copying/rotating one quadrant to others (enumerate a small ruleset).
* Emit if a single rule explains all pairs.

**Integration:** After B7.

---

### (Note on Tiling/Mask)

You already have tiling+mask working. As a follow‑up, **expand the mask template set** (used in induction only; masks must be *input‑only*):

* `ALL`, `NONZERO`, `BACKGROUND`, `NOT_LARGEST`, `PARITY(anchor)`, `BORDER_RING(k)`, `STRIPES(axis,k)`
  Each template must unify across *all* train pairs and still yield residual==0.

---

## D) Fail‑Driven Autobuilder (keep B fed continuously)

**Purpose:** When a task fails, the system proposes **new, concrete operators/inducers** that are likely to solve that failure cluster. You approve, implement, and re‑run. This converts manual discovery into a loop.

**Workflow:**

1. After a public run, read `receipts.jsonl` to gather failed tasks.
2. For each failed task, compute a **fail signature** (same as A2).
3. Based on the signature, fire **pattern miners** that propose parameterized operators with verifying inducers.

### D1. Δ‑Pattern Miner → `COPY_BY_DELTAS`

* For each train pair, detect matched objects (same color & size or shape hash).
* Compute displacement vectors `(Δr, Δc)` between input and output counterparts.
* If the **same Δ** (or small set like ±Δ) unifies across all pairs, propose:

  * `COPY_BY_DELTAS(global Δ)` or
  * `COPY_BY_DELTAS_PER_COLOR({color_i: Δ_i})`
* Include an inducer that verifies residual==0 on all pairs.

### D2. Motif Miner → `REPEAT_TILE_ON_MASK`

* Extract small motif `M` present in output.
* Search mask templates **from input only** (background, nonzero, not_largest, parity, stripes).
* If one mask template with `M` explains **all** train pairs (IoU == 1.0 between edited set and mask), propose operator + inducer.

### D3. Mask Miner → Recolor/Border/Stripe fills

* If residuals lie on borders, stripes, or parity sets, propose:

  * `RECOLOR_BY_MASK(template, target_color_strategy)`
  * `BORDER_FILL(k)`
* The inducer selects template + target color (e.g., mode on border pixels) and proves residual==0.

**Deliverable:**
`d_suggestions.json` — list of `{task, signature, proposed_operator_spec, expected_inducer_checks}` for manual acceptance.

**Acceptance:**

* For ≥80% of fails in top 2 clusters, D proposes at least one operator spec that, once implemented, solves examples representative of that cluster.

---

## Wiring & Order of Execution

1. **Implement A (A1–A3)**

   * Logging (JSONL), coverage runner, minimal cache.
   * **Outcome:** You can see where to spend time in B.

2. **Implement B operators (in order):**
   B1 KEEP_LARGEST_COMPONENT → B2 OUTLINE_OBJECTS → B3 AXIS_PROJECTION_FILL →
   B4 SYMMETRY_COMPLETION → B5 LINE_FROM_EXTREMA → B6 RECOLOR_BY_ROLE → B7 MORPH (k=1) → B8 QUADRANT_MAPPING.

   * Integrate each with its inducer into `autobuild_operators(train_pairs)` (cheap to expensive).
   * After each two families, run coverage, inspect receipts, choose next.

3. **Enable D (continuous):**

   * After each public run, generate `fail_clusters.json`, run miners (D1–D3), produce `d_suggestions.json`.
   * Approve and implement the highest‑value suggestions, then re‑run coverage.

4. **Optional later:**

   * **C (Guided beam):** add a node score `(sum_residual, -pairs_at_zero, program_length)` and tiny operator priors.
   * **E (Perf):** cache candidate ops per task signature; parallel jobs if desired.

---

## Code Sketches (for Claude to implement)

> These are short sketches; integrate with your existing helpers (`components`, `equal`, etc.) and array type (list or numpy). Ensure all inducers **prove residual==0 across *all* train pairs** before emitting.

### KEEP_LARGEST_COMPONENT

```python
def KEEP_LARGEST_COMPONENT(bg: int = 0):
    def f(z):
        H, W = z.shape
        out = np.zeros_like(z)
        objs = components(z, bg=bg)  # returns list with .pixels, .size, .color, .bbox
        if not objs: return out
        largest = max(objs, key=lambda o: o.size)
        for (r, c) in largest.pixels:
            out[r, c] = largest.color
        return out
    return f

def induce_KEEP_LARGEST(train, bg=0):
    P = KEEP_LARGEST_COMPONENT(bg)
    for x, y in train:
        if not np.array_equal(P(x), y):
            return []
    return [Operator("KEEP_LARGEST_COMPONENT", {"bg": bg}, P, "keep-largest")]
```

### OUTLINE_OBJECTS

```python
def OUTLINE_OBJECTS(thickness=1, mode="outer", scope="all"):
    def neighbors4(r,c,H,W):
        if r>0:   yield r-1,c
        if r+1<H: yield r+1,c
        if c>0:   yield r,c-1
        if c+1<W: yield r,c+1
    def f(z):
        H, W = z.shape
        out = np.zeros_like(z)
        objs = components(z, bg=0)
        targets = [max(objs, key=lambda o:o.size)] if scope == "largest" else objs
        for o in targets:
            pix_set = set(o.pixels)
            for (r,c) in o.pixels:
                for (nr,nc) in neighbors4(r,c,H,W):
                    if (nr,nc) not in pix_set:  # touching background => boundary
                        out[r,c] = o.color
                        break
        return out
    return f

def induce_OUTLINE(train):
    candidates = [("outer","largest"), ("outer","all")]
    for mode, scope in candidates:
        P = OUTLINE_OBJECTS(1, mode, scope)
        if all(np.array_equal(P(x), y) for x,y in train):
            return [Operator("OUTLINE_OBJECTS", {"t":1,"mode":mode,"scope":scope}, P, "outline")]
    return []
```

### PROJECT_ALONG_AXIS

```python
def PROJECT_ALONG_AXIS(axis="row", mode="to_border", scope="all"):
    def f(z):
        H, W = z.shape
        out = z.copy()
        objs = components(z, bg=0)
        targets = [max(objs, key=lambda o:o.size)] if scope=="largest" else objs
        for o in targets:
            for (r,c) in o.pixels:
                if axis == "row":
                    step = [(r,k) for k in range(0, c)] if mode=="to_border" else [(r,c-1)]
                    for (rr,cc) in step: out[rr,cc] = o.color
                    step = [(r,k) for k in range(c+1, W)] if mode=="to_border" else [(r,c+1)]
                    for (rr,cc) in step: out[rr,cc] = o.color
                else:  # col
                    step = [(k,c) for k in range(0, r)] if mode=="to_border" else [(r-1,c)]
                    for (rr,cc) in step: out[rr,cc] = o.color
                    step = [(k,c) for k in range(r+1, H)] if mode=="to_border" else [(r+1,c)]
                    for (rr,cc) in step: out[rr,cc] = o.color
        return out
    return f

def induce_PROJECT(train):
    for axis in ("row","col"):
        for mode in ("to_border","to_obstacle"):  # start with to_border
            for scope in ("largest","all"):
                P = PROJECT_ALONG_AXIS(axis, mode, scope)
                if all(np.array_equal(P(x), y) for x,y in train):
                    return [Operator("PROJECT_ALONG_AXIS", {"axis":axis,"mode":mode,"scope":scope}, P, "project")]
    return []
```

*(Similarly implement SYMMETRY_COMPLETION, LINE_FROM_EXTREMA, RECOLOR_BY_ROLE, DILATE/ERODE, QUADRANT_MAPPING with the same pattern.)*

---

## Integration Order in `autobuild_operators(train_pairs)` (cheap → expensive)

```python
def autobuild_operators(train):
    ops = []
    # existing simple ops...
    ops += induce_KEEP_LARGEST(train, bg=0)         # B1
    ops += induce_OUTLINE(train)                    # B2
    ops += induce_PROJECT(train)                    # B3
    ops += induce_REFLECT_COMPLETE(train)           # B4
    ops += induce_DRAW_LINE(train)                  # B5
    ops += induce_RECOLOR_BY_ROLE(train)            # B6
    ops += induce_MORPH(train)                      # B7
    ops += induce_MAP_QUADRANTS(train)              # B8
    # existing tiling/mask (with expanded mask templates)
    return ops
```

---

## Running Loop

1. **Public run (with A enabled):**

   ```
   python ui_clean.py --arc_public_dir <path> --out_json runs/<date>/predictions.json --beam 160 --depth 6 --jobs 8
   ```

   (Adjust flags to your runner’s interface.)

2. **Inspect outputs:**

   * `runs/<date>/receipts.jsonl`
   * `runs/<date>/fail_clusters.json`

3. **Add next operators (B)**

   * Follow the ordered list; after each pair of operator families, re‑run public and observe deltas.

4. **Invoke D (continuous)**

   * Run miners on fails, generate `d_suggestions.json`.
   * Approve one or two high‑value suggestions; implement (with verifying inducers) and re‑run.

5. *(Optional)* **C: Guided beam**

   * If expansion is heavy: add node score `(sum_residual, -pairs_at_zero, program_length)` and small operator priors to order expansions.

6. *(Optional)* **E: Perf**

   * Cache candidate ops per task signature; optionally parallelize.

---

## Acceptance & Milestones

* **Milestone 1 (A complete):**

  * `receipts.jsonl` and `fail_clusters.json` produced; micro‑cache active; you can grep for top fail signatures.

* **Milestone 2 (B1–B3 live):**

  * KEEP_LARGEST, OUTLINE, PROJECT implemented with inducers and integrated; **coverage up materially** (expect +tens of solves on public).

* **Milestone 3 (B4–B6 live):**

  * Reflection completion, line drawing, recolor by role added; another coverage bump.

* **Milestone 4 (D loop active):**

  * `d_suggestions.json` emitted after runs; at least one suggestion per top cluster; accepted suggestions convert to new inducers that verify residual==0 across pairs.

* **Definition of Done (per operator family):**

  1. Operator callable implemented.
  2. Inducer proves residual==0 across all train pairs or rejects.
  3. Integrated into `autobuild_operators` in the specified order.
  4. Unit tests on synthetic pairs + at least 2 public tasks solved primarily by this operator.

---

## Notes for Claude Code

* **No partial acceptance:** an inducer either unifies parameters across **all** train pairs and verifies residual==0, or it returns `[]` (do not add operator).
* **Keep logs tiny:** JSONL only, one line per task; no plotting/UI required.
* **Prefer input‑only masks** in all mask‑based inducers (no peeking at ground truth except for verification).
* **When in doubt, be conservative:** prefer operators that preserve palette/structure unless receipts prove otherwise.

---

Use this plan verbatim to start: **implement A1–A3, then B1–B3, then enable D miners**.
Re‑run public after each step, and let receipts + fail clusters tell you the next operator to add.
---
1. A1 — Hash computation

* task_sha: SHA-256 over a canonical JSON of all train pairs. Serialize as

  ```
  {"train":[{"in": grid_to_list(x), "out": grid_to_list(y)}, ...]}
  ```

  Use ints only, no numpy types, sorted keys, no whitespace variance. Example:

  ```python
  def task_sha(train_pairs):
      payload = {"train":[{"in":x.tolist(), "out":y.tolist()} for x,y in train_pairs]}
      return sha256(json.dumps(payload, sort_keys=True).encode())
  ```
* program_sha: SHA-256 over the operator sequence with names and params, in order.

  ```python
  def program_sha(ops):
      payload = [{"name":op.name, "params":op.params} for op in ops]  # params must be JSON serializable and with sorted keys
      return sha256(json.dumps(payload, sort_keys=True).encode())
  ```

2. A1 — Invariants tracking

* Yes, add helpers in utils.py:

  * `compute_palette_delta(x, y) -> {"preserved": bool, "delta": {color: int}}` where delta is count_y[color] − count_x[color].
  * `compute_component_delta(x, y) -> {"count_delta": int, "largest_kept": bool, "size_largest_x": int, "size_largest_y": int}` using your existing components routine.
* Where to track: inside `solve_with_beam` just before returning a solution or declaring failure. Compute per pair then aggregate:

  * palette_invariants: preserved is `all(p["preserved"] for p in per_pair)` and include a merged delta by summing deltas.
  * component_invariants: count_delta could be mode or list, largest_kept is `all`.
* Include these fields in the JSONL record.

3. A2 — Coverage runner integration

* Prefer importable runner. Create `run_solver(dataset_dir, out_dir) -> None` in `ui_clean.py` that writes receipts.jsonl to `out_dir` and returns nothing.
* `scripts/coverage.py` should import `run_solver` and then read `receipts.jsonl` to produce `fail_clusters.json`.
* Only use subprocess if import is not feasible. Import keeps logs and errors cleaner.

4. B — Integration order with existing inducers
   Interleave by cost and payoff. Use this order in `autobuild_operators(train)`:

1) COLOR_PERM
2) ROT_FLIP
3) PARITY_CONST
4) CROP_KEEP
5) KEEP_LARGEST_COMPONENT   ← B1
6) OUTLINE_OBJECTS          ← B2
7) PROJECT_ALONG_AXIS       ← B3
8) SYMMETRY_COMPLETION      ← B4
9) LINE_FROM_EXTREMA        ← B5
10) RECOLOR_BY_ROLE         ← B6
11) MORPH_DILATE_ERODE      ← B7
12) QUADRANT_MAPPING        ← B8
13) TILING                  (with input-only mask induction)
14) HOLE_FILL
15) COPY_BY_DELTAS
    Reasoning: 1–3 are ultra cheap and often decisive. 4–7 are cheap to moderate and unlock many public tasks. 8–12 are moderate. 13–15 can be heavier or more brittle, so keep later.

5. A — Complete before B?

* Do A1 first. It is small and gives you receipts immediately.
* You can start B in parallel right after A1. Do not block on A2 or A3.
* Add A2 minimal clustering after your first B1–B3 pass to guide the next operator choices.
* Add A3 micro-cache when you notice beam expansion costs rising.

If you want a snippet for the JSONL writer or the exact invariant helpers, say the word and I will drop them in ready to paste.
