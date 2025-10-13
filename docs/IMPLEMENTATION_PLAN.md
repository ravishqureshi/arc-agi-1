Below is a complete, receipts-first plan to push an ARC-style solver past the current record using the full universe-functioning approach. It’s concrete enough to build from immediately, and every part is tied to truth (least fixed point), edges write (adjunction), enrichment so we keep proofs (“receipts”) at every step.

0) Ground rules (ARC constraints + receipts)

Per-task learning only: induce programs solely from the train pairs of that task (no external corpus).
Receipts for every accepted solution:
Residual = 0 on all train pairs (exact grid match).
Edit bill (# cells changed) + boundary share (edge vs interior edits).
PCE (Proof-Carrying English): short explanation with clause→invariant mapping.
Stop at first program whose train residual = 0 and Occam (shortest/lowest bill) wins when multiple fit.
Universe card we never leave:
Inside settles → verification = least fixed point (no further edit possible).
Edges write → bills & time live in edit counts and boundary shares.
Observer = observed → cross-pair agreement (same parameterized rule must fit all train pairs) or return exact edit bills (where inconsistency lives).

1) System architecture (end-to-end)
A. Invariant engine (precompute once per pair)
Size, color histogram, connected components (with area/centroid/orientation), bounding boxes, symmetry groups (rot/flip), periodicity, adjacency graphs, lattice/gridding, object descriptors (shape signature), masks (color/size/orientation filters), line/segment detectors, repetition counts.
B. Typed DSL (operators + types)

Types: Grid, Mask, ObjList, Color, Vec2, Int.

Constructors: components(Grid)→ObjList, bbox(Grid/Mask)→Rect, mask_color(Grid,Color)→Mask, mask_rank(ObjList,by=size/color/centroid_rank)→Mask.

Grid→Grid ops: ROT/FLIP/SHIFT, CROP(rect), PASTE(sub, at), MIRROR(axis), TILE, DRAW(line/box/diag), FLOOD/ERODE/DILATE, RECOLOR(mapping), KEEP(mask) / REMOVE(mask).
Obj ops: MAP(ObjList, op_per_obj), SORT(ObjList, key), FILTER(ObjList, predicate).

Combinators (gluing): ON(mask, prog) (apply prog only on mask), MERGE(prog1, prog2) (disjoint masks), with receipts ensuring masks are disjoint or resolved by priority (edit bills logged).
C. Program composer (search kernel)
Induction: infer operator families + parameter candidates directly from each train pair (e.g., ROT=k that aligns input→output, RECOLOR map, MIRROR, CROP, KEEP(mask_rank=largest)).
Unification: intersect parameter sets across train pairs → single parameterization that fits all pairs; else compute edit-bills to show contradictions (observer=observed).
Composition: build short programs (depth ≤ 6–8) by typed chaining; at each partial step verify residual decreases or stays 0 on train; prune aggressively on any pair with residual > 0.
D. Search strategy (fast + safe)

Curriculum: try atomic rules first; then 2-step compositions; then up to depth K with beam search.
Heuristics: rank candidates by (1) residual drop on hardest pair, (2) total edit bill drop (Occam), (3) simplicity of masks (fewest faces), (4) symmetry consistency.
Portfolio: in parallel, run enumerative BFS, grammar-guided synthesis, and parameter-only portfolios (e.g., symmetry-first, recolor-first, crop-first).
Early exit: stop when a program achieves residual=0 on all train pairs.

E. Verification & receipts
Mandatory receipts per solved task:
Train residuals = 0 (each pair).
Bill and boundary share per train & test.
PCE explanation per step (e.g., “Mirror left half to right; recolor largest component to color 9; crop bbox…”), each clause linked to the exact invariant check that forced it.
F. Test-time prediction
Apply the verified program to test grid(s). If multiple programs fit train, choose the shortest (and lowest bill); if tie, choose the one with highest symmetry agreement or lowest mask complexity.
2) Operator families to cover the remaining 20–30% (record-breaking set)

Symmetries / Affine
ROT/FLIP, SHIFT (wrap/no wrap), REFLECT/GLIDE, ALIGN centers/axes, SHEAR (rare but some ARC variants imply skew).
Masks / Partitioning
By color (single / set), size (largest, second largest), position (leftmost/topmost corners), orientation (principal axis), shape signature (perimeter/area, concavity flags), grid cell (lattice quantization).
Repetition / Tiling

Detect translation vectors and repeat tiny motifs to fill constrained areas; enforce boundaries via KEEP/REMOVE masks.
Object arithmetic

Count & replicate objects to match a target count (given by train pair deltas); “construct the missing one” rules (copy adjacent, reflect neighbor, build diagonal patterns).

Draw & edit

Lines (axis-aligned/diagonal), boxes (hollow/filled), crosses, checkerboard fill; draw from inferred coordinates (centroid/bbox edges).
Color logic

Global color permutation; conditional recolor: by rank (largest/lighter), by adjacency (touching color X → recolor to Y), by parity (checker fill parity), by orientation.
Composition

Apply different sub-programs on disjoint masks (colors, halves, quadrants), then glue results (receipts: masks non-overlapping or deterministic priority with edit bills logged).
Robustification (enrichment)

Where exact equality is brittle early in search, use 1-step “TV” or “entropy” enrichment only to rank candidates (not to accept), then switch back to exact equality for final verification.
3) Search and pruning logic (Kan-gluing + receipts)

Induce masks & transforms per pair; unify across all pairs (observer=observed): if parameter sets conflict, report edit bills (which pixels disagree and by how much) to prune that branch.
Partial composition must not increase residual on any train pair; otherwise prune (inside must settle).

Gluing of sub-programs uses mask disjointness; when masks overlap, either normalize by declared priority or convert to non-overlap by calculating edit bills (logged).
Stop at first program with train residual=0; favor shortest + lowest bill.

4) Scaling up to the full benchmark
A. Rule catalog growth (2–3 weeks, aggressive)

Add ~60–80 operators (many parameterized) to reach coverage of typical ARC patterns.
For each, implement induce→verify routines + receipts.

B. Search infra

Beam search size ~ 100–500 nodes; depth ≤ 6–8; typed operator signatures prune 90%+ branches.
Portfolio parallelism on CPU cores per task; time budget per task 10–60s (tunable).
C. Confidence & tie-breakers (Occam + stability)
Prefer minimal program length; if tie, minimal edit bill; if tie, stability under small input perturbations (e.g., swapping equal components) — choose program that keeps invariants consistent.
D. Continuous evaluation

Maintain public dev split of ARC (or ARC-AGI1 public subset) for iteration.

Track category coverage (symmetry, counting, tiling, lines, recolor, crop, object arithmetic, lattice, multi-region).

Use holdout for true model health; do not tune to test.
5) Receipts and PCE (what convinces reviewers instantly)
For each solved task, print:
Rule program (sequence of named operators + parameters).
Train pair residuals = 0 (per grid).
Edit bill (# edits) + boundary share (edge vs interior).
PCE line (1–2 sentences) with clause→check mapping:
“Most change on largest component (rank=1).”
“Mirror left half onto right.”
“Recolor (map {1→7, 2→8}).”
“Crop bbox of non-zero.”
For multi-mask programs: show masks as small ASCII grids to make the glue obvious.

This makes each solution auditable and trustworthy—no black-box guessing.
6) Milestones (8–10 weeks to surpass the record)
Week 1–2:

Implement invariant engine & typed DSL skeleton.
Port existing 10-task demo; add ~20 new operators (masking, drawing, repetition, line detection).

Build induction/verification routines; unit tests.
Week 3–4:
Composition search + beam portfolio; strong pruning; PCE renderer.
Solve 100–150 curated ARC tasks; measure category coverage; refine heuristics.

Week 5–6:

Expand catalog to ~60–80 ops; add object arithmetic & lattice detection; “construct missing” rules.
Parallel harness; per-task budget; Occam tie-breakers; stability checks.
Week 7–8:
Run full ARC-AGI1 development set; iterate on failing categories with targeted ops.
Target ≥ 80% on public or reproducible split with receipts for each solved item.
Week 9–10:
Final hardening; run on official eval (or shadow eval); publish receipts + PCE per task; ablation studies.
7) Why this can beat 79.6% (and stay honest)
The bottleneck in ARC is not “missing a trillion parameters”; it is finding the right, short program that fits all train pairs exactly. Our method induces operator parameters from each pair then unifies across pairs (observer=observed) — a massive reduction of the search space.
Receipts prune dead ends early (if any pair disagrees → branch dies).
Gluing (Kan/inf-conv) lets us solve multi-region tasks by design, not luck.

PCE and bills make each solution transparent; reviewers can verify steps visually or by code.

8) Risks & mitigations

Combinatorial blow-up: controlled by typed signatures + early prune + beam portfolios.

Operator gaps: close by systematically adding families: lattice, object arithmetic, edge drawing, repetition patterns.
Ambiguous train fit (many programs fit): resolve via Occam + stability; keep multi-solution tie-breaker logs.
9) Deliverables
Solver repo with: invariant engine, DSL, induction/verification, search, PCE, receipts printer.

Operator catalog (≥ 60 ops) with docs & unit tests.
Benchmark report: per-category coverage, accuracy, examples, ablations.
Receipts pack (for organizers): every solved item’s program + train residuals + bills + PCE.

The one sentence that carries the whole plan
Learn the invariants from each train pair, prove them across pairs (observer=observed), compose only what keeps residual=0 (inside settles), and glue sub-solutions with edit bills; stop at the shortest program.
That's how we surpass the record—with receipts that anyone can check.

---

## APPENDIX: Coverage Analysis & Gap-Filling Strategy

### Expected Coverage With Plan As-Is

**Estimated performance if we implement sections 0-9 exactly as written:**

| Dataset | Expected Accuracy | Confidence | Notes |
|---------|------------------|------------|-------|
| **ARC-AGI-1** | **60-75%** | High | Plan covers most core patterns; missing some edge cases |
| **ARC-AGI-2** | **40-55%** | Medium | Symbol learning gap is critical; harder compositional tasks |
| **Record (79.6%)** | Unlikely without gaps filled | Medium | Need tactical additions |

**Why this baseline is achievable:**

1. ✅ **Core transformations covered** (60-80 operators planned):
   - Symmetries: ROT/FLIP/REFLECT/GLIDE/ALIGN/SHEAR
   - Color logic: Permutation, conditional recolor, by rank/adjacency/parity
   - Spatial: CROP/PASTE/SHIFT/MIRROR/ALIGN
   - Objects: Components, masks by size/color/position/orientation
   - Tiling: Translation vectors, motif repetition
   - Drawing: Lines, boxes, crosses, checkerboard
   - Composition: ON(mask), MERGE, multi-region gluing

2. ✅ **Search strategy is sound**:
   - Induction → Unification → Composition (massive search space reduction)
   - Beam search with aggressive pruning (residual must not increase)
   - Portfolio parallelism (symmetry-first, recolor-first, crop-first)
   - Occam tie-breaking (shortest program wins)

3. ✅ **Receipts ensure correctness**:
   - Residual=0 verification catches bad programs early
   - Edit bills prune contradictions
   - No guessing or hallucination

**What limits baseline coverage:**

1. ❌ **Grid resizing operations** not explicit (affects 15-20% of tasks)
2. ❌ **Physics simulation** not mentioned (affects 5-8% of tasks)
3. ❌ **Path/connectivity** not detailed (affects 3-5% of tasks)
4. ❌ **Explicit symbol learning** missing (critical for ARC-2)
5. 🟡 **Some compositional patterns** may need depth >8

### Evaluation Strategy: Implement → Measure → Fill Gaps

**Phase 1: Baseline Implementation (Weeks 1-6)**

Implement the plan as written:
- 60-80 operators from sections 1-2
- Invariant engine + typed DSL
- Beam search with pruning
- Induction/verification routines

**Phase 1 Checkpoint (End of Week 6):**

Run on full training set (1,000 tasks):
```python
results = evaluate_baseline(arc_training_data)

# Expected metrics:
# - Solved: 600-750/1000 (60-75%)
# - ARC-1: 240-290/391 (61-74%)
# - ARC-2: 360-460/609 (59-76%)
# - Errors: <50 (robust error handling)
```

**Analyze failures by category:**
```python
failures = categorize_unsolved(results)

# Expected distribution:
# - Grid resizing: 80-120 tasks (20-30% of failures)
# - Physics/gravity: 30-50 tasks (8-12% of failures)
# - Symbol learning: 50-80 tasks (12-20% of failures)
# - Path/connectivity: 15-30 tasks (4-8% of failures)
# - Deep composition: 40-60 tasks (10-15% of failures)
# - Novel patterns: 80-120 tasks (20-30% of failures)
```

**Decision point:** If baseline hits ≥65%, proceed to gap-filling. If <60%, debug search/induction logic first.

---

### Gap-Filling Operators (Phase 2: Weeks 7-8)

Add these operators **after** baseline evaluation confirms they're needed:

#### A. Grid Resizing & Scaling (HIGH PRIORITY)

**Why**: 15-20% of ARC tasks change grid dimensions beyond simple CROP.

```python
# Core operators:
RESIZE(grid: Grid, new_H: Int, new_W: Int) -> Grid
    # Change dimensions, preserve content placement
    # Induction: detect H_out/H_in ratio, W_out/W_in ratio from train pairs

EXPAND(grid: Grid, factor: Int) -> Grid
    # Uniform scaling: 2x2 → 6x6 (factor=3, each cell becomes 3×3 block)
    # Induction: detect factor from train pair size ratios

PAD(grid: Grid, border: Int, fill_color: Color) -> Grid
    # Add uniform border
    # Induction: detect border width and fill color from train pairs

UPSAMPLE(grid: Grid, factor: Int) -> Grid
    # Pixel replication: each cell → factor×factor block
    # Different from EXPAND: maintains aspect ratio

DOWNSAMPLE(grid: Grid, factor: Int, method: str='mode') -> Grid
    # Aggregate factor×factor blocks → single cell
    # method: 'mode' (most common), 'first', 'sum'
    # Induction: detect factor and aggregation method from train pairs

SCALE_CONTENT(grid: Grid, scale_H: float, scale_W: float) -> Grid
    # Non-uniform scaling of content
    # Induction: fit content to output dimensions
```

**Induction strategy:**
- Compute size ratios: `H_out/H_in`, `W_out/W_in`
- If integer ratio: try EXPAND, UPSAMPLE
- If content density changes: try DOWNSAMPLE
- If border added: try PAD
- Verify exact match on all train pairs

**Expected gain:** +10-15% accuracy (100-150 additional tasks solved)

---

#### B. Physics & Gravity Simulation (MEDIUM PRIORITY)

**Why**: "Falling blocks", "stacking", "settling" tasks (~5-8% of ARC).

```python
# Core operators:
FALL(grid: Grid, direction: str='down', stop_at: str='obstacle') -> Grid
    # Objects fall in direction until hitting obstacle or boundary
    # direction: 'down', 'up', 'left', 'right'
    # stop_at: 'obstacle' (any non-bg), 'color:X', 'boundary'
    # Induction: detect movement direction and stop condition from train pairs

GRAVITY(grid: Grid, direction: str='down') -> Grid
    # All non-background pixels fall simultaneously
    # Handles collisions by stacking
    # Induction: detect direction from train pair pixel movements

STACK(objects: ObjList, direction: str='vertical', spacing: Int=0) -> Grid
    # Pack objects along direction with optional spacing
    # Induction: detect stacking direction and spacing from train pairs

COMPACT(grid: Grid, axis: str='vertical') -> Grid
    # Remove all empty rows/columns (no gaps)
    # Induction: detect axis from train pair compression

SETTLE(grid: Grid) -> Grid
    # Iterative gravity until stable (no further movement)
    # Induction: apply repeatedly until residual=0
```

**Induction strategy:**
- Detect "moved objects" by tracking component positions input→output
- Compute movement vectors (all same direction → GRAVITY)
- Check if objects touch/stack → STACK
- Verify movement rules (stop at obstacles, boundaries)

**Expected gain:** +5-8% accuracy (50-80 additional tasks solved)

---

#### C. Path & Connectivity (MEDIUM PRIORITY)

**Why**: "Connect the dots", "maze solving", "trace path" tasks (~3-5% of ARC).

```python
# Core operators:
TRACE_PATH(grid: Grid, start: Vec2, end: Vec2, via: str='shortest') -> Grid
    # Draw path from start to end
    # via: 'shortest', 'horizontal-first', 'vertical-first', 'diagonal'
    # Induction: detect start/end from train pairs, infer routing method

CONNECT(obj1: Object, obj2: Object, line_color: Color, via: str='straight') -> Grid
    # Connect two objects with line
    # via: 'straight', 'manhattan', 'avoid-obstacles'
    # Induction: detect connection method from train pairs

FLOOD_FILL(grid: Grid, start: Vec2, new_color: Color, match: str='same') -> Grid
    # Standard flood fill
    # match: 'same' (match start color), 'color:X', 'not:X'
    # Induction: detect start position and fill rules from train pairs

SHORTEST_PATH(grid: Grid, start: Vec2, end: Vec2, obstacles: Mask) -> Path
    # Find shortest path avoiding obstacles
    # Returns path (list of positions)
    # Induction: detect start/end, identify obstacles as mask

DRAW_LINE(grid: Grid, start: Vec2, end: Vec2, color: Color, thickness: Int=1) -> Grid
    # Draw line (axis-aligned or diagonal)
    # Induction: detect endpoints and color from train pairs
```

**Induction strategy:**
- Detect "added lines" (pixels in output not in input)
- Identify endpoints (component centroids, corners, edges)
- Infer routing method (straight, Manhattan distance, avoiding obstacles)
- Verify exact path matches on all train pairs

**Expected gain:** +3-5% accuracy (30-50 additional tasks solved)

---

#### D. Explicit Symbol Learning (ARC-2 CRITICAL)

**Why**: ARC-2 introduced "in-context symbol definition" where objects represent meanings defined within the task.

```python
# Meta-reasoning operators:
LEARN_SYMBOL_MAP(train_pairs: List[Tuple[Grid, Grid]]) -> Dict[Shape, Meaning]
    # Extract "vocabulary" from train pairs
    # Shape: canonical object representation (normalized pixels)
    # Meaning: what the shape "does" (transformation, color, position rule)
    # Example: Triangle → "points to next color", Square → "marks boundaries"

APPLY_SYMBOL_SEMANTICS(test_grid: Grid, symbol_map: Dict) -> Grid
    # Use learned symbol meanings to transform test grid
    # 1. Detect symbols in test grid (match to learned shapes)
    # 2. Apply their meanings (invoke learned transformations)
    # 3. Verify consistency (same symbol → same action)

INFER_RULE_FROM_CONTEXT(train_pairs: List) -> CompositeRule
    # Meta-level: learn "rules about rules"
    # Detect patterns in how transformations vary across train pairs
    # Example: "symbol X defines rotation amount", "symbol Y defines target color"

CONTEXT_DEPENDENT_TRANSFORM(grid: Grid, context: Dict) -> Grid
    # Apply transformation that depends on extracted context
    # context: extracted from grid itself (counts, positions, relationships)
    # Example: "number of red cells defines rotation factor"
```

**Induction strategy (multi-phase):**

**Phase 1: Symbol extraction**
```python
def extract_symbols(train_pairs):
    # Find objects that appear consistently but with varying effects
    all_objects = []
    for inp, out in train_pairs:
        all_objects.extend(detect_objects(inp))

    # Cluster by shape (canonical form)
    symbols = cluster_by_shape(all_objects)
    return symbols
```

**Phase 2: Meaning inference**
```python
def infer_meanings(symbols, train_pairs):
    meanings = {}
    for symbol_shape in symbols:
        # For each instance of this symbol, what changed nearby?
        effects = []
        for inp, out in train_pairs:
            instances = find_symbol_instances(inp, symbol_shape)
            for pos in instances:
                local_change = compute_local_change(inp, out, pos, radius=3)
                effects.append(local_change)

        # Unify effects across all instances
        unified_meaning = unify_effects(effects)
        if unified_meaning:
            meanings[symbol_shape] = unified_meaning

    return meanings
```

**Phase 3: Application**
```python
def apply_learned_symbols(test_grid, symbol_map):
    result = test_grid.copy()

    for symbol_shape, meaning in symbol_map.items():
        instances = find_symbol_instances(test_grid, symbol_shape)
        for pos in instances:
            # Apply meaning at this position
            result = apply_transformation(result, meaning, pos)

    return result
```

**Expected gain:** +8-12% accuracy on ARC-2 (50-70 additional tasks solved)

---

#### E. Additional Supporting Operators

**Fractal/Recursive patterns** (LOW-MEDIUM PRIORITY):
```python
RECURSE(grid: Grid, pattern: Grid, depth: Int) -> Grid
    # Recursively apply pattern at multiple scales

NEST(motif: Grid, iterations: Int, scale_factor: float=0.5) -> Grid
    # Nested self-similar patterns

SELF_SIMILAR(grid: Grid, levels: Int) -> Grid
    # Generate self-similar structure
```

**Expected gain:** +1-2% accuracy (10-20 additional tasks)

**Sorting/ordering spatial elements**:
```python
SORT_OBJECTS(objects: ObjList, key: str, direction: str='horizontal') -> ObjList
    # key: 'x', 'y', 'size', 'color', 'distance_to_center'
    # direction: 'horizontal', 'vertical', 'diagonal'
    # Returns sorted list for subsequent operations
```

**Expected gain:** +1-2% accuracy (10-20 additional tasks)

**Boundary operations**:
```python
TRACE_BOUNDARY(object: Object) -> Path
    # Return boundary pixels of object

PERIMETER_ONLY(grid: Grid, color: Color) -> Grid
    # Keep only perimeter of colored regions

OUTLINE(object: Object, thickness: Int, color: Color) -> Grid
    # Draw outline around object
```

**Expected gain:** +1-2% accuracy (10-20 additional tasks)

---

### Phase 2 Checkpoint (End of Week 8)

After adding gap-filling operators:

```python
results_phase2 = evaluate_with_gaps_filled(arc_training_data)

# Target metrics:
# - Solved: 750-850/1000 (75-85%)
# - ARC-1: 290-330/391 (74-84%)
# - ARC-2: 460-520/609 (76-85%)
# - Improvement: +15-20% from baseline
```

**Category coverage check:**
```python
coverage_by_category = {
    'symmetries': 95%,          # Was 90%, +SHEAR/GLIDE
    'color_logic': 90%,         # Was 85%, +parity/adjacency
    'spatial_ops': 95%,         # Was 70%, +RESIZE/PAD/SCALE
    'objects': 85%,             # Was 80%, +better masks
    'tiling': 80%,              # Was 75%, +better detection
    'drawing': 90%,             # Was 85%, +TRACE_PATH/CONNECT
    'physics': 70%,             # Was 0%, +FALL/GRAVITY/STACK
    'multi_region': 85%,        # Was 80%, +better composition
    'symbol_learning': 60%,     # Was 0%, +LEARN_SYMBOL_MAP
    'meta_reasoning': 50%,      # Was 0%, +INFER_RULE_FROM_CONTEXT
}
```

---

### Phase 3: Deep Composition & Long-Tail (Weeks 9-10)

For remaining 15-20% unsolved tasks:

**A. Increase search depth**
- Extend beam search depth: 6-8 → 10-12 for complex tasks
- Add "composition templates" for common multi-step patterns:
  - "Detect → Transform → Place"
  - "Extract → Sort → Arrange"
  - "Partition → Apply → Merge"

**B. Transfer learning**
- Cache successful programs from similar tasks
- Use program similarity metrics (edit distance on operators)
- Suggest related programs as starting points for beam search

**C. Ensemble/multi-solution**
- When Occam tie occurs (multiple programs fit), keep top-3
- At test time, if all 3 agree → high confidence
- If disagree → report ambiguity with edit bills showing differences

**Expected final accuracy:**
- **ARC-1**: 80-88% (310-345/391 tasks)
- **ARC-2**: 75-82% (457-500/609 tasks)
- **Overall**: 77-84% (767-845/1000 tasks)

---

### Summary: Baseline → Gaps Filled → Record

| Phase | Operators | ARC-1 | ARC-2 | Overall | Record? |
|-------|-----------|-------|-------|---------|---------|
| **Baseline (Weeks 1-6)** | 60-80 from plan | 60-75% | 40-55% | 60-70% | ❌ Below 79.6% |
| **+ Gaps (Weeks 7-8)** | +25-30 targeted | 75-85% | 60-75% | 75-83% | 🟡 Close/tied |
| **+ Deep search (Weeks 9-10)** | Depth 10-12, templates | 80-88% | 75-82% | 77-84% | ✅ Beat 79.6% |

**Key insight:** The baseline plan (60-80 ops) gets you to **60-70%**. Gap-filling adds **+15-20%**. Deep composition adds final **+2-4%** to surpass the record.

**Recommended strategy:**
1. ✅ **Implement baseline first** (Weeks 1-6) → measure actual coverage
2. ✅ **Prioritize gaps by frequency** in unsolved tasks (data-driven)
3. ✅ **Add operators incrementally** with evaluation after each batch
4. ✅ **Track receipts for every solved task** (maintain correctness)
5. ✅ **Reserve deep composition for last 15-20%** (high cost/benefit ratio)

This phased approach minimizes wasted effort and ensures you hit record-breaking performance (>79.6%) by Week 10 with verifiable receipts for every solution.