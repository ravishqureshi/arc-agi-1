# Anti-Hardcode & Implementation Review

## Verdict

**PASS**

## Blockers (must fix to submit)

None found.

## High-Value Issues (should fix)

None found.

## Findings (evidence)

### 1. Parametricity: AXIS_PROJECTION (lines 511-759)

**Closure Implementation (lines 511-631):**
- **No hardcoded dimensions**: Uses `x_input.shape` at runtime (line 560)
  ```python
  H, W = x_input.shape  # Per-input dimensions
  ```
- **No hardcoded colors**: Uses `bg` parameter (line 525), inferred per-input if None (line 527)
- **mode="to_obstacle" logic (lines 576-612)**: Stops at first non-bg, non-obj-color pixel. No hardcoded stop positions.
  - Line 584-587: Left direction stops when `pixel_color != bg and pixel_color != obj.color`
  - Line 590-594: Right direction stops when `pixel_color != bg and pixel_color != obj.color`
  - Line 600-604: Up direction stops when `pixel_color != bg and pixel_color != obj.color`
  - Line 607-611: Down direction stops when `pixel_color != bg and pixel_color != obj.color`
- **scope="per_object" (lines 553-556)**: Enumerates all objects without hardcoded indices

**Unifier (lines 634-759):**
- **Parametric enumeration**: Enumerates axes ∈ {row, col}, scopes ∈ {largest, all, per_object}, modes ∈ {to_border, to_obstacle} (lines 656-658)
- **Unified params across train pairs**: All candidates must work on ALL train pairs (lines 668, 718: `preserves_y + compatible_to_y`)
- **Mask-local check (lines 669-707, 719-757)**:
  - Uses train outputs (y) for validation - **ALLOWED** (not test-peeking)
  - ONE-STEP apply (lines 676, 726) - no fixed-point iteration in unifier
  - Validates M = (x != y) mask-local property correctly
  - Lines 684-698: Outside M, U1 must equal {x} (no edits); on M, y must be in U1 and not empty

**✓ PASS**: All parameters unified across train pairs. No per-pair drift.

---

### 2. Parametricity: MOD_PATTERN (lines 1012-1246)

**Closure Implementation (lines 1012-1079):**
- **No hardcoded dimensions**: Operates on x_input.shape (line 1052)
  ```python
  H_in, W_in = x_input.shape  # Per-input dimensions
  ```
- **Parametric modulo arithmetic** (lines 1063-1064):
  ```python
  i = (r - ar) % p
  j = (c - ac) % q
  ```
  Where `p, q, ar, ac` are parameters (not constants)
- **class_map from params**: Line 1050 extracts class_map from params (not hardcoded)
- **Fallback for missing classes**: Lines 1067-1071 allow all colors if class not in map (safe default)

**Unifier (lines 1082-1246):**
- **Anchor generation (lines 1117-1142)**:
  - Line 1119: `H, W = x0.shape` - uses first train input dimensions
  - Lines 1139-1142: Uses `H//2, W//2` as anchor candidates
  - **Analysis**: This is a **search heuristic**, not hardcoding. The final anchor is unified across ALL train pairs (lines 1169-1181).
  - Rationale: Anchor candidates are derived from first input geometry to generate reasonable search space. Final selected anchor must work across all train pairs (verified by mask-local check).

- **Unified class_map (lines 1148-1186)**:
  - Line 1165: `class_colors[(i, j)].add(int(x[r, c]))` - builds from INPUT x only
  - Lines 1169-1181: Intersects class color sets across ALL train pairs
    ```python
    for pair_colors in class_colors_per_pair[1:]:
        for (i, j) in list(class_map.keys()):
            if (i, j) in pair_colors:
                class_map[(i, j)] &= pair_colors[(i, j)]
    ```
  - Line 1184: Rejects candidates with empty class colors (safe guard)

- **Mask-local check (lines 1203-1244)**:
  - Line 1210: `U_x = init_from_grid(x)` - input only
  - Line 1213: ONE-STEP apply (no fixed-point iteration)
  - Lines 1219-1241: Validates M = (x != y) mask-local property
  - Uses train y for validation (ALLOWED)

**✓ PASS**: Anchor candidates are search heuristics. Final params unified across all train pairs. No output-peeking in mask computation (class_map built from x only).

---

### 3. Determinism

**Tie-breaking:**
- Line 46 (infer_bg_from_border): `min(..., key=lambda c: (-counts[c], c))` - deterministic (lowest color on ties)
- Line 109 (KEEP_LARGEST): `max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))` - deterministic bbox tie-breaking
- Line 213 (OUTLINE_OBJECTS): Same deterministic tie-breaking
- Line 549 (AXIS_PROJECTION): Same deterministic tie-breaking
- Line 896 (SYMMETRY_COMPLETION): Same deterministic tie-breaking

**No RNG usage**: No `random` imports or `numpy.random` calls in closures.py

**Iteration order**:
- Line 1145: `for ar, ac in sorted(anchors):` - deterministic anchor enumeration
- Lines 1146-1147: `for p in range(2, 7): for q in range(2, 7):` - deterministic period enumeration
- Lines 656-658: `axes = ["row", "col"]; scopes = ["largest", "all", "per_object"]; modes = ["to_border", "to_obstacle"]` - fixed lists

**Dict iteration safety**:
- Lines 1172-1181: Dict iteration over class_map uses set intersection (`&=`), which is commutative - order doesn't affect result
- Line 1184: Check `any(len(colors) == 0 for colors in class_map.values())` - order-independent

**✓ PASS**: All operations deterministic. No nondeterministic sources.

---

### 4. Engine: Fixed-Point vs Beam

**Primary Runtime (closure_engine.py:205-253)**:
```python
def run_fixed_point(closures: List[Closure], x_input: Grid, ...) -> Tuple[SetValuedGrid, Dict]:
    """Run fixed-point iteration until convergence."""
    # Tarski fixed-point iteration
    U = init_top(H, W)
    for iteration in range(max_iters):
        U_prev = U.copy()
        for closure in closures:
            U = closure.apply(U, x_input)  # Monotone shrinking
        if U == U_prev:
            return U, stats  # Convergence
```

**Solver Entry Point (search.py:298-476)**:
```python
def solve_with_closures(inst: ARCInstance):
    """Solve ARC instance with fixed-point closure engine."""
    closures = autobuild_closures(inst.train)  # Unifiers
    U_final, _ = run_fixed_point(closures, x, canvas=canvas)  # Fixed-point runtime
```

**Unifiers only validate, don't run FP**:
- Line 676 (AXIS_PROJECTION): `U1 = candidate_none.apply(U_x, x)` - ONE-STEP apply
- Line 1213 (MOD_PATTERN): `U1 = candidate.apply(U_x, x)` - ONE-STEP apply
- No `run_fixed_point()` calls in unifiers

**Beam Search**: Exists at search.py:53 but is NOT the primary Master Operator runtime.

**✓ PASS**: Fixed-point engine is the primary runtime for closure compositions.

---

### 5. Data-Leak Check

**No evaluation/test data reads**:
```bash
$ grep -r "arc-agi_evaluation\|arc-agi_test" src/arc_solver/ tests/
# No matches
```

**Unifiers use TRAINING data only**:
- Line 1152 (MOD_PATTERN): `for x, y in train:` - iterates training pairs
- Line 671 (AXIS_PROJECTION): `for x, y in train:` - iterates training pairs
- Mask-local checks use train outputs (y) for **validation**, not test outputs

**Closures are input-only**:
- AXIS_PROJECTION.apply (line 520): Only reads `x_input`, never accesses test outputs
- MOD_PATTERN.apply (line 1027): Only reads `x_input`, never accesses test outputs
- class_map built from train INPUTS (line 1165): `class_colors[(i, j)].add(int(x[r, c]))`

**✓ PASS**: No test/evaluation data leakage. Unifiers validate on train y (ALLOWED), closures operate on input only.

---

### 6. Mask-Local Correctness

**Implementation Pattern (used in AXIS_PROJECTION and MOD_PATTERN)**:
```python
# Build U_x: init from x (lines 673, 1210)
U_x = init_from_grid(x)

# Apply ONE step only (lines 676, 1213)
U1 = candidate.apply(U_x, x)

# Check mask-local: M = (x != y) on overlap region
for r in range(H_check):
    for c in range(W_check):
        x_color = int(x[r, c])
        y_color = int(y[r, c])

        if x_color == y_color:
            # Outside M: U1 must equal {x} (no edits)
            if U1.get_set(r, c) != {x_color}:
                mask_local_ok = False
        else:
            # On M: y must be in U1 and not empty
            if y_color not in U1.get_set(r, c) or len(U1.get_set(r, c)) == 0:
                mask_local_ok = False
```

**Correctness**:
- ONE-STEP apply prevents FP-iteration side effects in unifier
- M = (x != y) correctly identifies edit region
- Validates y ∈ U1[M] without adding bits (only intersect operations)
- Outside M, closure must not edit (U1 = {x})

**✓ PASS**: Mask-local implementation correct. Closures only clear bits (monotone shrinking).

---

## Minimal Patch Suggestions (inline diffs)

None required. All fixes pass anti-hardcode validation.

---

## Recheck Guidance

After any future changes to these closures:

1. **Unified params**: Verify all parameters work across ALL train pairs (not per-pair tuning)
   - Check: `grep -n "for x, y in train" src/arc_solver/closures.py` should show unifiers validating on all pairs
   - Check: No per-pair parameter storage (no `params_per_pair = []`)

2. **No hardcoded dimensions**: Use `x_input.shape` at runtime, not constants from first train input
   - Check: `grep -n "shape\[0\].*=" src/arc_solver/closures.py` should only show assignments from runtime input
   - Avoid: Literals like `if H == 30:` or hardcoded bounds

3. **Deterministic tie-breaking**: Ensure all `max()`, `min()`, `Counter` operations have deterministic tie-breaks
   - Check: `grep -n "max(.*key=lambda" src/arc_solver/closures.py` should show composite keys like `(o.size, -o.bbox[0], -o.bbox[1])`
   - Check: `grep -n "min(.*key=lambda" src/arc_solver/closures.py` should show composite keys like `(-counts[c], c)`

4. **Fixed-point runtime**: Verify `run_fixed_point()` is the primary execution path (not beam)
   - Check: `grep -n "run_fixed_point" src/arc_solver/search.py` should show it in `solve_with_closures()`
   - Check: No `run_fixed_point()` calls in unifiers (ONE-STEP apply only)

5. **No test-peeking**: Unifiers may use train outputs for validation, but NEVER test outputs
   - Check: `grep -r "arc-agi_evaluation\|arc-agi_test" src/arc_solver/` should return nothing
   - Check: Closures.apply() only reads `x_input`, never `y` or test data

6. **Mask-local**: ONE-STEP apply in unifier, validate M = (x != y) without adding bits
   - Check: Lines 676, 1213 call `candidate.apply(U_x, x)` exactly once per pair
   - Check: Mask-local loop validates outside-M cells have U1 = {x} (no edits)

---

## Summary

All three fixes (AXIS_PROJECTION mode="to_obstacle", AXIS_PROJECTION scope="per_object", MOD_PATTERN mask-local) pass anti-hardcode validation:

- ✅ No hardcoded training dimensions (use x_input.shape at runtime)
- ✅ No per-pair parameter drift (all params unified across train pairs)
- ✅ Deterministic (sorted anchors, deterministic tie-breaking, no RNG)
- ✅ Fixed-point engine as primary runtime (not beam)
- ✅ No test data leakage (unifiers use train y for validation, closures operate on input only)
- ✅ Mask-local correctness (one-step apply, M = (x != y) validation)
- ✅ class_map built from INPUT x only (no output-peeking in mask computation)

**PASS - Ready for submission.**
