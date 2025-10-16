# Math Closure-Soundness Review: Blocker Fixes

**Reviewer**: Claude (math-closure-soundness-reviewer)
**Date**: 2025-10-16
**Implementation Report**: `/Users/ravishq/code/arc-agi-1/IMPLEMENTATION_REPORT_BLOCKER_FIXES.md`
**Context Packs**: `20251015_CANVAS_AWARE.md`, `20251015_CANVAS_GREEDY_VERIFY_FIX.md`

---

## Verdict

**PASS**

All mathematical contracts are preserved. The blocker fixes correctly eliminate parametricity violations by using only multiplicative factors (k_h, k_w) instead of hard-coded dimensions. The closure composition laws hold, and the implementation is mathematically sound.

---

## Blockers (must fix to preserve correctness)

**None.** No mathematical correctness violations found.

---

## High-Value Issues (should fix soon)

### 1. Test Suite Expects Removed SAME Strategy

**Location**: `tests/test_closures_minimal.py:1444-1459, 1572-1583`

**Issue**: Two tests expect the `SAME` strategy which was deliberately removed to fix parametricity violations:
- `test_canvas_size_same_shape()` - expects `strategy == "SAME"` with hard-coded `H, W`
- `test_canvas_size_same_shape_regression()` - expects `strategy == "SAME"` with hard-coded `H, W`

**Current behavior**: Both tests **FAIL** with:
```
AssertionError: Should detect SAME strategy
assert 'TILE_MULTIPLE' == 'SAME'
```

**Root cause**: Tests are outdated. The implementation correctly uses `TILE_MULTIPLE` with `k_h=k_w=1` for same-shape tasks (parametric), but tests expect the old `SAME` strategy (non-parametric).

**Evidence**:
```python
# Same-shape task (2x2 -> 2x2)
Params: {'strategy': 'TILE_MULTIPLE', 'k_h': 1, 'k_w': 1}
Has H in params: False  # ✓ No hard-coded dimensions
Has W in params: False  # ✓ No hard-coded dimensions
```

**Recommendation**: Update both tests to expect `TILE_MULTIPLE` with `k_h=1, k_w=1` and assert NO `H` or `W` in params.

**Proposed fix**:
```diff
# tests/test_closures_minimal.py
def test_canvas_size_same_shape():
-   assert canvas_closure.params["strategy"] == "SAME", "Should use SAME strategy"
-   assert canvas_closure.params["H"] == 2, "Should have H=2"
-   assert canvas_closure.params["W"] == 2, "Should have W=2"
+   assert canvas_closure.params["strategy"] == "TILE_MULTIPLE", "Should use TILE_MULTIPLE"
+   assert canvas_closure.params["k_h"] == 1 and canvas_closure.params["k_w"] == 1, "Should have k=1 for same-shape"
+   assert "H" not in canvas_closure.params and "W" not in canvas_closure.params, "Should NOT store absolute H,W"
```

---

## Closure Law Table

| name        | one-line law                                                          | shrinking? | idempotent? | unified params? | input-only mask? | train exact? | verdict |
|-------------|-----------------------------------------------------------------------|------------|-------------|-----------------|------------------|--------------|---------|
| CANVAS_SIZE | ∀(x,y)∈train: y.shape = (k_h×x.H, k_w×x.W) with constant k_h, k_w   | yes        | yes         | yes             | yes (identity)   | n/a (meta)   | PASS    |

**Detailed verification**:

### CANVAS_SIZE[strategy=TILE_MULTIPLE]

**Law**: `∀ (x, y) ∈ train: y.shape = (k_h × x.shape[0], k_w × x.shape[1])`
where k_h and k_w are constant across all train pairs.

**Shrinking**: ✓ YES
- `apply(U) = U` (identity function)
- `U ⊆ U` trivially holds
- No bits are added or cleared

**Idempotent**: ✓ YES
- `apply(apply(U)) = apply(U) = U`
- Trivial for identity function

**Unified params**: ✓ YES
- Params: `{strategy: "TILE_MULTIPLE", k_h: int, k_w: int}`
- Multipliers k_h, k_w are **constant across all train pairs**
- Verified in unifier (lines 1529-1537): `len(set(multipliers)) == 1`
- **NO hard-coded H, W dimensions stored**

**Input-only mask**: ✓ YES
- Identity function: no mask computation, no dependency on y
- Canvas computed per-input: `H_out = k_h × H_in, W_out = k_w × W_in`
- `y` used only in unifier for verification (not in apply)

**Train exact**: n/a (meta-closure)
- CANVAS_SIZE is metadata-only (`is_meta=True`)
- Provides canvas dimensions for engine, doesn't constrain cells
- Train exactness requires composition with constraining closures

**Data files used**:
- Unifier uses only `train` parameter (from `data/arc-agi_training_challenges.json`)
- No references to `evaluation_*` or `test_*` files

**Parametricity proof**:
1. Same-shape (3×3 → 3×3): `k_h=1, k_w=1` → Test input 4×4 → Canvas 4×4 ✓
2. Shape-change (2×2 → 6×6, 3×3 → 9×9): `k_h=3, k_w=3` → Test input 5×5 → Canvas 15×15 ✓
3. Non-integer (3×3 → 5×7): Returns `[]` (no closure) ✓

---

## Evidence

### 1. CANVAS_SIZE is identity (shrinking & idempotent)

**File**: `src/arc_solver/closures.py:1484-1490`

```python
def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
    """
    Identity/no-op - this closure only carries metadata for the engine.

    Returns U unchanged (trivially monotone, shrinking, idempotent).
    """
    return U
```

**Verification**:
```
$ python verify_identity.py
U == U_after: True          # ✓ Identity
Bits before: 10
Bits after: 10              # ✓ No bits changed (shrinking)
is_meta flag: True          # ✓ Meta-closure
```

### 2. TILE_MULTIPLE unifier enforces constant multipliers

**File**: `src/arc_solver/closures.py:1511-1537`

```python
# Check if all pairs have same non-None multiplier
multipliers = []
for x, y in train:
    H_in, W_in = x.shape
    H_out, W_out = y.shape
    if H_in == 0 or W_in == 0:
        multipliers.append(None)
        continue
    # Check if H_out is integer multiple of H_in
    if H_out % H_in == 0 and W_out % W_in == 0:
        k_h = H_out // H_in
        k_w = W_out // W_in
        multipliers.append((k_h, k_w))
    else:
        multipliers.append(None)

# Check if all pairs have same non-None multiplier
if all(m is not None for m in multipliers) and len(set(multipliers)) == 1:
    k_h, k_w = multipliers[0]
    candidate = CANVAS_SIZE_Closure(
        f"CANVAS_SIZE[strategy=TILE_MULTIPLE,k_h={k_h},k_w={k_w}]",
        {"strategy": "TILE_MULTIPLE", "k_h": k_h, "k_w": k_w}  # ✓ NO H, W
    )
```

**Verification**:
```
$ python verify_params.py
Test 1: Same-shape (2x2 -> 2x2)
  Strategy: TILE_MULTIPLE
  Params: {'strategy': 'TILE_MULTIPLE', 'k_h': 1, 'k_w': 1}
  Has H in params: False  # ✓ No hard-coded H
  Has W in params: False  # ✓ No hard-coded W

Test 2: Shape-changing (3x multiplier)
  Strategy: TILE_MULTIPLE
  Params: {'strategy': 'TILE_MULTIPLE', 'k_h': 3, 'k_w': 3}
  Has H in params: False  # ✓ No hard-coded H
  Has W in params: False  # ✓ No hard-coded W
```

### 3. _compute_canvas computes per-input canvas correctly

**File**: `src/arc_solver/closure_engine.py:351-372`

```python
def _compute_canvas(x_input: Grid, canvas_params: Dict) -> Dict:
    """
    Compute canvas size for a given input based on strategy.

    Args:
        x_input: Input grid
        canvas_params: CANVAS_SIZE closure params with "strategy": "TILE_MULTIPLE", "k_h", "k_w"

    Returns:
        Dict with "H" and "W" keys for output shape
    """
    strategy = canvas_params["strategy"]

    if strategy == "TILE_MULTIPLE":
        # Per-input canvas: H_out = k_h * H_in, W_out = k_w * W_in
        k_h = canvas_params["k_h"]
        k_w = canvas_params["k_w"]
        H_in, W_in = x_input.shape
        return {"H": k_h * H_in, "W": k_w * W_in}  # ✓ Computed from current input
    else:
        # Unknown strategy - fallback to input shape (backward compatible)
        return {"H": x_input.shape[0], "W": x_input.shape[1]}
```

**Verification**:
```
$ python verify_canvas.py
Canvas for 2x2 input: {'H': 6, 'W': 6}  # ✓ 3 × 2 = 6
Canvas for 3x3 input: {'H': 9, 'W': 9}  # ✓ 3 × 3 = 9
```

### 4. verify_closures_on_train handles per-input canvas

**File**: `src/arc_solver/closure_engine.py:260-296`

```python
def verify_closures_on_train(closures: List[Closure],
                              train: List[Tuple[Grid, Grid]]) -> bool:
    # CANVAS-AWARE: Extract canvas strategy from CANVAS_SIZE closure
    canvas_params = None
    for closure in closures:
        if closure.name.startswith("CANVAS_SIZE"):
            canvas_params = closure.params  # ✓ Params contain only k_h, k_w
            break

    for x, y in train:
        # Compute canvas per-input for TILE_MULTIPLE strategy
        canvas = _compute_canvas(x, canvas_params) if canvas_params else None  # ✓ Per-input

        U_final, _ = run_fixed_point(closures, x, canvas=canvas)

        # Check if U_final is fully determined
        if not U_final.is_fully_determined():
            return False

        # Check if U_final equals expected output
        y_pred = U_final.to_grid()
        if y_pred is None or not np.array_equal(y_pred, y):
            return False

    return True
```

### 5. verify_consistent_on_train allows TOP cells (meta-closures)

**File**: `src/arc_solver/closure_engine.py:299-348`

```python
def verify_consistent_on_train(closures: List[Closure],
                                train: List[Tuple[Grid, Grid]]) -> bool:
    """
    Verify closures don't create contradictions with train outputs.

    Unlike verify_closures_on_train, this does NOT require U to be fully determined.
    Only checks:
    1. No cell is empty (contradiction)
    2. For cells where x[r,c] == y[r,c], closure doesn't remove y's color
    """
    # Extract canvas params if present
    canvas_params = None
    for closure in closures:
        if closure.name.startswith("CANVAS_SIZE"):
            canvas_params = closure.params
            break

    for x, y in train:
        # Compute canvas per-input
        canvas = _compute_canvas(x, canvas_params) if canvas_params else None  # ✓ Per-input

        # Run fixed point
        U_final, _ = run_fixed_point(closures, x, canvas=canvas)

        # Check for contradictions (not full determination)
        for r in range(U_final.H):
            for c in range(U_final.W):
                # Skip cells outside y bounds (canvas may be larger than y)
                if r >= y.shape[0] or c >= y.shape[1]:
                    continue  # ✓ Cells outside y are allowed to be multi-valued

                allowed = U_final.get_set(r, c)
                if len(allowed) == 0:
                    return False  # Empty cell = contradiction

                # If x and y agree on this cell, y's color must still be allowed
                if r < x.shape[0] and c < x.shape[1]:
                    if int(x[r, c]) == int(y[r, c]):
                        if int(y[r, c]) not in allowed:
                            return False  # Removed correct color

    return True  # No contradictions (may have TOP cells)
```

**Mathematical property**: Allows multi-valued cells (TOP), only checks for:
1. No empty cells (∅) — would be a contradiction
2. Preserved correct colors where x[r,c] == y[r,c]

This is correct for incremental composition verification where meta-closures (like CANVAS_SIZE) may leave cells undetermined.

### 6. Synthetic mini-grids verification

**Test 1: Same-shape (backward compatibility)**
```json
Input:  [[1, 2], [3, 4]]
Output: [[5, 6], [7, 8]]
Closure: CANVAS_SIZE[strategy=TILE_MULTIPLE,k_h=1,k_w=1]
Canvas for 2x2 input: {"H": 2, "W": 2}
Canvas for 10x10 input: {"H": 10, "W": 10}
✓ Parametric: works on any same-shape input
```

**Test 2: Shape-changing (3× expansion)**
```json
Input:  [[1, 2], [3, 4]]
Output: 6×6 grid
Closure: CANVAS_SIZE[strategy=TILE_MULTIPLE,k_h=3,k_w=3]
Canvas for 2x2 input: {"H": 6, "W": 6}
Canvas for 5x5 input: {"H": 15, "W": 15}
✓ Parametric: generalizes to test inputs of any size
```

**Test 3: Non-integer multiple (rejection)**
```json
Input:  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
Output: 5×7 grid
Closures found: 0
✓ Correctly rejects non-integer shape changes
```

---

## Minimal Patch Suggestions (inline diffs)

### Update outdated tests expecting SAME strategy

```diff
# tests/test_closures_minimal.py:1444-1459
def test_canvas_size_same_shape():
-   """Test: CANVAS_SIZE with SAME strategy (backward compatibility)."""
+   """Test: CANVAS_SIZE with TILE_MULTIPLE (k=1) for same-shape tasks."""
    # All train pairs have same output shape
    x1 = np.array([[1, 2], [3, 4]])
    y1 = np.array([[5, 6], [7, 8]])
    x2 = np.array([[9, 0], [1, 2]])
    y2 = np.array([[3, 4], [5, 6]])
    train = [(x1, y1), (x2, y2)]

    closures = unify_CANVAS_SIZE(train)
    assert len(closures) == 1, "Should find CANVAS_SIZE closure"

    canvas_closure = closures[0]
-   assert canvas_closure.params["strategy"] == "SAME", "Should use SAME strategy"
-   assert canvas_closure.params["H"] == 2, "Should have H=2"
-   assert canvas_closure.params["W"] == 2, "Should have W=2"
+   assert canvas_closure.params["strategy"] == "TILE_MULTIPLE", "Should use TILE_MULTIPLE"
+   assert canvas_closure.params["k_h"] == 1 and canvas_closure.params["k_w"] == 1, \
+       "Should have k_h=k_w=1 for same-shape tasks"
+   assert "H" not in canvas_closure.params and "W" not in canvas_closure.params, \
+       "Should NOT store absolute H,W (parametricity)"

    # Verify identity behavior
    U = init_top(2, 2)
    U_after = canvas_closure.apply(U, x1)
    assert U == U_after, "CANVAS_SIZE should be identity/no-op"
```

```diff
# tests/test_closures_minimal.py:1572-1583
def test_canvas_size_same_shape_regression():
-   """CANVAS_SIZE should still work for same-shape tasks (baseline coverage)."""
+   """CANVAS_SIZE uses TILE_MULTIPLE with k=1 for same-shape tasks."""
    x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y1 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    train = [(x1, y1)]

    canvas_closures = unify_CANVAS_SIZE(train)
    assert len(canvas_closures) == 1, "Should get one CANVAS_SIZE closure"
    canvas = canvas_closures[0]
-   assert canvas.params["strategy"] == "SAME", "Should detect SAME strategy"
-   assert canvas.params["H"] == 3 and canvas.params["W"] == 3, "Should store H=3, W=3 for SAME"
+   assert canvas.params["strategy"] == "TILE_MULTIPLE", "Should use TILE_MULTIPLE"
+   assert canvas.params["k_h"] == 1 and canvas.params["k_w"] == 1, \
+       "Should have k_h=k_w=1 for same-shape"
+   assert "H" not in canvas.params and "W" not in canvas.params, \
+       "Should NOT store absolute H,W (parametricity)"
```

---

## Notes to Implementer

### 1. Registration order verified

**File**: `src/arc_solver/search.py:236-240`

```python
# CANVAS-AWARE: Extract canvas closure separately (metadata-only, exempt from back-off)
canvas_closure = None
canvas_candidates = unify_CANVAS_SIZE(train)
if canvas_candidates:
    canvas_closure = canvas_candidates[0]  # At most one CANVAS_SIZE closure
```

CANVAS_SIZE is extracted **before** all other unifiers and prepended to final list (line 287). This is correct: output shape must be known before other closures can be applied.

### 2. Meta-closure exemption from greedy back-off

**File**: `src/arc_solver/search.py:265-276`

```python
# Phase 1: Separate meta and non-meta; verify incrementally with consistency check
for c in candidates:
    if c.is_meta:
        kept_meta.append(c)
        continue  # Don't verify meta-only (they're metadata, may be identity)

    # Try adding non-meta closure
    trial = kept_meta + kept_nonmeta + [c]
    if verify_consistent_on_train(trial, train):  # Consistency, not full exactness
        kept_nonmeta.append(c)
    # else skip c (don't drop meta or prior nonmeta)
```

**Correct**: Meta-closures (CANVAS_SIZE) are never dropped during greedy back-off. They provide essential metadata for the fixed-point engine.

### 3. Parametricity discipline maintained

**Before (BLOCKER)**:
```python
# SAME strategy (violated parametricity)
{"strategy": "SAME", "H": 6, "W": 6}  # ✗ Hard-coded training dimensions
```

**After (FIXED)**:
```python
# TILE_MULTIPLE strategy (parametric)
{"strategy": "TILE_MULTIPLE", "k_h": 1, "k_w": 1}  # ✓ Multiplicative factors only
```

**Generalization proof**:
- Train: 3×3 → 3×3 (k=1)
- Test: 5×5 → ?
- Predicted: 5×5 (5 × 1 = 5) ✓

### 4. Coverage impact analysis

**From Implementation Report**:
- Coverage maintained: 10/1000 (1.0% baseline)
- CANVAS_SIZE in receipts: 731/1000 tasks (73.1%)
- Solved tasks: All 10 have CANVAS_SIZE + constraining closures

**Mathematical interpretation**: CANVAS_SIZE alone is insufficient (meta-only), but composes correctly with constraining closures. This is expected behavior.

### 5. Determinism verified

**From Implementation Report**:
- Timing differences only (OS scheduling)
- Closures and predictions byte-identical ✓
- No RNG, no I/O in closure logic ✓

---

## Summary

**PASS**: All mathematical contracts preserved. The blocker fixes successfully eliminate parametricity violations by:

1. ✓ Removing SAME and MAX_TRAIN strategies (hard-coded dimensions)
2. ✓ Using TILE_MULTIPLE with parametric multipliers (k_h, k_w) only
3. ✓ Computing canvas per-input: H_out = k_h × H_in
4. ✓ Preserving identity property (shrinking, idempotent)
5. ✓ Maintaining train exactness via composition with constraining closures
6. ✓ No test/eval data peeking (training-only induction)

**Only issue**: Two tests expect removed SAME strategy (high-value fix, not a blocker).

---

**Reviewed by**: Claude (math-closure-soundness-reviewer)
**Timestamp**: 2025-10-16T00:00:00Z
