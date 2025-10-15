# Math Closure-Soundness Review

**Closure**: OUTLINE_OBJECTS
**Date**: 2025-10-15
**Reviewer**: Claude Code (Mathematical Soundness Analysis)
**Scope**: M1.3 OUTLINE_OBJECTS implementation verification

---

## Verdict

**VERIFIED**

All mathematical properties required for fixed-point convergence are satisfied. Implementation is sound and production-ready.

---

## Blockers (must fix to preserve correctness)

**NONE**

---

## High-Value Issues (should fix soon)

**NONE**

---

## Closure Law Table

| name             | one-line law                                                                                              | shrinking? | idempotent? | unified params? | input-only mask? | train exact? | verdict  |
|------------------|-----------------------------------------------------------------------------------------------------------|------------|-------------|-----------------|------------------|--------------|----------|
| OUTLINE_OBJECTS  | Object pixels adjacent (4-connected) to background retain color; interior pixels forced to {bg}          | yes        | yes         | yes             | yes              | yes          | PASS     |

---

## Evidence

### Property 1: Monotonicity (F(U) ⊆ F(V) when U ⊆ V)

**Status**: VERIFIED

**Code**: `/Users/ravishq/code/arc-agi-1/src/arc_solver/closures.py:129-205`

**Analysis**:
- All masks derived exclusively from `x_input` (lines 138, 164-186)
- Component extraction: `components(x_input, bg=bg)` - deterministic from input (line 138)
- Outline computation: 4-neighbor checks on `x_input` only (lines 172-185)
- No dependency on current set-valued grid `U` for mask computation
- Only uses `intersect()` operations (lines 197, 200, 203)

**Mathematical proof**:
```
Given: U ⊆ V (i.e., ∀(r,c), U[r,c] ⊆ V[r,c])
For any (r,c):
  F(U)[r,c] = U[r,c] ∩ mask[r,c]
  F(V)[r,c] = V[r,c] ∩ mask[r,c]
  where mask[r,c] is computed from x_input only

Since U[r,c] ⊆ V[r,c] and mask is constant:
  F(U)[r,c] = U[r,c] ∩ mask ⊆ V[r,c] ∩ mask = F(V)[r,c]

Therefore: F(U) ⊆ F(V) ✓
```

**Test evidence**:
```
Test 5: Monotonicity
  U ⊆ V: True
  F(U) ⊆ F(V): True
```

**Property test**: All monotonicity assumptions verified through shrinking test (line 495 in test_closures_minimal.py)

---

### Property 2: Shrinking (F(U) ⊆ U)

**Status**: VERIFIED

**Code**: `/Users/ravishq/code/arc-agi-1/src/arc_solver/closures.py:188-203`

**Line-by-line verification**:
- Line 188: `U_new = U.copy()` - starts with exact copy
- Lines 197, 200, 203: `U_new.intersect(r, c, mask)` - only shrinking operations
- No lines add colors - only intersection (bitwise AND)

**Critical observation**: All operations are of form `U_new.data[r, c] &= mask` (closure_engine.py:62)
- Bitwise AND can only clear bits, never set new ones
- Therefore: ∀(r,c), F(U)[r,c] ⊆ U[r,c]

**Test evidence**:
```
Test 1: Shrinking property
  Bits before: 250 (10 colors × 25 cells)
  Bits after: 25 (90% reduction)
  Shrinking: True
```

**Empirical verification**: 5×5 grid with 3×3 blob
- Initial: 250 bits (all colors allowed in all cells)
- After closure: 25 bits (singletons only)
- Reduction: 90% (strong shrinking)

**Property test**: `test_shrinking_outline()` passes (line 495-523)

---

### Property 3: Idempotence (F(F(U)) = F(U))

**Status**: VERIFIED

**Code**: `/Users/ravishq/code/arc-agi-1/src/arc_solver/closures.py:159-186`

**Determinism sources**:
1. **Component selection** (line 152):
   ```python
   selected_objs = [max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))]
   ```
   - Deterministic tie-breaking: size → uppermost → leftmost
   - Same input → same component selection

2. **Outline computation** (lines 172-185):
   ```python
   for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
       # Fixed iteration order: down, up, right, left
   ```
   - 4-neighbor check in fixed order
   - Deterministic pixel classification

3. **Mask application** (lines 193-203):
   - Masks are constant for given `x_input`
   - Second application recomputes identical masks

**Mathematical proof**:
```
F(U) = U ∩ M(x_input)     where M is mask function

F(F(U)) = F(U ∩ M)
        = (U ∩ M) ∩ M
        = U ∩ (M ∩ M)     [associativity]
        = U ∩ M           [idempotence of set intersection]
        = F(U) ✓
```

**Test evidence**:
```
Test 2: Idempotence
  F(U) == F(F(U)): True

Test 3: Convergence
  Iterations: 2
  Multi-valued cells: 0
```

**Property test**: `test_idempotence_outline()` passes (line 526-550)

---

### Property 4: Unified Parameters (one param set for ALL train pairs)

**Status**: VERIFIED

**Code**: `/Users/ravishq/code/arc-agi-1/src/arc_solver/closures.py:208-244`

**Unifier algorithm**:
```python
def unify_OUTLINE_OBJECTS(train):
    valid_closures = []
    mode = "outer"              # Fixed for M1
    scopes = ["largest", "all"] # Two options
    bgs = range(10)             # All colors

    for scope in scopes:
        for bg in bgs:
            candidate = OUTLINE_OBJECTS_Closure(
                f"OUTLINE_OBJECTS[mode={mode},scope={scope},bg={bg}]",
                {"mode": mode, "scope": scope, "bg": bg}
            )
            if verify_closures_on_train([candidate], train):
                valid_closures.append(candidate)

    return valid_closures
```

**Properties verified**:
1. **Exhaustive search**: 2 scopes × 10 backgrounds = 20 candidates
2. **Single parameterization**: Each candidate has ONE fixed (mode, scope, bg) tuple
3. **All-pairs verification**: `verify_closures_on_train` checks ALL train pairs
4. **Strict rejection**: Only returns params that work on ALL pairs

**Test evidence**:
```
Test 8: Unified parameters
  Found 2 valid closures
  OUTLINE_OBJECTS[mode=outer,scope=largest,bg=0]
  OUTLINE_OBJECTS[mode=outer,scope=all,bg=0]
  Pair 1: exact=True, iters=2
  Pair 2: exact=True, iters=2

Test 9: Unifier rejects bad params
  Mismatched pairs rejected: True
```

**Property test**: `test_train_exactness_outline_multiple()` passes (line 585-630)

---

### Property 5: Train Exactness (all pairs reach singleton(y))

**Status**: VERIFIED

**Code**:
- Unifier: `/Users/ravishq/code/arc-agi-1/src/arc_solver/closures.py:241`
- Verifier: `/Users/ravishq/code/arc-agi-1/src/arc_solver/closure_engine.py:252-276`

**Verification logic** (closure_engine.py):
```python
def verify_closures_on_train(closures, train):
    for x, y in train:  # ALL pairs (no early exit)
        U_final, _ = run_fixed_point(closures, x)

        # Check 1: All cells are singletons
        if not U_final.is_fully_determined():
            return False

        # Check 2: Exact equality with expected output
        y_pred = U_final.to_grid()
        if y_pred is None or not np.array_equal(y_pred, y):
            return False

    return True
```

**Exactness requirements**:
1. **Fully determined**: All cells must be singletons (checked via bit-trick: `mask & (mask-1) == 0`)
2. **Exact match**: `np.array_equal(y_pred, y)` - no approximation
3. **All pairs**: Loop over entire train set - no partial success

**Test evidence**:
```
Test 4: Expected output (outline)
Input:
[[0, 0, 0, 0, 0],
 [0, 3, 3, 3, 0],
 [0, 3, 3, 3, 0],
 [0, 3, 3, 3, 0],
 [0, 0, 0, 0, 0]]

Output:
[[0, 0, 0, 0, 0],
 [0, 3, 3, 3, 0],
 [0, 3, 0, 3, 0],  <- Center (2,2) cleared to 0
 [0, 3, 3, 3, 0],
 [0, 0, 0, 0, 0]]

Center pixel (2,2) = 0 (expected 0) ✓
Edge pixel (1,1) = 3 (expected 3) ✓
```

**Property test**: `test_train_exactness_outline_single()` passes (line 553-582)

---

### Property 6: Convergence (true fixed-point detection)

**Status**: VERIFIED

**Code**: `/Users/ravishq/code/arc-agi-1/src/arc_solver/closure_engine.py:204-245`

**Fixed-point iteration**:
```python
def run_fixed_point(closures, x_input, max_iters=100):
    H, W = x_input.shape
    U = init_top(H, W)  # U₀ = ⊤ (all colors allowed)

    for iteration in range(max_iters):
        U_prev = U.copy()

        # Apply all closures in sequence
        for closure in closures:
            U = closure.apply(U, x_input)

        # Check convergence (EXACT equality)
        if U == U_prev:
            return U, {"iters": iteration + 1, ...}

    return U, {"iters": max_iters, ...}
```

**Convergence guarantees**:
1. **Tarski theorem**: Finite lattice + monotone shrinking → finite convergence
2. **True equality**: `U == U_prev` checks bitwise equality (np.array_equal on data)
3. **Not heuristic**: No "small change" approximation

**Convergence speed for OUTLINE_OBJECTS**:
- **Iteration 1**: Apply closure, narrow from ⊤ to constrained set
- **Iteration 2**: Verify no change (U == U_prev) → converged
- **Guaranteed**: ≤2 iterations (deterministic masks from x_input)

**Test evidence**:
```
Test 3: Convergence
  Iterations: 2
  Multi-valued cells: 0
  Fully determined: True
```

**Property test**: `test_convergence_outline()` passes (line 633-648)

---

### Property 7: Determinism (repeatable component/outline selection)

**Status**: VERIFIED

**Code**: `/Users/ravishq/code/arc-agi-1/src/arc_solver/closures.py:152, 174`

**Determinism sources**:

1. **Component tie-breaking** (line 152):
   ```python
   max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))
   ```
   - Primary: largest size
   - Tie-break 1: uppermost (smallest bbox[0], hence `-bbox[0]` for max)
   - Tie-break 2: leftmost (smallest bbox[1], hence `-bbox[1]` for max)
   - Result: Deterministic selection for equal-size components

2. **4-neighbor iteration** (line 174):
   ```python
   for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
       # Fixed order: down, up, right, left
   ```
   - No random or hash-order iteration
   - Same input → same outline pixels

3. **Component extraction** (utils.py:53-86):
   - Row-major traversal: `for r in range(H): for c in range(W)`
   - BFS with deque (FIFO) - deterministic
   - Same grid → same component ordering

**Test evidence**:
```
Test 7: Deterministic tie-breaking
  U1 == U2: True
  y1 == y2: True
  Output:
  [[1 1 0 0 0]  <- Leftmost component kept
   [0 0 0 0 0]]
```

**Property test**: All 19/19 tests pass with identical results on repeated runs

---

## Synthetic Mini-Grid Tests

### Grid 1: Solid 3×3 blob with interior

**Input**:
```json
[[0, 0, 0, 0, 0],
 [0, 3, 3, 3, 0],
 [0, 3, 3, 3, 0],
 [0, 3, 3, 3, 0],
 [0, 0, 0, 0, 0]]
```

**Expected (outline only)**:
```json
[[0, 0, 0, 0, 0],
 [0, 3, 3, 3, 0],
 [0, 3, 0, 3, 0],
 [0, 3, 3, 3, 0],
 [0, 0, 0, 0, 0]]
```

**Result**: PASS
- Iterations: 2
- Fully determined: True
- Exact match: True
- Center (2,2) cleared to 0 ✓

---

### Grid 2: Single pixel (edge case)

**Input**: `[[1]]`

**Expected**: `[[1]]` (out-of-bounds treated as background → pixel is outline)

**Result**: PASS
- Pixel (0,0) set: {1}
- Correctly identified as outline ✓

---

### Grid 3: Equal-size components (tie-breaking)

**Input**:
```json
[[1, 1, 0, 2, 2],
 [0, 0, 0, 0, 0]]
```

**Expected (scope="largest")**: `[[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]]`

**Result**: PASS
- Leftmost component (color 1) selected ✓
- Deterministic on repeated runs ✓

---

### Grid 4: Ring object (thin components)

**Input**:
```json
[[0, 1, 1, 0],
 [0, 1, 0, 1],
 [0, 1, 1, 0]]
```

**Expected (scope="all")**: Same as input (all pixels are outline)

**Result**: PASS
- Outline is entire object ✓
- Exact match ✓

---

## Minimal Patch Suggestions (inline diffs)

**NONE REQUIRED**

The implementation is mathematically sound and requires no patches.

---

## Notes to Implementer

### Edge Cases Handled Correctly

1. **Out-of-bounds neighbors** (lines 177-179):
   ```python
   if not (0 <= nr < H and 0 <= nc < W):
       is_outline = True
       break
   ```
   - Treated as background → triggers outline classification
   - **Correct**: Edge pixels are adjacent to implicit "background" beyond grid

2. **Empty grid** (lines 140-147):
   - No components found → entire grid becomes background
   - **Correct**: No objects to outline

3. **Single-pixel component** (lines 177-179):
   - Always marked as outline (no interior possible)
   - **Correct**: All neighbors are background or out-of-bounds

4. **Equal-size components** (line 152):
   - Deterministic tie-break by (size, -bbox[0], -bbox[1])
   - **Correct**: Ensures repeatability

---

### Registration Order

**File**: `/Users/ravishq/code/arc-agi-1/src/arc_solver/search.py`

OUTLINE_OBJECTS must be registered after KEEP_LARGEST in `autobuild_closures()`:

```python
def autobuild_closures(train):
    closures = []
    # B1: KEEP_LARGEST_COMPONENT
    closures += unify_KEEP_LARGEST(train)
    # B2: OUTLINE_OBJECTS (after KEEP_LARGEST)
    closures += unify_OUTLINE_OBJECTS(train)
    return closures
```

**Rationale**:
- Order doesn't affect fixed-point (closures commute mathematically)
- Order affects discovery efficiency (try cheap closures first)
- KEEP_LARGEST is cheaper (no outline computation)

---

### Law Statement (Formal)

**OUTLINE_OBJECTS Law (One-Liner)**:
Object pixels adjacent (4-connected) to background retain their color; all other object pixels are forced to {bg}.

**Formal Definition**:
```
Given:
  - x_input: Grid (H×W)
  - params: {mode: "outer", scope: "largest"|"all", bg: int}

Let:
  - O = components(x_input, bg) filtered by scope
  - P_outline = {(r,c) ∈ O | ∃(dr,dc) ∈ {(±1,0), (0,±1)}: x[r+dr,c+dc] = bg ∨ out_of_bounds}
  - P_interior = O \ P_outline

Apply:
  ∀(r,c) ∈ P_outline:  U'[r,c] = U[r,c] ∩ {x[r,c]}  (keep object color)
  ∀(r,c) ∈ P_interior: U'[r,c] = U[r,c] ∩ {bg}      (force background)
  ∀(r,c) ∉ O:          U'[r,c] = U[r,c] ∩ {bg}      (background stays)

Result: U' ⊆ U (shrinking, monotone, idempotent)
```

---

### Test Suite Summary

**File**: `/Users/ravishq/code/arc-agi-1/tests/test_closures_minimal.py:492-672`

All 6 OUTLINE_OBJECTS property tests pass:
- ✓ `test_shrinking_outline()` - Shrinking property verified
- ✓ `test_idempotence_outline()` - Idempotence verified
- ✓ `test_train_exactness_outline_single()` - Single pair exactness
- ✓ `test_train_exactness_outline_multiple()` - Multi-pair unification
- ✓ `test_convergence_outline()` - ≤2 iteration convergence
- ✓ `test_outline_scope_all()` - Scope="all" correctness

**Total test results**: 19/19 tests pass (13 KEEP_LARGEST + 6 OUTLINE_OBJECTS)

**Run command**:
```bash
python /Users/ravishq/code/arc-agi-1/tests/test_closures_minimal.py
```

---

## Final Assessment

### Mathematical Soundness: VERIFIED

The OUTLINE_OBJECTS closure satisfies all seven required properties:

1. ✓ **Monotonicity**: Masks from x_input only, intersection preserves order
2. ✓ **Shrinking**: Only bitwise AND operations (U' ⊆ U), 90% bit reduction observed
3. ✓ **Idempotence**: Deterministic masks from x_input (F∘F = F)
4. ✓ **Unified Parameters**: Single param set across all train pairs (2 scopes × 10 bgs)
5. ✓ **Train Exactness**: All pairs reach singleton(y), exact equality
6. ✓ **Convergence**: Fixed-point detected in ≤2 iterations
7. ✓ **Determinism**: Repeatable tie-breaking and neighbor iteration

### Tarski Fixed-Point Compliance

The implementation complies with Tarski's fixed-point theorem:
- **Domain**: Finite complete lattice (P(C)^{H×W}, ⊆)
- **Function**: Monotone (F(U) ⊆ F(V) when U ⊆ V)
- **Property**: Shrinking (F(U) ⊆ U)
- **Guarantee**: Least fixed-point exists and computable in finite steps

### Production Readiness: READY

- **Blockers**: 0
- **High-value issues**: 0
- **Edge cases**: All handled correctly
- **Test coverage**: Comprehensive (6 property tests)
- **Alignment**: Follows context pack spec exactly
- **Pattern**: Matches KEEP_LARGEST_COMPONENT gold standard

---

## References

- **Context Pack**: `/Users/ravishq/code/arc-agi-1/context_packs/20251015_M1.3_outline_objects.md`
- **Master Operator**: `/Users/ravishq/code/arc-agi-1/docs/core/arc_agi_master_operator.md`
- **Implementation**: `/Users/ravishq/code/arc-agi-1/src/arc_solver/closures.py:119-244`
- **Engine**: `/Users/ravishq/code/arc-agi-1/src/arc_solver/closure_engine.py:1-277`
- **Tests**: `/Users/ravishq/code/arc-agi-1/tests/test_closures_minimal.py:492-672`

---

**Signature**: Mathematical soundness review complete. No violations found. Implementation verified.
