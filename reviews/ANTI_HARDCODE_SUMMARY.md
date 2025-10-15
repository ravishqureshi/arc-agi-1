# Anti-Hardcode Audit: Executive Summary

**Date**: 2025-10-15
**Auditor**: Claude (Anti-Hardcode & Implementation Auditor)
**Scope**: Fixed-Point Closure Engine (B0) + KEEP_LARGEST_COMPONENT (B1)

---

## TL;DR

**Mathematical Foundation**: PASS (monotone shrinking, true convergence, Tarski guarantees hold)
**Parametricity**: FAIL (critical bg=0 hardcodes throughout)
**Train Verification**: PASS (all pairs checked, no shortcuts)

**Verdict**: NEEDS_WORK - Fix 3 critical parametricity violations (~35 minutes), then READY FOR SUBMISSION

---

## Files Audited

1. `/Users/ravishq/code/arc-agi-1/src/arc_solver/closure_engine.py` (245 lines)
   - SetValuedGrid (lattice with 10-bit masks)
   - run_fixed_point() (lfp iterator)
   - verify_closures_on_train() (ALL pairs check)

2. `/Users/ravishq/code/arc-agi-1/src/arc_solver/closures.py` (107 lines)
   - KEEP_LARGEST_COMPONENT_Closure
   - unify_KEEP_LARGEST() unifier

3. `/Users/ravishq/code/arc-agi-1/src/arc_solver/search.py` (351 lines)
   - autobuild_closures()
   - solve_with_closures()

---

## What's CORRECT (Mathematical Correctness)

### 1. Lattice Structure & Monotone Shrinking
- `SetValuedGrid.data` uses 10-bit masks (bits 0-9 = colors 0-9)
- `intersect(r, c, mask)` uses bitwise AND (`&=`) - true lattice meet
- Result: Closures can ONLY shrink sets, never grow them

### 2. True Convergence (Not Just Iteration Limit)
```python
# closure_engine.py:224
if U == U_prev:
    return U, stats
```
- Checks equality of grids, not just "ran N times"
- Guarantees least fixed point per Tarski theorem

### 3. Train Verification on ALL Pairs
```python
# closure_engine.py:255
for x, y in train:
    U_final, _ = run_fixed_point(closures, x)
    if not equal(U_final.to_grid(), y):
        return False
```
- Loops through EVERY train pair
- No shortcuts, no "just check first pair" toys

### 4. Closure.apply() is Parametric (Structure)
- `KEEP_LARGEST_COMPONENT_Closure.apply()` reads `self.params["bg"]`
- Does NOT use global hardcode inside apply()
- Structure is correct (params flow from unifier → closure → apply)

### 5. Fixed-Point Stats Logged
```python
stats = {"iters": iteration+1, "cells_multi": U.count_multi_valued_cells()}
```
- Provides receipts for convergence iteration and under-constrained cells

---

## What's BROKEN (Parametricity Violations)

### CRITICAL #1: Unifier Defaults bg to 0
**File**: `closures.py:88`
```python
def unify_KEEP_LARGEST(train: List[Tuple[Grid, Grid]], bg: int = 0)
```

**Problem**: Default parameter is a hardcode. If caller omits bg, silently uses 0.

**Impact**: Any ARC task with bg != 0 (e.g., bg=7) will fail.

**Fix**: Remove default; try all bg values {0-9}:
```python
def unify_KEEP_LARGEST(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    valid_closures = []
    for bg in range(10):
        candidate = KEEP_LARGEST_COMPONENT_Closure(f"KEEP_LARGEST[bg={bg}]", {"bg": bg})
        if verify_closures_on_train([candidate], train):
            valid_closures.append(candidate)
    return valid_closures
```

---

### CRITICAL #2: Caller Passes Hardcoded bg=0
**File**: `search.py:222`
```python
closures += unify_KEEP_LARGEST(train, bg=0)
```

**Problem**: Even if unifier accepted multiple bg values, this call forces bg=0.

**Impact**: Blocks parameter discovery.

**Fix**: Remove argument:
```python
closures += unify_KEEP_LARGEST(train)
```

---

### CRITICAL #3: Unsafe Default in Closure.apply()
**File**: `closures.py:52`
```python
bg = self.params.get("bg", 0)
```

**Problem**: `.get()` with default 0 masks unification bugs. If params missing "bg", silently uses 0.

**Impact**: Hides unifier failures; defeats fail-loudly principle.

**Fix**: Remove default (let KeyError propagate):
```python
bg = self.params["bg"]  # Fail loudly if missing
```

---

## Lesser Issues (Non-Blocking but Should Fix)

### MEDIUM #1: Shape Assumption in run_fixed_point
**File**: `closure_engine.py:213`
```python
H, W = x_input.shape
U = init_top(H, W)
```

**Problem**: Assumes output shape = input shape. Not true for crop/pad/tile tasks.

**Current Impact**: B1 (KEEP_LARGEST) preserves shape, so this works NOW. But will break for future closures (crop, tile expansion).

**Recommendation**: Add ARCHITECTURE_DEBT note; fix before B2+ closures that change shape.

---

### MEDIUM #2: Magic Number max_iters=100
**File**: `closure_engine.py:201`
```python
def run_fixed_point(..., max_iters: int = 100)
```

**Problem**: Arbitrary constant without rationale.

**Fix**: Make it a documented module constant:
```python
DEFAULT_MAX_ITERS = 100  # Tarski guarantees finite convergence; this is safety net
```

---

### MEDIUM #3: Empty Cell Handling Hardcodes bg=0
**File**: `closure_engine.py:105`
```python
if mask == 0:
    result[r, c] = 0  # Empty → background
```

**Problem**: Assumes bg=0 for under-constrained cells.

**Current Impact**: Rarely hit (only when lfp fails to converge to singletons).

**Fix**: Pass bg as parameter to `to_grid_deterministic()`:
```python
def to_grid_deterministic(self, fallback='lowest', bg=0):
    ...
    if mask == 0:
        result[r, c] = bg
```

---

## What About Beam Search bg=0 Hardcodes?

**File**: `search.py:32,34,37-39`

Multiple calls like `induce_KEEP_LARGEST(train, bg=0)`, `induce_HOLE_FILL(train, bg=0)`.

**Status**: Out of scope for this audit (legacy beam code). Same parametricity principle applies, but:
- Beam search is being phased out (master operator is replacing it)
- Closures are the primary submission path
- Fix beam code in separate refactor

**Recommendation**: Document as tech debt; focus on closures first.

---

## Test Coverage & Verification

**Question**: Are there unit tests for B0/B1?

**Finding**: No test files found in `/Users/ravishq/code/arc-agi-1/src/` or repo root.

**Recommendation**: After applying patch, create minimal smoke test:
```python
# tests/test_closure_b1.py
def test_keep_largest_multiple_bg():
    # Task with bg=7
    x = np.array([[7,7,7],[7,3,3],[7,3,3]])
    y = np.array([[7,7,7],[7,3,3],[7,3,3]])  # Largest component is 3s
    train = [(x, y)]

    closures = unify_KEEP_LARGEST(train)
    assert len(closures) > 0
    assert any(c.params["bg"] == 7 for c in closures)
```

---

## Determinism Check

**Question**: Are predictions stable across runs?

**Status**: No evidence of RNG usage in closure_engine.py or closures.py. All operations are deterministic:
- Bitwise AND for intersect
- Equality check for convergence
- components() (from utils) uses numpy (deterministic if input is same)

**Verification Needed**: Run solver twice on same task, confirm outputs identical.

**Command** (after patch):
```bash
python -m arc_solver.main --task data/training/0a1d4ef8.json --output pred1.json
python -m arc_solver.main --task data/training/0a1d4ef8.json --output pred2.json
diff pred1.json pred2.json  # Should be empty
```

---

## Submission Readiness Checklist

- [ ] **C1 Fixed**: unify_KEEP_LARGEST tries all bg values (closures.py:88)
- [ ] **C2 Fixed**: autobuild_closures removes bg=0 argument (search.py:222)
- [ ] **C3 Fixed**: Closure.apply() uses params["bg"] without default (closures.py:52)
- [ ] **M2 Fixed**: max_iters is named constant (closure_engine.py:201)
- [ ] **M3 Fixed**: to_grid_deterministic accepts bg param (closure_engine.py:105)
- [ ] **Arch Debt Noted**: Shape assumption documented in run_fixed_point (closure_engine.py:213)
- [ ] **Determinism Verified**: Two runs produce identical outputs
- [ ] **Receipts Working**: receipts.jsonl logs closures + fp stats
- [ ] **Train Exactness**: All train pairs reach singleton(y) before solving test

---

## Next Steps (Immediate)

1. **Apply patch** (`reviews/ANTI_HARDCODE_PATCH.diff`)
   - Fixes C1, C2, C3, M2, M3
   - Adds ARCHITECTURE_DEBT comment
   - ~35 minutes implementation time

2. **Run smoke test on B1 tasks**
   - Find ARC tasks where KEEP_LARGEST is the correct closure
   - Verify solver produces correct outputs
   - Check receipts.jsonl has correct closure params

3. **Run determinism check**
   - Solve same task twice
   - Confirm predictions identical
   - Confirm receipts.jsonl has identical closure_set_sha

4. **Update CONTEXT_INDEX.md**
   - Add `reviews/` directory with audit findings
   - No file moves in this patch, so no CONTEXT_INDEX changes needed

---

## Long-Term Recommendations (Beyond B1)

### 1. Background Inference Heuristic
Instead of trying all bg ∈ {0-9}, add smart heuristic:
```python
def infer_bg_candidates(train):
    """Return likely bg values ordered by frequency."""
    all_grids = [x for x, y in train] + [y for x, y in train]
    color_counts = defaultdict(int)
    for g in all_grids:
        for color, count in zip(*np.unique(g, return_counts=True)):
            color_counts[color] += count
    # Most common color is likely bg
    return sorted(color_counts, key=color_counts.get, reverse=True)
```

### 2. Shape Parametricity
For future closures (crop, tile), shape must be parametric:
- Unifier infers output shape from train outputs
- run_fixed_point accepts `out_shape` parameter
- Closure params include shape transforms

### 3. Closure Composition Order
Current order: apply closures sequentially in list order. Consider:
- Dependency analysis (some closures must run before others)
- Commutativity detection (independent closures can run in any order)
- Log which closure caused which cells to narrow (for debug/receipts)

---

## Final Verdict

**MATHEMATICAL CORRECTNESS**: 10/10
- Lattice structure is sound
- Monotone shrinking guaranteed
- Tarski convergence proven
- Train verification is exhaustive

**PARAMETRICITY**: 4/10 (Before Patch) → 9/10 (After Patch)
- Critical hardcodes present but localized
- Fix is straightforward (35 min)
- Architecture debt noted for future work

**SUBMISSION READINESS**: BLOCKED → READY (After Patch)

---

## Patch Application Instructions

```bash
cd /Users/ravishq/code/arc-agi-1

# Review patch
cat reviews/ANTI_HARDCODE_PATCH.diff

# Apply patch (dry run first)
git apply --check reviews/ANTI_HARDCODE_PATCH.diff

# Apply for real
git apply reviews/ANTI_HARDCODE_PATCH.diff

# Verify changes
git diff src/arc_solver/closures.py
git diff src/arc_solver/closure_engine.py
git diff src/arc_solver/search.py

# Run smoke test (after creating test file)
pytest tests/test_closure_b1.py -v

# Run on real ARC task
python -m arc_solver.main --dataset training --output runs/b1-patched/predictions.json

# Verify determinism
python -m arc_solver.main --task data/training/0a1d4ef8.json --output pred1.json
python -m arc_solver.main --task data/training/0a1d4ef8.json --output pred2.json
diff pred1.json pred2.json  # Should be empty
```

---

**Sign-off**: After applying this patch and verifying determinism, the B0/B1 implementation is SUBMISSION-READY for ARC-AGI Kaggle.
