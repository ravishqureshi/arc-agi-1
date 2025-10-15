# Anti-Hardcode Audit: Fixed-Point Closure Engine (B0/B1)

**Date**: 2025-10-15
**Scope**: `closure_engine.py`, `closures.py`, `search.py`
**Verdict**: NEEDS_WORK - Critical parametricity violations found

---

## Executive Summary

The fixed-point closure engine (B0) has correct **mathematical structure** (monotone shrinking, true convergence, train verification) but suffers from **parametricity violations** that break the master operator paradigm. The primary issue: **background color is hardcoded to 0** throughout, when it should be **inferred and unified from train pairs**.

**Key Principle Violated**: "Observer = observed - same parametrization must fit ALL train pairs"

---

## CRITICAL Issues (BLOCKERS)

### C1. Background hardcode in closure unifier
**File**: `closures.py:88`
**Code**: `def unify_KEEP_LARGEST(train: List[Tuple[Grid, Grid]], bg: int = 0)`

**Violation**: Background defaults to 0. This is a hardcode that defeats parametric unification.

**Fix**: Unifier should try bg ∈ {0-9} and return closures for ALL bg values that satisfy train exactness:
```python
def unify_KEEP_LARGEST(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    """Try all possible bg values, return those that verify on train."""
    closures = []
    for bg in range(10):
        candidate = KEEP_LARGEST_COMPONENT_Closure(
            "KEEP_LARGEST_COMPONENT", {"bg": bg}
        )
        if verify_closures_on_train([candidate], train):
            closures.append(candidate)
    return closures
```

**Severity**: CRITICAL - Breaks on any task where bg ≠ 0 (e.g., bg=7)

---

### C2. Hardcoded bg in autobuild_closures
**File**: `search.py:222`
**Code**: `closures += unify_KEEP_LARGEST(train, bg=0)`

**Violation**: Caller passes hardcoded bg=0 instead of letting unifier discover it.

**Fix**: Remove bg parameter entirely:
```python
closures += unify_KEEP_LARGEST(train)
```

**Severity**: CRITICAL - Prevents discovery of correct bg parameter

---

### C3. Shape assumption in run_fixed_point
**File**: `closure_engine.py:213-214`
**Code**:
```python
H, W = x_input.shape
U = init_top(H, W)
```

**Violation**: Assumes output shape equals input shape. Many ARC tasks have different output shapes (cropping, tiling expansion, padding).

**Current Impact**: For B1 (KEEP_LARGEST), output shape DOES equal input shape, so this works. But engine is not general.

**Fix Required for General Use**:
- Unifiers must return (closures, output_shape_fn)
- run_fixed_point must accept target shape: `run_fixed_point(closures, x_input, out_shape)`

**Severity**: CRITICAL for generality; OK for current B1-only usage

**Recommendation**: Add ARCHITECTURE_DEBT.md note; fix before adding crop/tile closures

---

### C4. Unsafe default in closure apply
**File**: `closures.py:52`
**Code**: `bg = self.params.get("bg", 0)`

**Violation**: Silently defaults to 0 if params missing. Should fail loudly.

**Fix**:
```python
bg = self.params["bg"]  # KeyError if missing = correct
```

**Severity**: HIGH - Masks unification bugs by silently using wrong parameter

---

## HIGH Issues (Must Fix Before Production)

### H1. All autobuild_operators calls hardcode bg=0
**File**: `search.py:32,34,37-39`
**Code**: Multiple calls with `bg=0`:
```python
induce_CROP_KEEP(train, bg=0)
induce_KEEP_LARGEST(train, bg=0)
induce_HOLE_FILL(train, bg=0)
induce_COPY_BY_DELTAS(train, rank=-1, group='global', bg=0)
```

**Violation**: These are beam-search operators (not closures), but same parametricity principle applies.

**Status**: Out of scope for this audit (beam code is legacy). Note for future refactor.

---

## MEDIUM Issues (Should Fix)

### M1. Magic number max_iters
**File**: `closure_engine.py:201`
**Code**: `max_iters: int = 100`

**Violation**: Arbitrary limit. Should be configurable or adaptive.

**Fix**: Make it a module constant with rationale:
```python
DEFAULT_MAX_ITERS = 100  # Tarski guarantees finite convergence; this is safety net
```

**Severity**: MEDIUM - Current default is reasonable, but should be explicit

---

### M2. Empty cell handling hardcodes bg=0
**File**: `closure_engine.py:105`
**Code**:
```python
if mask == 0:
    result[r, c] = 0  # Empty → background
```

**Violation**: Assumes background is 0 when converting empty cells.

**Current Impact**: Empty cells only occur if closures fail (under-constrained). This line is fallback for tie-breaking.

**Fix**: Pass bg as parameter to `to_grid_deterministic()`:
```python
def to_grid_deterministic(self, fallback: str = 'lowest', bg: int = 0) -> Grid:
    ...
    if mask == 0:
        result[r, c] = bg  # Use actual bg
```

**Severity**: MEDIUM - Rarely hit in practice (only on under-constrained grids)

---

## LOW Issues (Nice to Have)

### L1. Benign TODO
**File**: `search.py:223`
**Code**: `# TODO: Add more closure families as they're implemented`

**Status**: Acceptable placeholder for future work. Not a blocker.

---

## Positive Findings (What's CORRECT)

### ✓ SetValuedGrid.intersect is truly monotone
**File**: `closure_engine.py:56-58`
```python
def intersect(self, r: int, c: int, mask: int):
    """Intersect cell (r, c) with mask (monotone shrinking)."""
    self.data[r, c] &= mask
```
**Verdict**: CORRECT - Bitwise AND implements lattice meet (∩)

---

### ✓ run_fixed_point achieves true convergence
**File**: `closure_engine.py:224`
```python
if U == U_prev:
    ...
    return U, stats
```
**Verdict**: CORRECT - Checks equality, not iteration limit. Convergence is genuine.

---

### ✓ verify_closures_on_train checks ALL pairs
**File**: `closure_engine.py:255`
```python
for x, y in train:
    ...
```
**Verdict**: CORRECT - Loops through ALL train pairs, not just first one.

---

### ✓ KEEP_LARGEST_COMPONENT_Closure.apply is parametric
**File**: `closures.py:51-85`
```python
bg = self.params.get("bg", 0)
objs = components(x_input, bg)
...
```
**Verdict**: CORRECT (except for unsafe .get) - Uses self.params["bg"], not a global hardcode.

---

### ✓ Fixed-point stats are logged
**File**: `closure_engine.py:225-228`
```python
stats = {
    "iters": iteration + 1,
    "cells_multi": U.count_multi_valued_cells()
}
```
**Verdict**: CORRECT - Provides receipts (convergence iteration, multi-valued cell count)

---

## Mathematical Correctness (PASS)

1. **Lattice structure**: SetValuedGrid with bitwise ops is a valid complete lattice ✓
2. **Monotone shrinking**: intersect() only removes bits (∩), never adds ✓
3. **Idempotence**: Closures return new grid (functional), so F(F(U)) = F(U) by convergence ✓
4. **Tarski convergence**: Finite lattice + monotone ⇒ lfp exists and is reached ✓
5. **Train verification**: verify_closures_on_train checks U → singleton(y) for ALL pairs ✓

---

## Summary of Required Fixes

| Issue | File:Line | Severity | Fix Effort |
|-------|-----------|----------|------------|
| bg hardcode in unifier | closures.py:88 | CRITICAL | 15 min |
| bg hardcode in caller | search.py:222 | CRITICAL | 2 min |
| Unsafe .get() default | closures.py:52 | HIGH | 2 min |
| Empty cell → 0 | closure_engine.py:105 | MEDIUM | 10 min |
| max_iters magic | closure_engine.py:201 | MEDIUM | 5 min |
| Shape assumption | closure_engine.py:213 | ARCH_DEBT | Note only |

**Total effort**: ~35 minutes to remove all blockers

---

## Verdict

**NEEDS_WORK** - Implementation is mathematically sound but violates parametricity discipline.

**Must fix before submission**:
1. C1 (bg inference in unifier)
2. C2 (remove bg=0 from caller)
3. C4 (remove unsafe default)

**Can defer**:
- C3 (shape assumption) - OK for B1, but document as ARCH_DEBT
- M1, M2 (magic numbers) - Non-critical improvements
- H1 (beam bg hardcodes) - Out of scope (legacy beam code)

**Once fixed**: The closure engine will be **fully parametric** and **submission-ready** for B1 tasks.
