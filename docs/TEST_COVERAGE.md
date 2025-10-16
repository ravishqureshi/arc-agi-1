# ARC-AGI Test Coverage Report

**Run**: A+B Composition-Safe Unifiers + Per-Input Background
**Date**: 2025-10-15
**Dataset**: `data/arc-agi_training_challenges.json` (1000 tasks)
**Output**: `runs/20251015_AB/`

---

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| **Solved** | 6 | 0.6% |
| **Under-constrained** | 0 | 0.0% |
| **Failed** | 994 | 99.4% |
| **TOTAL** | 1000 | 100.0% |

---

## Test Execution Stats

**Tests Ran**: 1000/1000 tasks (100% completion)
**Tests Passed**: 6 tasks (all match ground truth)
**Tests Failed**: 994 tasks

**Implemented Closures**: 5 families
- KEEP_LARGEST_COMPONENT (M1.2)
- OUTLINE_OBJECTS (M1.3)
- OPEN_CLOSE (M2.1)
- AXIS_PROJECTION (M2.2)
- SYMMETRY_COMPLETION (M2.3)

**New Infrastructure** (A+B):
- Composition-safe unifier gates (preserves_y + compatible_to_y)
- Per-input background inference via border flood-fill
- Greedy composition with back-off in autobuild_closures

---

## Solved Tasks (6)

### 1. `4347f46a.json` - OUTLINE_OBJECTS

**Closure Applied:**
```json
{
  "name": "OUTLINE_OBJECTS[mode=outer,scope=all,bg=None]",
  "params": {"bg": null, "mode": "outer", "scope": "all"}
}
```

**Fixed-Point Stats:**
- Iterations: 2
- Multi-valued cells: 0 (exact solution)
- **Per-Input BG**: ✓ Uses border flood-fill (was bg=0 before A+B)

**Pattern**: Extract outer outline of all objects (4-connectivity), clear interior pixels to background.

---

### 2. `496994bd.json` - SYMMETRY_COMPLETION

**Closure Applied:**
```json
{
  "name": "SYMMETRY_COMPLETION[axis=h,scope=global,bg=None]",
  "params": {"axis": "h", "scope": "global", "bg": null}
}
```

**Fixed-Point Stats:**
- Iterations: 2
- Multi-valued cells: 0 (exact solution)
- **Per-Input BG**: ✓ Uses border flood-fill (was bg=0 before A+B)

**Pattern**: Horizontal reflection (mirror across horizontal center) and union with original to complete symmetric pattern.

---

### 3. `6f8cd79b.json` - OPEN_CLOSE

**Closure Applied:**
```json
{
  "name": "OPEN_CLOSE[mode=close,bg=8]",
  "params": {"bg": 8, "mode": "close"}
}
```

**Fixed-Point Stats:**
- Iterations: 2
- Multi-valued cells: 0 (exact solution)
- **Explicit BG**: Still uses bg=8 (per-input inference not needed)

**Pattern**: CLOSE operation (DILATE then ERODE) fills small gaps in foreground objects.

---

### 4. `9565186b.json` - KEEP_LARGEST_COMPONENT

**Closure Applied:**
```json
{
  "name": "KEEP_LARGEST_COMPONENT[bg=5]",
  "params": {"bg": 5}
}
```

**Fixed-Point Stats:**
- Iterations: 2
- Multi-valued cells: 0 (exact solution)
- **Explicit BG**: Still uses bg=5 (per-input inference not needed)

**Pattern**: Keep only largest connected component, remove all others.

---

### 5. `b8825c91.json` - SYMMETRY_COMPLETION

**Closure Applied:**
```json
{
  "name": "SYMMETRY_COMPLETION[axis=v,scope=global,bg=4]",
  "params": {"axis": "v", "scope": "global", "bg": 4}
}
```

**Fixed-Point Stats:**
- Iterations: 2
- Multi-valued cells: 0 (exact solution)
- **Explicit BG**: Still uses bg=4 (per-input inference not needed)

**Pattern**: Vertical reflection (mirror across vertical center) and union with original to complete symmetric pattern.

---

### 6. `f25ffba3.json` - SYMMETRY_COMPLETION

**Closure Applied:**
```json
{
  "name": "SYMMETRY_COMPLETION[axis=h,scope=global,bg=None]",
  "params": {"axis": "h", "scope": "global", "bg": null}
}
```

**Fixed-Point Stats:**
- Iterations: 2
- Multi-valued cells: 0 (exact solution)
- **Per-Input BG**: ✓ Uses border flood-fill (was bg=0 before A+B)

**Pattern**: Horizontal reflection (mirror across horizontal center) and union with original to complete symmetric pattern.

---

## Failed Tasks (994)

**Reasons for failure:**
- No closure (or composition) unifies across all train pairs
- Pattern requires closures not yet implemented (M3-M4)
- Complex multi-step transformations beyond current closure set

**Sample failed tasks** (showing status and fp stats):

```
0d3d703e.json: failed, fp.iters=0, fp.cells_multi=-1
007bbfb7.json: failed, fp.iters=0, fp.cells_multi=-1
00d62c1b.json: failed, fp.iters=0, fp.cells_multi=-1
```

**Note**: Failed tasks have `fp.iters=0` because no closures (or compositions) unified. This is expected behavior—unifiers correctly reject when train exactness cannot be proven.

---

## Under-Constrained Tasks (0)

No tasks reached fixed-point with multi-valued cells remaining. All closures either:
- Reached exact singleton solution (solved), or
- Failed to unify (failed)

---

## Coverage Analysis

### By Closure Family

| Closure | Solved | Notes |
|---------|--------|-------|
| KEEP_LARGEST_COMPONENT | 1 | Handles noise removal patterns |
| OUTLINE_OBJECTS | 1 | Handles outline extraction patterns (now uses bg=None) |
| OPEN_CLOSE | 1 | Fills gaps/removes spurs via morphology |
| AXIS_PROJECTION | 0 | Extends pixels along axis to border (no matches yet) |
| SYMMETRY_COMPLETION | 3 | Completes symmetric patterns via reflection (2 now use bg=None) |
| **Total Unique** | 6 | No overlap, **no composition yet** |

### A+B Contribution

**New Solved**: 0 tasks (coverage unchanged at 0.6%)
**Infrastructure Improvements**:
- **Part A (Composition-Safe Gates)**: Infrastructure ready for multi-closure solutions
  - `preserves_y()`: Validates closure doesn't contradict output
  - `compatible_to_y()`: Validates closure doesn't introduce spurious colors
  - Greedy composition with back-off in `autobuild_closures`
- **Part B (Per-Input BG)**: 3/6 tasks now use bg=None (more flexible)
  - Border flood-fill deterministic (majority vote + tie-break to lowest color)
  - Reduces incidental parameters (bg=None vs explicit bg values)

**Why No Coverage Increase?**
- All 6 solvable tasks can be solved with single closures
- No training tasks require composition of multiple closures yet
- Future closures (M3-M4) expected to benefit from composition infrastructure

**Per-Input BG Usage**:
- Tasks using bg=None: 3 (4347f46a, 496994bd, f25ffba3)
- Tasks using explicit bg: 3 (6f8cd79b, 9565186b, b8825c91)
- Border flood-fill working correctly for all bg=None cases

---

## Receipts Analysis

**Sample receipts.jsonl entry** (A+B with bg=None):

```json
{
  "task": "496994bd.json",
  "status": "solved",
  "closures": [
    {
      "name": "SYMMETRY_COMPLETION[axis=h,scope=global,bg=None]",
      "params": {"axis": "h", "scope": "global", "bg": null}
    }
  ],
  "fp": {"iters": 2, "cells_multi": 0},
  "timing_ms": {"fp": 0, "total": 8, "unify": 7},
  "hashes": {
    "task_sha": "...",
    "closure_set_sha": "..."
  },
  "invariants": {
    "component_delta": {...},
    "palette_delta": {...}
  }
}
```

All receipts include:
- Task ID
- Status (solved/failed)
- Closure list with parameters (bg can be null for per-input inference)
- Fixed-point statistics
- Timing (unify + total + fp)
- Hashes (task + closure set)
- Invariants (component + palette deltas)

---

## Validation

**Schema Validation**: Run `python scripts/submission_validator.py runs/20251015_AB/predictions.json`

**Determinism Check**: Run `bash scripts/determinism.sh data/arc-agi_training_challenges.json`

**Predictions Format**: Valid Kaggle submission format
- Task IDs: Properly formatted with `.json` extension
- Grids: Python lists (not numpy arrays)
- Values: Integers 0-9

---

## Next Steps

1. **Implement M3-M4** - Additional closure families likely to benefit from composition infrastructure
2. **Monitor composition usage** - Track when multiple closures are needed
3. **Run on evaluation set** (`data/arc-agi_evaluation_challenges.json`, 120 tasks) for final validation
4. **Optimize unifiers** - Potential to expand candidate sets with composition-safe gates

---

## Files

- **Predictions**: `runs/20251015_AB/predictions.json`
- **Receipts**: `runs/20251015_AB/receipts.jsonl`
- **This Report**: `docs/TEST_COVERAGE.md`

---

**Status**: A+B complete. Infrastructure upgraded for composition + per-input BG. Coverage: 6/1000 (0.6%). Ready for M3-M4.
