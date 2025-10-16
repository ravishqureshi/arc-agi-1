# ARC-AGI Test Coverage Report

**Run**: M2.3 SYMMETRY_COMPLETION
**Date**: 2025-10-15
**Dataset**: `data/arc-agi_training_challenges.json` (1000 tasks)
**Output**: `runs/20251015_m2.3/`

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
**Tests Passed**: 6 tasks
**Tests Failed**: 994 tasks

**Implemented Closures**: 5 families
- KEEP_LARGEST_COMPONENT (M1.2)
- OUTLINE_OBJECTS (M1.3)
- OPEN_CLOSE (M2.1)
- AXIS_PROJECTION (M2.2)
- SYMMETRY_COMPLETION (M2.3) ← **NEW**

---

## Solved Tasks (6)

### 1. `4347f46a.json` - OUTLINE_OBJECTS

**Closure Applied:**
```json
{
  "name": "OUTLINE_OBJECTS[mode=outer,scope=all,bg=0]",
  "params": {"bg": 0, "mode": "outer", "scope": "all"}
}
```

**Fixed-Point Stats:**
- Iterations: 2
- Multi-valued cells: 0 (exact solution)

**Pattern**: Extract outer outline of all objects (4-connectivity), clear interior pixels to background.

---

### 2. `496994bd.json` - SYMMETRY_COMPLETION ← **NEW in M2.3**

**Closure Applied:**
```json
{
  "name": "SYMMETRY_COMPLETION[axis=h,scope=global,bg=0]",
  "params": {"axis": "h", "scope": "global", "bg": 0}
}
```

**Fixed-Point Stats:**
- Iterations: 2
- Multi-valued cells: 0 (exact solution)

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

**Pattern**: CLOSE operation (DILATE then ERODE) fills small gaps in foreground objects (bg=8).

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

**Pattern**: Keep only largest connected component, remove all others.

---

### 5. `b8825c91.json` - SYMMETRY_COMPLETION ← **NEW in M2.3**

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

**Pattern**: Vertical reflection (mirror across vertical center) and union with original to complete symmetric pattern.

---

### 6. `f25ffba3.json` - SYMMETRY_COMPLETION ← **NEW in M2.3**

**Closure Applied:**
```json
{
  "name": "SYMMETRY_COMPLETION[axis=h,scope=global,bg=0]",
  "params": {"axis": "h", "scope": "global", "bg": 0}
}
```

**Fixed-Point Stats:**
- Iterations: 2
- Multi-valued cells: 0 (exact solution)

**Pattern**: Horizontal reflection (mirror across horizontal center) and union with original to complete symmetric pattern.

---

## Failed Tasks (994)

**Reasons for failure:**
- No closure unifies across all train pairs
- Pattern requires closures not yet implemented (M3-M4)
- Complex multi-step transformations beyond single-closure composition

**Sample failed tasks** (showing status and fp stats):

```
0d3d703e.json: failed, fp.iters=0, fp.cells_multi=-1
007bbfb7.json: failed, fp.iters=0, fp.cells_multi=-1
00d62c1b.json: failed, fp.iters=0, fp.cells_multi=-1
```

**Note**: Failed tasks have `fp.iters=0` because no closures unified. This is expected behavior—unifiers correctly reject when train exactness cannot be proven.

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
| OUTLINE_OBJECTS | 1 | Handles outline extraction patterns |
| OPEN_CLOSE | 1 | Fills gaps/removes spurs via morphology |
| AXIS_PROJECTION | 0 | Extends pixels along axis to border (no matches yet) |
| SYMMETRY_COMPLETION | 3 | Completes symmetric patterns via reflection |
| **Total Unique** | 6 | No overlap |

### M2.3 Contribution

**New Solved**: +3 tasks (496994bd.json, b8825c91.json, f25ffba3.json)
**Coverage Increase**: 0.3% → 0.6% (+100% relative improvement)

**Symmetry Breakdown**:
- Horizontal (axis=h): 2 tasks (496994bd, f25ffba3)
- Vertical (axis=v): 1 task (b8825c91)
- Diagonal axes (diag/anti): 0 tasks (no matches yet)
- Scope variants: All 3 used global scope

---

## Receipts Analysis

**Sample receipts.jsonl entry** (M2.3 solved task):

```json
{
  "task": "496994bd.json",
  "status": "solved",
  "closures": [
    {
      "name": "SYMMETRY_COMPLETION[axis=h,scope=global,bg=0]",
      "params": {"axis": "h", "scope": "global", "bg": 0}
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
- Status (solved/under_constrained/failed)
- Closure list with parameters
- Fixed-point statistics
- Timing (unify + total + fp)
- Hashes (task + closure set)
- Invariants (component + palette deltas)

---

## Validation

**Schema Validation**: Run `python scripts/submission_validator.py runs/20251015_m2.3/predictions.json`

**Determinism Check**: Run `bash scripts/determinism.sh data/arc-agi_training_challenges.json`

**Predictions Format**: Valid Kaggle submission format
- Task IDs: Properly formatted with `.json` extension
- Grids: Python lists (not numpy arrays)
- Values: Integers 0-9

---

## Next Steps

1. **Complete M2** - M2 milestone complete (3 closures implemented)
2. **Continue M3** (MOD_PATTERN, DIAGONAL_REPEAT) for periodic/diagonal patterns
3. **Continue M4** (TILING, COPY_BY_DELTAS) for final baseline submission
4. **Run on evaluation set** (`data/arc-agi_evaluation_challenges.json`, 120 tasks) for final validation
5. **Monitor receipts** for under-constrained cases (multi-valued cells at fixed-point)

---

## Files

- **Predictions**: `runs/20251015_m2.3/predictions.json`
- **Receipts**: `runs/20251015_m2.3/receipts.jsonl`
- **This Report**: `docs/TEST_COVERAGE.md`

---

**Status**: M2 complete (OPEN_CLOSE + AXIS_PROJECTION + SYMMETRY_COMPLETION). Coverage: 6/1000 (0.6%). Ready for M3.
