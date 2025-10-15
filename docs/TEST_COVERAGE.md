# ARC-AGI Test Coverage Report

**Run**: M2.1 OPEN_CLOSE
**Date**: 2025-10-15
**Dataset**: `data/arc-agi_training_challenges.json` (1000 tasks)
**Output**: `runs/20251015_m2.1/`

---

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| **Solved** | 3 | 0.3% |
| **Under-constrained** | 0 | 0.0% |
| **Failed** | 997 | 99.7% |
| **TOTAL** | 1000 | 100.0% |

---

## Test Execution Stats

**Tests Ran**: 1000/1000 tasks (100% completion)
**Tests Passed**: 3 tasks
**Tests Failed**: 997 tasks

**Implemented Closures**: 3 families
- KEEP_LARGEST_COMPONENT (M1.2)
- OUTLINE_OBJECTS (M1.3)
- OPEN_CLOSE (M2.1) ← **NEW**

---

## Solved Tasks (3)

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

### 2. `6f8cd79b.json` - OPEN_CLOSE ← **NEW in M2.1**

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

**Task Details:**
- Train pairs: 3
- Test pairs: 1
- Palette delta: {"0": -50, "8": 50}

**Pattern**: CLOSE operation (DILATE then ERODE) fills small gaps in foreground objects (bg=8).

---

### 3. `9565186b.json` - KEEP_LARGEST_COMPONENT

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

## Failed Tasks (997)

**Reasons for failure:**
- No closure unifies across all train pairs
- Pattern requires closures not yet implemented (M2.2-M2.3, M3-M4)
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
| **Total Unique** | 3 | No overlap |

### M2.1 Contribution

**New Solved**: +1 task (6f8cd79b.json)
**Coverage Increase**: 0.2% → 0.3% (+50% relative improvement)

---

## Receipts Analysis

**Sample receipts.jsonl entry** (M2.1 solved task):

```json
{
  "task": "6f8cd79b.json",
  "status": "solved",
  "closures": [
    {
      "name": "OPEN_CLOSE[mode=close,bg=8]",
      "params": {"bg": 8, "mode": "close"}
    }
  ],
  "fp": {"iters": 2, "cells_multi": 0},
  "timing_ms": {"fp": 0, "total": 4, "unify": 3},
  "hashes": {
    "task_sha": "f850cfa985e2f36e69aac89172c12e36f8d13d7d84a49247088cc954d0f61c8d",
    "closure_set_sha": "6447571ebdc05267724c88ed7a5578c37b1d0dec3e2dba57aca5915d7cffd834"
  },
  "invariants": {
    "component_delta": {"count_delta": 0, "largest_kept": false},
    "palette_delta": {"delta": {"0": -50, "8": 50}, "preserved": true}
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

**Schema Validation**: Run `python scripts/submission_validator.py runs/20251015_m2.1/predictions.json`

**Determinism Check**: Run `bash scripts/determinism.sh data/arc-agi_training_challenges.json`

**Predictions Format**: Valid Kaggle submission format
- Task IDs: Properly formatted with `.json` extension
- Grids: Python lists (not numpy arrays)
- Values: Integers 0-9

---

## Next Steps

1. **Implement M2.2-M2.3** (AXIS_PROJECTION, SYMMETRY_COMPLETION) to increase coverage
2. **Run on evaluation set** (`data/arc-agi_evaluation_challenges.json`, 120 tasks) for final validation
3. **Monitor receipts** for under-constrained cases (multi-valued cells at fixed-point)
4. **Continue M3-M4** (MOD_PATTERN, DIAGONAL_REPEAT, TILING, COPY_BY_DELTAS)

---

## Files

- **Predictions**: `runs/20251015_m2.1/predictions.json`
- **Receipts**: `runs/20251015_m2.1/receipts.jsonl`
- **This Report**: `docs/TEST_COVERAGE.md`

---

**Status**: M2.1 complete. Ready for M2.2 (AXIS_PROJECTION).
