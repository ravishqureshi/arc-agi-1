# ARC Solver - Test Coverage Report

**Generated:** 2025-10-13
**Dataset:** ARC-AGI Training Set (1,000 tasks, 1,076 test outputs)
**Solver Version:** Modular v1.0.0 (3 active rules)

---

## Summary Statistics

| Metric | Value | Percentage |
|--------|-------|------------|
| **Total Test Outputs** | 1,076 | 100.0% |
| **Code Runs Successfully** | 1,076 | 100.0% |
| **Tests Passing** | 6 | 0.6% |
| **ARC-1 Passing** | 6 / 407 | 1.5% |
| **ARC-2 Passing** | 0 / 669 | 0.0% |

---

## Breakdown by Rule

| Rule | Tests Passing | Tasks Solved |
|------|---------------|--------------|
| ROT | 3 | 3 |
| FLIP | 2 | 2 |
| CROP_BBOX_NONZERO | 1 | 1 |

---

## Passing Tests (Detailed)

| Task ID | ARC Version | Test # | Rule | Pred Shape | Truth Shape | Residual | Status |
|---------|-------------|--------|------|------------|-------------|----------|--------|
| 1cf80156 | ARC-1 | 0 | CROP_BBOX_NONZERO | 4x6 | 4x6 | 0 | ✅ PASS |
| 3c9b0459 | ARC-1 | 0 | ROT | 3x3 | 3x3 | 0 | ✅ PASS |
| 6150a2bd | ARC-1 | 0 | ROT | 3x3 | 3x3 | 0 | ✅ PASS |
| 67a3c6ac | ARC-1 | 0 | FLIP | 3x3 | 3x3 | 0 | ✅ PASS |
| 68b16354 | ARC-1 | 0 | FLIP | 3x3 | 3x3 | 0 | ✅ PASS |
| ed36ccf7 | ARC-1 | 0 | ROT | 3x3 | 3x3 | 0 | ✅ PASS |

---

## Failing Tests - Sample (First 10 by Task ID)

| Task ID | ARC Version | Test # | Rule | Pred Shape | Truth Shape | Residual | Issue |
|---------|-------------|--------|------|------------|-------------|----------|-------|
| 00576224 | ARC-2 | 0 | None | 7x7 | 7x7 | 49 | No rule matched |
| 00576224 | ARC-2 | 1 | None | 7x7 | 7x7 | 49 | No rule matched |
| 0064b7a9 | ARC-2 | 0 | None | 13x13 | 14x7 | 182 | No rule + shape mismatch |
| 007bbfb7 | ARC-1 | 0 | None | 11x11 | 11x11 | 121 | No rule matched |
| 00950b68 | ARC-2 | 0 | None | 10x10 | 3x3 | 109 | No rule + shape mismatch |
| 00dbd492 | ARC-2 | 0 | None | 14x14 | 14x14 | 196 | No rule matched |
| 017c7c7b | ARC-2 | 0 | None | 5x5 | 5x5 | 25 | No rule matched |
| 025d127b | ARC-1 | 0 | None | 3x3 | 3x3 | 9 | No rule matched |
| 0293b340 | ARC-2 | 0 | None | 30x20 | 30x15 | 600 | No rule + shape mismatch |
| 045e512c | ARC-2 | 0 | None | 7x7 | 7x7 | 49 | No rule matched |

**Note:** 1,070 tests are failing (99.4%). Most common issue: "No rule matched" (no induction routine fits the training pairs).

---

## Error Analysis

### Why Only 0.6% Passing?

Currently, the modular solver has **only 3 active induction routines**:

1. **induce_symmetry_rule** - ROT(0,1,2,3) + FLIP('h','v')
2. **induce_crop_nonzero_rule** - CROP_BBOX_NONZERO
3. **induce_keep_nonzero_rule** - KEEP(MASK_NONZERO)

These are extremely simple transformations that only work on very basic tasks.

### Missing Rules (from legacy demo)

These 4 rules were in `arc_demo.py` but not yet migrated:

- **induce_color_perm** - Global color permutation (could solve ~6 more tasks)
- **induce_move_bbox_to_origin** - Move bbox to (0,0) while preserving grid size
- **induce_mirror_left_to_right** - Mirror left half to right half
- **induce_component_size_recolor** - Recolor connected components by size rank

### ARC-2 Challenge

**0/669 ARC-2 tests passing (0.0%)**

ARC-2 tasks are significantly harder:
- Symbol learning (objects represent abstract concepts)
- In-context definition (meanings defined within the task)
- More complex multi-step compositions
- Novel patterns not seen in ARC-1

All 6 passing tests are from ARC-1, confirming that:
- Simple symmetry/crop rules work on some ARC-1 tasks
- ARC-2 requires more sophisticated operators (Phase 2+)

---

## Next Steps to Improve Coverage

### Phase 0: Restore Legacy Rules (Target: ~12 tasks, 1.2%)

Add 4 missing rules from `arc_demo.py`:

- [ ] Color permutation
- [ ] Move bbox to origin
- [ ] Mirror operations
- [ ] Component rank recolor

**Expected:** 12 tasks solved (1.2% accuracy), matching legacy demo performance

### Phase 1: Implement Baseline (Target: 600-750 tasks, 60-75%)

Add 60-80 operators per IMPLEMENTATION_PLAN.md:

**Spatial operators:**
- SHIFT(dx, dy, wrap=True/False)
- PASTE(sub, at)
- RESIZE(new_H, new_W)
- PAD(border, fill_color)

**Color operators:**
- RECOLOR(mapping)
- COLOR_PERM() with induction
- Conditional recolor by rank/adjacency/parity

**Object operators:**
- MAP(ObjList, op_per_obj)
- SORT(ObjList, key='size'/'color'/'x'/'y')
- FILTER(ObjList, predicate)

**Tiling operators:**
- TILE(pattern, repeat_x, repeat_y)
- Detect translation vectors
- Motif repetition with boundaries

**Drawing operators:**
- DRAW_LINE(start, end, color)
- DRAW_BOX(rect, filled=True/False)
- FLOOD_FILL(start, color)

**Expected:** 600-750 tasks solved (60-75% accuracy)

### Phase 2: Gap-Filling (Target: 750-850 tasks, 75-85%)

Add 25-30 targeted operators for common failure modes:

**Grid resizing (15-20% of tasks):**
- EXPAND(factor) - Uniform scaling
- UPSAMPLE(factor) - Pixel replication
- DOWNSAMPLE(factor, method='mode')

**Physics simulation (5-8% of tasks):**
- FALL(direction='down')
- GRAVITY(direction)
- STACK(objects, direction='vertical')
- SETTLE() - Iterative gravity until stable

**Path/connectivity (3-5% of tasks):**
- TRACE_PATH(start, end, via='shortest')
- CONNECT(obj1, obj2, line_color)
- SHORTEST_PATH(start, end, obstacles)

**Symbol learning (8-12% of ARC-2 tasks):**
- LEARN_SYMBOL_MAP(train_pairs)
- APPLY_SYMBOL_SEMANTICS(test_grid, symbol_map)

**Expected:** 750-850 tasks solved (75-85% accuracy)

### Phase 3: Deep Search (Target: 770-840 tasks, 77-84%)

Compositional improvements:

- Increase beam search depth: 6-8 → 10-12
- Add composition templates ("Detect → Transform → Place")
- Implement transfer learning from similar tasks
- Multi-solution ensembles with voting

**Expected:** 770-840 tasks solved (77-84% accuracy) → **Beat 79.6% record** 🏆

---

## How to Update This Report

### Automatic Update (Recommended)

```bash
# Run test coverage generator
python scripts/generate_test_coverage.py

# This will:
# 1. Test all 1,000 tasks
# 2. Save results to docs/test_coverage_data.json
# 3. Regenerate this markdown report
# 4. Show before/after comparison
```

### Manual Update

```bash
# Generate raw data
python -c "
import sys; sys.path.insert(0, 'src')
import json
from arc_solver import G, solve_instance, ARCInstance
from arc_version_split import is_arc1_task

# ... (see scripts/generate_test_coverage.py for full code)
"

# Edit docs/TEST_COVERAGE.md manually
```

---

## Tracking Progress

| Phase | Target Accuracy | Current | Tasks to Add | Status |
|-------|-----------------|---------|--------------|--------|
| **Modular v1.0** | 1.2% (12 tasks) | 0.6% (6 tasks) | 6 | 🔄 In Progress |
| **Phase 1 Baseline** | 60-75% (600-750) | 0.6% (6) | 594-744 | 🔲 Not Started |
| **Phase 2 Gap-Fill** | 75-85% (750-850) | 0.6% (6) | 744-844 | 🔲 Not Started |
| **Phase 3 Deep Search** | 77-84% (770-840) | 0.6% (6) | 764-834 | 🔲 Not Started |
| **Record (79.6%)** | 79.6% (796) | 0.6% (6) | 790 | 🎯 Goal |

---

## Data Files

- **Raw data:** `docs/test_coverage_data.json` (1,076 test results)
- **Task ID lists:** `arc1_task_ids.txt`, `arc2_new_task_ids.txt`
- **Solutions:** `data/arc-agi_training_solutions.json`
- **Challenges:** `data/arc-agi_training_challenges.json`

---

## Comparison with Legacy Demo

| Metric | Legacy (arc_demo.py) | Modular v1.0 | Delta |
|--------|---------------------|--------------|-------|
| Rules Implemented | 6 | 3 | -3 |
| Tasks Solved | 12 | 6 | -6 |
| Accuracy | 1.2% | 0.6% | -0.6% |
| Lines of Code | 500 (monolithic) | 1000 (modular, 15 files) | +500 |
| Maintainability | Low | High | ✅ |
| Scalability | Low | High | ✅ |
| Test Coverage | None | Full (1,076 tests) | ✅ |

**Trade-off:** Temporarily lower accuracy for much better architecture that can scale to 60-80+ operators.

---

**Last Updated:** 2025-10-13
**Data File:** `docs/test_coverage_data.json`
**Tasks Tested:** 1,000 / 1,000 (100%)
**Code Status:** All tests run successfully (no crashes) ✅
