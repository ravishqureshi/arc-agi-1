# ARC Demo Expansion: Added 2 Rules (4 → 6)

**Date**: 2025-10-13
**Source**: `docs/core/2_ui_arc_press_demo.md`
**Target**: `src/arc_demo.py`

---

## Summary

Extracted and integrated 2 new transformation rules from the press demo into the working `arc_demo.py` code. The rules work correctly but **do not solve additional real ARC tasks** in the current dataset.

---

## New Rules Added

### Rule 4: `induce_move_bbox_to_origin`

**What it does**: Extracts the bounding box of non-zero content and moves it to the top-left corner (0,0), preserving the original grid size.

**Key difference from `crop_bbox`**: Maintains grid dimensions instead of resizing.

**Example**:
```
Input (3×4):              Output (bbox at origin):
[[0, 0, 0, 0],      →     [[5, 5, 0, 0],
 [0, 5, 5, 0],             [5, 0, 0, 0],
 [0, 5, 0, 0]]             [0, 0, 0, 0]]
```

**Implementation** (lines 207-231):
- Shape-preserving (validates `x.shape == y.shape`)
- Uses `np.full_like()` to fill with background
- Exact verification on all training pairs

### Rule 5: `induce_mirror_left_to_right`

**What it does**: Mirrors the left half of the grid to the right half.

**Constraint**: Requires even width (rejects odd-width grids).

**Example**:
```
Input (2×4, even width):    Output (left mirrored):
[[1, 0, 0, 0],        →     [[1, 0, 0, 1],
 [0, 1, 0, 0]]               [0, 1, 1, 0]]
```

**Implementation** (lines 233-258):
- Checks `W % 2 == 0` (raises `ValueError` if odd)
- Uses `np.fliplr()` on left half
- Wrapped in try/except for safe rule rejection

---

## Testing Results

### Synthetic Demo Tasks

✅ **Both rules work correctly** on synthetic test cases:
- `move_bbox_to_origin`: Verified with 3×4 → 3×5 grid
- `mirror_left_to_right`: Verified with 2×4 grids

### Real ARC Training Data (1,000 tasks)

| Metric | 4 Rules (Before) | 6 Rules (After) | Change |
|--------|------------------|-----------------|--------|
| **Solved** | 12/1,000 (1.2%) | 12/1,000 (1.2%) | **+0** |
| **Errors** | 234 | 234 | +0 |
| **move_bbox_to_origin matches** | N/A | 0 | - |
| **mirror_left_to_right matches** | N/A | 0 | - |

⚠️ **Neither new rule matches any real ARC training task.**

### Analysis: Why No Improvement?

1. **ARC task distribution**: These specific spatial patterns (bbox-to-origin, left-right mirror) are not present in the training set
2. **Pattern rarity**: Only 12/1,000 tasks (1.2%) match ANY of the 6 rules
3. **Complexity gap**: Real ARC tasks require combinations of rules, not single transformations

### Solved Tasks Breakdown (Still 12 tasks)

| Rule | Count |
|------|-------|
| color_perm | 4 |
| symmetry:rot180 | 2 |
| component_size_recolor | 2 |
| crop_bbox | 1 |
| symmetry:flip_h | 1 |
| symmetry:flip_v | 1 |
| symmetry:rot90 | 1 |

**Note**: All 12 solved tasks are from ARC-1 (0 from ARC-2).

---

## Code Quality Improvements

✅ **Better error handling** than press_demo:
- `move_bbox_to_origin`: Checks shape preservation
- `mirror_left_to_right`: Graceful rejection of odd-width grids (try/except)

✅ **Consistent with existing style**:
- Same naming convention (`induce_*`)
- Same verification pattern (exact match on training)
- Same docstring format

✅ **No bugs introduced**:
- Avoided http:// bug from press_demo
- Avoided assert-based crashes from press_demo

---

## Catalog Status

**Updated catalog** (lines 298-305):
```python
CATALOG = [
    induce_symmetry,              # 6 transforms (rot90/180/270, flip_h/v, id)
    induce_color_perm,            # Global color mapping
    induce_crop_bbox,             # Extract bounding box
    induce_move_bbox_to_origin,   # NEW: Move bbox to origin
    induce_mirror_left_to_right,  # NEW: Mirror left to right
    induce_component_size_recolor,# Recolor by component size
]
```

**Total rules**: 6 (up from 4)

---

## Next Steps for Real Impact

To improve from 1.2% → >10% accuracy, need:

### 1. Pattern Analysis (Data-Driven)
- Analyze the 988 unsolved tasks
- Identify common transformation patterns
- Prioritize rules that appear frequently

### 2. High-Value Rule Categories
- **Tiling/Repetition**: Repeat patterns in grid
- **Object operations**: Extract, transform, place objects
- **Grid arithmetic**: Add/subtract grids
- **Masking**: Apply transformations conditionally
- **Multi-step**: Combine multiple operations

### 3. Rule Composition
- Allow chaining rules (e.g., crop → color_perm → mirror)
- Search for best rule combination
- Verify each step exactly (maintain receipts)

### 4. Error Analysis
- 234 tasks crash (23.4%)
- Add shape/dimension validation
- Handle edge cases (empty grids, single pixels)

---

## Lessons Learned

1. **Rules must match data distribution**: Random rule additions won't help without data analysis
2. **Synthetic demos ≠ real performance**: Press demo's 10/10 success doesn't transfer
3. **ARC is compositional**: Single-rule transformations are rare (<2%)
4. **Framework is solid**: Adding rules is straightforward, need strategic selection

---

## Files Updated

1. ✅ `src/arc_demo.py` - Added 2 rules (lines 207-258, 298-305)
2. ✅ `README.md` - Updated rule count (4 → 6)
3. ✅ `docs/arc_demo_expansion_summary.md` - This file

---

## Recommendation

**Stop random rule additions**. Start data-driven development:

1. Run analysis on unsolved tasks → identify patterns
2. Build rules for top 10 most common patterns
3. Add rule composition framework
4. Target 10-20% accuracy before expanding further

The framework is ready. Now need strategic rule selection based on actual ARC task distribution.
