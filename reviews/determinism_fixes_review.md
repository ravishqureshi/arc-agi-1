# Determinism Fixes Review: MOD_PATTERN & AXIS_PROJECTION

## Verdict
**PASS**

## Summary
All three fix areas demonstrate correct determinism implementation:
- AXIS_PROJECTION to_obstacle mode: deterministic iteration and color assignment
- unify_AXIS_PROJECTION mask-local check: deterministic set operations and comparisons
- unify_MOD_PATTERN mask-local check: deterministic dict iteration via insertion order

No blockers found. All potential non-determinism sources properly addressed.

## Detailed Analysis

### 1. AXIS_PROJECTION_Closure: to_obstacle mode (lines 576-612)

**What was fixed**: Added "to_obstacle" mode that extends projection until hitting another object instead of going to border.

**Determinism check**:
```python
# Line 583-588: Left direction
for col in range(c, -1, -1):
    pixel_color = int(x_input[r, col])
    if pixel_color != bg and pixel_color != obj.color:
        break
    projected_pixels[(r, col)] = obj.color
```

✓ **PASS**: Loop iteration order is deterministic (range with explicit bounds)
✓ **PASS**: Dictionary insertion order is deterministic (pixels processed row-by-row, col-by-col)
✓ **PASS**: Break conditions are deterministic (based on input grid values, not sets/dicts)
✓ **PASS**: All four directions (left, right, up, down) use same deterministic pattern

**Tie-breaking** (line 550):
```python
selected_objs = [max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))]
```
✓ **PASS**: Deterministic tie-breaking by size, then top-left position

**Scope per_object** (lines 553-555):
```python
elif scope == "per_object":
    # Each object projects independently
    selected_objs = objs
```
✓ **PASS**: Processes all objects in deterministic order from `components()` which returns list

### 2. unify_AXIS_PROJECTION: mask-local check (lines 669-707, 719-756)

**What was fixed**: Added mask-local validation to ensure closure only edits cells where x != y.

**Determinism check**:
```python
# Line 669-707
mask_local_ok = True
for x, y in train:
    U_x = init_from_grid(x)
    U1 = candidate_none.apply(U_x, x)

    H_check = min(x.shape[0], y.shape[0], U1.H)
    W_check = min(x.shape[1], y.shape[1], U1.W)

    for r in range(H_check):
        for c in range(W_check):
            x_color = int(x[r, c])
            y_color = int(y[r, c])

            if x_color == y_color:
                allowed = U1.get_set(r, c)
                if allowed != {x_color}:  # Line 690
                    mask_local_ok = False
                    break
            else:
                allowed = U1.get_set(r, c)
                if len(allowed) == 0 or y_color not in allowed:  # Line 696
                    mask_local_ok = False
                    break
```

✓ **PASS**: List iteration over `train` is deterministic
✓ **PASS**: Nested range loops are deterministic
✓ **PASS**: Set comparison `allowed != {x_color}` (line 690) is deterministic (equality check with singleton)
✓ **PASS**: Set membership `y_color not in allowed` (line 696) is deterministic
✓ **PASS**: No unordered set/dict iteration that affects control flow

**Parameter enumeration** (lines 656-663):
```python
axes = ["row", "col"]
scopes = ["largest", "all", "per_object"]
modes = ["to_border", "to_obstacle"]

for axis in axes:
    for scope in scopes:
        for mode in modes:
            # ... build candidate
```
✓ **PASS**: Lists enumerated in fixed order

### 3. unify_MOD_PATTERN: mask-local check (lines 1202-1244)

**What was fixed**: Added mask-local validation similar to AXIS_PROJECTION.

**Determinism check**:

**Anchor enumeration** (lines 1120-1145):
```python
anchors = {(0, 0)}  # Line 1120
# ... add more anchors to set ...
for ar, ac in sorted(anchors):  # Line 1145 - SORTED!
```
✓ **PASS**: Set sorted before iteration

**Class map building** (lines 1152-1182):
```python
for x, y in train:
    H_cur, W_cur = x.shape
    class_colors = {}  # Line 1154

    for r in range(H_cur):  # Line 1156
        for c in range(W_cur):
            i = (r - ar) % p
            j = (c - ac) % q

            if (i, j) not in class_colors:
                class_colors[(i, j)] = set()

            class_colors[(i, j)].add(int(x[r, c]))  # Line 1165
```
✓ **PASS**: Dict keys `(i,j)` inserted in deterministic order (row-by-row, col-by-col iteration)
✓ **PASS**: In Python 3.7+, dict maintains insertion order

**Class map intersection** (lines 1172-1181):
```python
class_map = {}
for key in class_colors_per_pair[0]:  # Line 1172
    class_map[key] = class_colors_per_pair[0][key].copy()

for pair_colors in class_colors_per_pair[1:]:
    for (i, j) in list(class_map.keys()):  # Line 1177
        if (i, j) in pair_colors:
            class_map[(i, j)] &= pair_colors[(i, j)]
        else:
            class_map[(i, j)] = set()
```
✓ **PASS**: Line 1172 iterates over dict with deterministic insertion order (from first train pair)
✓ **PASS**: Line 1177 iterates over class_map keys, but set intersection is commutative so order doesn't affect result
✓ **PASS**: Set operations (`&=`) are deterministic for fixed inputs

**Mask-local check** (lines 1204-1241):
```python
mask_local_ok = True
for x, y in train:
    U_x = init_from_grid(x)
    U1 = candidate.apply(U_x, x)

    H_check = min(x.shape[0], y.shape[0], U1.H)
    W_check = min(x.shape[1], y.shape[1], U1.W)

    for r in range(H_check):
        for c in range(W_check):
            x_color = int(x[r, c])
            y_color = int(y[r, c])

            if x_color == y_color:
                allowed = U1.get_set(r, c)
                if allowed != {x_color}:
                    mask_local_ok = False
                    break
            else:
                allowed = U1.get_set(r, c)
                if len(allowed) == 0 or y_color not in allowed:
                    mask_local_ok = False
                    break
```
✓ **PASS**: Same deterministic pattern as unify_AXIS_PROJECTION

## Determinism Checklist

### 1. Non-deterministic tie-breaking
✓ **PASS**: All max() calls use explicit tuple keys with deterministic ordering
- Line 550 (AXIS_PROJECTION): `(o.size, -o.bbox[0], -o.bbox[1])`

### 2. Set/dict iteration without sorting
✓ **PASS**: All potential issues addressed
- Line 1145 (MOD_PATTERN): `sorted(anchors)` before iteration
- Line 1172 (MOD_PATTERN): Dict iteration uses deterministic insertion order
- Lines 690, 696 (AXIS_PROJECTION): Set operations are checks/membership, not iteration
- Lines 1227, 1233 (MOD_PATTERN): Same pattern as AXIS_PROJECTION

### 3. RNG, timers, wall-clock
✓ **PASS**: No random number generation, no time-based operations

### 4. Filesystem/IO order dependencies
✓ **PASS**: No filesystem operations, no I/O

### 5. Race conditions
✓ **PASS**: Single-threaded execution, no parallel processing

## Code Quality Notes

### Strengths
1. **Consistent patterns**: Mask-local checks implemented identically in both unifiers
2. **Explicit ordering**: Uses `range()` with explicit bounds throughout
3. **Deterministic tie-breaking**: All max() operations include secondary sort keys
4. **Safe set operations**: Set equality and membership checks instead of iteration

### Design correctness
1. **to_obstacle mode**: Correctly extends in both directions from each pixel
2. **per_object scope**: Each object projects independently without interference
3. **Mask-local validation**: Ensures closures are truly local to edit mask (x != y)
4. **Class map building**: Correctly uses input-only information (MOD_PATTERN)

## Conclusion

All three fix areas pass determinism review. The implementation correctly:
- Uses deterministic iteration order (sorted sets, range loops, list iteration)
- Implements deterministic tie-breaking (tuple keys for max operations)
- Avoids non-deterministic sources (no RNG, no I/O, no wall-clock)
- Maintains same-input → same-output guarantee

No changes required.
