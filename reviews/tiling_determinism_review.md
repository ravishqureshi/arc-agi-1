# TILING Unifier Determinism Review

## Verdict
**PASS**

## Summary
All four unifier functions (`unify_TILING`, `unify_COLOR_PERM`, `unify_RECOLOR_ON_MASK`, and helper functions `_compute_template_mask`, `_compute_target_color`, `_compute_tiling_mask`) are deterministic. No blockers found.

## Blockers
None.

## High-Value Issues
None.

## Findings (Evidence)

### 1. Counter Tie-Breaking (CORRECT)
All Counter usage employs deterministic tie-breaking:

**Line 1364-1366 (unify_TILING):**
```python
counts = Counter(colors_on_mask)
mode_color = min(counts.keys(), key=lambda c: (-counts[c], c))
```
- Uses `min()` with tuple key `(-count, color)` instead of `Counter.most_common()`
- Ties broken by lowest color value
- Deterministic across all runs

**Line 1738-1739 (_compute_target_color):**
```python
counts = Counter(colors_on_mask)
mode_color = min(counts.keys(), key=lambda c: (-counts[c], c))
```
- Identical pattern - deterministic tie-breaking

### 2. Set Iteration Order (CORRECT)
Sets are handled properly:

**Line 1289-1311 (unify_TILING):**
```python
anchors = {(0, 0)}  # Set for deduplication
# ... add more anchors ...
anchors = sorted(anchors)[:8]  # Convert to sorted list before iteration
```
- Set used only for deduplication
- Converted to sorted list before any iteration
- Deterministic

**Line 1649, 1213 (largest_pixels sets):**
```python
largest_pixels = set(largest.pixels)
# ... later ...
for r, c in largest_pixels:  # Only for membership testing
    mask[r, c] = False
```
- Sets used only for membership testing (`if (r, c) in largest_pixels`)
- Not iterated in order-dependent way
- Deterministic

### 3. Component Selection (CORRECT)
Largest component selection uses deterministic tie-breaking:

**Lines 1648, 1212, 775, 213, 550, 109:**
```python
largest = max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))
```
- Primary sort: size (largest first)
- Tie-break 1: -bbox[0] (topmost)
- Tie-break 2: -bbox[1] (leftmost)
- Fully deterministic

### 4. Dictionary Iteration (CORRECT)
Dictionaries built and iterated in deterministic order:

**Line 1515-1531 (unify_COLOR_PERM):**
```python
color_map = {}
for x, y in train:  # Deterministic order
    for r in range(H_overlap):  # Deterministic order
        for c in range(W_overlap):  # Deterministic order
            color_map[x_c] = y_c
# ... later ...
used_y_colors = list(color_map.values())  # Insertion order preserved (Python 3.7+)
```
- Dict populated in deterministic nested-loop order
- Values extracted in same insertion order
- Deterministic (requires Python 3.7+, which is standard now)

**Line 1050-1060 (unify_MOD_PATTERN):**
```python
class_map = {}
for key in class_colors_per_pair[0]:  # Dict iteration
    class_map[key] = class_colors_per_pair[0][key].copy()
```
- Iterates over dict keys directly (insertion order)
- Safe because source dict built in deterministic order
- Deterministic

### 5. Loop Enumeration (CORRECT)
All enumeration uses deterministic structures:

**Lines 1292-1303, 1317-1331, 1334-1337:**
- Loops use `range()`, explicit tuples, or sorted lists
- No filesystem or I/O ordering
- Deterministic

**Line 1326-1327 (PARITY anchor2 variants):**
```python
for ar2, ac2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
```
- Explicit tuple list - deterministic

**Line 1329-1331 (STRIPES variants):**
```python
for axis in ["row", "col"]:
    for k in [2, 3]:
```
- Explicit lists - deterministic

### 6. No RNG, Timers, or Filesystem Dependencies
- No `random`, `time.time()`, or wall-clock usage
- No filesystem traversal (`os.listdir`, `glob`, etc.)
- All data flows from input parameters only
- Deterministic

### 7. Early Exit Handling (ACCEPTABLE)
**Line 1442-1443:**
```python
if len(valid) >= 10:
    return valid
```
- Early exit after 10 candidates
- Safe because iteration order is deterministic
- Same candidates in same order every run
- Deterministic

## Code Quality Observations

### Strengths
1. **Consistent tie-breaking patterns** - All Counter usage and component selection use the same deterministic patterns
2. **Proper set handling** - Sets converted to sorted lists before order-dependent operations
3. **Clean enumeration** - All loops use deterministic structures (range, explicit lists)
4. **No external dependencies** - Pure function of input data

### Pattern Compliance
The code follows all determinism best practices:
- Counter with `min(counts.keys(), key=lambda c: (-counts[c], c))`
- Component selection with `max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))`
- Set → sorted list conversion before iteration
- No RNG, timers, or filesystem I/O

## Test Recommendations

While the code passes static analysis, recommend runtime verification:

1. **Parallel determinism test:**
   ```bash
   for i in {1..5}; do
     python scripts/run_public.py --tasks 5 --jobs 1 > run1_$i.json
     python scripts/run_public.py --tasks 5 --jobs 4 > run4_$i.json
     diff run1_$i.json run4_$i.json
   done
   ```

2. **Hash comparison:**
   - Run same task multiple times
   - Hash predictions.json byte-for-byte
   - Verify identical hashes

## Conclusion

All TILING-related unifiers are **fully deterministic**. No changes required.

The fixes correctly address:
- Mask-local check (ONE-STEP, no fixed-point iteration)
- Canvas computation once at start
- Shape-aware overlap region checking
- Expanded template support (PARITY, STRIPES, BORDER_RING)
- Deterministic tie-breaking throughout

**Status**: Ready for submission.
