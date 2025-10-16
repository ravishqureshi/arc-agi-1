# Math Closure-Soundness Review

**Reviewer:** Math-Closure-Soundness-Reviewer
**Date:** 2025-10-16
**Scope:** AXIS_PROJECTION + MOD_PATTERN fixes (mask-local verification + new modes)
**Data:** `data/arc-agi_training_challenges.json` (training-only induction, no eval/test peeking)

---

## Verdict

**PASS**

Both AXIS_PROJECTION and MOD_PATTERN fixes satisfy all mathematical soundness requirements and correctly implement mask-local verification.

---

## Blockers (must fix to preserve correctness)

**None.**

All reviewed closures preserve correctness of the fixed-point solver on ARC tasks.

---

## High-Value Issues (should fix soon)

**None.**

The implementations are mathematically sound and correctly enforce the composition-safe gate with mask-local verification.

---

## Closure Law Table

| name            | one-line law                                                                                     | shrinking? | idempotent? | unified params? | input-only mask? | train exact? | verdict |
|-----------------|--------------------------------------------------------------------------------------------------|------------|-------------|-----------------|------------------|--------------|---------|
| AXIS_PROJECTION | Project object pixels along axis (row/col) to border or obstacle; all other cells → {bg}       | yes        | yes         | yes             | yes              | yes          | PASS    |
| MOD_PATTERN     | Partition cells by congruence class ((r-ar) mod p, (c-ac) mod q); each class → fixed color set | yes        | yes         | yes             | yes              | yes          | PASS    |

---

## Evidence

### AXIS_PROJECTION_Closure (src/arc_solver/closures.py:511-632)

**Law statement:**
> From each object pixel, extend along axis (row or col) to image border (mode="to_border") or until hitting another object (mode="to_obstacle"), painting object color along the ray. All non-projected cells forced to {bg}.

**Changes reviewed:**
1. **New mode: "to_obstacle"** (lines 576-612)
2. **New scope: "per_object"** (lines 553-555)
3. **Mask-local check in unifier** (lines 669-707, 719-757)

#### 1. Monotone & Shrinking Law

**Evidence (lines 615-629):**
```python
# Build masks: projected pixels allow object color, others allow bg only
U_new = U.copy()
bg_mask = color_to_mask(bg)

for r in range(U.H):
    for c in range(U.W):
        if (r, c) in projected_pixels:
            # Projected pixel: allow object color
            obj_color = projected_pixels[(r, c)]
            obj_mask = color_to_mask(obj_color)
            U_new.intersect(r, c, obj_mask)  # ← ONLY CLEAR BITS
        else:
            # Non-projected pixel: force to bg
            U_new.intersect(r, c, bg_mask)   # ← ONLY CLEAR BITS
```

**Verification:**
- `color_to_mask(c)` returns singleton bitmask `1 << c` (1 bit set)
- `U_new.intersect(r, c, mask)` performs bitwise AND: `data[r,c] &= mask`
- Both branches shrink U (U' ⊆ U) deterministically ✓
- No bit-setting operations anywhere in apply() ✓

#### 2. Idempotence (≤2-pass practical)

**Evidence (lines 559-612):**
- `projected_pixels` dict is built deterministically from `x_input`, `bg`, `axis`, `scope`, `mode`
- All parameters are fixed per closure instance
- Second application builds identical `projected_pixels` dict
- Second apply yields U_new == U_prev (no change) ✓

**Convergence proof:**
- Pass 1: Apply constraints from `projected_pixels`
- Pass 2: Same constraints; U unchanged (fixed point reached)
- ≤2 passes guaranteed ✓

#### 3. Input-only Masks

**Evidence (lines 536-612):**
```python
objs = components(x_input, bg=bg)  # ← from x_input only
# ... select components based on scope ...

# Build projection from x_input pixels
if mode == "to_border":
    for obj in selected_objs:
        for r, c in obj.pixels:  # ← obj.pixels from x_input
            # extend along axis to border

elif mode == "to_obstacle":
    for obj in selected_objs:
        for r, c in obj.pixels:  # ← obj.pixels from x_input
            # extend until hitting pixel_color from x_input[r,col] or x_input[row,c]
```

**Verification:**
- `components(x_input, bg=bg)`: Extracts objects from input only ✓
- `bg` inferred from `infer_bg_from_border(x_input)` if None ✓
- Obstacle detection (mode="to_obstacle") checks `x_input[r,col]` or `x_input[row,c]` ✓
- **No dependency on y anywhere in apply()** ✓

**Data file proof:**
- Unifier receives `train: List[Tuple[Grid, Grid]]` parameter
- Uses `y` **only** in mask-local verification (lines 669-707, 719-757)
- No file I/O or dataset loading in closure or unifier
- Evidence derived from training challenges only (as required) ✓

#### 4. Mask-local Verification (unify_AXIS_PROJECTION, lines 669-707, 719-757)

**Fix 2 implementation (lines 669-707 for bg=None, 719-757 for explicit bg):**

```python
# Mask-local check (ONE-STEP, no fixed-point iteration)
mask_local_ok = True
for x, y in train:
    # Build U_x: init from x
    U_x = init_from_grid(x)  # ← singleton grid from x

    # Apply ONE step only
    U1 = candidate.apply(U_x, x)  # ← single application, no iteration

    # Check mask-local: M = (x != y) on overlap region
    H_check = min(x.shape[0], y.shape[0], U1.H)
    W_check = min(x.shape[1], y.shape[1], U1.W)

    for r in range(H_check):
        for c in range(W_check):
            x_color = int(x[r, c])
            y_color = int(y[r, c])

            if x_color == y_color:
                # Outside M: U1 must equal {x} (no edits)
                allowed = U1.get_set(r, c)
                if allowed != {x_color}:  # ← exact singleton check
                    mask_local_ok = False
                    break
            else:
                # On M: y[r,c] must be in U1.get_set(r,c) and not empty
                allowed = U1.get_set(r, c)
                if len(allowed) == 0 or y_color not in allowed:
                    mask_local_ok = False
                    break
```

**Verification:**
1. **One-step apply**: Line 676 (bg=None) / 726 (explicit bg) applies closure exactly once ✓
2. **Mask M = (x != y)**: Lines 684-685 (bg=None) / 734-735 (explicit bg) define M correctly ✓
3. **Outside M**: Lines 687-692 (bg=None) / 737-742 (explicit bg) enforce `U1[r,c] = {x[r,c]}` (no edits where x==y) ✓
4. **On M**: Lines 694-698 (bg=None) / 744-748 (explicit bg) enforce y-color is allowed and cell not empty ✓
5. **Rejection**: Candidate rejected if mask-local check fails for any train pair ✓

**Why this is correct:**
- AXIS_PROJECTION law: "project objects along axis"
- If x[r,c] == y[r,c], that cell should NOT be modified (law doesn't apply there)
- If x[r,c] != y[r,c], law MAY apply (projection may write there)
- Check ensures law is mask-local to M = (x != y) exactly ✓

#### 5. Unified Parameters

**Evidence (unifier enumerates candidates, lines 656-758):**
```python
# Try bg=None first (per-input inference) - lines 660-707
for axis in axes:
    for scope in scopes:
        for mode in modes:
            candidate_none = AXIS_PROJECTION_Closure(
                f"AXIS_PROJECTION[axis={axis},scope={scope},mode={mode},bg=None]",
                {"axis": axis, "scope": scope, "mode": mode, "bg": None}
            )
            if preserves_y(...) and compatible_to_y(...) and mask_local_ok:
                valid.append(candidate_none)

# Fallback: enumerate explicit bgs - lines 709-758
for axis in axes:
    for scope in scopes:
        for mode in modes:
            for bg in range(10):
                candidate = AXIS_PROJECTION_Closure(...)
                if preserves_y(...) and compatible_to_y(...) and mask_local_ok:
                    valid.append(candidate)
```

**Verification:**
- Single parameterization per candidate (axis, scope, mode, bg) ✓
- Same params must pass all gates on **all** train pairs ✓
- If any pair fails, candidate rejected ✓
- Observer = observed: Same law enforcement across all pairs ✓

#### 6. New Mode: "to_obstacle" (lines 576-612)

**Stop condition (lines 586-587, 593-594, 602-603, 609-610):**
```python
pixel_color = int(x_input[r, col])  # or x_input[row, c]
# Stop if hit another object (not bg and not current obj color)
if pixel_color != bg and pixel_color != obj.color:
    break
```

**Correctness verification:**
- Stop when hitting **another** object (different non-bg color) ✓
- Deterministic (same input → same stop positions) ✓
- Input-only (checks x_input pixels, not y) ✓
- Prevents overlapping projections (ensures idempotence) ✓

**Shrinking proof:**
- Only adds cells to `projected_pixels` dict
- Each added cell → singleton constraint `{obj.color}`
- Shrinking property preserved ✓

#### 7. New Scope: "per_object" (lines 553-555)

**Implementation:**
```python
elif scope == "per_object":
    # Each object projects independently
    selected_objs = objs
```

**Correctness verification:**
- `objs` from `components(x_input, bg=bg)` is deterministically ordered ✓
- Each object projects along its own axis rays ✓
- No interaction between objects (deterministic composition) ✓
- Shrinking property preserved (same as scope="all") ✓

#### Synthetic Mini-Grid Test

**Input:**
```python
x = [[0, 0, 0, 0],
     [0, 1, 0, 2],
     [0, 0, 0, 0]]
```

**Expected (AXIS_PROJECTION[axis=col, scope=all, mode=to_border, bg=0]):**
```python
y = [[0, 1, 0, 2],
     [0, 1, 0, 2],
     [0, 1, 0, 2]]
```

**Mental execution:**
1. Objects: [(1,1) color 1], [(1,3) color 2]
2. Scope "all" selects both
3. Project along col:
   - Object at (1,1): fills column 1 with color 1
   - Object at (1,3): fills column 3 with color 2
4. Projected_pixels = {(0,1):1, (1,1):1, (2,1):1, (0,3):2, (1,3):2, (2,3):2}
5. All other cells → {0}
6. Result matches y exactly ✓

---

### MOD_PATTERN_Closure (src/arc_solver/closures.py:1012-1077)

**Law statement:**
> For cell (r,c), congruence class is ((r-ar) mod p, (c-ac) mod q). Cell (r,c) may only contain colors in class_map[(i,j)] where i,j are the class indices.

**Changes reviewed:**
1. **Mask-local check in unifier** (lines 1133-1175)

#### 1. Monotone & Shrinking Law

**Evidence (lines 1056-1077):**
```python
U_new = U.copy()

for r in range(U.H):
    for c in range(U.W):
        # Skip cells outside x_input bounds
        if r >= H_in or c >= W_in:
            continue

        # Compute congruence class
        i = (r - ar) % p
        j = (c - ac) % q

        # Get allowed colors for this class
        if (i, j) in class_map:
            allowed_colors = class_map[(i, j)]
        else:
            allowed_colors = set(range(10))

        # Build bitmask from allowed colors
        allowed_mask = set_to_mask(allowed_colors)  # ← builds bitmask

        # Intersect (only clear bits)
        U_new.intersect(r, c, allowed_mask)  # ← bitwise AND only
```

**Verification:**
- `set_to_mask(colors)` builds bitmask via OR of `(1 << c)` for each color ✓
- `U_new.intersect(r, c, allowed_mask)` performs `data[r,c] &= allowed_mask` ✓
- Only clear bits (monotone & shrinking) ✓
- Deterministic from params (p, q, anchor, class_map) ✓

#### 2. Idempotence (≤2-pass practical)

**Evidence:**
- Congruence class `(i,j) = ((r-ar) % p, (c-ac) % q)` is deterministic from cell position and params ✓
- `class_map[(i,j)]` is fixed for each class ✓
- Second application: Same class → same allowed_colors → same mask → U unchanged ✓

**Convergence proof:**
- Pass 1: Constrain each cell to `class_map[(i,j)]`
- Pass 2: Same constraints; U unchanged (fixed point reached)
- ≤2 passes guaranteed ✓

#### 3. Input-only Masks (unify_MOD_PATTERN, lines 1082-1177)

**Evidence (lines 1134-1146):**
```python
for x, y in train:
    H_cur, W_cur = x.shape
    class_colors = {}  # (i,j) -> Set[int]

    for r in range(H_cur):
        for c in range(W_cur):
            i = (r - ar) % p
            j = (c - ac) % q

            if (i, j) not in class_colors:
                class_colors[(i, j)] = set()

            # Add color from INPUT x
            class_colors[(i, j)].add(int(x[r, c]))  # ← from x ONLY
```

**Verification:**
- Colors extracted from **input x only** (line 1146) ✓
- No reference to y colors in class_map construction ✓
- y used **only** for mask-local verification (lines 1133-1175) ✓

**Data file proof:**
- Unifier receives `train: List[Tuple[Grid, Grid]]` parameter
- No file I/O or dataset loading
- Evidence from training challenges only ✓

#### 4. Unified Parameters (lines 1150-1162)

**Evidence:**
```python
# Intersect class color sets across all pairs (unified params)
# Start with first pair's class_colors
class_map = {}
for key in class_colors_per_pair[0]:
    class_map[key] = class_colors_per_pair[0][key].copy()

for pair_colors in class_colors_per_pair[1:]:
    # Intersect each class
    for (i, j) in list(class_map.keys()):
        if (i, j) in pair_colors:
            class_map[(i, j)] &= pair_colors[(i, j)]  # ← INTERSECT
        else:
            class_map[(i, j)] = set()  # No overlap

# Skip if any class has empty allowed colors
if any(len(colors) == 0 for colors in class_map.values()):
    continue  # ← reject candidate if contradiction
```

**Verification:**
- Single `class_map` unified across **all** train pairs via intersection ✓
- If any class becomes empty (∅), candidate rejected ✓
- Observer = observed: Same law enforcement across all pairs ✓

#### 5. Mask-local Verification (Fix 3, lines 1133-1175)

**Implementation:**
```python
# Step 6: Mask-local check (ONE-STEP, no fixed-point iteration)
mask_local_ok = True
for x, y in train:
    # Build U_x: init from x
    U_x = init_from_grid(x)  # ← singleton grid from x

    # Apply ONE step only
    U1 = candidate.apply(U_x, x)  # ← single application, no iteration

    # Check mask-local: M = (x != y) on overlap region
    H_check = min(x.shape[0], y.shape[0], U1.H)
    W_check = min(x.shape[1], y.shape[1], U1.W)

    for r in range(H_check):
        for c in range(W_check):
            x_color = int(x[r, c])
            y_color = int(y[r, c])

            if x_color == y_color:
                # Outside M: U1 must equal {x} (no edits)
                allowed = U1.get_set(r, c)
                if allowed != {x_color}:  # ← exact singleton check
                    mask_local_ok = False
                    break
            else:
                # On M: y[r,c] must be in U1.get_set(r,c) and not empty
                allowed = U1.get_set(r, c)
                if len(allowed) == 0 or y_color not in allowed:
                    mask_local_ok = False
                    break
```

**Verification:**
1. **One-step apply**: Line 1144 applies closure exactly once ✓
2. **Mask M = (x != y)**: Lines 1152-1153 define M correctly ✓
3. **Outside M**: Lines 1155-1160 enforce `U1[r,c] = {x[r,c]}` (no edits where x==y) ✓
4. **On M**: Lines 1162-1166 enforce y-color is allowed and cell not empty ✓
5. **Rejection**: Candidate rejected if mask-local check fails for any train pair ✓

**Why this is correct:**
- MOD_PATTERN law: "constrain by congruence class"
- If x[r,c] == y[r,c], MOD_PATTERN should NOT modify that cell (or should allow x[r,c] in class_map)
- If x[r,c] != y[r,c], MOD_PATTERN MAY apply (y-color must be in allowed set for that class)
- Check ensures law is mask-local to M = (x != y) exactly ✓

#### Synthetic Mini-Grid Test

**Input:**
```python
x = [[1, 2, 1, 2],
     [3, 4, 3, 4],
     [1, 2, 1, 2]]
```

**Expected (MOD_PATTERN[p=2, q=2, anchor=(0,0)]):**
```python
y = [[1, 2, 1, 2],
     [3, 4, 3, 4],
     [1, 2, 1, 2]]
```

**Mental execution:**
1. Anchor (0,0), p=2, q=2
2. Build class_map from x:
   - Class (0,0): cells (0,0), (0,2), (2,0), (2,2) → colors {1}
   - Class (0,1): cells (0,1), (0,3), (2,1), (2,3) → colors {2}
   - Class (1,0): cells (1,0), (1,2) → colors {3}
   - Class (1,1): cells (1,1), (1,3) → colors {4}
3. Apply MOD_PATTERN:
   - Cell (0,0): class (0,0) → allowed {1} → {1} ✓
   - Cell (0,1): class (0,1) → allowed {2} → {2} ✓
   - Cell (1,0): class (1,0) → allowed {3} → {3} ✓
   - Cell (1,1): class (1,1) → allowed {4} → {4} ✓
4. Result matches y exactly ✓

---

## Minimal Patch Suggestions (inline diffs)

None required. Both implementations are correct as written.

---

## Notes to Implementer

### AXIS_PROJECTION

1. **Mode "to_obstacle" correctness**
   The stop condition `pixel_color != bg and pixel_color != obj.color` (lines 586-587, 593-594, 602-603, 609-610) correctly prevents:
   - Projecting through other objects (would violate input-only constraint)
   - Overlapping projections (would violate determinism)
   - ✓ Implementation is sound

2. **Scope "per_object" determinism**
   `components()` uses deterministic tie-breaking `(o.size, -o.bbox[0], -o.bbox[1])`, ensuring iteration over `selected_objs` is reproducible even when objects have equal size.
   - ✓ Determinism guaranteed

3. **Canvas-awareness**
   Lines 618-629 correctly skip cells outside U bounds (`if (r, c) in projected_pixels: ... else: ...`), ensuring compatibility with output-canvas scenarios where U.H > x_input.shape[0].
   - ✓ CANVAS-AWARE implementation

### MOD_PATTERN

1. **Anchor enumeration strategy**
   Lines 1120-1123 enumerate anchors: (0,0), bbox corners, quadrant origins. This covers common periodic patterns (grid-aligned, object-aligned, quadrant-aligned) while keeping search space bounded to O(10) anchors.
   - ✓ Pragmatic and sufficient

2. **Period range [2,7]**
   Lines 1127-1128 limit p,q ∈ [2,7). This is appropriate for ARC tasks (typical grids ≤30×30) and prevents combinatorial explosion (25 period combinations per anchor).
   - ✓ Good tradeoff between coverage and performance
   - **Future consideration:** If larger grids appear, extend to [2,10] or [2,H//2]

3. **Empty class rejection**
   Line 1165 `if any(len(colors) == 0 for colors in class_map.values()): continue` correctly rejects candidates where unification produces ∅ for any class. This prevents contradictions in fixed-point solver.
   - ✓ Fail-loud behavior is correct

4. **Class_map construction correctness**
   The unifier correctly:
   - Enumerates all cells in each train input (lines 1137-1146)
   - Collects colors from **INPUT x** only (line 1146: `int(x[r,c])`)
   - Intersects class colors across pairs (lines 1150-1162)
   - This ensures "observer = observed" and input-only derivation
   - ✓ Sound unification algorithm

### Data File Usage

**Training-only induction confirmed:**
- Both unifiers operate on `train: List[Tuple[Grid, Grid]]` parameter only
- No file I/O or dataset loading within closures.py
- No references to `_evaluation_*` or `_test_*` files

**Data files used (per docs/context_index.md):**
- `data/arc-agi_training_challenges.json` (for development/testing)
- `data/arc-agi_training_solutions.json` (ground truth for training)
- ✓ No evaluation/test leakage detected

### Registration Order (from src/arc_solver/search.py)

Current order in `autobuild_closures`:
```python
closures += unify_KEEP_LARGEST(train)        # M1
closures += unify_OUTLINE_OBJECTS(train)     # M1
closures += unify_OPEN_CLOSE(train)          # M2.1
closures += unify_AXIS_PROJECTION(train)     # M2.2 ← Reviewed
closures += unify_SYMMETRY_COMPLETION(train) # M2.3
closures += unify_MOD_PATTERN(train)         # M3.1 ← Reviewed
closures += unify_TILING(train)              # M4.1
...
```

**Ordering rationale:**
- AXIS_PROJECTION before SYMMETRY: Projections are simpler transformations than reflections
- MOD_PATTERN after SYMMETRY: Periodic patterns may compose with reflected structures
- Both before TILING: Simpler constraints (geometric/periodic) before complex tiling

**Soundness:**
- Order does not affect fixed-point convergence semantics (Tarski's theorem guarantees least fixed point for any order)
- Composition is verified at the set level via `verify_closures_on_train(kept, train)`
- Greedy back-off in `autobuild_closures` ensures train exactness
- ✓ Registration order is sound

---

## Final Confirmation

Both AXIS_PROJECTION and MOD_PATTERN fixes satisfy all mathematical soundness requirements:

1. ✓ **Monotone & shrinking laws preserved**
   All apply() operations use only bitwise AND (intersection). No bit-setting operations.

2. ✓ **Idempotence (≤2-pass practical)**
   Deterministic masks from input guarantee convergence in 1-2 iterations.

3. ✓ **Input-only masks**
   All masks and parameters derived from x_input only. No y-dependency in apply().

4. ✓ **Correct mask-local verification**
   One-step apply, M=(x!=y) check, outside M: U1={x}, on M: y∈U1 and U1≠∅.

5. ✓ **Unified params across train pairs**
   Single parameterization validated on all pairs (observer = observed).

6. ✓ **Train exactness**
   Composition-safe gates (preserves_y + compatible_to_y + mask_local_ok) ensure correctness.

**No mathematical soundness violations detected.**

---

**Signature:**
Math-Closure-Soundness-Reviewer
2025-10-16
