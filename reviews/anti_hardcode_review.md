# Anti-Hardcode & Implementation Review

## Verdict

**PASS**

## Blockers (must fix to submit)

None found. The implementation demonstrates excellent parametric discipline.

## High-Value Issues (should fix)

### 1. Edge pixel outline semantics (minor ambiguity)

**Location:** `src/arc_solver/closures.py:177-178`

**Issue:**
```python
if not (0 <= nr < H and 0 <= nc < W):
    is_outline = True
    break
```

The code treats out-of-bounds neighbors as triggering outline status. This means pixels at grid edges are always considered outline if they're part of an object. While reasonable, the spec (line 83-84 of context pack) says:

> "compute **outline** = pixels with any 4-neighbor equal to `bg`"

**Impact:** Interpretation difference. If objects touch the grid edge, those pixels are marked as outline even if the "out of bounds" space isn't explicitly `bg`. This is a defensible design choice (edge = implicit background) but should be documented or verified against ARC-AGI ground truth.

**Recommendation:** Add comment documenting this choice, or modify to only check in-bounds neighbors if spec requires strict interpretation.

### 2. Multiple valid parameter sets returned

**Location:** `src/arc_solver/closures.py:235-242`

**Issue:**
```python
for scope in scopes:
    for bg in bgs:
        candidate = OUTLINE_OBJECTS_Closure(...)
        if verify_closures_on_train([candidate], train):
            valid_closures.append(candidate)
```

If multiple (scope, bg) combinations pass train verification, ALL are returned. The spec says "Returns: List of OUTLINE_OBJECTS closures (usually 0 or 1)".

**Impact:** If multiple param sets work, they all get added to the closure set and applied together during fixed-point iteration. While this is idempotent (they all converge to same output), it's wasteful and potentially confusing.

**Recommendation:** Add early return after finding first valid closure, or implement canonical ordering (prefer lowest bg, then "largest" over "all") and return at most one.

## Findings (evidence)

### Parametricity: All parameters properly discovered

**Source enumeration:**
- `mode`: Fixed to "outer" for M1 (line 231) - documented in spec
- `scope`: Enumerates both ["largest", "all"] (line 232) 
- `bg`: Enumerates all 0-9 (line 233) 

**Unification across train pairs:**
- Line 241: `verify_closures_on_train([candidate], train)` checks ALL pairs
- `closure_engine.py:252-276`: Verification loops `for x, y in train:` - confirms ALL pairs tested
- Single parameter set must work for all pairs - no per-pair drift 

**Parameter usage:**
- Line 131-133: Direct dictionary access `self.params["mode"]`, `self.params["scope"]`, `self.params["bg"]`
- No `.get()` with defaults - fails loudly if params missing 
- Line 135-136: Validates mode="outer" (M1 constraint) 
- Line 155-156: Validates scope in ["largest", "all"] - fails loudly on invalid 

### Engine: Fixed-point is primary runtime, not beam

**Entry point verification:**
- `search.py:230-372`: `solve_with_closures` is the master solver
- `search.py:241`: Uses `run_fixed_point(closures, x)` - not beam
- `search.py:218`: Import shows `unify_OUTLINE_OBJECTS` registered
- `search.py:224`: Closure added to autobuild pipeline

**Fixed-point guarantee:**
- `closure_engine.py:204-246`: `run_fixed_point` iterates until `U == U_prev`
- Line 225-232: Applies all closures in sequence each iteration
- Tarski convergence guaranteed by monotone/shrinking properties

**Beam relegated to legacy:**
- `search.py:53-104`: `beam_search` exists but not used in closure path
- `search.py:107-206`: `solve_with_beam` is separate function, not called by `solve_with_closures`

### Determinism: All sources stable

**Tie-breaking:**
- Line 152: `max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))` - deterministic largest selection 
- Matches KEEP_LARGEST pattern (closures.py:66) - consistent approach 

**Iteration order:**
- Line 174: 4-neighbor iteration `[(1, 0), (-1, 0), (0, 1), (0, -1)]` - fixed order 
- Line 165-169: Component iteration via `for obj in selected_objs` - deterministic (list order)
- Line 172-185: Pixel iteration via `for r, c in obj.pixels` - deterministic (from BFS in utils.py:53-86)

**No non-deterministic sources:**
- No `random` module usage 
- No `time.time()` or clock used for logic 
- No unordered dict iteration affecting outputs 
- No network or I/O during closure application 

**RNG seeding:**
- Not applicable - no randomness used 

### Monotonicity & Shrinking: Only removes possibilities

**Intersection-only operations:**
- Line 188: `U_new = U.copy()` - starts from current state
- Line 189: `bg_mask = color_to_mask(bg)` - builds constraint mask
- Line 197: `U_new.intersect(r, c, obj_mask)` - shrinks to object color 
- Line 200: `U_new.intersect(r, c, bg_mask)` - shrinks to background 
- Line 203: `U_new.intersect(r, c, bg_mask)` - shrinks to background 

**No bit additions:**
- `SetValuedGrid.intersect` (closure_engine.py:60-62): `self.data[r, c] &= mask` - bitwise AND only 
- No operations that expand possibility sets 

**Test coverage:**
- `test_closures_minimal.py:495-523`: `test_shrinking_outline` verifies F(U) † U
- Line 513: Confirms subset property
- Line 522: Confirms bit count decreases

### Idempotence: Stabilizes in d2 passes

**Test verification:**
- `test_closures_minimal.py:526-550`: `test_idempotence_outline` verifies F(F(U)) = F(U)
- Line 548: Confirms second application doesn't change result 

**Convergence speed:**
- `test_closures_minimal.py:633-648`: `test_convergence_outline` verifies d2 iterations
- Line 647: Asserts `stats["iters"] <= 2` 

### Train Exactness: All pairs verified, no shortcuts

**Unifier discipline:**
- Line 235-242: Enumerates all candidates, verifies each against train
- Line 241: `verify_closures_on_train([candidate], train)` - checks ALL pairs
- No early return on first pair - must pass all 

**Verification implementation:**
- `closure_engine.py:264`: `for x, y in train:` - loops all pairs 
- Line 268-269: Checks fully determined (no multi-valued cells)
- Line 272-274: Checks exact equality `np.array_equal(y_pred, y)` 
- Returns False on ANY pair failure - no approximations 

**Test coverage:**
- `test_closures_minimal.py:585-631`: `test_train_exactness_outline_multiple` with 2 pairs
- Line 617: `train = [(x1, y1), (x2, y2)]` - multiple pairs
- Line 625-630: Manually verifies each pair after unification 

## Minimal Patch Suggestions (inline diffs)

### Optional: Document edge handling

```diff
# src/arc_solver/closures.py
@@ -173,6 +173,8 @@
             for r, c in obj.pixels:
                 is_outline = False
                 for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
+                    # Note: Out-of-bounds neighbors treated as background
+                    # (edge pixels are outline if object touches grid boundary)
                     nr, nc = r + dr, c + dc
                     if not (0 <= nr < H and 0 <= nc < W):
```

### Optional: Return at most one closure

```diff
# src/arc_solver/closures.py
@@ -228,6 +228,8 @@
     valid_closures = []

     # Enumerate parameters
+    # Note: Prefer canonical ordering (lowest bg first, then "largest" before "all")
+    # to ensure at most one closure returned when multiple params work
     mode = "outer"  # Fixed for M1
     scopes = ["largest", "all"]
     bgs = range(10)
@@ -240,6 +242,8 @@
             )
             if verify_closures_on_train([candidate], train):
                 valid_closures.append(candidate)
+                # Early return on first valid (single param set per family)
+                return valid_closures

     return valid_closures
```

## Recheck Guidance

After applying optional fixes:

1. **Re-run determinism check:**
   ```bash
   bash /Users/ravishq/code/arc-agi-1/scripts/determinism.sh /Users/ravishq/code/arc-agi-1/data
   ```

2. **Re-run tests:**
   ```bash
   python /Users/ravishq/code/arc-agi-1/tests/test_closures_minimal.py
   ```

3. **Verify single closure return:**
   - Check receipts.jsonl to confirm at most one OUTLINE_OBJECTS closure per task
   - Verify no performance degradation from multiple closure applications

4. **Smoke test on eval set:**
   ```bash
   python /Users/ravishq/code/arc-agi-1/scripts/run_public.py \
       --arc_public_dir /Users/ravishq/code/arc-agi-1/data \
       --out /Users/ravishq/code/arc-agi-1/runs/20251015_review
   ```

---

## Summary

**Parametricity Score: 9.5/10**

This is an exemplary implementation of parametric discipline:

-  All parameters enumerated (no hard-coded values)
-  Unified across train pairs (single param set)
-  Fails loudly on missing params (no silent defaults)
-  Fixed-point runtime (not beam)
-  Deterministic (stable tie-breaks, fixed iteration order)
-  Verifies ALL train pairs (no shortcuts)
-  Monotone/shrinking (intersection-only)
-  Comprehensive test coverage

**Deductions:**
- -0.3: Edge handling not explicitly documented (minor ambiguity)
- -0.2: Multiple param sets may be returned (wasteful but not incorrect)

**Recommendation:** Approve for submission. Optional patches improve clarity but are not required for correctness.
