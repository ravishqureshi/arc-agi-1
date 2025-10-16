# Anti-Hardcode & Implementation Review

## Verdict

**PASS**

## Blockers (must fix to submit)

None.

## High-Value Issues (should fix)

None.

## Findings (evidence)

### 1. Parametricity: where parameters come from, and proof of unification

**Fix 1: unify_TILING (lines 1256-1444)**
- Canvas params computed ONCE at start (lines 1281-1282) via `unify_CANVAS_SIZE(train)` - no longer per-candidate
- Anchors derived from input geometry only: (0,0), bbox corners from x0, quadrant origins from x0 (lines 1289-1311)
  - `H_in, W_in = x0.shape` used to derive quadrants, NOT `H_out, W_out` (line 1287 reads y0 but never uses it in logic)
- Motif size bounded by parametric constant `MAX_MOTIF_SIZE = 6` (line 1315) - not training-specific
- Motif colors derived from INPUT x0 only:
  - Global mode: `motif[mr, mc] = int(x0[r_in, c_in])` (line 1347) - wraps around x0 with modulo
  - Mask mode: `colors_on_mask` collected from x0 (line 1358), mode color computed with deterministic tie-breaking (line 1366)
- Mask-local check (lines 1392-1436):
  - Canvas computed per-input from x (line 1395) - deterministic from input
  - ONE-STEP apply (line 1408) - no fixed-point iteration in unifier (critical fix)
  - Overlap region: `H_check = min(x.shape[0], y.shape[0], U1.H)` (line 1411) - shape-aware
  - Verification uses residual mask M=(x!=y) on overlap only (lines 1414-1430)
- **Proof of unification**: All params (anchor, motif, mask_template) are the same across all train pairs. Each candidate is validated with `preserves_y + compatible_to_y` gates (line 1388) then mask-local check across ALL train pairs (lines 1393-1436). Unified params stored in single closure instance.

**Fix 2: unify_COLOR_PERM (lines 1497-1548)**
- Shape guard removed - now operates on overlap region only (lines 1518-1519)
- Color map built from ALL train pairs via intersection: `if color_map[x_c] != y_c: return []` (lines 1526-1529) - contradiction forces early exit
- Bijective constraint: `len(set(used_y_colors)) != len(used_y_colors)` (lines 1534-1535) ensures no color collisions
- **Proof of unification**: Single `color_map` dict unified across all train pairs. If any pair has conflicting mapping, returns empty list (line 1529). Bijective check ensures permutation property holds globally.

**Fix 3: unify_RECOLOR_ON_MASK (lines 1746-1869)**
- Shape guard removed - evaluates on overlap region (lines 1806-1807, 1849-1850)
- Template enumeration is parametric:
  - PARITY: anchor2 in [(0,0), (0,1), (1,0), (1,1)] (lines 1773-1774) - exhaustive 2x2 grid
  - STRIPES: axis in ["row", "col"], k in [2, 3] (lines 1776-1778) - small parametric space
  - BORDER_RING: k=1 (line 1780) - fixed width parameter
- Template masks computed from INPUT x only (lines 1800, 1839) via `_compute_template_mask(x, ...)`
- Residual mask check on overlap: `T_overlap = T[:H_overlap, :W_overlap]` (line 1814) compares against `M = (x != y)` (line 1811)
- Strategy verification: target_color derived from x+T (lines 1843, 1587-1597), checked against y on overlap (lines 1851-1858)
- **Proof of unification**: Template and strategy must match ALL train pairs. First loop checks template matches residual M for all pairs (lines 1791-1818), second loop checks strategy produces correct colors for all pairs (lines 1831-1861). Both must pass before candidate is added.

**Fix 4: _compute_template_mask (lines 1618-1707)**
- All templates derived from input x shape (H, W = x.shape, line 1633):
  - PARITY: `(r + c) % 2 == (ar2 + ac2) % 2` (lines 1664-1665) - pure modulo arithmetic from anchor2 param
  - STRIPES: `(r // k) % 2 == 0` or `(c // k) % 2 == 0` (lines 1677-1684) - pure parametric formula
  - BORDER_RING: `min(r, H - 1 - r, c, W - 1 - c) < k` (line 1700) - distance to edge, parametric in k
- No training-specific dimensions hardcoded
- No output y used - all masks computed from x only

### 2. Engine: show fixed-point path is the runtime, not beam

Evidence from code:
- **unify_TILING**: Removed `run_fixed_point()` call from candidate loop (line 1408 now does ONE-STEP apply only)
- **Unifiers only validate, don't solve**: All unifiers use composition-safe gates (`preserves_y + compatible_to_y`) to check candidates, not fixed-point iteration
- **Runtime path**: Fixed-point engine (`closure_engine.py`) applies closures iteratively until convergence. Unifiers return validated closure instances; engine runs them.
- **No beam at runtime**: None of the reviewed functions call `beam_search` or `solve_with_beam`

### 3. Determinism: seeds, stable iteration order, no nondeterministic sources

**Deterministic sources:**
- Tie-breaking in motif color selection: `min(counts.keys(), key=lambda c: (-counts[c], c))` (line 1366) - highest count, then lowest color
- Anchor sorting: `anchors = sorted(anchors)[:8]` (line 1311) - deterministic sort before limiting
- Template/strategy enumeration: Fixed order lists (lines 1767-1780, 1322-1331) - same order every run
- Overlap region computation: `min(x.shape[0], y.shape[0])` (lines 1518, 1806) - deterministic from input shapes
- Component selection: `max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))` (lines 1212, 1415, 1648) - deterministic tie-breaking by bbox position

**No nondeterministic sources:**
- No RNG calls (no `random.` or `np.random.`)
- No I/O during parameter induction (only reads train data passed as args)
- No clock/timestamp dependencies
- No network calls
- Dict iteration: All critical dict iterations are over `dict.keys()` or `dict.items()` where order doesn't affect correctness (e.g., checking all classes in class_map)

**Stable iteration:**
- Sorted anchors before enumeration (line 1311)
- Fixed-order template/strategy lists (lines 1767-1780)
- Deterministic mode color selection with tie-breaking (line 1366)

## Minimal Patch Suggestions (inline diffs)

None required. All changes are clean.

## Recheck Guidance

After any future modifications to these functions, re-confirm:

1. **Parametricity check:**
   - Grep for hardcoded shapes: `grep -n "shape\[0\].*[0-9]" closures.py` should only show comparisons, not assignments
   - Verify motif derivation uses only x0 (INPUT): `grep -n "motif.*y0" closures.py` should return nothing
   - Check anchor computation: `grep -n "anchors.*train\[" closures.py` should only use `train[0]` for geometry, not all pairs

2. **Fixed-point runtime check:**
   - Verify no `run_fixed_point()` in unifiers: `grep -n "run_fixed_point.*unify_" closures.py` should return nothing
   - Confirm ONE-STEP apply in mask-local check: verify line 1408 calls `candidate.apply()` exactly once per pair

3. **Determinism check:**
   - Verify tie-breaking in all `min()`/`max()` calls: `grep -n "min(.*key=lambda" closures.py` should show composite keys like `(-counts[c], c)`
   - Check for RNG: `grep -n "random\." closures.py` should return nothing
   - Verify sorted anchors: confirm line 1311 has `sorted()` before slicing

4. **Shape-awareness check:**
   - Verify overlap region usage: `grep -n "H_overlap.*min(" closures.py` should show all unifiers computing overlap
   - Check no absolute shape assumptions: `grep -n "shape.*==" closures.py` should only show dynamic comparisons, not literals like `== 30`
