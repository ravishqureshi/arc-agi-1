# Closure Law Checklist - Math Correctness Review

**Date**: 2025-10-15
**Reviewer**: Claude (Sonnet 4.5)

## Format
`name | one-line law | monotone? | idempotent? | unifier exact? | mask input-only? | edge cases tried | verdict`

---

## B0: Fixed-Point Engine

**Component**: `SetValuedGrid` + `run_fixed_point()`

| Property | Law/Spec | Verified | Evidence |
|----------|----------|----------|----------|
| Lattice order | `U ⊆ V ⇔ ∀(r,c): U.data[r,c] ⊆ V.data[r,c]` | ✓ | Bitwise subset via uint16 masks |
| Top element | `⊤ = 0x3FF (all colors {0-9})` | ✓ | `init_top(H,W)` line 150 |
| Intersect operation | `U.data[r,c] &= mask → U' ⊆ U` | ✓ | AND never adds bits (line 58) |
| Convergence check | `U_{n+1} = F(U_n) until U == U_prev` | ✓ | `np.array_equal()` (line 224) |
| Max iterations | Safety bound = 100 | ✓ | Pragmatic (never reached in practice) |
| Deterministic order | Closures applied in list order | ✓ | Line 220-221 |

**Verdict**: ✓ PASS - Engine is mathematically sound

---

## B1: KEEP_LARGEST_COMPONENT

| Property | Status | Evidence |
|----------|--------|----------|
| **One-line law** | `g(r,c) = x(r,c) if (r,c) ∈ largest_component(x, bg) else bg` | - |
| **Monotone?** | ✓ YES | Masks computed from `x_input` only (lines 52-68); independent of U |
| **Shrinking?** | ✓ YES | Only `intersect()` calls (lines 78, 83); no bit additions |
| **Idempotent?** | ✓ YES | Deterministic `components()` + AND idempotence: `(a & b) & b = a & b` |
| **Unifier exact?** | ✓ YES | Single `{bg}` param (line 98); ALL pairs tested (line 103) |
| **Mask input-only?** | ✓ YES | Masks: `color_to_mask(x[r,c])`, `color_to_mask(bg)`; no U or y dependence |
| **Edge cases tried** | ✓ YES | See below |

### Edge Cases Examined

1. **Empty grid (no components)**
   - Handler: lines 55-62
   - Behavior: All cells → {bg}
   - Verdict: ✓ Correct

2. **Single component**
   - Behavior: All non-bg cells keep input color
   - Verdict: ✓ Correct (largest = only)

3. **Multiple equal-size components**
   - Tie-break: `max()` selects first (deterministic)
   - Idempotence: ✓ Preserved (same largest each time)

4. **U with cells already restricted**
   - Behavior: `intersect()` further restricts
   - Verdict: ✓ Monotone shrinking preserved

5. **Empty cell (mask=0)**
   - Detection: `to_grid()` returns None (line 82)
   - Verdict: ✓ Correct (signals unsolvable)

**Verdict**: ✓ PASS - Closure is mathematically exact

---

## Future Closures (Template)

### OUTLINE (B2 - planned)

| Property | Status | Notes |
|----------|--------|-------|
| **One-line law** | `g(r,c) = c_fg if (r,c) on boundary(obj, bg) else bg` | - |
| **Monotone?** | TODO | Check mask computation |
| **Shrinking?** | TODO | Verify only intersect() used |
| **Idempotent?** | TODO | Boundary detection deterministic? |
| **Unifier exact?** | TODO | Single params, ALL pairs? |
| **Mask input-only?** | TODO | No U or y dependence? |
| **Edge cases** | TODO | Empty objs, 1-pixel objs, nested objs |

---

### SYMMETRY_COMPLETION (planned)

| Property | Status | Notes |
|----------|--------|-------|
| **One-line law** | `g(r,c) = x(R(r,c)) where R ∈ {ROT90, FLIP_H, FLIP_V}` | - |
| **Monotone?** | TODO | Reflection deterministic from x? |
| **Shrinking?** | TODO | Intersect with reflected values? |
| **Idempotent?** | TODO | R(R(U)) = U for reflections? |
| **Unifier exact?** | TODO | Same R for ALL train pairs? |
| **Mask input-only?** | TODO | Reflection mapping from x only? |

---

### TILING_ON_MASK (planned)

| Property | Status | Notes |
|----------|--------|-------|
| **One-line law** | `g(r,c) = motif[r % h, c % w] if (r,c) ∈ mask else x(r,c)` | - |
| **Monotone?** | TODO | Motif and mask from x? |
| **Shrinking?** | TODO | Only intersect on mask pixels? |
| **Idempotent?** | TODO | Same tiling each pass? |
| **Unifier exact?** | TODO | Same motif+mask for ALL pairs? |
| **Mask input-only?** | TODO | Mask = x ≠ y (derived from train only)? |

---

## Review Checklist for New Closures

Before approving any new closure T_i, verify:

1. [ ] **Law stated in one line** (executable math, not English)
2. [ ] **Monotonicity proof**: Masks independent of U's state?
3. [ ] **Shrinking proof**: Only bitwise AND / set intersection?
4. [ ] **Idempotence proof**: Same masks on 2nd application?
5. [ ] **Unifier exactness**: ONE param set, tested on ALL train pairs?
6. [ ] **Train verification**: `verify_closures_on_train()` called?
7. [ ] **Singleton check**: `is_fully_determined()` required?
8. [ ] **Exact equality**: `np.array_equal(y_pred, y)` used?
9. [ ] **Edge cases**: Empty grids, single object, equal-size objects?
10. [ ] **No heuristics**: No approximate equality, no per-pair params?

---

## Pass/Fail Criteria

### PASS ✓
- Law is stated as exact equality (not heuristic)
- Monotone: masks from input only
- Shrinking: only AND/intersect operations
- Idempotent: deterministic mask computation
- Unifier returns [] if ANY train pair fails
- Train exactness: singletons + exact match on ALL pairs

### FAIL ✗
- Law is approximate or heuristic
- Masks depend on U's current state
- Any operation adds bits (OR, union, color creation)
- Non-deterministic behavior (random, hash-order dependent)
- Per-pair parameters (violates observer = observed)
- Partial train solutions accepted

---

## Summary

**B0 Engine**: ✓ PASS (6/6 properties verified)
**B1 KEEP_LARGEST**: ✓ PASS (6/6 properties verified, 5/5 edge cases handled)

**Next**: Implement B2+ closures with same rigor; use this checklist for each.
