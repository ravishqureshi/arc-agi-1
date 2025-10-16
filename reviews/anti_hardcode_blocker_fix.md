# Anti-Hardcode & Implementation Review

## Verdict

**PASS**

## Blockers (must fix to submit)

None. All critical parametricity violations have been fixed.

## High-Value Issues (should fix)

None identified. The implementation is submission-ready.

## Findings (evidence)

### Parametricity: PASS

**CANVAS_SIZE closure params (receipts analysis):**
- 731 CANVAS_SIZE closures across all tasks
- ALL use parametric multipliers only: `{"k_h": int, "k_w": int, "strategy": "TILE_MULTIPLE"}`
- ZERO closures contain hard-coded H or W dimensions
- Same-shape tasks use `k_h=1, k_w=1` (not hard-coded dimensions)
- Shape-changing example: `k_h=3, k_w=3` for 2×2→6×6 tasks

**Parameter unification (src/arc_solver/closures.py:1493-1540):**
```python
# Lines 1528-1534: Only k_h, k_w stored
if all(m is not None for m in multipliers) and len(set(multipliers)) == 1:
    k_h, k_w = multipliers[0]
    candidate = CANVAS_SIZE_Closure(
        f"CANVAS_SIZE[strategy=TILE_MULTIPLE,k_h={k_h},k_w={k_w}]",
        {"strategy": "TILE_MULTIPLE", "k_h": k_h, "k_w": k_w}  # ✓ No H, W
    )
```

**Runtime canvas computation (src/arc_solver/closure_engine.py:351-372):**
```python
def _compute_canvas(x_input: Grid, canvas_params: Dict) -> Dict:
    if strategy == "TILE_MULTIPLE":
        k_h = canvas_params["k_h"]
        k_w = canvas_params["k_w"]
        H_in, W_in = x_input.shape
        return {"H": k_h * H_in, "W": k_w * W_in}  # ✓ Per-input computation
```

**Proof of generalization:**
- Training input 2×2 with k_h=3, k_w=3 → output 6×6
- Test input 4×4 with same k_h=3, k_w=3 → output 12×12 (correct generalization)
- No hard-coded dimensions in params; multipliers apply to ANY input shape

### Engine: PASS (fixed-point is primary)

**Fixed-point engine (src/arc_solver/closure_engine.py:205-253):**
- `run_fixed_point()` is the primary runtime
- Guaranteed convergence via Tarski theorem (monotone + shrinking)
- Max iterations: 100 (safety net)

**CANVAS_SIZE as meta-closure:**
- `is_meta=True` set in constructor (line 1482)
- Always kept first in composition (src/arc_solver/search.py:236-287)
- Identity apply (returns U unchanged) - only provides metadata

**Beam search separation:**
- Beam search in `search.py:53-104` is separate legacy runtime
- NOT used for closure-based solver (`solve_with_closures`)
- No cross-contamination between engines

### Determinism: PASS

**No random sources:**
- No `random`, `Random`, `np.random`, or `shuffle` usage in closures
- Only comment mentioning RNG is in docstring (line 921): "no RNG, no I/O, no wall-clock"

**Stable iteration order:**
1. MOD_PATTERN anchor enumeration uses `sorted(anchors)` (closures.py:1024)
2. Deterministic tie-breaking in all max() operations:
   - Largest component: `max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))` (line 109)
   - Background inference: `min(counts.keys(), key=lambda c: (-counts[c], c))` (line 46)
3. Dict iteration deterministic where order affects output (sorted or stable key order)

**Receipt determinism check:**
- Timing variations (43ms vs 42ms) are acceptable (OS scheduling)
- ALL closures and predictions byte-identical across runs
- No nondeterministic parameter drift detected

### Data Leakage: PASS

**Source code audit:**
- ZERO references to `arc-agi_evaluation` or `arc-agi_test` in src/arc_solver/
- All unifiers use only training data (data/arc-agi_training_challenges.json)
- No test output peeking in parameter induction

**Data-use compliance:**
- Allowed: data/arc-agi_training_challenges.json ✓
- Forbidden: data/arc-agi_evaluation_*, data/arc-agi_test_* ✓
- Submission schema: {task.json: [grid...]} with ints 0-9 ✓

## Minimal Patch Suggestions (inline diffs)

No patches required. Implementation is correct.

## Recheck Guidance

### Post-fix verification checklist:

1. **Parametricity recheck:**
   ```bash
   # Verify no H or W in CANVAS_SIZE params
   jq -r 'select(.closures != null) | .closures[] |
          select(.name | startswith("CANVAS_SIZE")) |
          .params | keys | join(",")' runs/blocker_fix/receipts.jsonl |
          sort | uniq
   # Expected: k_h,k_w,strategy (only)
   ```

2. **Data leakage recheck:**
   ```bash
   # Verify no eval/test data references
   grep -r "arc-agi_evaluation\|arc-agi_test" src/arc_solver/
   # Expected: (no matches)
   ```

3. **Determinism recheck:**
   ```bash
   # Run determinism check
   bash scripts/determinism.sh data/arc-agi_training_challenges.json
   # Expected: PASS (ignore timing_ms diffs)
   ```

4. **Coverage sanity check:**
   ```bash
   # Verify CANVAS_SIZE presence
   grep -c "CANVAS_SIZE" runs/blocker_fix/receipts.jsonl
   # Expected: ~700+ (73.1% observed)

   # Verify solved tasks
   grep -c '"status":"solved"' runs/blocker_fix/receipts.jsonl
   # Expected: 10+ (1.0% baseline maintained)
   ```

### Future architectural notes:

1. **Greedy composition issue (not a blocker):**
   - 721/1000 tasks have CANVAS_SIZE alone (meta-only, no constraining closures)
   - `verify_consistent_on_train` may be too restrictive
   - Separate fix needed to improve composition retention

2. **Non-integer shape strategies (future milestone):**
   - Tasks with non-integer scaling (e.g., 3×3→5×7) return no closure
   - Future: add crop/pad strategies beyond TILE_MULTIPLE

3. **SAME and MAX_TRAIN strategies removed (by design):**
   - These stored absolute H,W dimensions from training (parametricity violation)
   - Correctly replaced by TILE_MULTIPLE with k_h=k_w=1 for same-shape tasks

---

## Conclusion

All critical parametricity violations have been **FIXED**. The CANVAS_SIZE implementation:

1. **Stores only parametric multipliers** (k_h, k_w) - no hard-coded dimensions
2. **Computes canvas at runtime** per-input (generalizes to test)
3. **Uses fixed-point engine** as primary runtime (deterministic, sound)
4. **No data leakage** (training-only induction)
5. **Fully deterministic** (stable tie-breaking, no RNG)

**Coverage:** 10/1000 (1.0% baseline maintained)
**Parametricity:** 0/731 CANVAS_SIZE closures contain hard-coded H or W
**Engine:** Fixed-point (not beam) is runtime for closures
**Determinism:** Byte-identical predictions (timing variations acceptable)

**Status:** READY FOR SUBMISSION

---

**Reviewed by:** Claude (Anti-Hardcode Reviewer)
**Review Date:** 2025-10-16
**Codebase State:** `/Users/ravishq/code/arc-agi-1` (main branch, blocker fixes applied)
