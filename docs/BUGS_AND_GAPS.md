# Bugs and Gaps Tracking

**Purpose:** Master list of known bugs, implementation gaps, and optimization opportunities discovered during ARC-AGI solver development.

**Philosophy:** Zero compromises, pure mathematics approach. Every issue is tracked, assessed, and either fixed or marked as not needed with clear rationale.

**Status Legend:**
- `[ ]` **Open** - Known issue, not yet addressed
- `[!]` **Critical** - Blocks progress or causes incorrect results
- `[x]` **Fixed** - Resolved and verified
- `[~]` **Not Needed** - Assessed and determined unnecessary
- `[?]` **Investigating** - Root cause being analyzed

---

## Critical Bugs

### [x] Per-Color Recolor Logic Bug
**Discovered:** 2025-10-13
**Location:** `src/arc_solver/core/induction.py:90-149` (induce_recolor_obj_rank)
**Status:** FIXED 2025-10-13

**Problem:**
The per-color recolor induction routine had flawed logic. It tried to find a single uniform target color across all masked regions, but per-color recolor means:
- Each source color's largest object gets recolored to a (potentially different) target color
- Example: Color 1's largest → 7, Color 2's largest → 8

**Root Cause:**
```python
# Old logic (WRONG):
target_color = None  # Single target color
for x, y in train:
    vals = y[m]  # m includes objects from ALL colors
    c = most_common(vals)  # Tries to find uniform color
    if target_color != c: return None  # Fails if colors differ
```

**Fix:**
Build a per-color mapping `{src_color: tgt_color}` instead of single target.

**Verification:**
- Manual test case passes: `[[1,1,0,2,2],[1,0,0,2,0]]` → `[[7,7,0,8,8],[7,0,0,8,0]]`
- Full test sweep on 1,076 tasks
- PCE generation updated

**Impact:** Affects 5-10% of tasks using per-color object transformations.

### [x] COPY Operation Identity Transform Bug
**Discovered:** 2025-10-13
**Location:** `src/arc_solver/core/induction.py:302-400` (induce_move_or_copy_obj_rank)
**Status:** FIXED 2025-10-13

**Problem:**
When learning COPY operations, the induction routine would find delta=(0,0) first (identity transform) instead of the actual copy delta. This happened because COPY operations have both source and target objects in the output grid, and the algorithm checked the source object first.

**Example:**
```python
# Input:  [[3,3,0,0,0,0],
#          [3,3,0,0,0,0]]
# Output: [[3,3,0,3,3,0],  # Source at (0,0) AND copy at (0,3)
#          [3,3,0,3,3,0]]
# Algorithm found delta=(0,0) from source instead of delta=(0,3) from copy
```

**Root Cause:**
```python
for j in cand:  # Candidates include both source and target objects
    dx = centroid_diff(Yobjs[j], Xobjs[xi])
    if match: found = (dx, dy); break  # First match wins, which is (0,0)
```

**Fix:**
Skip delta=(0,0) since identity is not a MOVE/COPY operation:
```python
if (dx, dy) == (0, 0):
    continue  # Skip identity transform
```

**Verification:**
- Unit test `test_copy_obj_rank()` passes
- Full test sweep: +2 test outputs (task 25ff71a9)

**Impact:** Critical for COPY pattern recognition. Without fix, 0/2 COPY tasks would pass.

### [x] MOVE/COPY Shape Mismatch Crash
**Discovered:** 2025-10-13
**Location:** `src/arc_solver/core/induction.py:283` (shape_mask helper)
**Status:** FIXED 2025-10-13

**Problem:**
The induction routine crashed with `IndexError: index 2 is out of bounds for axis 1 with size 2` when train pairs had different shapes (e.g., X is 3×3, Y is 3×2).

**Root Cause:**
```python
def shape_mask(objs, idx, H, W):
    m = np.zeros((H, W), dtype=bool)
    for (r, c) in objs[idx].pixels:
        m[r, c] = True  # Crash if object pixels exceed grid bounds
```

Used X's shape `(H, W)` to create mask but tried to index Y's object pixels that could be outside bounds.

**Fix:**
Early shape check - MOVE/COPY operations require same input/output shape:
```python
for x, y in train:
    if x.shape != y.shape:
        return None  # MOVE/COPY not applicable
```

**Verification:**
- Full test sweep completes without crashes (1,076/1,076 tasks)
- Test coverage script runs successfully

**Impact:** Critical - prevented test suite from running. Fixed in first test run.

---

## Implementation Gaps

### [ ] Remaining Legacy Rules (from arc_demo.py)

**Status:** 3/5 legacy rules migrated

**Completed:**
- [x] Color permutation (added 2025-10-13, +3 tasks)
- [x] Component rank recolor (added 2025-10-13, +1 task after per-color fix)
- [x] Move/copy objects (added 2025-10-13, +1 task)

**Remaining:**
- [ ] Move bbox to origin
- [ ] Mirror left to right

**Assessment:** Low priority - these are simple patterns covered in IMPLEMENTATION_PLAN.md Phase 1. Expected total impact: +1-2 tasks (1.2-1.3% accuracy).

**Current Status:** 12/~14 tasks solved (1.1%). Nearly at parity with legacy demo (1.2%).

---

## Optimization Opportunities

### [ ] Induction Catalog Performance

**Current:** Try 8 induction routines sequentially on every task
**Cost:** ~0.5s per task × 1,000 tasks = 500s total
**Opportunity:** Most tasks fail fast (no rule matches), but we still try all 8

**Potential optimizations:**
1. Early exit if train pairs have shape mismatch (60% of tasks)
2. Cache connected_components results (computed 4× per task)
3. Parallel induction (try all rules concurrently)

**Expected gain:** 2-3× speedup (500s → 150-200s)
**Priority:** Low (not blocking accuracy improvements)

---

## Phase 1 Blockers

_None currently identified_

---

## Design Decisions Log

### Why Object-Rank Didn't Improve Coverage (2025-10-13)

**Result:** Added object-rank infrastructure (9 files changed, +200 LOC), but 0 new tasks solved.

**Root Causes:**
1. **Pattern rarity** - Simple "recolor largest object" tasks are rare in ARC-1/2
2. **Insufficient variations** - Only tried 4 patterns (rank-0 global/per-color, top-1/2)
3. **Need compositions** - Real tasks combine object-rank with other ops (SHIFT, TILE, etc.)

**Key Learning:** Infrastructure ≠ immediate coverage gains. Object-rank is foundational but needs Phase 1 operators to be effective.

**Decision:** Continue with Grok's path (remaining legacy rules), then jump to Phase 1 high-impact operators.

---

## Test Failures Analysis

### Systematic Failure Patterns (from TEST_COVERAGE.md)

**Pattern 1: Shape mismatch (600+ tasks)**
- Input shape ≠ Output shape
- Requires: RESIZE, CROP, PASTE, PAD operators

**Pattern 2: No rule matched (450+ tasks)**
- Training pairs don't fit any current induction routine
- Requires: More operators in Phase 1

**Pattern 3: ARC-2 complete failure (0/669)**
- All ARC-2 tests fail
- Requires: Multi-step compositions, in-context definition operators

---

### MOVE/COPY Object Migration Success (2025-10-13)

**Result:** Added MOVE_OBJ_RANK and COPY_OBJ_RANK operators, +1 task solved (25ff71a9, 2 test outputs).

**What worked:**
1. **Careful migration** - Followed Grok's reference implementation, adapted to modular architecture
2. **Shape-aware** - Added early shape check (MOVE/COPY requires same input/output shape)
3. **Identity handling** - Skipped delta=(0,0) to avoid false positives on COPY patterns
4. **Comprehensive testing** - Unit tests + full sweep caught both bugs immediately

**Bugs found and fixed:**
1. Identity transform bug (delta=(0,0) matched first in COPY)
2. Shape mismatch crash (IndexError when X and Y differ in shape)

**Architecture changes:**
- Added to `spatial.py`: MOVE_OBJ_RANK, COPY_OBJ_RANK, in_bounds helper
- Added to `induction.py`: induce_move_or_copy_obj_rank, shape_mask, shift_mask helpers
- Updated `solver.py`: PCE generation for MOVE/COPY rules
- Added `test_arc_solver.py`: 2 new unit tests
- Created `scripts/generate_test_coverage.py`: Automated test sweep script

**Impact:**
- Accuracy: 0.9% → 1.1% (+0.2%)
- Tasks: 10 → 12 (+2 test outputs from 1 task)
- Rules: 8 → 10 variations (+2)
- **Reached parity with legacy demo (12 tasks)**

**Learnings:**
- Centroid-based matching works well for MOVE/COPY patterns
- Shape consistency is critical for spatial transformations
- Identity transform (delta=0) needs explicit handling
- Full test sweep (1,000 tasks) takes ~90 seconds

---

### Beam Search Migration Success (2025-10-13)

**Result:** Added beam search for multi-step compositions. Infrastructure complete, 0 new tasks solved (expected).

**What worked:**
1. **Refactored induction** - Changed from `Optional[Rule]` to `List[Rule]` for beam compatibility
2. **Added PCE field** - All rules now have Proof-Carrying English explanations
3. **Universal primitives** - Generate ROT, FLIP, CROP, KEEP regardless of train match
4. **Beam search** - Composese primitives up to depth K, prunes on residual increase
5. **Comprehensive tests** - 3 beam-specific unit tests, all passing

**Architecture changes:**
- Added to `induction.py`: 7 new beam-compatible induction functions (+350 LOC)
- Added to `solver.py`: Node, compose, beam_search, solve_with_beam (+160 LOC)
- Updated `Rule` dataclass: added `pce` field
- Added 3 unit tests to `test_arc_solver.py`

**Impact:**
- Accuracy: 12/1076 → 12/1076 (no change, expected)
- Tasks: Same 12 tasks
- Multi-step: ROT90 → CROP composition verified in tests
- **Infrastructure ready** for Phase 1 operators

**Why no improvement:**
- Only 10 primitive operations available
- Multi-step compositions like "ROT90 → CROP" are rare in ARC
- Real gains will come when we add Phase 1 operators (SHIFT, TILE, PASTE, etc.)
- Beam search is FOUNDATIONAL - will be critical when operator catalog grows

**Learnings:**
- Beam search works correctly (tests pass, compositions verified)
- Universal primitive generation is key for multi-step search
- Need 60-80 operators (Phase 1) before beam search shows coverage gains
- Test on first 50 tasks: single-step 1/50, beam 1/50 (0 beam-only wins)

**Next:** Add remaining legacy rules (2), then jump to Phase 1 high-impact operators.

---

### TILE and DRAW_LINE Operators Success (2025-10-13)

**Result:** Added TILE and DRAW_LINE operators with induction routines, +1 task solved (a416b8f3).

**What worked:**
1. **Empirical discovery** - Used pattern discovery script to identify tiling/drawing patterns in training data
2. **General operators** - Built TILE, TILE_SUBGRID, DRAW_LINE, DRAW_BOX, FLOOD_FILL operators
3. **Observer=observed** - Induction routines learn parameters from EACH task's training pairs
4. **Beam integration** - Added beam-compatible versions for multi-step compositions
5. **Comprehensive testing** - 3 new unit tests, all passing

**Architecture changes:**
- Added `operators/tiling.py`: TILE, TILE_SUBGRID (+50 LOC)
- Added `operators/drawing.py`: DRAW_LINE, DRAW_BOX, FLOOD_FILL (+140 LOC)
- Added to `induction.py`: induce_tile_rule, induce_draw_line_rule (+120 LOC)
- Added to `induction.py`: induce_tile, induce_draw_line (beam versions, +100 LOC)
- Updated `solver.py`: PCE generation for TILE/DRAW_LINE, candidate_rules integration (+15 LOC)
- Added 3 unit tests to `test_arc_solver.py`
- Created `scripts/discover_patterns.py`: Pattern discovery from training data (+230 LOC)

**Impact:**
- Accuracy: 12/1076 → 13/1076 (+0.09%)
- Tasks: 12 → 13 (+1 task: a416b8f3)
- Rules: 12 → 14 variations (+2)
- Pattern coverage: TILE matched, DRAW_LINE ready but no matches yet

**Catalog ordering lesson:**
- TILE_SUBGRID can match COPY patterns (false positive)
- Solution: Order CATALOG by specificity - try object operations before tiling/drawing
- Principle: More constrained operations first (Occam's razor with tie-breaking)

**Learnings:**
- Tiling patterns exist but are rare (~1% of first 100 tasks)
- Drawing patterns more common (~15% of first 100 tasks) but detection needs refinement
- Subgrid extraction search space is large (need bounds on position enumeration)
- Catalog ordering critical to avoid false positives from general operators

**Next:** Continue with empirical discovery approach - analyze more failing tasks to identify high-impact operators.

---

## Update Protocol

**When to add entries:**
1. Discovery of any bug (logic error, incorrect result, crash)
2. Identification of missing functionality blocking progress
3. Recognition of systematic failure pattern in tests
4. Design decision with non-obvious tradeoffs

**Entry format:**
```markdown
### [Status] Brief Description
**Discovered:** YYYY-MM-DD
**Location:** file.py:line or module
**Status:** Current state

**Problem:** What is wrong?
**Root Cause:** Why does it happen?
**Fix/Plan:** How to resolve?
**Impact:** How many tasks affected?
**Verification:** How to test fix?
```

**Review cadence:**
- After each major feature addition
- When test coverage plateaus
- Before Phase transitions (0→1, 1→2, etc.)

---

## Related Documentation

- **TEST_COVERAGE.md** - Current solver performance and passing/failing tests
- **IMPLEMENTATION_PLAN.md** - Roadmap and operator catalog
- **CONTEXT_INDEX.md** - Navigation map for entire repo
- **architecture.md** - System design and module responsibilities

---

**Last Updated:** 2025-10-13
**Current Coverage:** 13/1076 (1.21%)
**Critical Bugs:** 0 open, 3 fixed
**Open Gaps:** 2 legacy rules, beam search + tiling/drawing operators ready
**Recent Change:** Added TILE and DRAW_LINE operators (+1 task: a416b8f3)
