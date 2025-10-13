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

---

## Implementation Gaps

### [ ] Remaining Legacy Rules (from arc_demo.py)

**Status:** 2/4 legacy rules migrated

**Completed:**
- [x] Color permutation (added 2025-10-13, +3 tasks)
- [x] Component rank recolor (added 2025-10-13, infrastructure only, +0 tasks)

**Remaining:**
- [ ] Move bbox to origin
- [ ] Mirror left to right

**Assessment:** Low priority - these are simple patterns covered in IMPLEMENTATION_PLAN.md Phase 1. Expected total impact: +2-3 tasks (1.0-1.2% accuracy).

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
**Current Coverage:** 9/1076 (0.8%)
**Critical Bugs:** 0 open, 1 fixed
**Open Gaps:** 2 legacy rules, performance optimizations
