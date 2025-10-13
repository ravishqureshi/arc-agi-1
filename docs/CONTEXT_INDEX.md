# Documentation Context Index

**Purpose:** Fast navigation map for locating docs/code. Pointers only - details in the actual files.

**Last updated:** 2025-10-13 (Post-tiling/drawing operators, beam search)

---

## Top-Level

- **README.md** - Project overview, setup, quickstart
- **data/README.md** - Dataset documentation (challenges, solutions)

---

## Core Documentation (docs/)

### Planning & Progress

- **IMPLEMENTATION_PLAN.md** - Master roadmap (Weeks 1-10, 60-80 operators, 79.6% target)
- **TEST_COVERAGE.md** - Coverage report (13/1076 = 1.21%)
- **test_coverage_data.json** - Raw test results (1,076 test outputs)
- **BUGS_AND_GAPS.md** ⚠️ - Known bugs, gaps, design decisions (zero compromises tracking)

### Submission

- **SUBMISSION_REQUIREMENTS.md** - Kaggle format, constraints, scoring
- **CODE_STRUCTURE.md** - Notebook organization (%%writefile pattern)
- **arc-agi-kaggle-docs.md** - Competition rules, dataset, timeline

### Architecture

- **architecture.md** - Modular design (23 files, folder structure, operator families)
- **modularization_complete.md** - Modularization summary (500 lines → 15 files)

### Verification

- **test_verification.md** - Test correctness verification
- **modular_test_results.md** - Post-modularization test results
- **legacy_rules_coverage.md** - Legacy rules in IMPLEMENTATION_PLAN analysis
- **arc_solver_v1_review.md** - arc_solver_v1.py code review
- **arc_demo_expansion_summary.md** - Adding rules to arc_demo.py summary

---

## Universe Intelligence Theory (docs/core/)

- **universe-intelligence-unified.md** - Complete UI guide (ORDER/ENTROPY/QUADRATIC)
- **ui_arc_demo.md** - Original 6-rule demo explanation
- **2_ui_arc_press_demo.md** - Press demo (move_bbox_to_origin, mirror rules)
- **\*.md** - Other UI enrichment versions (v1, v2 - superseded by unified)

---

## Source Code (src/)

### Core Framework

- **universe_intelligence.py** - UI implementation (HornKB, softmax, schur_dtn, etc.)

### ARC Solver Package (arc_solver/)

**Main:**
- **\_\_init\_\_.py** - Package exports (all public APIs)
- **arc_solver_v1.py** - Monolithic reference (500 lines, kept for comparison)

**Core (core/):**
- **types.py** - Grid, Mask, ObjList types + helpers (G, copy_grid, asserts)
- **invariants.py** - Invariant engine (histogram, bbox, components, symmetries, exact_equals)
- **receipts.py** - Receipts (edit_counts, residual, PCE, verification)
- **induction.py** - Rule induction from train pairs (CATALOG, beam-compatible versions)
- **solver.py** - Main solver harness (solve_instance, beam_search, solve_with_beam)

**Operators (operators/):**
- **symmetry.py** - ROT, FLIP
- **spatial.py** - BBOX, CROP, MOVE_OBJ_RANK, COPY_OBJ_RANK
- **masks.py** - MASK_COLOR, MASK_NONZERO, MASK_OBJ_RANK, KEEP, REMOVE
- **color.py** - COLOR_PERM, infer_color_mapping, merge_mappings
- **tiling.py** - TILE, TILE_SUBGRID
- **drawing.py** - DRAW_LINE, DRAW_BOX, FLOOD_FILL
- **composition.py** - ON, SEQ

**Legacy (legacy/):**
- **arc_demo.py** - 6-rule demo (12/1000 tasks, 1.2% accuracy, reference only)

**Tests (tests/):**
- **test_arc_solver.py** - Unit tests (13 tests, all passing ✅)

---

## Scripts (scripts/)

- **generate_test_coverage.py** - Test coverage generator (runs 1,076 tasks, outputs JSON + summary)
- **discover_patterns.py** - Empirical pattern discovery (tiling, drawing, shift detection)

---

## Data Files (data/)

- **arc-agi_training_challenges.json** - 1,000 tasks (391 ARC-1 + 609 ARC-2)
- **arc-agi_training_solutions.json** - Ground truth solutions
- **arc-agi_evaluation_challenges.json** - Evaluation set (no solutions)
- **arc-agi_test_challenges.json** - Test set (placeholder)
- **sample_submission.json** - Submission format example

---

## Utilities

- **arc_version_split.py** - ARC-1 vs ARC-2 task distinction (is_arc1_task, is_arc2_new_task)
- **example_version_tracking.py** - Version performance tracking example
- **arc1_task_ids.txt** - 391 ARC-1 task IDs
- **arc2_new_task_ids.txt** - 609 ARC-2 task IDs
- **arc1_removed_task_ids.txt** - 9 removed ARC-1 task IDs

---

## Quick Navigation

**"I need to..."**

- **understand the project** → README.md, docs/architecture.md
- **understand the theory** → docs/core/universe-intelligence-unified.md
- **understand the plan** → docs/IMPLEMENTATION_PLAN.md
- **check test coverage** → docs/TEST_COVERAGE.md, docs/test_coverage_data.json
- **check for bugs/gaps** → docs/BUGS_AND_GAPS.md
- **add a new operator** → docs/architecture.md, src/arc_solver/operators/, docs/IMPLEMENTATION_PLAN.md Section 2
- **add a new induction routine** → src/arc_solver/core/induction.py (add to CATALOG)
- **run tests** → `python src/tests/test_arc_solver.py`
- **generate test coverage** → `python scripts/generate_test_coverage.py`
- **discover patterns** → `python scripts/discover_patterns.py`
- **understand receipts** → src/arc_solver/core/receipts.py, docs/core/universe-intelligence-unified.md
- **prepare submission** → docs/SUBMISSION_REQUIREMENTS.md, docs/CODE_STRUCTURE.md
- **track progress** → docs/TEST_COVERAGE.md (progress table), docs/BUGS_AND_GAPS.md (learnings)

---

## Conventions

**Mathematical Discipline:**
- Receipts-first (residual=0 on train, edit bills, PCE)
- Universe Intelligence: "Inside settles", "Edges write", "Observer=observed"

**Code Organization:**
- Modular: 23 files, ~65 lines each
- By responsibility: core/ (architecture), operators/ (DSL), legacy/ (reference)
- Type-safe: Grid, Mask, ObjList with validation

**Naming:**
- Operators: UPPERCASE (ROT, FLIP, CROP, MASK_COLOR)
- Functions: lowercase_underscore (induce_symmetry_rule, edit_counts)
- Types: PascalCase (Grid, Mask, Invariants, Receipts)

**Testing:**
- Unit: src/tests/test_arc_solver.py (13 tests, all passing)
- Full coverage: docs/TEST_COVERAGE.md (1,076 test outputs)
- Ground truth: data/arc-agi_training_solutions.json

---

## Update Protocol

When adding/moving docs or changing structure, update:
1. This CONTEXT_INDEX.md (pointers only, keep concise)
2. docs/architecture.md (if code structure changes)
3. docs/TEST_COVERAGE.md (if coverage changes)
4. docs/BUGS_AND_GAPS.md (if bugs/gaps/decisions discovered)
5. README.md (if top-level structure changes)

---

**Maintained by:** Claude Code
**Repo:** ARC-AGI 2025 Pure Mathematics Solver
