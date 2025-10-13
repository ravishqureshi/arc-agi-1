# Documentation Context Index

**Purpose:** Fast navigation map for humans and AIs to locate the right document/code quickly. This index lists every major document and what questions it answers.

**Last updated:** 2025-10-13 (Post-modularization)

---

## Top-Level

### README.md
- **What:** Project overview, goals, dataset info, quickstart
- **When to read:** First contact with the repo
- **Answers:**
  - "What is this project?"
  - "What dataset are we using (ARC-1 vs ARC-2)?"
  - "How do I run the demo?"
  - "What is Universe Intelligence?"

### data/README.md
- **What:** Dataset documentation (challenges, solutions, versions)
- **Answers:**
  - "What JSON files exist and what do they contain?"
  - "How big are the data files?"
  - "What is the format of challenges vs solutions?"

---

## Core Documentation (docs/)

### docs/IMPLEMENTATION_PLAN.md
- **What:** Master plan for 79.6%+ accuracy (Weeks 1-10)
- **Answers:**
  - "What's the roadmap?"
  - "What operators do we need?"
  - "What's the target accuracy per phase?"
  - "How many operators for baseline/gap-filling/deep search?"
- **Key sections:**
  - Section 0: Ground rules (receipts-first, per-task learning)
  - Section 1: System architecture (Invariant engine, Typed DSL, Search)
  - Section 2: Operator families (60-80 operators)
  - Section 6-8: Milestones and expected accuracy
  - Appendix: Coverage analysis, gap-filling operators
- **Cross-reference:** architecture.md (implementation), TEST_COVERAGE.md (progress tracking)

### docs/SUBMISSION_REQUIREMENTS.md
- **What:** Kaggle submission format and constraints
- **Answers:**
  - "What format must submission.json have?"
  - "What are the time limits?"
  - "What packages are allowed?"
  - "How is scoring calculated?"
- **Use when:** Preparing final submission

### docs/CODE_STRUCTURE.md
- **What:** How to organize code for Kaggle notebook submission
- **Answers:**
  - "How to use %%writefile pattern?"
  - "How to structure notebook cells?"
  - "How to avoid import issues?"
- **Use when:** Creating submission notebook

### docs/arc-agi-kaggle-docs.md
- **What:** Competition documentation (rules, dataset, timeline)
- **Answers:**
  - "What are the competition rules?"
  - "What is the evaluation metric?"
  - "What is the prize structure?"

---

## Architecture & Design (docs/)

### docs/architecture.md
- **What:** Modular architecture design for 25-30 files
- **Answers:**
  - "What is the folder structure?"
  - "How are files organized by responsibility?"
  - "What goes in core/ vs operators/ vs legacy/?"
  - "How do I add new operators?"
- **Key sections:**
  - Design principles (receipts-first, modular, type-safe)
  - Folder structure with file counts
  - Core components (types, invariants, receipts, induction, solver)
  - Operator families by category
  - Migration guide from monolithic to modular
- **Cross-reference:** IMPLEMENTATION_PLAN.md (what to build), src/arc_solver/ (implementation)

### docs/modularization_complete.md
- **What:** Summary of modularization process (500 lines → 15 files)
- **Answers:**
  - "What was changed during modularization?"
  - "What are the benefits of modular architecture?"
  - "What tests were run to verify correctness?"
- **Use when:** Understanding why we refactored, what changed

---

## Test Coverage & Results (docs/)

### docs/TEST_COVERAGE.md
- **What:** Comprehensive test coverage report for all 1,000 tasks
- **Answers:**
  - "How many tasks are we solving?"
  - "What is our accuracy on ARC-1 vs ARC-2?"
  - "Which rules are passing/failing?"
  - "What are the most common failure modes?"
  - "How does our performance compare to legacy demo?"
- **Key sections:**
  - Summary statistics (6/1076 passing = 0.6%)
  - Breakdown by rule (ROT, FLIP, CROP_BBOX_NONZERO)
  - Passing tests (detailed table with shapes, residuals)
  - Failing tests (sample with issue types)
  - Progress tracking table (Phase 0-3 targets)
- **Data file:** docs/test_coverage_data.json (1,076 test results)
- **Update method:** Run test generator, regenerate report
- **Cross-reference:** IMPLEMENTATION_PLAN.md (targets), test_verification.md (validation)

### docs/test_verification.md
- **What:** Verification that tests correctly compare to ground truth
- **Answers:**
  - "Do tests compare predictions to solutions file?"
  - "Are residuals computed correctly?"
  - "Are all 6 solved tasks verified?"
- **Use when:** Verifying test correctness, debugging comparison logic

### docs/modular_test_results.md
- **What:** Test results after modularization
- **Answers:**
  - "Did modularization break anything?"
  - "What is the current performance?"

### docs/legacy_rules_coverage.md
- **What:** Analysis of which legacy rules are in IMPLEMENTATION_PLAN.md
- **Answers:**
  - "Are missing rules from arc_demo.py covered in the plan?"
  - "Where will color_perm, move_bbox_to_origin, component_recolor be implemented?"
  - "How many times is each rule mentioned in the plan?"
- **Key finding:** All 4 missing rules are covered (4-5 mentions each)

---

## Universe Intelligence Theory (docs/core/)

### docs/core/universe-intelligence-unified.md
- **What:** Complete guide to Universe Intelligence framework
- **Answers:**
  - "What are the 3 enrichments (ORDER/ENTROPY/QUADRATIC)?"
  - "How do Horn clauses and least fixed points work?"
  - "What is the Dirichlet-to-Neumann operator?"
  - "How do I use receipts (residual=0, gaps ≈ 1e-12)?"
- **Key sections:**
  - ORDER: Horn clauses, least fixed points, logic reasoning
  - ENTROPY: KL divergence, Fenchel-Young, probabilistic reasoning
  - QUADRATIC: Laplacian, Green identity, spatial geometry
  - Usage examples for each enrichment
- **Cross-reference:** src/universe_intelligence.py (implementation)

### docs/core/universe-intelligence.md
- **What:** V1 - Quadratic enrichment only
- **Status:** Superseded by universe-intelligence-unified.md

### docs/core/universe-intelligence-v2.md
- **What:** V2 - All three enrichments
- **Status:** Superseded by universe-intelligence-unified.md

### docs/core/ui_arc_demo.md
- **What:** Original ARC demo explanation with 6 rules
- **Answers:**
  - "How does the legacy demo work?"
  - "What are the 6 transformation rules?"
  - "How are receipts generated?"
- **Cross-reference:** src/arc_solver/legacy/arc_demo.py (implementation)

### docs/core/2_ui_arc_press_demo.md
- **What:** Press demo with move_bbox_to_origin and mirror rules
- **Answers:**
  - "What are the 2 additional rules from press demo?"
  - "How does move_bbox_to_origin differ from crop?"

---

## Code Reviews & Analysis (docs/)

### docs/arc_solver_v1_review.md
- **What:** Comprehensive review of arc_solver_v1.py (Phase 1 starter)
- **Answers:**
  - "Is arc_solver_v1.py mathematically correct?"
  - "What are the key features?"
  - "How does it compare to arc_demo.py?"
- **Verdict:** 100% mathematically correct, production-ready

### docs/arc_demo_expansion_summary.md
- **What:** Summary of adding 2 rules to arc_demo.py
- **Answers:**
  - "What rules were added?"
  - "Did they improve performance?"
- **Result:** New rules work but solve 0 additional tasks

### docs/temp_code_file.md
- **What:** Temporary starter code (now in src/arc_solver_v1.py)
- **Status:** Historical reference only

---

## Source Code (src/)

### src/universe_intelligence.py
- **What:** Core Universe Intelligence implementation
- **Exports:**
  - HornKB (ORDER enrichment)
  - softmax, entropy_phi, KL (ENTROPY enrichment)
  - schur_dtn, solve_dirichlet, green_gap (QUADRATIC enrichment)
- **Use when:** Need mathematical reasoning with receipts
- **Cross-reference:** docs/core/universe-intelligence-unified.md (theory)

### src/arc_solver/ (Main Package)

#### src/arc_solver/__init__.py
- **What:** Main package exports (all public APIs)
- **Exports:** Types, invariants, receipts, induction, solver, operators
- **Use when:** Importing solver functionality

#### src/arc_solver/arc_solver_v1.py
- **What:** Monolithic Phase 1 starter (kept for reference)
- **Sections:**
  - Section 0: Types (Grid, Mask)
  - Section 1: Invariant engine
  - Section 2: Receipts
  - Section 3: Core DSL (ROT, FLIP, CROP, MASK, KEEP, REMOVE, ON, SEQ)
  - Section 4: Induction routines
  - Section 5: Solver harness
  - Section 6: Unit tests
- **Status:** Working baseline, use modular version for new features

### src/arc_solver/core/ (Architecture)

#### src/arc_solver/core/types.py
- **What:** Type definitions (Grid, Mask, ObjList)
- **Exports:** G() helper, copy_grid(), assert_grid(), assert_mask()
- **Use when:** Working with grid types

#### src/arc_solver/core/invariants.py
- **What:** Invariant detection engine
- **Exports:**
  - Invariants dataclass (shape, histogram, bbox, components, symmetries)
  - invariants() - Main function
  - color_histogram(), bbox_nonzero(), connected_components()
  - rot90(), rot180(), rot270(), flip_h(), flip_v()
  - exact_equals()
- **Use when:** Computing grid properties, detecting symmetries

#### src/arc_solver/core/receipts.py
- **What:** Receipts & verification (edit bills, residuals, PCE)
- **Exports:**
  - Receipts dataclass
  - edit_counts() - Returns (total, boundary, interior) edits
  - residual() - Hamming distance to target
  - generate_pce() - Proof-Carrying English
  - verify_train_residuals_zero()
- **Use when:** Verifying transformations, generating proofs
- **Universe Intelligence:** "Edges write" - tracks where changes happen

#### src/arc_solver/core/induction.py
- **What:** Rule induction from training pairs
- **Exports:**
  - Rule dataclass (name, params, prog)
  - induce_symmetry_rule() - ROT + FLIP
  - induce_crop_nonzero_rule() - CROP_BBOX_NONZERO
  - induce_keep_nonzero_rule() - KEEP(MASK_NONZERO)
  - CATALOG - List of all induction routines
  - induce_rule() - Try all routines
- **Use when:** Learning transformations from examples
- **Universe Intelligence:** "Observer=observed" - unify across train pairs

#### src/arc_solver/core/solver.py
- **What:** Main solver harness with receipts + PCE
- **Exports:**
  - ARCInstance dataclass (train, test_in, test_out)
  - SolveResult dataclass (rule, preds, receipts, acc_exact)
  - solve_instance() - Main solver function
  - pce_for_rule() - Generate explanations
- **Use when:** Solving ARC tasks end-to-end
- **Universe Intelligence:** Residual=0 on train before predicting test

### src/arc_solver/operators/ (DSL Operators)

#### src/arc_solver/operators/symmetry.py
- **What:** Rotation and reflection operators
- **Exports:**
  - ROT(k) - Rotate k*90 degrees (k=0,1,2,3)
  - FLIP(axis) - Flip horizontal ('h') or vertical ('v')
- **Use when:** Applying symmetry transformations

#### src/arc_solver/operators/spatial.py
- **What:** Spatial transformations and cropping
- **Exports:**
  - BBOX(bg) - Get bounding box
  - CROP(rect_fn) - Crop to rectangle
  - CROP_BBOX_NONZERO(bg) - Convenience: crop to non-zero content
- **Use when:** Cropping, extracting regions

#### src/arc_solver/operators/masks.py
- **What:** Mask creation and application
- **Exports:**
  - MASK_COLOR(c) - Select cells of color c
  - MASK_NONZERO(bg) - Select non-background cells
  - KEEP(mask_fn) - Keep only masked cells (others → 0)
  - REMOVE(mask_fn) - Remove masked cells (set to 0)
- **Use when:** Selecting regions by color or content

#### src/arc_solver/operators/composition.py
- **What:** Operator composition and gluing
- **Exports:**
  - ON(mask_fn, prog) - Apply prog only to masked region
  - SEQ(p1, p2) - Sequential composition (p1 then p2)
- **Use when:** Combining transformations
- **Universe Intelligence:** "Edges write" - changes only in masked region

### src/arc_solver/legacy/

#### src/arc_solver/legacy/arc_demo.py
- **What:** Original 6-rule demo (12/1000 tasks solved)
- **Rules:** symmetry, color_perm, crop_bbox, move_bbox_to_origin, mirror_left_to_right, component_size_recolor
- **Status:** Reference only, use modular version for new work
- **Performance:** 12 tasks solved (1.2% accuracy)

### src/tests/

#### src/tests/test_arc_solver.py
- **What:** Unit tests for modular solver
- **Tests:**
  - test_invariants() - Invariant detection
  - test_symmetry_induction() - Symmetry rule learning
  - test_crop_bbox() - Crop rule learning
  - test_demo_tasks() - 3 demo tasks (rot180, flip_h, crop)
- **Status:** ALL PASSING ✅
- **Run:** `python src/tests/test_arc_solver.py`

---

## Data Files (data/)

### data/arc-agi_training_challenges.json
- **What:** 1,000 training tasks (391 ARC-1 + 609 ARC-2)
- **Format:** `{task_id: {"train": [...], "test": [...]}}`

### data/arc-agi_training_solutions.json
- **What:** Ground truth solutions for training tasks
- **Format:** `{task_id: [[output_grid], ...]}`

### data/arc-agi_evaluation_challenges.json
- **What:** Evaluation set (no solutions provided)

### data/arc-agi_test_challenges.json
- **What:** Test set (placeholder, will be released during competition)

### data/sample_submission.json
- **What:** Example submission format

---

## Utilities

### arc_version_split.py
- **What:** Distinguish ARC-1 vs ARC-2 tasks
- **Functions:**
  - is_arc1_task(task_id) → bool
  - is_arc2_new_task(task_id) → bool
  - get_version_stats() → dict
- **Data files:** arc1_task_ids.txt (391), arc2_new_task_ids.txt (609)

### example_version_tracking.py
- **What:** Example of tracking performance by ARC version
- **Function:** compute_separate_scores(predictions, solutions)

---

## How to Navigate by Task

### "I need to understand the project"
→ README.md, docs/architecture.md

### "I need to understand the theory"
→ docs/core/universe-intelligence-unified.md

### "I need to understand the implementation plan"
→ docs/IMPLEMENTATION_PLAN.md (complete roadmap with phases)

### "I need to check test coverage"
→ docs/TEST_COVERAGE.md (1,076 test results), docs/test_coverage_data.json (raw data)

### "I need to add a new operator"
→ docs/architecture.md (how to add), src/arc_solver/operators/ (examples), docs/IMPLEMENTATION_PLAN.md Section 2 (operator families)

### "I need to add a new induction routine"
→ src/arc_solver/core/induction.py (add to CATALOG), docs/architecture.md (migration guide)

### "I need to run tests"
→ `python src/tests/test_arc_solver.py` (unit tests), docs/TEST_COVERAGE.md (full coverage)

### "I need to solve an ARC task"
```python
from arc_solver import G, solve_instance, ARCInstance, rot180

x = G([[0,1],[2,0]])
y = rot180(x)
task = ARCInstance("demo", [(x,y)], [G([[0,3],[4,0]])], [rot180(G([[0,3],[4,0]]))])

result = solve_instance(task)
print(f"Rule: {result.rule.name}, Accuracy: {result.acc_exact}")
```

### "I need to understand receipts"
→ src/arc_solver/core/receipts.py (implementation), docs/core/universe-intelligence-unified.md (theory)

### "I need to check if legacy rules are covered in plan"
→ docs/legacy_rules_coverage.md (all 4 missing rules are covered)

### "I need to prepare Kaggle submission"
→ docs/SUBMISSION_REQUIREMENTS.md (format), docs/CODE_STRUCTURE.md (notebook structure)

### "I need to understand modularization"
→ docs/modularization_complete.md (summary), docs/architecture.md (design)

### "I need to verify test correctness"
→ docs/test_verification.md (verification report)

### "I need to track progress"
→ docs/TEST_COVERAGE.md (progress tracking table with Phase 0-3 targets)

### "I need dataset information"
→ README.md (ARC-1 vs ARC-2 breakdown), data/README.md (file details)

### "I need to understand Universe Intelligence"
→ docs/core/universe-intelligence-unified.md (complete guide), src/universe_intelligence.py (implementation)

---

## Conventions

### Mathematical Discipline
- **Receipts-first:** Every transformation has residual=0 on train, edit bills, PCE
- **Universe Intelligence principles:**
  - "Inside settles" - Verification = least fixed point
  - "Edges write" - All changes at boundaries
  - "Observer = observed" - Cross-pair agreement

### Code Organization
- **Modular:** 15 files, avg 65 lines each (vs 500-line monolith)
- **By responsibility:** core/ (architecture), operators/ (DSL), legacy/ (reference)
- **Type-safe:** Grid, Mask, ObjList with validation

### Testing
- **Unit tests:** src/tests/test_arc_solver.py (4 tests, all passing)
- **Full coverage:** docs/TEST_COVERAGE.md (1,076 test outputs tracked)
- **Ground truth:** All tests compare to data/arc-agi_training_solutions.json

### Naming
- **Operators:** UPPERCASE (ROT, FLIP, CROP, MASK_COLOR)
- **Functions:** lowercase_underscore (induce_symmetry_rule, edit_counts)
- **Types:** PascalCase (Grid, Mask, Invariants, Receipts)

---

## Update Protocol

If a document is added, moved, or a major section changes:

1. Update this CONTEXT_INDEX.md
2. Update docs/architecture.md (if code structure changes)
3. Update docs/TEST_COVERAGE.md (if adding operators/improving accuracy)
4. Update README.md (if top-level structure changes)

---

**Maintained by:** Claude Code
**Format inspired by:** Opoch-OO reference_context_index.md
**Repo:** ARC-AGI 2025 Pure Mathematics Solver
