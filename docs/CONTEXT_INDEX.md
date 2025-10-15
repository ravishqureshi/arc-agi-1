# Context Index - Repository Navigation

**Last Updated**: 2025-10-15
**Coverage**: 10/1000 = 1.00% (baseline - about to increase with closures!)
**Paradigm**: Master Operator / Fixed-Point Closures

---

## Quick Start

```bash
# NEW: Run solver on evaluation set and generate predictions.json
python scripts/run_public.py --dataset=data/arc-agi_evaluation_challenges.json --output=runs/submission

# Validate submission format
python scripts/submission_validator.py runs/submission/predictions.json

# Check determinism
bash scripts/determinism.sh data/arc-agi_evaluation_challenges.json

# OLD: Run solver on training set (uses beam search, not closures)
python src/ui_clean.py
```

## Architecture Flow

**NEW (Master Operator Paradigm):**
1. Load task → 2. `autobuild_closures()` tries unifiers → 3. `run_fixed_point()` until convergence → 4. All cells singleton = solved

**OLD (Beam Search - still present):**
1. Load task → 2. `autobuild_operators()` → 3. `beam_search()` → 4. Verify train residual=0

## Code Structure

### Core Package: `src/arc_solver/`

| File | Purpose | Paradigm | Key Content |
|------|---------|----------|-------------|
| `types.py` | Dataclasses | Both | Grid, Operator, ARCInstance, Node, Obj |
| `utils.py` | Helpers | Both | equal, residual, components, task_sha, program_sha, closure_set_sha, log_receipt |
| **`closure_engine.py`** | **Fixed-Point Engine** | **NEW** | **SetValuedGrid, Closure, run_fixed_point [B0]** |
| **`closures.py`** | **Closure Implementations** | **NEW** | **KEEP_LARGEST_COMPONENT [B1], unifiers** |
| `search.py` | Search/Solve | Both | solve_with_closures (NEW), solve_with_beam (OLD) |
| `operators.py` | Operators | OLD | ROT, FLIP, COLOR_PERM, CROP, PARITY, TILING, HOLE_FILL |
| `inducers.py` | Inducers | OLD | 9 inducer functions for beam search |

### Entry Points

| File | Purpose | Paradigm |
|------|---------|----------|
| **`scripts/run_public.py`** | **Main submission script** | **NEW - Closures** |
| `scripts/submission_validator.py` | Validate predictions.json | Utility |
| `scripts/determinism.sh` | Determinism check | Utility |
| `src/ui_clean.py` | Training set runner | OLD - Beam Search |

## Current Capabilities

### NEW Paradigm (Fixed-Point Closures) - 1 closure:
1. ✅ **B1: KEEP_LARGEST_COMPONENT** - Keep only largest 4-connected component

**Pending (high priority):**
- B2: OUTLINE_OBJECTS
- B7: OPEN/CLOSE (k=1) - morphology
- B3: AXIS_PROJECTION_FILL
- B9: MOD_PATTERN - general parity
- B10: DIAGONAL_REPEAT

### OLD Paradigm (Beam Search Inducers) - 9 inducers:
1. COLOR_PERM - Color mappings
2. ROT_FLIP - Rotations and flips
3. PARITY_CONST - Parity patterns
4. CROP_KEEP - Crop and keep operations
5. KEEP_LARGEST (old operator version)
6. TILING - Tile motifs
7. HOLE_FILL - Fill holes
8. COPY_BY_DELTAS (global) - Copy by offsets
9. COPY_BY_DELTAS (per_color) - Copy by color

**Baseline solved tasks**: 0d3d703e, 1cf80156, 3c9b0459, 6150a2bd, 67a3c6ac, 68b16354, b1948b0a, c8f0f002, d511f180, ed36ccf7

## Adding New Closures (NEW Paradigm)

**4 steps:**
1. Add closure class (inherits from Closure) → `src/arc_solver/closures.py`
2. Implement `apply(U, x_input)` - monotone & shrinking
3. Add unifier function that verifies ALL train pairs → same file
4. Add to `autobuild_closures()` → `src/arc_solver/search.py`

**See**: `docs/IMPLEMENTATION_PLAN_v2.md` for detailed specs

## Data Locations

```
data/
├── arc-agi_training_challenges.json  # 1000 training tasks
├── arc-agi_training_solutions.json   # Expected outputs
├── arc-agi_evaluation_challenges.json
└── arc-agi_evaluation_solutions.json

arc1_task_ids.txt        # 391 ARC-1 tasks
arc2_new_task_ids.txt    # 609 ARC-2 tasks
```

## Documentation

| File | Purpose |
|------|---------|
| **`docs/IMPLEMENTATION_PLAN_v2.md`** | **CURRENT PLAN - Fixed-point closures roadmap (A → B)** |
| `docs/core/arc_agi_master_operator.md` | Master Operator paradigm explanation |
| `docs/core/universe-intelligence-added-clarification.md` | Original architecture (OLD beam search) |
| `docs/IMPLEMENTATION_PLAN.md` | OLD plan (beam search + beam expansion) |
| `docs/ARC_TEST_COVERAGE_REPORT.md` | Full test results (1000 tasks table) |
| `docs/SUBMISSION_REQUIREMENTS.md` | Kaggle submission format |
| `OBJECTIVE_ASSESSMENT.md` | Why we deleted 27 operators |

## Key Principles

1. **Receipts-First**: Train residual == 0 required
2. **Observer=Observed**: One parameterization fits ALL train pairs
3. **Inside Settles**: No guessing
4. **Inducer Discipline**: Operators learned from data
5. **Beam Pruning**: Only keep compositions that reduce residual

## Test Commands

```bash
# Run solver
python src/ui_clean.py

# Test on first N tasks
python -c "from src.ui_clean import run_solver; run_solver(out_dir='runs/test')"

# Check receipts
cat runs/2025-10-13/receipts.jsonl | head -3 | python -m json.tool
```
---

**Navigation**: Start with README.md → this file → `src/arc_solver/` for code
