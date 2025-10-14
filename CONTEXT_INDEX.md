# Context Index - Repository Navigation

**Last Updated**: 2025-10-13
**Coverage**: 10/1000 = 1.00%

---

## Quick Start

```bash
python src/ui_clean.py  # Run solver on all 1000 tasks
```

## Architecture Flow

1. Load task → 2. `autobuild_operators()` → 3. `beam_search()` → 4. Verify train residual=0

## Code Structure

### Core Package: `src/arc_solver/`

| File | Purpose | Key Content |
|------|---------|-------------|
| `types.py` | Dataclasses | Grid, Operator, ARCInstance, Node, Obj |
| `utils.py` | Helpers | equal, residual, components, rank_by_size |
| `operators.py` | Operators | ROT, FLIP, COLOR_PERM, CROP, KEEP, PARITY, TILING, HOLE_FILL, COPY |
| `inducers.py` | Inducers | 8 inducer functions (induce_COLOR_PERM, etc.) |
| `search.py` | Search | autobuild_operators, beam_search, solve_with_beam |

### Entry Point: `src/ui_clean.py` (52 lines)

Main script - imports arc_solver package and runs on all tasks.

## Current Inducers (8)

| Name | Operator | Solved |
|------|----------|--------|
| induce_COLOR_PERM | COLOR_PERM | 4/10 |
| induce_ROT_FLIP | ROT, FLIP | 5/10 |
| induce_CROP_KEEP | CROP_BBOX_NONZERO, KEEP_NONZERO | 1/10 |
| induce_PARITY_CONST | RECOLOR_PARITY_CONST | 0/10 |
| induce_TILING_AND_MASK | REPEAT_TILE | 0/10 |
| induce_HOLE_FILL | HOLE_FILL_ALL | 0/10 |
| induce_COPY_BY_DELTAS | COPY_OBJ_RANK_BY_DELTAS | 0/10 |

**Solved tasks**: 0d3d703e, 1cf80156, 3c9b0459, 6150a2bd, 67a3c6ac, 68b16354, b1948b0a, c8f0f002, d511f180, ed36ccf7

## Adding New Inducers

1. Add operator function → `src/arc_solver/operators.py`
2. Add inducer function → `src/arc_solver/inducers.py`
3. Add to `autobuild_operators()` → `src/arc_solver/search.py`

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
| `docs/core/universe-intelligence-added-clarification.md` | Architecture source (original doc) |
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

# Generate test coverage
python generate_test_coverage.py

# Quick test
python -c "from src.arc_solver import G, ARCInstance, solve_with_beam; ..."
```

## Files to Ignore

These were deleted (proved useless):
- `src/arc_solver/` (old 27 operators)
- `src/tests/`
- `scripts/`
- `src/ui_solver.py`

---

**Navigation**: Start with README.md → this file → `src/arc_solver/` for code
