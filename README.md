# ARC-AGI Solver - Universe Intelligence Approach

**🗺️ New to this repo? Start with [CONTEXT_INDEX.md](CONTEXT_INDEX.md) for navigation**

## Current Status

**Coverage**: 10/1000 = 1.00% on ARC training set

**Architecture**: Inducer-based operator discovery from Universe Intelligence framework

## What This Is

Implementation of Universe Intelligence (UI) mathematics for ARC-AGI:
- **Inducers** learn operators from training pairs
- **Autobuilder** tries all inducers per task
- **Beam search** composes operators with residual=0 pruning
- **Receipts-first discipline**: Only accept train residual=0

## Project Structure

```
├── data/                          # ARC dataset
├── src/
│   ├── arc_solver/               # Modular solver package
│   │   ├── types.py              # Grid, Operator, ARCInstance
│   │   ├── utils.py              # Helper functions
│   │   ├── operators.py          # All operator functions
│   │   ├── inducers.py           # All inducer functions
│   │   └── search.py             # Beam search & autobuilder
│   └── ui_clean.py               # Main entry point (52 lines)
├── docs/
│   ├── core/                     # Universe Intelligence docs
│   ├── ARC_TEST_COVERAGE_REPORT.md  # Full test results table
│   ├── DATA_SPLIT_ANALYSIS.md    # ARC-1 vs ARC-2 split
│   └── SUBMISSION_REQUIREMENTS.md   # Kaggle format
├── CONTEXT_INDEX.md              # Navigation map
├── OBJECTIVE_ASSESSMENT.md       # Decision rationale
└── generate_test_coverage.py     # Test coverage generator
```

## Running the Solver

```bash
python src/ui_clean.py
```

Output: Shows progress every 100 tasks, prints solved tasks with operator chains

## Current Inducers (8 total)

1. **COLOR_PERM** - Learn color mappings
2. **ROT/FLIP** - Try all symmetries
3. **CROP_BBOX_NONZERO** - Crop to non-zero bounding box
4. **KEEP_NONZERO** - Keep only non-zero cells
5. **PARITY_CONST** - Recolor by parity pattern
6. **TILING** - Extract and repeat tile motifs
7. **HOLE_FILL** - Fill enclosed holes in objects
8. **COPY_BY_DELTAS** - Copy objects by centroid deltas (global + per_color)

## Next Steps

1. **Triage failing tasks** - Understand missing patterns
2. **Add new inducers** - Data-driven: only add when pattern identified
3. **Target**: 10% on ARC-1, 1% on ARC-2

## Key Files

- `CONTEXT_INDEX.md` - **Navigation map** (start here for AI assistants)
- `src/arc_solver/` - **Modular solver package** (all logic here)
- `src/ui_clean.py` - Main entry point
- `docs/core/universe-intelligence-added-clarification.md` - Architecture source
- `docs/ARC_TEST_COVERAGE_REPORT.md` - Full test coverage (1000 tasks)
- `OBJECTIVE_ASSESSMENT.md` - Why we started from scratch
- `generate_test_coverage.py` - Regenerate test coverage

## Philosophy

**Receipts-First**: Every claim has a proof:
- Train residual=0 → operator fits ALL train pairs
- Beam search prunes on residual increase → monotone descent
- No hallucinations, no guessing

**Observer=Observed**: One parameterization must fit ALL train pairs. Contradictions prune immediately.

**Inside Settles**: Accept program ONLY if train residual == 0.

---

**Previous work** (27 hand-coded operators, old architecture) was deleted. It added zero value. We started from mathematician's proven architecture with modular structure for easy extension.
