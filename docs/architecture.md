# ARC Solver Architecture

## Overview

This document describes the modular architecture for the production ARC solver, designed to implement the full [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) with 25-30 files organized by responsibility.

## Folder Structure

```
src/
├── universe_intelligence.py    # Core UI framework (standalone)
├── arc_solver/                 # Main ARC solver package
│   ├── __init__.py
│   ├── core/                   # Core architecture (7 files)
│   │   ├── __init__.py
│   │   ├── types.py            # Grid, Mask, ObjList type definitions
│   │   ├── invariants.py       # Invariant detection (histogram, bbox, components, symmetries)
│   │   ├── receipts.py         # Edit counting, residual computation, PCE generation
│   │   ├── induction.py        # Base induction routines
│   │   ├── search.py           # Beam search with pruning
│   │   └── solver.py           # Main solver harness
│   ├── operators/              # DSL operators by family (11+ files)
│   │   ├── __init__.py
│   │   ├── symmetry.py         # ROT, FLIP
│   │   ├── spatial.py          # CROP, BBOX, SHIFT
│   │   ├── masks.py            # MASK_COLOR, MASK_NONZERO, MASK_RECT, MASK_COMPONENTS
│   │   ├── selection.py        # KEEP, REMOVE
│   │   ├── composition.py      # ON, SEQ
│   │   ├── color.py            # RECOLOR, COLOR_PERM
│   │   ├── grid_ops.py         # RESIZE, TILE, PAD, TRIM
│   │   ├── physics.py          # GRAVITY, MOVE, FILL
│   │   ├── path.py             # CONNECT, EXTEND_LINE
│   │   ├── symbols.py          # SYMBOL_REWRITE, IN_CONTEXT_DEF
│   │   └── patterns.py         # PERIODICITY, REPEAT
│   ├── utils/                  # Helper utilities
│   │   ├── __init__.py
│   │   ├── grid_utils.py       # Grid manipulation helpers
│   │   └── debug.py            # Debugging and visualization
│   └── legacy/                 # Previous implementations
│       ├── __init__.py
│       ├── arc_demo.py         # 6-rule demo (12/1000 solved)
│       └── arc_solver_v1.py    # Phase 1 starter
├── tests/                      # Unit tests (5 files)
│   ├── __init__.py
│   ├── test_invariants.py
│   ├── test_operators.py
│   ├── test_induction.py
│   ├── test_search.py
│   └── test_end_to_end.py
└── demos/                      # Demo scripts (3 files)
    ├── __init__.py
    ├── demo_baseline.py        # Run baseline on training data
    └── demo_receipts.py        # Demonstrate receipts-first discipline
```

## Design Principles

### 1. Receipts-First Discipline
Every transformation must provide a mathematical receipt:
- **Residual = 0** on training pairs (exact proof)
- **Edit bill** (total/boundary/interior edits)
- **PCE** (Proof-Carrying English explanation)

### 2. Modular Operators
Each operator family is self-contained:
- Clear input/output types (Grid → Grid, Grid → Mask, etc.)
- Independent unit tests
- Composable via ON and SEQ

### 3. Type Safety
Three core types:
- **Grid**: `np.ndarray` of integers (color values)
- **Mask**: `np.ndarray` of booleans (True = selected)
- **ObjList**: List of connected components

### 4. Phased Implementation
Follow [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md):
- **Phase 1 (Weeks 1-6)**: Baseline 60-80 operators → 60-75% accuracy
- **Phase 2 (Weeks 7-8)**: Gap-filling +25-30 operators → 75-85% accuracy
- **Phase 3 (Weeks 9-10)**: Deep search depth 10-12 → 77-84% (beat record)

## Core Components

### `core/types.py`
Type definitions and type checking:
```python
Grid = np.ndarray      # Shape (H, W), dtype int
Mask = np.ndarray      # Shape (H, W), dtype bool
ObjList = List[Grid]   # List of connected components
```

### `core/invariants.py`
Invariant detection for each grid:
```python
class Invariants:
    histogram: Dict[int, int]      # Color frequencies
    bbox: Tuple[int,int,int,int]   # Bounding box (r0,c0,r1,c1)
    components: ObjList             # Connected components
    symmetries: Set[str]            # {'rot90', 'flip_h', ...}
```

### `core/receipts.py`
Proof and edit tracking:
```python
def edit_counts(a: Grid, b: Grid) -> Tuple[int,int,int]
    # Returns (total_edits, boundary_edits, interior_edits)
    # Returns (-1, -1, -1) for shape mismatch

def compute_residual(task: Dict, program: Callable) -> int
    # Count mismatched outputs vs expected on train pairs

def generate_pce(task_name: str, rule_name: str, params: dict, edits: Tuple) -> str
    # Generate Proof-Carrying English explanation
```

### `core/induction.py`
Induction routines:
```python
def induce_symmetry(train_pairs) -> Optional[Transform]
def induce_crop(train_pairs) -> Optional[Transform]
def induce_color_perm(train_pairs) -> Optional[Transform]
def induce_mask_operation(train_pairs) -> Optional[Transform]
```

### `core/search.py`
Beam search with pruning:
```python
def beam_search(
    train_pairs: List[Tuple[Grid, Grid]],
    operators: List[Callable],
    max_depth: int = 5,
    beam_width: int = 10,
    prune_threshold: float = 0.0  # Residual must not increase
) -> List[Program]
```

### `core/solver.py`
Main solver harness:
```python
def solve_task(
    task: Dict,
    operators: List[Callable],
    search_config: dict
) -> Tuple[List[Grid], List[str]]
    # Returns (predictions, pce_explanations)
```

## Operator Families

### Symmetry (`operators/symmetry.py`)
- `ROT(degrees)` - Rotate 90/180/270
- `FLIP(axis)` - Flip horizontal/vertical

### Spatial (`operators/spatial.py`)
- `CROP(bbox)` - Crop to bounding box
- `SHIFT(dx, dy, wrap=False)` - Shift grid
- `BBOX_NONZERO()` - Get bounding box of non-zero cells

### Masks (`operators/masks.py`)
- `MASK_COLOR(color)` - Mask cells of specific color
- `MASK_NONZERO()` - Mask non-zero cells
- `MASK_RECT(r0,c0,r1,c1)` - Mask rectangular region
- `MASK_COMPONENTS()` - Separate into connected components

### Selection (`operators/selection.py`)
- `KEEP(mask)` - Keep only masked cells (others → 0)
- `REMOVE(mask)` - Remove masked cells (set to 0)

### Composition (`operators/composition.py`)
- `ON(mask, program)` - Apply program only to masked region
- `SEQ(programs)` - Sequential composition

### Color (`operators/color.py`)
- `RECOLOR(mapping)` - Recolor via dictionary
- `COLOR_PERM()` - Permute colors globally

## Migration from Legacy

### From `arc_solver_v1.py` → Modular Structure

1. **Extract Section 0 (Types)** → `core/types.py`
2. **Extract Section 1 (Invariants)** → `core/invariants.py`
3. **Extract Section 2 (Receipts)** → `core/receipts.py`
4. **Extract Section 3 (Core DSL)** → `operators/` by family
5. **Extract Section 4 (Induction)** → `core/induction.py`
6. **Extract Section 5 (Solver)** → `core/solver.py`
7. **Extract Section 6 (Tests)** → `tests/test_*.py`

### Testing After Migration

Run legacy tests to ensure nothing breaks:
```bash
python src/arc_solver/legacy/arc_solver_v1.py  # Should still pass 3/3
```

Run new modular tests:
```bash
python -m pytest src/tests/  # All tests should pass
```

## Next Steps

1. ✅ **Folder structure created** - All directories and `__init__.py` files in place
2. ✅ **Legacy files moved** - `arc_demo.py` and `arc_solver_v1.py` in `legacy/`
3. 🔄 **Extract components** - Split `arc_solver_v1.py` into modular files
4. 🔄 **Add unit tests** - Create `tests/test_*.py` for each module
5. 🔄 **Implement baseline** - Add 60-80 operators per Phase 1 plan
6. 🔄 **Evaluate on training data** - Measure accuracy on 1000 tasks
7. 🔄 **Analyze failures** - Data-driven operator selection for Phase 2
8. 🔄 **Implement gap-filling** - Add 25-30 targeted operators
9. 🔄 **Deep search** - Increase beam depth to 10-12
10. 🔄 **Beat 79.6% record** - Final optimization and submission

## References

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Complete implementation roadmap
- [SUBMISSION_REQUIREMENTS.md](SUBMISSION_REQUIREMENTS.md) - Kaggle submission format
- [universe-intelligence-unified.md](core/universe-intelligence-unified.md) - UI framework guide
- [arc_solver_v1_review.md](arc_solver_v1_review.md) - Phase 1 starter review
