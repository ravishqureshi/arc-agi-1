# ARC AGI 2025 - Pure Mathematics Approach

Solving the Abstraction and Reasoning Corpus using mathematical principles.

**Competition**: [ARC Prize 2025](https://www.kaggle.com/competitions/arc-prize-2025)
**Goal**: Beat ARC AGI with numpy, scipy, and pure mathematics - no LLMs

📚 **Quick Navigation:** See [docs/CONTEXT_INDEX.md](docs/CONTEXT_INDEX.md) for a complete map of all documentation and code.

## Project Structure

```
arc-agi-1/
├── data/                          # Competition data files
│   ├── README.md                  # Data files documentation
│   ├── arc-agi_training_challenges.json    (3.8 MB)
│   ├── arc-agi_training_solutions.json     (643 KB)
│   ├── arc-agi_evaluation_challenges.json  (962 KB)
│   ├── arc-agi_evaluation_solutions.json   (219 KB)
│   ├── arc-agi_test_challenges.json        (991 KB - placeholder)
│   └── sample_submission.json              (19 KB)
│
├── docs/                          # Documentation
│   ├── SUBMISSION_REQUIREMENTS.md # Source of truth for submission format
│   ├── CODE_STRUCTURE.md          # How to organize code (%%writefile pattern)
│   ├── arc-agi-kaggle-docs.md     # Competition documentation
│   └── core/                      # Universe Intelligence framework docs
│       ├── universe-intelligence.md        # V1: Quadratic enrichment
│       ├── universe-intelligence-v2.md     # V2: All three enrichments
│       ├── universe-intelligence-unified.md # Complete guide
│       └── ui_arc_demo.md         # ARC demo explanation
│
├── notebooks/                     # Development notebooks
│   └── submission.ipynb           # Final submission notebook (future)
│
├── src/                           # Source code
│   ├── universe_intelligence.py   # Core UI: ORDER/ENTROPY/QUADRATIC enrichments
│   ├── arc_solver/                # Main ARC solver package (MODULAR)
│   │   ├── __init__.py            # Main package exports
│   │   ├── arc_solver_v1.py       # Monolithic starter (for reference)
│   │   ├── core/                  # Core architecture
│   │   │   ├── types.py           # Grid, Mask, ObjList types
│   │   │   ├── invariants.py      # Invariant engine (histogram, components, symmetries)
│   │   │   ├── receipts.py        # Edit bills, residual, PCE
│   │   │   ├── induction.py       # Rule induction from train pairs
│   │   │   └── solver.py          # Main solver harness
│   │   ├── operators/             # DSL operators by family
│   │   │   ├── symmetry.py        # ROT, FLIP
│   │   │   ├── spatial.py         # CROP, BBOX
│   │   │   ├── masks.py           # MASK_COLOR, MASK_NONZERO, KEEP, REMOVE
│   │   │   └── composition.py     # ON, SEQ
│   │   └── legacy/                # Previous implementations (reference only)
│   │       └── arc_demo.py        # 6-rule demo (12/1000 solved)
│   ├── tests/                     # Unit tests
│   │   └── test_arc_solver.py     # Modular tests (ALL PASSING ✅)
│   └── demos/                     # Demo scripts
│
├── scripts/                       # Helper scripts
│   └── build_notebook.py          # Auto-generate notebook from src/ (future)
│
├── arc_version_split.py           # Utility: Distinguish ARC-1 vs ARC-2 tasks
├── example_version_tracking.py    # Example: Track ARC-1 vs ARC-2 scores separately
├── arc1_task_ids.txt              # 391 ARC-1 tasks (present in ARC-2)
├── arc2_new_task_ids.txt          # 609 new tasks in ARC-2
├── arc1_removed_task_ids.txt      # 9 ARC-1 tasks removed in ARC-2
│
├── venv/                          # Python virtual environment
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Setup

### 1. Python Environment
```bash
# Virtual environment already created with Python 3.11
source venv/bin/activate

# Installed packages:
# - kaggle (CLI for competition)
# - numpy (numerical computing)
# - scipy (scientific computing)
```

### 2. Kaggle API (Already configured)
Credentials in `~/.kaggle/kaggle.json`

### 3. Competition Data (Already downloaded)
Located in `data/` folder - see `data/README.md` for details

## Dataset: ARC-AGI-2 (includes ARC-1)

**Your training data contains ARC-AGI-2** - the harder, calibrated benchmark released in 2025.

### Composition

| Dataset | Tasks | Percentage | Description |
|---------|-------|------------|-------------|
| **ARC-AGI-1 tasks** | 391 | 39.1% | Original tasks from 2019 benchmark |
| **ARC-AGI-2 new tasks** | 609 | 60.9% | New tasks designed to challenge frontier AI |
| **Total training** | 1,000 | 100% | Combined dataset |

**Note**: 9 tasks from original ARC-1 (400 tasks) were removed in ARC-2 (susceptible to brute force).

### Key Differences (ARC-1 → ARC-2)

- **Size**: 400 → 1,000 training tasks
- **Difficulty**: Calibrated via 400+ human participants (avg 2.7 min/task)
- **New concepts**: In-context symbol definition (objects represent other things)
- **Cleaned**: Removed brute-forceable tasks
- **Performance**: Pure LLMs score 0% on ARC-2 (vs higher on ARC-1)

### Track ARC-1 vs ARC-2 Performance Separately

Use the provided utilities to measure performance on each version:

```python
from arc_version_split import is_arc1_task, is_arc2_new_task
from example_version_tracking import compute_separate_scores

# Check task version
if is_arc1_task('007bbfb7'):
    print("This task is from ARC-AGI-1")

# Compute separate scores
results = compute_separate_scores(predictions, solutions)
print(f"ARC-1 Score: {results['arc1_score']:.1%}")
print(f"ARC-2 Score: {results['arc2_score']:.1%}")
```

**Files**:
- `arc_version_split.py` - Utility functions to check task versions
- `example_version_tracking.py` - Example usage and score tracking
- `arc1_task_ids.txt` - List of 391 ARC-1 tasks in dataset
- `arc2_new_task_ids.txt` - List of 609 new ARC-2 tasks
- `arc1_removed_task_ids.txt` - List of 9 ARC-1 tasks removed

## Core Framework: Universe Intelligence

This project uses **Universe Intelligence (UI)** - a mathematical reasoning framework with three enrichments:

### 1. `src/universe_intelligence.py` - Core Implementation

**Three mathematical tools in one framework:**

| Enrichment | What | Use For | Receipt |
|------------|------|---------|---------|
| **ORDER** | Horn clauses, least fixed points | Logic, rules, pattern inference | residual = 0 |
| **ENTROPY** | KL divergence, Fenchel-Young | Probability, uncertainty, consensus | gap ≈ 1e-12 |
| **QUADRATIC** | Laplacian, Green identity | Spatial transforms, geometry | gap ≈ 1e-12 |

**Key classes:**
- `HornKB` - Horn clause knowledge base with least fixed point computation
- `DtN` - Dirichlet-to-Neumann operator for boundary problems
- Functions: `softmax`, `entropy_phi`, `KL`, `schur_dtn`, `solve_dirichlet`, `green_gap`

**Usage:**
```python
import src.universe_intelligence as ui

# ORDER: Logic reasoning
kb = ui.HornKB(facts=['edge(A,B)'], rules=[...])
derived_facts, steps, residual = kb.lfp()  # residual = 0 (exact proof)

# ENTROPY: Probabilistic reasoning
p = ui.softmax([1.0, 2.0, 0.5])
val, s, conj, inner, gap = ui.entropy_phi(p)  # gap ≈ 1e-12

# QUADRATIC: Spatial geometry
L = ui.random_connected_laplacian(n=50)
dtn = ui.schur_dtn(L, boundary_indices=[0, 10, 20])
gap, Ein, Ebd = ui.green_gap(L, dtn, boundary_values)  # gap ≈ 1e-14
```

**Documentation**: See `docs/core/universe-intelligence-unified.md` for complete guide.

### 2. `src/arc_demo.py` - Demo ARC Solver

**Rule-based solver with 6 transformation rules:**
1. **Symmetry** - rot90/180/270, flip horizontal/vertical
2. **Color permutation** - Global color mapping
3. **Crop bbox** - Extract bounding box of non-zero content
4. **Move bbox to origin** - Move bounding box to top-left corner (preserves grid size)
5. **Mirror left to right** - Mirror left half to right half (requires even width)
6. **Component recolor** - Recolor connected components by size

**Features:**
- Learns from training pairs (exact match required)
- Generates predictions with **receipts**: bills, boundary edits, proof-carrying explanations
- Demo tasks: 3/3 accuracy on test cases

**Run demo:**
```bash
source venv/bin/activate
python src/arc_demo.py
```

**Output:**
```
[color_perm test#0] ok=True  bill=4 (boundary 4, interior 0)
[rot90 test#0] ok=True  bill=7 (boundary 7, interior 0)
[crop_bbox test#0] ok=True  bill=-1 (boundary -1, interior -1)
DEMO accuracy: 3/3 test grids matched exactly.
```

## Development Workflow

1. **Explore data** in `data/` folder
2. **Read requirements** in `docs/SUBMISSION_REQUIREMENTS.md`
3. **Study UI framework** in `docs/core/universe-intelligence-unified.md`
4. **Test demo solver**: `python src/arc_demo.py`
5. **Develop algorithm** locally with training data
6. **Validate** with evaluation data
7. **Create Kaggle notebook** for submission (use `%%writefile` pattern from `docs/CODE_STRUCTURE.md`)
8. **Submit** (max 1 per day)

## Submission Requirements

- **Format**: Kaggle Notebook
- **Output**: `submission.json`
- **Time limit**: 12 hours
- **Internet**: Disabled
- **Scoring**: Exact match, 2 attempts per test output

See `docs/SUBMISSION_REQUIREMENTS.md` for complete details.

## Key Constraints

- Only numpy, scipy, and standard library
- No pre-trained models or LLMs
- Must work on unseen test tasks
- Notebook must be self-contained

## Mathematical Approach

**Philosophy**: Solve ARC with **provable reasoning** instead of probabilistic guessing.

**Universe Intelligence bedrock:**
- **Inside settles** - No surprises in the middle (least fixed points)
- **Edges write** - All change happens at boundaries
- **Receipts** - Every answer has mathematical proof (residual = 0, gaps ≈ 1e-12)

**Why this beats LLMs:**
1. **Deterministic** - No hallucinations, every claim has a receipt
2. **Compositional** - Combine logic + probability + geometry as needed
3. **Efficient** - CPU-only, millisecond queries (no GPU needed)
4. **Interpretable** - Each step has a mathematical certificate
5. **Provable** - If there's no proof, system refuses to answer

## Resources

- Competition: https://www.kaggle.com/competitions/arc-prize-2025
- ARC Prize: https://ARCprize.org
- Interactive tasks: https://ARCprize.org (explore the challenges)

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Test Universe Intelligence framework
python -c "import src.universe_intelligence as ui; print('✅ UI loaded')"

# Run modular test suite
python src/tests/test_arc_solver.py

# Run legacy demos
python src/arc_solver/legacy/arc_demo.py
python src/arc_solver/arc_solver_v1.py

# Explore training data
python -c "import json; d=json.load(open('data/arc-agi_training_challenges.json')); print(f'{len(d)} training tasks')"
```

## Next Steps

1. ✅ Framework ready (`universe_intelligence.py` with all three enrichments)
2. ✅ Demo solver working (`arc_demo.py` - 3/3 accuracy on test tasks)
3. 🔄 Expand rule catalog (add more transformation patterns)
4. 🔄 Build production solver for full ARC benchmark
5. 🔄 Create Kaggle submission notebook
6. 🔄 Submit and iterate (1 submission per day limit)
