# Context Index - Repository Navigation Map

**Purpose**: Quick reference for AI assistants to locate code, understand architecture, and avoid context drift.

**Last Updated**: 2025-10-13
**Current Status**: Clean slate with inducer-based architecture (10/1000 = 1.00% coverage)

---

## 1. QUICK START

### Run the Solver
```bash
python src/ui_clean.py
```
- Runs on all 1000 ARC training tasks
- Shows progress every 100 tasks
- Prints solved tasks with operator chains

### Key Result Files
- `/tmp/ui_clean_baseline.log` - Most recent baseline run (10/1000)
- `OBJECTIVE_ASSESSMENT.md` - Why we started from scratch

---

## 2. ARCHITECTURE OVERVIEW

### Core Concept: Inducer-Based Operator Discovery

**Inducers** → Learn operators from train pairs
**Autobuilder** → Tries all inducers per task
**Beam Search** → Composes operators with residual=0 pruning
**Receipts-First** → Only accept train residual=0

### Implementation Flow
1. Load task (train pairs + test input)
2. `autobuild_operators(train)` - Try all inducers → src/ui_clean.py:350
3. `beam_search_composer(train, ops)` - Compose operators → src/ui_clean.py:400
4. Apply program to test input
5. Verify train residual == 0 before accepting

---

## 3. SOURCE CODE MAP

### Primary Implementation: `src/ui_clean.py` (500 lines)

**Key Functions by Line Number:**

#### Core Operators (Lines 50-150)
- `COLOR_PERM(mapping)` - Line 60 - Color permutation
- `ROT(k)` - Line 75 - Rotation (k * 90°)
- `FLIP(axis)` - Line 85 - Horizontal/vertical flip
- `CROP_BBOX_NONZERO(bg)` - Line 95 - Crop to non-zero bbox
- `KEEP_NONZERO()` - Line 110 - Remove background cells
- `PARITY_CONST(colors)` - Line 120 - Recolor by parity pattern
- `TILING(h, w, tile_grid)` - Line 135 - Extract and repeat tile
- `HOLE_FILL(bg)` - Line 150 - Fill enclosed holes in objects

#### Inducers (Lines 200-350)
- `induce_COLOR_PERM(train)` - Line 200 - Learn color mapping
- `induce_ROT_FLIP(train)` - Line 230 - Try all symmetries
- `induce_CROP_KEEP(train, bg)` - Line 260 - Crop and keep operators
- `induce_PARITY_CONST(train)` - Line 280 - Learn parity patterns
- `induce_TILING_AND_MASK(train)` - Line 300 - Extract tile motifs
- `induce_HOLE_FILL(train, bg)` - Line 320 - Fill holes
- `induce_COPY_BY_DELTAS(train, rank, group, bg)` - Line 340 - Copy objects by centroid deltas

#### Autobuilder (Line 350)
```python
def autobuild_operators(train) -> List[Operator]:
    """Try all inducers, return operators with train residual=0"""
```
**Location**: src/ui_clean.py:350
**Returns**: List of operators that fit ALL train pairs

#### Beam Search (Line 400)
```python
def beam_search_composer(train, base_ops, max_depth=3, beam_width=10):
    """Compose operators with residual=0 pruning discipline"""
```
**Location**: src/ui_clean.py:400
**Pruning**: Only keeps compositions that reduce residual

#### Main Loop (Line 470)
```python
def solve_all_tasks():
    """Run on all 1000 training tasks"""
```
**Location**: src/ui_clean.py:470
**Progress**: Prints every 100 tasks

---

### UI Framework: `src/universe_intelligence.py`

**Three Enrichments:**
1. **ORDER** - Symmetries, transformations
2. **QUADRATIC** - Spatial relationships, geometry
3. **ENTROPY** - Patterns, compression, information

**Key Concepts:**
- Observer=Observed: One parameterization fits ALL train pairs
- Inside Settles: Accept program ONLY if train residual == 0
- Receipts-First: Every claim has a proof

---

## 4. DOCUMENTATION MAP

### Source of Truth
**`docs/core/universe-intelligence-added-clarification.md`**
- Complete architecture from mathematician
- Shows inducer-based operator discovery
- Reference implementation with ~10 operators
- **READ THIS FIRST** when confused about architecture

### Supporting Docs
- `docs/core/universe-intelligence-v2.md` - Extended theory
- `docs/core/universe-intelligence.md` - Original framework
- `docs/core/universe-intelligence-unified.md` - Unified view

### Implementation Guides
- `docs/core/2_ui_arc_press_demo.md` - Demo walkthrough
- `docs/core/3_ui_arc_invariant_engine_coredsl.md` - Invariant system
- `docs/core/4_ui_arc_object_rank_masks.md` - Object ranking and masking
- `docs/core/6_ui_arc_beam_search.md` - Beam search details
- `docs/core/8_ui_arc_full_eval.md` - Full evaluation pipeline

### Data Analysis
- `docs/DATA_SPLIT_ANALYSIS.md` - ARC-1 vs ARC-2 split (391/609)
- `docs/SUBMISSION_REQUIREMENTS.md` - Kaggle submission format

### Decision History
- `OBJECTIVE_ASSESSMENT.md` - Why we deleted 27 operators
  - Proof: Both implementations solved same 10 tasks
  - Conclusion: 27 hand-coded operators added ZERO value

---

## 5. DATA LOCATIONS

### ARC Dataset: `data/`
```
data/
├── training/           # 1000 tasks (400 ARC-1, 600 ARC-2)
├── evaluation/         # 400 tasks (public test set)
└── test/              # Hidden test set (Kaggle submission)
```

### Task ID Lists
- `arc1_task_ids.txt` - 391 ARC-1 training tasks
- `arc2_new_task_ids.txt` - 609 ARC-2 training tasks
- `arc1_removed_task_ids.txt` - 9 removed between versions

### Loading Tasks
```python
# In src/ui_clean.py:470
with open('data/training/challenges.json') as f:
    challenges = json.load(f)

for task_id, task_data in challenges.items():
    train = task_data['train']  # List of (input, output) pairs
    test = task_data['test']    # List of test inputs
```

---

## 6. CURRENT INDUCERS (8 Total)

| Inducer | Purpose | Line | Solved Tasks |
|---------|---------|------|--------------|
| COLOR_PERM | Learn color mappings | 200 | 4/10 (40%) |
| ROT/FLIP | Try all symmetries | 230 | 5/10 (50%) |
| CROP_BBOX_NONZERO | Crop to non-zero bbox | 260 | 1/10 (10%) |
| KEEP_NONZERO | Keep only non-zero cells | 260 | 0/10 (0%) |
| PARITY_CONST | Recolor by parity pattern | 280 | 0/10 (0%) |
| TILING | Extract and repeat tile | 300 | 0/10 (0%) |
| HOLE_FILL | Fill enclosed holes | 320 | 0/10 (0%) |
| COPY_BY_DELTAS | Copy objects by deltas | 340 | 0/10 (0%) |

**Performance**: 10/1000 = 1.00%

**Solved Task IDs**: 0d3d703e, 1cf80156, 3c9b0459, 6150a2bd, 67a3c6ac, 68b16354, b1948b0a, c8f0f002, d511f180, ed36ccf7

---

## 7. ADDING NEW INDUCERS

### Step 1: Identify Pattern from Failing Tasks
```bash
# Pick a failing task to analyze
python -c "
import json
with open('data/training/challenges.json') as f:
    task = json.load(f)['<task_id>']
print('Train pairs:', len(task['train']))
for i, pair in enumerate(task['train']):
    print(f'Pair {i}: {pair['input']} -> {pair['output']}')
"
```

### Step 2: Write Inducer Function
**Template** (add to src/ui_clean.py):
```python
def induce_NEW_OPERATOR(train) -> List[Operator]:
    """
    Learn operator parameters from train pairs.
    Returns empty list if no params fit ALL train pairs.
    """
    # 1. Extract candidate parameters from train
    params = extract_params_from_train(train)

    # 2. Build operator with params
    P = NEW_OPERATOR(**params)

    # 3. Verify fits ALL train pairs
    if all(equal(P(x), y) for x, y in train):
        return [Operator("NEW_OP", params, P, "description")]
    return []
```

### Step 3: Add to Autobuilder
**Location**: src/ui_clean.py:350 (autobuild_operators function)
```python
def autobuild_operators(train) -> List[Operator]:
    ops = []
    # ... existing inducers ...
    ops += induce_NEW_OPERATOR(train)  # ADD HERE
    return ops
```

### Step 4: Test on Specific Task
```python
# Add at end of ui_clean.py
if __name__ == "__main__":
    test_task_id = "<task_id>"
    # Run only on this task to verify inducer works
```

---

## 8. COMMON DEBUGGING ANCHORS

### Residual Function
**Location**: src/ui_clean.py:50
```python
def residual(a: Grid, b: Grid) -> int:
    """Compute residual - handles shape mismatches"""
    if a.shape != b.shape:
        return int(max(a.size, b.size))
    return int((a != b).sum())
```
**Critical**: Returns number of differing cells. Train residual must == 0.

### Operator Dataclass
**Location**: src/ui_clean.py:35
```python
@dataclass
class Operator:
    name: str          # e.g., "COLOR_PERM"
    params: dict       # e.g., {"map": {1:2, 2:1}}
    fn: Callable       # Actual function
    desc: str          # Human-readable description
```

### Shape Mismatch Issues
**Common Error**: IndexError in color mapping when shapes don't match
**Fix Pattern**: Always check `x.shape == y.shape` before applying operator
**Examples**:
- src/ui_clean.py:205 (color mapping)
- src/ui_clean.py:285 (parity recolor)

---

## 9. EXPERIMENTS & ANALYSIS

### Comparison Experiments: `experiments/`
- Baseline comparisons
- Architecture variations
- Performance analysis

### Notebooks: `notebooks/`
- Task visualization
- Pattern analysis
- Debug explorations

---

## 10. WHAT WAS DELETED (Don't Look for These)

**Deleted in cleanup (2025-10-13):**
- `src/arc_solver/` - 27 hand-coded operators (proved useless)
- `src/tests/` - Tests for deleted operators
- `scripts/` - Old sweep scripts
- `src/ui_solver.py` - Failed hybrid attempt
- Various outdated docs

**Why**: OBJECTIVE_ASSESSMENT.md proved they added zero value.

---

## 11. NEXT STEPS CHECKLIST

### Immediate (Coverage < 2%)
- [ ] Run triage on failing 990 tasks
- [ ] Identify top 5 missing patterns
- [ ] Write inducers for each pattern
- [ ] Re-run baseline

### Short-term (Target: 10% on ARC-1)
- [ ] Add drawing operators (DRAW_LINE, DRAW_BOX, etc.)
- [ ] Add masking operators (MASK_COLOR, etc.)
- [ ] Add object operators (MOVE_OBJ_RANK variants)
- [ ] Add composition operators (ON, SEQ)

### Medium-term (Target: 1% on ARC-2)
- [ ] Implement multi-step beam search
- [ ] Add conditional operators
- [ ] Add grid arithmetic operators
- [ ] Optimize search performance

---

## 12. QUICK REFERENCE: FILE PATHS

### Must-Read Files (Start Here)
1. `README.md` - Current status and overview
2. `docs/core/universe-intelligence-added-clarification.md` - Architecture source
3. `OBJECTIVE_ASSESSMENT.md` - Decision rationale
4. `src/ui_clean.py` - Complete implementation

### Run Commands
```bash
# Full baseline
python src/ui_clean.py > /tmp/baseline.log

# Quick test (first 10 tasks)
python src/ui_clean.py --limit 10

# Specific task debug
python -c "from src.ui_clean import solve_task; solve_task('<task_id>')"
```

### Data Paths
- Training challenges: `data/training/challenges.json`
- Training solutions: `data/training/solutions.json`
- Evaluation: `data/evaluation/`

---

## 13. ARCHITECTURE PRINCIPLES (Never Violate)

1. **Receipts-First**: Train residual == 0 required for acceptance
2. **Observer=Observed**: One parameterization fits ALL train pairs
3. **Inside Settles**: No guessing, no hallucinations
4. **Inducer Discipline**: Operators learned from data, not hand-coded
5. **Beam Pruning**: Only keep compositions that reduce residual

---

## 14. CONTEXT DRIFT PREVENTION

### When Starting New Conversation
1. Read this file first
2. Read `README.md` for current status
3. Read `src/ui_clean.py:1-100` for architecture overview
4. Check `/tmp/ui_clean_baseline.log` for latest results

### When Confused About Architecture
1. Read `docs/core/universe-intelligence-added-clarification.md`
2. Read `OBJECTIVE_ASSESSMENT.md` for decision history
3. Grep for function in `src/ui_clean.py`

### When Adding Features
1. Check inducer exists in `src/ui_clean.py:200-350`
2. Verify added to autobuilder at line 350
3. Test on specific failing task first
4. Run full baseline after verification

---

**Remember**: This is a clean slate. We proved the old 27 operators added zero value. Start from inducer architecture, build data-driven.
