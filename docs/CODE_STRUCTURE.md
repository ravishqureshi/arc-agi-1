# Code Structure & Development Workflow

## Problem
Kaggle submission must be a **self-contained notebook** with no internet access. But we'll have 1000s of lines of code that need to be maintainable.

## Solution: Local Development + %%writefile Pattern

### Folder Structure

```
arc-agi-1/
├── src/                    # Local Python modules for development
│   ├── __init__.py         # Package marker
│   ├── *.py               # Your solver modules (create as needed)
│   └── (organize as you like - no forced structure)
│
├── notebooks/              # Kaggle notebooks
│   └── submission.ipynb    # Final submission notebook
│
├── scripts/                # Helper scripts (optional)
│   └── build_notebook.py   # Auto-generate notebook from src/ (if needed)
│
├── data/                   # Competition data (already setup)
└── docs/                   # Documentation (already setup)
```

### Development Workflow

#### 1. Local Development
- Write code in `src/` as normal Python modules
- Use your IDE with autocomplete, debugging, etc.
- Test locally with data from `data/` folder
- Run unit tests, iterate quickly

#### 2. Kaggle Notebook Structure
Use `%%writefile` magic to embed modules in notebook:

```python
# Cell 1: Overview
"""
ARC AGI Solver - Pure Mathematics Approach
This notebook creates Python modules and runs the solver.
"""

# Cell 2: Create module 1
%%writefile solver_utils.py
import numpy as np

def some_function():
    # ... your code here
    pass

# Cell 3: Create module 2
%%writefile pattern_detector.py
import numpy as np
from solver_utils import some_function

def detect_patterns():
    # ... your code here
    pass

# Cell 4: Create main solver
%%writefile solver.py
from pattern_detector import detect_patterns

def solve(input_grid, train_pairs):
    # ... your code here
    return output_grid

# Cell 5: Import and run
import solver
import json

# Load test data
with open('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
    test_challenges = json.load(f)

# Generate predictions
submission = {}
for task_id, task_data in test_challenges.items():
    # ... use solver.solve() here
    pass

# Save submission.json
with open('submission.json', 'w') as f:
    json.dump(submission, f)
```

### Key Points

1. **No forced file structure in src/** - Organize as makes sense for your approach
2. **Notebook is self-contained** - All code embedded via `%%writefile`
3. **Notebook is readable** - Acts as table of contents showing all modules
4. **Easy to update** - Copy code from `src/` to `%%writefile` cells when ready

### Optional: Auto-build Script
Create `scripts/build_notebook.py` to automatically:
- Read all `.py` files from `src/`
- Generate notebook with `%%writefile` cells
- Add boilerplate for loading data and saving submission.json

This keeps local dev clean while ensuring Kaggle submission works.

### What NOT to do
- ❌ Don't hardcode local file paths in src/
- ❌ Don't assume internet access
- ❌ Don't import external packages beyond numpy/scipy
- ❌ Don't create 10,000 line single notebook cells

### What TO do
- ✅ Keep modules focused and modular
- ✅ Test everything locally first
- ✅ Use `/kaggle/input/arc-prize-2025/` paths in final notebook
- ✅ Validate submission.json format before submitting
