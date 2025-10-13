# ARC AGI 2025 - Submission Requirements (Source of Truth)

## Critical Submission Requirements

### 1. Submission Format
- **Must be a Kaggle Notebook** (not a script or local code)
- Notebook must run completely within **12 hours** (CPU or GPU)
- **No internet access** allowed in the notebook
- Must generate a file named **`submission.json`** in the working directory

### 2. Output File Structure

#### File name
```
submission.json
```

#### JSON Structure
```json
{
  "task_id_1": [
    {"attempt_1": [[...]], "attempt_2": [[...]]}
  ],
  "task_id_2": [
    {"attempt_1": [[...]], "attempt_2": [[...]]}
  ],
  "task_id_3": [
    {"attempt_1": [[...]], "attempt_2": [[...]]},
    {"attempt_1": [[...]], "attempt_2": [[...]]}
  ]
}
```

#### Critical Rules for submission.json
1. **ALL task_ids** from input file (`arc-agi_test_challenges.json`) MUST be present
2. Each task_id maps to a **list** (even if only one test output)
3. Each test output needs **exactly 2 attempts**: `attempt_1` and `attempt_2`
4. Both attempts must be present even if identical
5. Order of predictions must match order of test inputs in the task
6. Grid values are integers 0-9 (inclusive)
7. Grid dimensions: minimum 1x1, maximum 30x30

#### Example with Multiple Test Outputs
```json
{
  "00576224": [
    {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}
  ],
  "12997ef3": [
    {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]},
    {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}
  ]
}
```

### 3. Input Data Files (Available in Notebook)

During development:
- `arc-agi_training_challenges.json` - training tasks with demo pairs
- `arc-agi_training_solutions.json` - training solutions
- `arc-agi_evaluation_challenges.json` - validation tasks
- `arc-agi_evaluation_solutions.json` - validation solutions
- `sample_submission.json` - example submission format

During submission (notebook rerun):
- `arc-agi_test_challenges.json` - **240 unseen test tasks** (replaces placeholder)

### 4. Task Structure

Each task in the input JSON:
```json
{
  "task_id": {
    "train": [
      {"input": [[...]], "output": [[...]]}  // typically 2-4 pairs
    ],
    "test": [
      {"input": [[...]]}  // 1-3 test inputs, need to predict outputs
    ]
  }
}
```

### 5. Scoring
- For each test output, if either `attempt_1` OR `attempt_2` exactly matches ground truth → score 1
- Otherwise → score 0
- Final score = sum of scores / total number of test outputs
- **Exact match required** (every cell must match)

### 6. Submission Limits
- Maximum **1 submission per day**
- Select up to **2 final submissions** for judging

### 7. Skeleton Notebook Must Have

```python
import json
import numpy as np

# Load test challenges
with open('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
    test_challenges = json.load(f)

# Initialize submission dictionary
submission = {}

# Process each task
for task_id, task_data in test_challenges.items():
    train_pairs = task_data['train']
    test_inputs = task_data['test']

    # Your algorithm here
    predictions = []
    for test_input in test_inputs:
        input_grid = test_input['input']

        # Generate 2 attempts
        attempt_1 = solve(input_grid, train_pairs)  # Your function
        attempt_2 = solve(input_grid, train_pairs)  # Can be same or different

        predictions.append({
            "attempt_1": attempt_1,
            "attempt_2": attempt_2
        })

    submission[task_id] = predictions

# Save submission
with open('submission.json', 'w') as f:
    json.dump(submission, f)
```

### 8. Key Validation Checks Before Submission

```python
# 1. Check all task IDs present
assert set(submission.keys()) == set(test_challenges.keys())

# 2. Check each task has correct number of predictions
for task_id in test_challenges:
    assert len(submission[task_id]) == len(test_challenges[task_id]['test'])

# 3. Check each prediction has both attempts
for task_id, preds in submission.items():
    for pred in preds:
        assert 'attempt_1' in pred
        assert 'attempt_2' in pred
        assert isinstance(pred['attempt_1'], list)
        assert isinstance(pred['attempt_2'], list)

# 4. Check grid values are 0-9
# 5. Check grid dimensions are <= 30x30

print("Validation passed!")
```

### 9. Competition Details
- **Competition**: ARC Prize 2025
- **Link**: https://www.kaggle.com/competitions/arc-prize-2025
- **Submission deadline**: November 3, 2025
- **CLI download**: `kaggle competitions download -c arc-prize-2025`

### 10. Quick Reference: What to Build

1. Create Kaggle notebook
2. Load `arc-agi_test_challenges.json`
3. For each task:
   - Use `train` pairs to understand the pattern
   - Generate 2 prediction attempts for each `test` input
4. Save as `submission.json` with exact format above
5. Validate structure
6. Submit (runs for up to 12 hours)
