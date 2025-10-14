#!/usr/bin/env python3
"""
Approach 1: Hand-coded COMPLETE_SYMMETRY operator
Traditional operator approach - explicit procedural code
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from arc_solver import G

def complete_horizontal_gaps(grid):
    """
    Fill horizontal gaps for each colored object.

    Algorithm:
    1. For each non-zero color
    2. For each row, find leftmost and rightmost cell of that color
    3. Fill all columns between them

    Receipt: residual = 0 if output matches expected pattern
    """
    result = grid.copy()
    colors = np.unique(grid)

    for color in colors:
        if color == 0:
            continue

        # Find all cells with this color
        mask = (grid == color)
        rows, cols = np.where(mask)

        if len(rows) == 0:
            continue

        # Group by row
        row_to_cols = {}
        for r, c in zip(rows, cols):
            if r not in row_to_cols:
                row_to_cols[r] = []
            row_to_cols[r].append(c)

        # For each row, fill gap between min and max column
        for r, cols_in_row in row_to_cols.items():
            min_col = min(cols_in_row)
            max_col = max(cols_in_row)

            # Fill entire span
            for c in range(min_col, max_col + 1):
                result[r, c] = color

    return result

def compute_receipt(pred, truth):
    """Compute receipt: residual = number of mismatched cells"""
    return {
        'residual': np.sum(pred != truth),
        'method': 'hand-coded'
    }

if __name__ == "__main__":
    import json
    from pathlib import Path

    # Load task
    data_dir = Path('data')
    with open(data_dir / 'arc-agi_training_challenges.json') as f:
        challenges = json.load(f)
    with open(data_dir / 'arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    task_id = '22168020'
    task = challenges[task_id]

    print("="*70)
    print("APPROACH 1: Hand-coded COMPLETE_SYMMETRY operator")
    print("="*70)

    # Test on training examples
    print("\nTraining examples:")
    for i, pair in enumerate(task['train']):
        inp = G(pair['input'])
        truth = G(pair['output'])
        pred = complete_horizontal_gaps(inp)
        receipt = compute_receipt(pred, truth)

        print(f"  Train {i}: residual={receipt['residual']} {'✓' if receipt['residual']==0 else '✗'}")

    # Test on test example
    print("\nTest example:")
    test_inp = G(task['test'][0]['input'])
    test_truth = G(solutions[task_id][0])
    test_pred = complete_horizontal_gaps(test_inp)
    test_receipt = compute_receipt(test_pred, test_truth)

    print(f"  Test 0: residual={test_receipt['residual']} {'✓' if test_receipt['residual']==0 else '✗'}")

    print("\n" + "="*70)
    print(f"Lines of code: ~50 (procedural, explicit)")
    print("="*70)
