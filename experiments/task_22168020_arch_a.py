#!/usr/bin/env python3
"""
Approach 2: Architecture A - Operator uses UI ORDER internally
Operator wrapper that formulates problem as Horn clauses and uses lfp
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from arc_solver import G
from universe_intelligence import HornKB

def complete_gaps_via_ui_order(grid):
    """
    Fill horizontal gaps using UI ORDER enrichment.

    Formulation as Horn clauses:
    - Facts: cell(r,c,color) for each colored cell in input
    - Rules: if cell(r,c1,color) and cell(r,c2,color) and c1<c<c2 then cell(r,c,color)
             [fill gaps between cells on same row]

    Receipt: residual from lfp convergence
    """
    result = grid.copy()
    colors = np.unique(grid)
    total_residual = 0

    for color in colors:
        if color == 0:
            continue

        # Build Horn KB for this color
        facts = []
        mask = (grid == color)
        rows, cols = np.where(mask)

        if len(rows) == 0:
            continue

        # Add facts: cell(r,c,color) for existing cells
        for r, c in zip(rows, cols):
            facts.append(f"cell({r},{c},{color})")

        # Group by row to find gaps
        row_to_cols = {}
        for r, c in zip(rows, cols):
            if r not in row_to_cols:
                row_to_cols[r] = []
            row_to_cols[r].append(c)

        # Add rules: fill gaps on each row
        rules = []
        for r, cols_in_row in row_to_cols.items():
            min_col = min(cols_in_row)
            max_col = max(cols_in_row)

            # For each column in between, add rule
            for c in range(min_col, max_col + 1):
                if c not in cols_in_row:
                    # Rule: cell(r,min_col,color) ∧ cell(r,max_col,color) → cell(r,c,color)
                    rules.append(
                        ([f"cell({r},{min_col},{color})", f"cell({r},{max_col},{color})"],
                         f"cell({r},{c},{color})")
                    )

        # Solve via lfp
        kb = HornKB(facts=facts, rules=rules)
        Fstar, steps, residual = kb.lfp()
        total_residual += residual

        # Extract result from Fstar
        for fact in Fstar:
            if fact.startswith(f"cell("):
                # Parse: cell(r,c,color)
                parts = fact[5:-1].split(',')
                r = int(parts[0])
                c = int(parts[1])
                result[r, c] = color

    return result, total_residual

def compute_receipt(pred, truth, lfp_residual):
    """Compute receipt: both grid residual and lfp residual"""
    return {
        'residual': np.sum(pred != truth),
        'lfp_residual': lfp_residual,
        'method': 'arch_a_ui_order'
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
    print("APPROACH 2: Architecture A - Operator uses UI ORDER")
    print("="*70)

    # Test on training examples
    print("\nTraining examples:")
    for i, pair in enumerate(task['train']):
        inp = G(pair['input'])
        truth = G(pair['output'])
        pred, lfp_res = complete_gaps_via_ui_order(inp)
        receipt = compute_receipt(pred, truth, lfp_res)

        print(f"  Train {i}: residual={receipt['residual']}, lfp_residual={receipt['lfp_residual']} {'✓' if receipt['residual']==0 else '✗'}")

    # Test on test example
    print("\nTest example:")
    test_inp = G(task['test'][0]['input'])
    test_truth = G(solutions[task_id][0])
    test_pred, lfp_res = complete_gaps_via_ui_order(test_inp)
    test_receipt = compute_receipt(test_pred, test_truth, lfp_res)

    print(f"  Test 0: residual={test_receipt['residual']}, lfp_residual={test_receipt['lfp_residual']} {'✓' if test_receipt['residual']==0 else '✗'}")

    print("\n" + "="*70)
    print(f"Lines of code: ~60 (operator wraps UI ORDER)")
    print(f"Receipt: lfp convergence residual={lfp_res}")
    print("="*70)
