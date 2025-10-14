#!/usr/bin/env python3
"""
Clean UI implementation - main entry point.

This is a thin wrapper around the arc_solver package.
All logic is in modular files for easy extension.
"""

import json
from pathlib import Path

# Import from arc_solver package
from arc_solver import (
    ARCInstance, G,
    beam_search, solve_with_beam
)

if __name__ == "__main__":
    print("=" * 70)
    print("UI CLEAN - From clarification doc (8 inducers)")
    print("=" * 70)

    # Load data
    data_dir = Path(__file__).parent.parent / 'data'
    with open(data_dir / 'arc-agi_training_challenges.json') as f:
        challenges = json.load(f)
    with open(data_dir / 'arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    # Test on ALL 1000 tasks
    solved = 0
    total = len(challenges)

    for idx, (task_id, task_data) in enumerate(challenges.items(), 1):
        train = [(G(p['input']), G(p['output'])) for p in task_data['train']]
        test_in = [G(p['input']) for p in task_data['test']]
        test_out = [G(g) for g in solutions[task_id]]

        inst = ARCInstance(task_id, train, test_in, test_out)

        if idx % 100 == 0:
            print(f"Progress: {idx}/{total} ({100*idx/total:.1f}%), Solved: {solved}")

        steps, preds, residuals = solve_with_beam(inst, max_depth=6, beam_size=160)

        if steps and all(r == 0 for r in residuals):
            solved += 1
            print(f"✓ {task_id}: {' → '.join(op.name for op in steps)}")

    print(f"\n{'='*70}")
    print(f"FINAL: Solved {solved}/{total} = {100*solved/total:.2f}%")
    print(f"{'='*70}")
