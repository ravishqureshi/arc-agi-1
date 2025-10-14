#!/usr/bin/env python3
"""
Clean UI implementation - main entry point.

This is a thin wrapper around the arc_solver package.
All logic is in modular files for easy extension.
"""

import json
from pathlib import Path
from datetime import datetime

# Import from arc_solver package
from arc_solver import (
    ARCInstance, G,
    solve_with_beam,
    task_sha, program_sha, log_receipt
)


def run_solver(dataset_dir=None, out_dir=None):
    """
    Run solver on ARC dataset and write receipts to out_dir.

    Args:
        dataset_dir: Path to dataset directory (default: data/)
        out_dir: Output directory for receipts (default: runs/YYYY-MM-DD)
    """
    if dataset_dir is None:
        dataset_dir = Path(__file__).parent.parent / 'data'
    else:
        dataset_dir = Path(dataset_dir)

    if out_dir is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_dir = f"runs/{date_str}"

    print("=" * 70)
    print("UI CLEAN - From clarification doc (8 inducers)")
    print(f"Output: {out_dir}")
    print("=" * 70)

    # Load data
    with open(dataset_dir / 'arc-agi_training_challenges.json') as f:
        challenges = json.load(f)
    with open(dataset_dir / 'arc-agi_training_solutions.json') as f:
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

        steps, preds, test_residuals, metadata = solve_with_beam(inst, max_depth=6, beam_size=160)

        # Build receipt record
        status = "solved" if (steps and all(r == 0 for r in test_residuals)) else "failed"
        program = [op.name for op in steps] if steps else []

        receipt = {
            "task": task_id,
            "status": status,
            "program": program,
            "total_residual": metadata["total_residual"],
            "residuals_per_pair": metadata["residuals_per_pair"],
            "palette_invariants": metadata["palette_invariants"],
            "component_invariants": metadata["component_invariants"],
            "beam": metadata["beam"],
            "timing_ms": metadata["timing_ms"],
            "hashes": {
                "task_sha": task_sha(train),
                "program_sha": program_sha(steps) if steps else ""
            }
        }

        log_receipt(receipt, out_dir=out_dir)

        if status == "solved":
            solved += 1
            print(f"✓ {task_id}: {' → '.join(program)}")

    print(f"\n{'='*70}")
    print(f"FINAL: Solved {solved}/{total} = {100*solved/total:.2f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_solver()
