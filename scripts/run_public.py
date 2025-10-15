#!/usr/bin/env python3
"""
Run solver on ARC public evaluation/test set and generate predictions.json.

This is the main entry point for Kaggle submission.
Produces:
- predictions.json (required for submission)
- receipts.jsonl (for debugging and analysis)

Usage:
    python scripts/run_public.py --dataset=data/arc-agi_evaluation_challenges.json --output=runs/submission
    python scripts/run_public.py --dataset=data/arc-agi_test_challenges.json --output=runs/test-submission
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from arc_solver import (
    ARCInstance, G,
    solve_with_closures,
    task_sha, closure_set_sha, log_receipt
)


def run_public(dataset_path: str, output_dir: str, verbose: bool = True):
    """
    Run solver on public dataset and write predictions.json.

    Args:
        dataset_path: Path to challenges JSON file
        output_dir: Output directory for predictions and receipts
        verbose: Print progress messages
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load challenges
    with open(dataset_path) as f:
        challenges = json.load(f)

    predictions = {}
    solved_count = 0
    total_count = len(challenges)

    if verbose:
        print("=" * 70)
        print(f"ARC-AGI Solver - Fixed-Point Closure Engine")
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_dir}")
        print(f"Tasks: {total_count}")
        print("=" * 70)

    for idx, (task_id, task_data) in enumerate(challenges.items(), 1):
        # Parse train and test data
        train = [(G(p['input']), G(p['output'])) for p in task_data['train']]
        test_in = [G(p['input']) for p in task_data['test']]

        # Create instance (no test_out for evaluation set)
        inst = ARCInstance(task_id, train, test_in, [])

        # Solve with closures
        closures, preds, test_residuals, metadata = solve_with_closures(inst)

        # Determine status
        if closures is not None:
            status = "solved"
            solved_count += 1
        else:
            status = "failed"

        # Convert predictions to Python lists (not numpy arrays)
        # Kaggle requires 2D lists of integers
        task_predictions = []
        if len(preds) > 0:
            for pred in preds:
                # Convert numpy array to list
                task_predictions.append(pred.tolist())
        else:
            # No predictions - generate dummy predictions (copy of test inputs)
            # Kaggle requires at least 1 attempt per test case
            for x_test in test_in:
                task_predictions.append(x_test.tolist())

        # Store predictions (task ID must include .json extension)
        if not task_id.endswith('.json'):
            task_key = f"{task_id}.json"
        else:
            task_key = task_id

        predictions[task_key] = task_predictions

        # Build receipt record
        receipt = {
            "task": task_key,
            "status": status,
            "closures": [{"name": c.name, "params": c.params} for c in closures] if closures else [],
            "fp": metadata.get("fp", {"iters": 0, "cells_multi": -1}),
            "timing_ms": metadata.get("timing_ms", {}),
            "hashes": {
                "task_sha": task_sha(train),
                "closure_set_sha": closure_set_sha(closures) if closures else ""
            },
            "invariants": {
                "palette_delta": metadata.get("palette_invariants", {}),
                "component_delta": metadata.get("component_invariants", {})
            }
        }

        # Log receipt
        log_receipt(receipt, out_dir=output_dir)

        # Progress reporting
        if verbose:
            if idx % 10 == 0 or status == "solved":
                progress_pct = 100 * idx / total_count
                print(f"[{idx}/{total_count}] ({progress_pct:.1f}%) {task_key}: {status} - Solved: {solved_count}")

    # Write predictions.json
    predictions_path = Path(output_dir) / "predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)

    if verbose:
        print("=" * 70)
        print(f"COMPLETE: Solved {solved_count}/{total_count} ({100*solved_count/total_count:.1f}%)")
        print(f"Predictions: {predictions_path}")
        print(f"Receipts: {Path(output_dir) / 'receipts.jsonl'}")
        print("=" * 70)

    return solved_count, total_count


def main():
    parser = argparse.ArgumentParser(description="Run ARC-AGI solver on public dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/arc-agi_evaluation_challenges.json",
        help="Path to challenges JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: runs/YYYY-MM-DD)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Default output directory
    if args.output is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_dir = f"runs/{date_str}"
    else:
        output_dir = args.output

    # Run solver
    solved, total = run_public(args.dataset, output_dir, verbose=not args.quiet)

    # Exit code: 0 if solved > 0, 1 otherwise
    sys.exit(0 if solved > 0 else 1)


if __name__ == "__main__":
    main()
