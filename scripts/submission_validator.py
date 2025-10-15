#!/usr/bin/env python3
"""
Validate predictions.json against Kaggle ARC-AGI submission schema.

This script ensures:
1. All task IDs are strings ending in .json
2. All predictions are 2D lists (not numpy arrays)
3. All values are integers in range 0-9
4. No NaN, Inf, or other invalid values
5. All grids have consistent row lengths

Usage:
    python scripts/submission_validator.py runs/submission/predictions.json
"""

import sys
import json
import argparse
from pathlib import Path


def validate_grid(grid, task_id: str, attempt_idx: int) -> bool:
    """
    Validate a single grid.

    Args:
        grid: 2D list to validate
        task_id: Task ID (for error messages)
        attempt_idx: Attempt index (for error messages)

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Must be a list
    if not isinstance(grid, list):
        raise ValueError(f"{task_id} attempt {attempt_idx}: Grid must be a list, got {type(grid)}")

    # Must be non-empty
    if len(grid) == 0:
        raise ValueError(f"{task_id} attempt {attempt_idx}: Grid must be non-empty")

    # All rows must be lists
    for row_idx, row in enumerate(grid):
        if not isinstance(row, list):
            raise ValueError(f"{task_id} attempt {attempt_idx} row {row_idx}: Row must be a list, got {type(row)}")

        # Row must be non-empty
        if len(row) == 0:
            raise ValueError(f"{task_id} attempt {attempt_idx} row {row_idx}: Row must be non-empty")

        # All values must be int 0-9
        for col_idx, val in enumerate(row):
            if not isinstance(val, int):
                raise ValueError(
                    f"{task_id} attempt {attempt_idx} [{row_idx},{col_idx}]: "
                    f"Value must be int, got {type(val)}: {val}"
                )
            if not (0 <= val <= 9):
                raise ValueError(
                    f"{task_id} attempt {attempt_idx} [{row_idx},{col_idx}]: "
                    f"Value must be 0-9, got {val}"
                )

    # All rows must have same length
    row_lengths = [len(row) for row in grid]
    if len(set(row_lengths)) > 1:
        raise ValueError(
            f"{task_id} attempt {attempt_idx}: "
            f"Inconsistent row lengths: {row_lengths}"
        )

    return True


def validate_predictions(predictions_path: str, verbose: bool = True) -> bool:
    """
    Validate predictions.json file.

    Args:
        predictions_path: Path to predictions.json
        verbose: Print validation messages

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Load predictions
    with open(predictions_path) as f:
        predictions = json.load(f)

    # Must be a dict
    if not isinstance(predictions, dict):
        raise ValueError(f"Predictions must be a dict, got {type(predictions)}")

    # Must be non-empty
    if len(predictions) == 0:
        raise ValueError("Predictions dict is empty")

    if verbose:
        print(f"Validating {len(predictions)} tasks...")

    # Validate each task
    for task_id, attempts in predictions.items():
        # Task ID must be string ending in .json
        if not isinstance(task_id, str):
            raise ValueError(f"Task ID must be string, got {type(task_id)}: {task_id}")

        if not task_id.endswith('.json'):
            raise ValueError(f"Task ID must end with .json: {task_id}")

        # Attempts must be a list
        if not isinstance(attempts, list):
            raise ValueError(f"{task_id}: Attempts must be a list, got {type(attempts)}")

        # Must have at least 1 attempt
        if len(attempts) == 0:
            raise ValueError(f"{task_id}: Must have at least 1 attempt")

        # Validate each attempt
        for attempt_idx, grid in enumerate(attempts):
            validate_grid(grid, task_id, attempt_idx)

    if verbose:
        print(f"✓ Validated {len(predictions)} tasks")
        print(f"✓ All grids are valid 2D lists of integers 0-9")
        print(f"✓ Schema is compliant with Kaggle ARC-AGI submission format")

    return True


def main():
    parser = argparse.ArgumentParser(description="Validate ARC-AGI predictions.json")
    parser.add_argument(
        "predictions_path",
        type=str,
        help="Path to predictions.json file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress validation messages"
    )

    args = parser.parse_args()

    # Check file exists
    if not Path(args.predictions_path).exists():
        print(f"ERROR: File not found: {args.predictions_path}", file=sys.stderr)
        sys.exit(1)

    # Validate
    try:
        validate_predictions(args.predictions_path, verbose=not args.quiet)
        sys.exit(0)
    except ValueError as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
