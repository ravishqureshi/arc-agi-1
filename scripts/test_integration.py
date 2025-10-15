#!/usr/bin/env python3
"""
Quick integration test to verify the submission pipeline works end-to-end.

Tests:
1. closure_set_sha() function
2. solve_with_closures() returns correct format
3. Numpy → list conversion
4. Schema validation
"""

import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from arc_solver import (
    ARCInstance, G,
    solve_with_closures,
    task_sha, closure_set_sha,
    KEEP_LARGEST_COMPONENT_Closure
)


def test_closure_set_sha():
    """Test closure_set_sha hash function."""
    print("Testing closure_set_sha()...")

    # Create test closures
    closure1 = KEEP_LARGEST_COMPONENT_Closure("KEEP_LARGEST_COMPONENT", {"bg": 0})
    closure2 = KEEP_LARGEST_COMPONENT_Closure("KEEP_LARGEST_COMPONENT", {"bg": 1})

    # Hash empty list
    hash_empty = closure_set_sha([])
    assert len(hash_empty) == 64, "Hash should be 64 chars (SHA-256 hex)"

    # Hash single closure
    hash1 = closure_set_sha([closure1])
    assert len(hash1) == 64

    # Hash should be deterministic
    hash1_repeat = closure_set_sha([closure1])
    assert hash1 == hash1_repeat, "Hash should be deterministic"

    # Different params should give different hash
    hash2 = closure_set_sha([closure2])
    assert hash1 != hash2, "Different params should give different hash"

    print("  ✓ closure_set_sha() works correctly")


def test_solve_with_closures_format():
    """Test solve_with_closures returns correct format."""
    print("Testing solve_with_closures() format...")

    # Create simple test case
    train = [
        (G([[1, 1, 0], [1, 0, 0]]), G([[1, 1, 0], [1, 0, 0]])),  # Identity (should fail to find closure)
    ]
    test_in = [G([[2, 2, 0], [2, 0, 0]])]

    inst = ARCInstance("test_task", train, test_in, [])

    # Solve
    closures, preds, residuals, metadata = solve_with_closures(inst)

    # Check return types
    assert isinstance(preds, list), "preds should be a list"
    assert len(preds) == 1, "Should have 1 prediction"
    assert preds[0].shape == (2, 3), "Prediction should match test input shape"

    # Check metadata structure
    assert "fp" in metadata
    assert "timing_ms" in metadata
    assert "iters" in metadata["fp"]
    assert "cells_multi" in metadata["fp"]

    print("  ✓ solve_with_closures() format correct")


def test_numpy_to_list_conversion():
    """Test numpy array to Python list conversion."""
    print("Testing numpy → list conversion...")

    # Create numpy grid
    grid = G([[0, 1, 2], [3, 4, 5]])

    # Convert to list
    grid_list = grid.tolist()

    # Check type
    assert isinstance(grid_list, list)
    assert isinstance(grid_list[0], list)
    assert isinstance(grid_list[0][0], int)

    # Check values
    assert grid_list == [[0, 1, 2], [3, 4, 5]]

    print("  ✓ Numpy → list conversion works")


def test_schema_validation():
    """Test predictions.json schema validation."""
    print("Testing schema validation...")

    # Create valid predictions
    predictions = {
        "task1.json": [
            [[0, 1, 2], [3, 4, 5]],
            [[0, 1, 2], [3, 4, 5]]
        ],
        "task2.json": [
            [[9, 8, 7]]
        ]
    }

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        temp_path = f.name

    try:
        # Validate
        sys.path.insert(0, str(Path(__file__).parent))
        from submission_validator import validate_predictions

        validate_predictions(temp_path, verbose=False)
        print("  ✓ Schema validation works")

    finally:
        Path(temp_path).unlink()


def main():
    print("=" * 70)
    print("INTEGRATION TEST - Submission Pipeline")
    print("=" * 70)
    print()

    try:
        test_closure_set_sha()
        test_solve_with_closures_format()
        test_numpy_to_list_conversion()
        test_schema_validation()

        print()
        print("=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("The submission pipeline is ready to use.")
        print()
        print("Next steps:")
        print("  1. Run on small dataset: python scripts/run_public.py --dataset=... --output=runs/test")
        print("  2. Validate: python scripts/submission_validator.py runs/test/predictions.json")
        print("  3. Check determinism: bash scripts/determinism.sh")
        print()
        sys.exit(0)

    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 70)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ ERROR: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
