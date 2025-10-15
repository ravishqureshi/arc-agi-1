#!/usr/bin/env python3
"""
Minimal pipeline test for determinism verification.

Tests:
1. Load a few tasks
2. Run solver
3. Verify predictions.json format
4. Run twice and compare outputs
"""

import sys
import json
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from arc_solver import ARCInstance, G, solve_with_closures, task_sha, closure_set_sha

def create_test_task():
    """Create a simple KEEP_LARGEST test task."""
    # Input: 4x4 grid with 3 components (sizes: 4, 4, 8)
    # Output: Keep largest (size 8) component
    x = G([[0, 0, 1, 1],
           [0, 0, 1, 1],
           [2, 2, 2, 2],
           [2, 2, 2, 2]])

    y = G([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [2, 2, 2, 2],
           [2, 2, 2, 2]])

    return x, y

def test_single_run():
    """Test: Single run produces valid output."""
    print("Test 1: Single run")

    x, y = create_test_task()
    train = [(x, y)]
    test_in = [x]

    inst = ARCInstance("test_task", train, test_in, [])

    closures, preds, residuals, metadata = solve_with_closures(inst)

    # Verify closures found
    assert closures is not None, "Should find closures"
    assert len(closures) == 1, "Should find exactly 1 closure (bg=0)"

    # Verify predictions format
    assert len(preds) == 1, "Should have 1 prediction"
    pred = preds[0]
    assert isinstance(pred, np.ndarray), "Prediction should be numpy array"
    assert pred.dtype == int, "Prediction should be int type"
    assert np.all((pred >= 0) & (pred <= 9)), "All values should be 0-9"

    # Convert to list (as in run_public.py)
    pred_list = pred.tolist()
    assert isinstance(pred_list, list), "tolist() should produce list"
    assert all(isinstance(row, list) for row in pred_list), "All rows should be lists"

    # Verify correctness
    assert np.array_equal(pred, y), "Prediction should match expected output"

    print(f"  ✓ Found closures: {[c.name for c in closures]}")
    print(f"  ✓ Prediction shape: {pred.shape}")
    print(f"  ✓ Prediction correct: {np.array_equal(pred, y)}")
    print(f"  ✓ Format valid: {type(pred_list)}")

    return True

def test_determinism():
    """Test: Multiple runs produce identical outputs."""
    print("\nTest 2: Determinism (5 runs)")

    x, y = create_test_task()
    train = [(x, y)]
    test_in = [x]

    inst = ARCInstance("test_task", train, test_in, [])

    results = []
    closure_names = []
    hashes = []

    for i in range(5):
        closures, preds, residuals, metadata = solve_with_closures(inst)
        results.append(preds[0].tolist())
        closure_names.append([c.name for c in closures])
        hashes.append(task_sha(train))

    # Check all results are identical
    all_same = all(np.array_equal(results[0], r) for r in results)
    all_same_closures = all(c == closure_names[0] for c in closure_names)
    all_same_hashes = len(set(hashes)) == 1

    print(f"  Predictions identical: {'PASS' if all_same else 'FAIL'}")
    print(f"  Closures identical: {'PASS' if all_same_closures else 'FAIL'}")
    print(f"  Hashes identical: {'PASS' if all_same_hashes else 'FAIL'}")

    if not all_same:
        print(f"  First run:  {results[0]}")
        print(f"  Second run: {results[1]}")

    assert all_same, "Results should be identical"
    assert all_same_closures, "Closures should be identical"
    assert all_same_hashes, "Hashes should be identical"

    print(f"  ✓ All 5 runs produced identical outputs")

    return True

def test_predictions_json_format():
    """Test: predictions.json format is correct."""
    print("\nTest 3: predictions.json format")

    x, y = create_test_task()
    train = [(x, y)]
    test_in = [x, x]  # 2 test cases

    inst = ARCInstance("test_task.json", train, test_in, [])

    closures, preds, residuals, metadata = solve_with_closures(inst)

    # Simulate run_public.py logic
    task_predictions = []
    for pred in preds:
        task_predictions.append(pred.tolist())

    # Create predictions dict (as in run_public.py)
    predictions = {"test_task.json": task_predictions}

    # Verify schema
    assert isinstance(predictions, dict), "Should be dict"
    assert "test_task.json" in predictions, "Should have task key"
    assert isinstance(predictions["test_task.json"], list), "Should be list of attempts"
    assert len(predictions["test_task.json"]) == 2, "Should have 2 predictions"

    # Verify each prediction
    for i, attempt in enumerate(predictions["test_task.json"]):
        assert isinstance(attempt, list), f"Attempt {i} should be list"
        assert all(isinstance(row, list) for row in attempt), f"Attempt {i} rows should be lists"
        assert all(isinstance(val, int) for row in attempt for val in row), f"Attempt {i} values should be int"
        assert all(0 <= val <= 9 for row in attempt for val in row), f"Attempt {i} values should be 0-9"

    # Test JSON serialization
    json_str = json.dumps(predictions, indent=2)
    loaded = json.loads(json_str)

    assert loaded == predictions, "Should round-trip through JSON"

    print(f"  ✓ Schema correct: dict[str, list[list[list[int]]]]")
    print(f"  ✓ Task key: 'test_task.json'")
    print(f"  ✓ Attempts: {len(predictions['test_task.json'])}")
    print(f"  ✓ JSON serialization: OK")

    return True

def main():
    print("=" * 60)
    print("PIPELINE TEST - DETERMINISM & SCHEMA")
    print("=" * 60)

    # Set deterministic seed (as in determinism.sh)
    os.environ["PYTHONHASHSEED"] = "0"

    tests = [
        test_single_run,
        test_determinism,
        test_predictions_json_format,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"OVERALL: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✓ Pipeline is DETERMINISTIC and SCHEMA-COMPLIANT")
        print("✓ Ready for submission")
        return 0
    else:
        print("\n✗ Some tests failed - review implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
