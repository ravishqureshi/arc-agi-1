#!/usr/bin/env python3
"""
Determinism verification for B1 (KEEP_LARGEST_COMPONENT) closure.

Tests:
1. Same input → same output across multiple runs
2. Tied component sizes → deterministic selection
3. Unifier finds same bg across runs

Usage:
    PYTHONPATH=src python scripts/verify_b1_determinism.py
"""

import sys
import os
import numpy as np

# Add src to path if not already there
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from arc_solver.closures import unify_KEEP_LARGEST, KEEP_LARGEST_COMPONENT_Closure
from arc_solver.closure_engine import run_fixed_point
from arc_solver.utils import G

def test_basic_determinism():
    """Test: Same input produces same output across runs."""
    print("Test 1: Basic determinism")

    x = G([[0, 0, 1, 1],
           [0, 0, 1, 1],
           [2, 2, 0, 0],
           [2, 2, 0, 0]])

    closure = KEEP_LARGEST_COMPONENT_Closure("TEST", {"bg": 0})

    results = []
    for i in range(10):
        U, stats = run_fixed_point([closure], x)
        y = U.to_grid_deterministic(fallback='lowest', bg=0)
        results.append(y.tolist())

    all_same = all(np.array_equal(results[0], r) for r in results)
    print(f"  10 runs: {'PASS - identical' if all_same else 'FAIL - different outputs'}")
    if not all_same:
        for i, r in enumerate(results[:3]):
            print(f"    Run {i+1}: {r}")
        return False
    return True

def test_tied_components():
    """Test: Tied component sizes produce deterministic output."""
    print("\nTest 2: Tied component sizes")

    # Two components, both size=2
    x = G([[0, 0, 0, 0],
           [0, 1, 1, 0],
           [0, 0, 2, 2],
           [0, 0, 0, 0]])

    closure = KEEP_LARGEST_COMPONENT_Closure("TEST", {"bg": 0})

    results = []
    for i in range(10):
        U, stats = run_fixed_point([closure], x)
        y = U.to_grid_deterministic(fallback='lowest', bg=0)
        results.append(y.tolist())

    all_same = all(np.array_equal(results[0], r) for r in results)
    print(f"  10 runs: {'PASS - identical' if all_same else 'FAIL - different outputs'}")

    # Show which component was selected
    if all_same:
        y = np.array(results[0])
        if np.any(y == 1):
            print(f"    Selected: color 1 (top-left component)")
        elif np.any(y == 2):
            print(f"    Selected: color 2 (bottom-right component)")

    return all_same

def test_unifier_determinism():
    """Test: Unifier finds same bg across runs."""
    print("\nTest 3: Unifier determinism")

    train = [
        (G([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]]),
         G([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])),
    ]

    bg_values = []
    for i in range(5):
        closures = unify_KEEP_LARGEST(train)
        if closures:
            bg_values.append(closures[0].params["bg"])
        else:
            bg_values.append(None)

    all_same = len(set(bg_values)) == 1
    print(f"  5 runs: {'PASS - same bg' if all_same else 'FAIL - different bg'}")
    print(f"    Found bg values: {bg_values}")

    return all_same

def test_unifier_exhaustiveness():
    """Test: Unifier tries all bg values."""
    print("\nTest 4: Unifier exhaustiveness")

    # Task where bg=0 works
    train_bg0 = [
        (G([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]]),
         G([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])),
    ]

    closures_0 = unify_KEEP_LARGEST(train_bg0)
    found_bg0 = any(c.params["bg"] == 0 for c in closures_0)
    print(f"  Task with bg=0: {'FOUND bg=0' if found_bg0 else 'MISSED bg=0'}")

    # Task where bg=5 works
    train_bg5 = [
        (G([[5, 5, 1, 1], [5, 5, 1, 1], [2, 2, 5, 5], [2, 2, 5, 5]]),
         G([[5, 5, 1, 1], [5, 5, 1, 1], [5, 5, 5, 5], [5, 5, 5, 5]])),
    ]

    closures_5 = unify_KEEP_LARGEST(train_bg5)
    found_bg5 = any(c.params["bg"] == 5 for c in closures_5)
    print(f"  Task with bg=5: {'FOUND bg=5' if found_bg5 else 'MISSED bg=5'}")

    return found_bg0 and found_bg5

def main():
    print("=" * 60)
    print("B1 DETERMINISM VERIFICATION")
    print("=" * 60)

    tests = [
        test_basic_determinism,
        test_tied_components,
        test_unifier_determinism,
        test_unifier_exhaustiveness,
    ]

    results = [test() for test in tests]

    print("\n" + "=" * 60)
    print(f"OVERALL: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✓ B1 implementation is DETERMINISTIC and PARAMETRIC")
        print("✓ Ready for submission after applying recommended fixes")
        return 0
    else:
        print("\n✗ Some tests failed - review implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
