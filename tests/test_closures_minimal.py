"""
Minimal property tests for closure soundness.

Tests mathematical properties required by Tarski fixed-point theorem:
1. Monotonicity: F(U) ⊆ F(V) when U ⊆ V
2. Shrinking: F(U) ⊆ U
3. Idempotence: F(F(U)) = F(U)
4. Train exactness: U* = singleton(y) for ALL train pairs
5. Convergence: Fixed-point reached in finite steps

These are decisive micro-tests (2-3 grids per family).
Each test should complete in <10ms.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arc_solver.closure_engine import (
    SetValuedGrid,
    init_top,
    init_from_grid,
    color_to_mask,
    run_fixed_point,
    verify_closures_on_train
)
from arc_solver.closures import (
    KEEP_LARGEST_COMPONENT_Closure,
    unify_KEEP_LARGEST,
    OUTLINE_OBJECTS_Closure,
    unify_OUTLINE_OBJECTS,
    OPEN_CLOSE_Closure,
    unify_OPEN_CLOSE,
    AXIS_PROJECTION_Closure,
    unify_AXIS_PROJECTION,
    SYMMETRY_COMPLETION_Closure,
    unify_SYMMETRY_COMPLETION
)


# ==============================================================================
# Helper Functions
# ==============================================================================

def grid_subset_or_equal(U: SetValuedGrid, V: SetValuedGrid) -> bool:
    """Check if U ⊆ V (U has fewer or equal possibilities)."""
    if U.H != V.H or U.W != V.W:
        return False
    for r in range(U.H):
        for c in range(U.W):
            # U[r,c] ⊆ V[r,c] ⇔ U[r,c] & V[r,c] == U[r,c]
            if (U.data[r, c] & V.data[r, c]) != U.data[r, c]:
                return False
    return True


def make_test_grid_1() -> np.ndarray:
    """
    Small test grid with two components:
    - Large component (5 cells, color 1)
    - Small component (2 cells, color 2)
    """
    return np.array([
        [0, 1, 1, 0],
        [1, 1, 1, 2],
        [0, 0, 2, 0]
    ], dtype=int)


def make_test_grid_2() -> np.ndarray:
    """
    Test grid with single large component (color 3).
    """
    return np.array([
        [3, 3, 0],
        [3, 3, 3],
        [0, 3, 3]
    ], dtype=int)


def make_expected_output_1() -> np.ndarray:
    """Expected output for test_grid_1 (keep only largest component)."""
    return np.array([
        [0, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0]
    ], dtype=int)


def make_expected_output_2() -> np.ndarray:
    """Expected output for test_grid_2 (entire component kept)."""
    return np.array([
        [3, 3, 0],
        [3, 3, 3],
        [0, 3, 3]
    ], dtype=int)


# ==============================================================================
# Property 1: Monotonicity
# ==============================================================================

def test_monotonicity_keep_largest():
    """
    Test: F(U) ⊆ F(V) when U ⊆ V

    Setup:
    - U = tight grid (singletons from input)
    - V = loose grid (all colors allowed)
    - U ⊆ V by construction

    Verify:
    - F(U) ⊆ F(V) after applying KEEP_LARGEST
    """
    x = make_test_grid_1()

    # U is tight (each cell = singleton from input)
    U = init_from_grid(x)

    # V is loose (each cell = all colors)
    V = init_top(x.shape[0], x.shape[1])

    # Verify precondition: U ⊆ V
    assert grid_subset_or_equal(U, V), "Precondition failed: U should be subset of V"

    # Apply closure
    closure = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})
    U_result = closure.apply(U, x)
    V_result = closure.apply(V, x)

    # Verify postcondition: F(U) ⊆ F(V)
    assert grid_subset_or_equal(U_result, V_result), \
        "Monotonicity violated: F(U) should be subset of F(V)"


# ==============================================================================
# Property 2: Shrinking
# ==============================================================================

def test_shrinking_keep_largest():
    """
    Test: F(U) ⊆ U (closure only removes possibilities)

    Setup:
    - U = top element (all colors allowed)

    Verify:
    - F(U) ⊆ U (result has fewer or equal possibilities)
    """
    x = make_test_grid_1()
    U = init_top(x.shape[0], x.shape[1])

    # Store original for comparison
    U_copy = U.copy()

    # Apply closure
    closure = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})
    U_result = closure.apply(U, x)

    # Verify: F(U) ⊆ U
    assert grid_subset_or_equal(U_result, U_copy), \
        "Shrinking violated: F(U) should be subset of U"

    # Extra check: Count bits before/after
    bits_before = sum(bin(int(U_copy.data[r, c])).count('1')
                      for r in range(U.H) for c in range(U.W))
    bits_after = sum(bin(int(U_result.data[r, c])).count('1')
                     for r in range(U.H) for c in range(U.W))

    assert bits_after <= bits_before, \
        f"Shrinking violated: bits increased from {bits_before} to {bits_after}"


# ==============================================================================
# Property 3: Idempotence
# ==============================================================================

def test_idempotence_keep_largest():
    """
    Test: F(F(U)) = F(U) (applying twice is same as once)

    Setup:
    - U = top element

    Verify:
    - F(U) = F(F(U)) (second application doesn't change result)
    """
    x = make_test_grid_1()
    U = init_top(x.shape[0], x.shape[1])

    closure = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})

    # Apply once
    U1 = closure.apply(U, x)

    # Apply twice
    U2 = closure.apply(U1, x)

    # Verify: F(U) = F(F(U))
    assert U1 == U2, "Idempotence violated: F(F(U)) != F(U)"

    # Extra check: Verify bitwise equality
    assert np.array_equal(U1.data, U2.data), \
        "Idempotence violated: data arrays differ"


def test_idempotence_stabilizes_in_2_passes():
    """
    Test: Fixed-point stabilizes in ≤2 passes for KEEP_LARGEST.

    This is a stronger property: not just idempotent, but converges quickly.
    """
    x = make_test_grid_1()
    closure = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})

    U, stats = run_fixed_point([closure], x, max_iters=10)

    # KEEP_LARGEST should stabilize in 1 pass (deterministic mask application)
    assert stats["iters"] <= 2, \
        f"Expected convergence in ≤2 iterations, got {stats['iters']}"


# ==============================================================================
# Property 4: Train Exactness
# ==============================================================================

def test_train_exactness_single_pair():
    """
    Test: Closure produces exact output on single train pair.

    Verify:
    - Fixed-point U* is fully determined (all singletons)
    - U* equals expected output exactly
    """
    x = make_test_grid_1()
    y = make_expected_output_1()

    closure = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})
    U, stats = run_fixed_point([closure], x)

    # Check 1: Fully determined
    assert U.is_fully_determined(), \
        f"Not fully determined: {stats['cells_multi']} cells still multi-valued"

    # Check 2: Exact match
    y_pred = U.to_grid()
    assert y_pred is not None, "to_grid() returned None"
    assert np.array_equal(y_pred, y), \
        f"Predicted output doesn't match expected.\nPred:\n{y_pred}\nExpected:\n{y}"


def test_train_exactness_multiple_pairs():
    """
    Test: Unifier verifies ALL train pairs.

    Verify:
    - Same closure works on multiple pairs
    - Returns [] if ANY pair fails
    """
    train = [
        (make_test_grid_1(), make_expected_output_1()),
        (make_test_grid_2(), make_expected_output_2())
    ]

    # Unifier discovers bg from data (should find bg=0)
    closures = unify_KEEP_LARGEST(train)

    assert len(closures) >= 1, f"Expected at least 1 closure, got {len(closures)}"
    # Should find bg=0 since both pairs use black background
    assert any(c.params["bg"] == 0 for c in closures), "Should find bg=0"

    # Verify each pair manually
    for x, y in train:
        U, _ = run_fixed_point(closures, x)
        y_pred = U.to_grid()
        assert y_pred is not None, f"Failed to solve train pair"
        assert np.array_equal(y_pred, y), \
            f"Train exactness violated.\nInput:\n{x}\nPred:\n{y_pred}\nExpected:\n{y}"


def test_train_exactness_rejects_bad_params():
    """
    Test: Unifier returns [] if parameters don't work on ALL pairs.

    Create mismatched train pairs where no single bg works.
    """
    # Pair 1: largest component is color 1
    x1 = np.array([[1, 1], [0, 2]])
    y1 = np.array([[1, 1], [0, 0]])  # Expects bg=0

    # Pair 2: largest component is color 2, but bg=1 needed
    x2 = np.array([[2, 2], [1, 0]])
    y2 = np.array([[2, 2], [0, 0]])  # Still expects bg=0

    train = [(x1, y1), (x2, y2)]

    # This should succeed (both use bg=0, unifier discovers it)
    closures = unify_KEEP_LARGEST(train)
    assert len(closures) >= 1, "Expected unifier to succeed"
    assert any(c.params["bg"] == 0 for c in closures), "Should find bg=0"

    # Now try with mismatched outputs that don't match KEEP_LARGEST logic
    train_bad = [
        (x1, y1),  # Works with bg=0
        (x1, np.array([[1, 1], [1, 2]]))  # Would need bg=2 (doesn't match y1)
    ]

    closures_bad = unify_KEEP_LARGEST(train_bad)
    assert len(closures_bad) == 0, "Expected unifier to reject mismatched train pairs"


# ==============================================================================
# Property 5: Convergence
# ==============================================================================

def test_convergence_detects_fixed_point():
    """
    Test: Fixed-point iteration detects true convergence.

    Verify:
    - Iteration stops when U_n = U_{n-1}
    - Applying closure one more time doesn't change U
    """
    x = make_test_grid_1()
    closure = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})

    U_final, stats = run_fixed_point([closure], x)

    # Apply one more time - should not change
    U_again = closure.apply(U_final, x)

    assert U_final == U_again, \
        "Convergence failed: applying closure to fixed point changed it"


def test_convergence_is_fast():
    """
    Test: KEEP_LARGEST converges in ≤2 iterations.

    Since masks are deterministic from input, should stabilize quickly.
    (Iteration 1: apply closure; Iteration 2: verify no change)
    """
    x = make_test_grid_1()
    closure = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})

    U, stats = run_fixed_point([closure], x)

    # Should converge in ≤2 passes (1 to apply, 1 to verify)
    assert stats["iters"] <= 2, \
        f"Expected ≤2 iterations for deterministic closure, got {stats['iters']}"


# ==============================================================================
# Edge Cases
# ==============================================================================

def test_edge_case_empty_grid():
    """
    Test: Empty grid (all background) handled correctly.
    """
    x = np.array([[0, 0], [0, 0]], dtype=int)
    y = np.array([[0, 0], [0, 0]], dtype=int)

    closure = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})
    U, _ = run_fixed_point([closure], x)

    y_pred = U.to_grid()
    assert y_pred is not None
    assert np.array_equal(y_pred, y), \
        f"Empty grid not handled correctly.\nPred:\n{y_pred}\nExpected:\n{y}"


def test_edge_case_single_component():
    """
    Test: Single component is trivially the largest.
    """
    x = np.array([[1, 1], [1, 0]], dtype=int)
    y = np.array([[1, 1], [1, 0]], dtype=int)  # Same as input

    closure = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})
    U, _ = run_fixed_point([closure], x)

    y_pred = U.to_grid()
    assert y_pred is not None
    assert np.array_equal(y_pred, y)


def test_edge_case_equal_size_components():
    """
    Test: Multiple equal-size components - picks first deterministically.
    """
    x = np.array([
        [1, 1, 0, 2, 2],  # Two 2-cell components
        [0, 0, 0, 0, 0]
    ], dtype=int)

    closure = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})

    # Apply twice - should get same result (deterministic tie-break)
    U1, _ = run_fixed_point([closure], x)
    U2, _ = run_fixed_point([closure], x)

    assert U1 == U2, "Equal-size component selection not deterministic"


# ==============================================================================
# Summary Test
# ==============================================================================

def test_all_properties_together():
    """
    Integration test: Verify all properties in realistic scenario.
    """
    x = make_test_grid_1()
    y = make_expected_output_1()

    # Create closure
    closure = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})

    # Property 1: Shrinking
    U = init_top(x.shape[0], x.shape[1])
    U_shrunk = closure.apply(U, x)
    assert grid_subset_or_equal(U_shrunk, U), "Not shrinking"

    # Property 2: Idempotent
    U_again = closure.apply(U_shrunk, x)
    assert U_shrunk == U_again, "Not idempotent"

    # Property 3: Train exact
    U_final, stats = run_fixed_point([closure], x)
    assert U_final.is_fully_determined(), "Not fully determined"
    y_pred = U_final.to_grid()
    assert np.array_equal(y_pred, y), "Not train exact"

    # Property 4: Fast convergence
    assert stats["iters"] <= 2, f"Too many iterations: {stats['iters']}"


# ==============================================================================
# Run Tests
# ==============================================================================

# ==============================================================================
# Test Runner
# ==============================================================================

def run_all_tests():
    # Run all tests
    tests = [
        # KEEP_LARGEST tests
        ("Monotonicity", test_monotonicity_keep_largest),
        ("Shrinking", test_shrinking_keep_largest),
        ("Idempotence", test_idempotence_keep_largest),
        ("Idempotence (2-pass)", test_idempotence_stabilizes_in_2_passes),
        ("Train exactness (single)", test_train_exactness_single_pair),
        ("Train exactness (multiple)", test_train_exactness_multiple_pairs),
        ("Train exactness (rejects bad)", test_train_exactness_rejects_bad_params),
        ("Convergence (fixed-point)", test_convergence_detects_fixed_point),
        ("Convergence (fast)", test_convergence_is_fast),
        ("Edge case (empty)", test_edge_case_empty_grid),
        ("Edge case (single)", test_edge_case_single_component),
        ("Edge case (equal-size)", test_edge_case_equal_size_components),
        ("Integration (all properties)", test_all_properties_together),
        # OUTLINE_OBJECTS tests
        ("OUTLINE: Shrinking", test_shrinking_outline),
        ("OUTLINE: Idempotence", test_idempotence_outline),
        ("OUTLINE: Train exactness (single)", test_train_exactness_outline_single),
        ("OUTLINE: Train exactness (multiple)", test_train_exactness_outline_multiple),
        ("OUTLINE: Convergence", test_convergence_outline),
        ("OUTLINE: Scope all", test_outline_scope_all),
        # OPEN_CLOSE tests
        ("OPEN_CLOSE: Shrinking", test_shrinking_open_close),
        ("OPEN_CLOSE: Idempotence", test_idempotence_open_close),
        ("OPEN_CLOSE: OPEN train exactness", test_train_exactness_open),
        ("OPEN_CLOSE: CLOSE train exactness", test_train_exactness_close),
        # AXIS_PROJECTION tests
        ("AXIS_PROJECTION: Shrinking", test_shrinking_axis_projection),
        ("AXIS_PROJECTION: Idempotence", test_idempotence_axis_projection),
        ("AXIS_PROJECTION: Train exactness (col)", test_train_exactness_axis_projection_col),
        ("AXIS_PROJECTION: Train exactness (row)", test_train_exactness_axis_projection_row),
        ("AXIS_PROJECTION: Scope largest vs all", test_axis_projection_scope_largest_vs_all),
        # SYMMETRY_COMPLETION tests
        ("SYMMETRY: Shrinking", test_shrinking_symmetry_completion),
        ("SYMMETRY: Idempotence", test_idempotence_symmetry_completion),
        ("SYMMETRY: Vertical global", test_train_exactness_symmetry_vertical),
        ("SYMMETRY: Largest only", test_train_exactness_symmetry_largest_only),
        ("SYMMETRY: Diagonal", test_train_exactness_symmetry_diagonal),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name}: EXCEPTION: {e}")
            failed += 1

    print(f"\n{passed}/{len(tests)} tests passed")

    if failed > 0:
        sys.exit(1)


# ==============================================================================
# OUTLINE_OBJECTS Property Tests
# ==============================================================================

def test_shrinking_outline():
    """
    Test: F(U) ⊆ U for OUTLINE_OBJECTS (closure only removes possibilities).
    """
    # Ring object (outline is entire object)
    x = np.array([
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0]
    ], dtype=int)

    U = init_top(x.shape[0], x.shape[1])
    U_copy = U.copy()

    closure = OUTLINE_OBJECTS_Closure("test", {"mode": "outer", "scope": "largest", "bg": 0})
    U_result = closure.apply(U, x)

    # Verify: F(U) ⊆ U
    assert grid_subset_or_equal(U_result, U_copy), \
        "Shrinking violated: F(U) should be subset of U"

    # Count bits before/after
    bits_before = sum(bin(int(U_copy.data[r, c])).count('1')
                      for r in range(U.H) for c in range(U.W))
    bits_after = sum(bin(int(U_result.data[r, c])).count('1')
                     for r in range(U.H) for c in range(U.W))

    assert bits_after <= bits_before, \
        f"Shrinking violated: bits increased from {bits_before} to {bits_after}"


def test_idempotence_outline():
    """
    Test: F(F(U)) = F(U) for OUTLINE_OBJECTS (applying twice is same as once).
    """
    # Solid blob
    x = np.array([
        [0, 0, 0],
        [0, 2, 2],
        [0, 2, 2]
    ], dtype=int)

    U = init_top(x.shape[0], x.shape[1])

    closure = OUTLINE_OBJECTS_Closure("test", {"mode": "outer", "scope": "largest", "bg": 0})

    # Apply once
    U1 = closure.apply(U, x)

    # Apply twice
    U2 = closure.apply(U1, x)

    # Verify: F(U) = F(F(U))
    assert U1 == U2, "Idempotence violated: F(F(U)) != F(U)"
    assert np.array_equal(U1.data, U2.data), \
        "Idempotence violated: data arrays differ"


def test_train_exactness_outline_single():
    """
    Test: OUTLINE_OBJECTS produces exact output on ring object (outline = entire object).
    """
    # Ring object (thin components - use scope="all" to capture all)
    x = np.array([
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0]
    ], dtype=int)

    y = np.array([
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0]
    ], dtype=int)  # Outline is entire object (thin ring)

    # Use scope="all" since the hole makes this multiple components
    closure = OUTLINE_OBJECTS_Closure("test", {"mode": "outer", "scope": "all", "bg": 0})
    U, stats = run_fixed_point([closure], x)

    # Check fully determined
    assert U.is_fully_determined(), \
        f"Not fully determined: {stats['cells_multi']} cells still multi-valued"

    # Check exact match
    y_pred = U.to_grid()
    assert y_pred is not None, "to_grid() returned None"
    assert np.array_equal(y_pred, y), \
        f"Predicted output doesn't match expected.\nPred:\n{y_pred}\nExpected:\n{y}"


def test_train_exactness_outline_multiple():
    """
    Test: Unifier verifies ALL train pairs for OUTLINE_OBJECTS.
    """
    # Pair 1: Solid blob 1 (outline clears interior)
    x1 = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ], dtype=int)
    y1 = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ], dtype=int)  # All pixels are outline (2x2 blob)

    # Pair 2: Solid blob 2 (larger, interior cleared)
    x2 = np.array([
        [0, 0, 0, 0],
        [0, 2, 2, 2],
        [0, 2, 2, 2],
        [0, 2, 2, 2]
    ], dtype=int)
    y2 = np.array([
        [0, 0, 0, 0],
        [0, 2, 2, 2],
        [0, 2, 0, 2],
        [0, 2, 2, 2]
    ], dtype=int)  # Interior (2,2) cleared

    train = [(x1, y1), (x2, y2)]

    # Unifier should find params that work
    closures = unify_OUTLINE_OBJECTS(train)

    assert len(closures) >= 1, f"Expected at least 1 closure, got {len(closures)}"

    # Verify each pair manually
    for x, y in train:
        U, _ = run_fixed_point(closures, x)
        y_pred = U.to_grid()
        assert y_pred is not None, f"Failed to solve train pair"
        assert np.array_equal(y_pred, y), \
            f"Train exactness violated.\nInput:\n{x}\nPred:\n{y_pred}\nExpected:\n{y}"


def test_convergence_outline():
    """
    Test: OUTLINE_OBJECTS converges in ≤2 iterations.
    """
    # Two components (all scope)
    x = np.array([
        [1, 1, 0, 3, 3],
        [1, 0, 0, 3, 3]
    ], dtype=int)

    closure = OUTLINE_OBJECTS_Closure("test", {"mode": "outer", "scope": "all", "bg": 0})
    U, stats = run_fixed_point([closure], x)

    # Should converge in ≤2 passes
    assert stats["iters"] <= 2, \
        f"Expected convergence in ≤2 iterations, got {stats['iters']}"


def test_outline_scope_all():
    """
    Test: scope="all" outlines all components.
    """
    # Two components
    x = np.array([
        [1, 1, 0, 3, 3],
        [1, 0, 0, 3, 3]
    ], dtype=int)
    y = np.array([
        [1, 1, 0, 3, 3],
        [1, 0, 0, 3, 3]
    ], dtype=int)  # Both outlined (no interior)

    closure = OUTLINE_OBJECTS_Closure("test", {"mode": "outer", "scope": "all", "bg": 0})
    U, _ = run_fixed_point([closure], x)

    y_pred = U.to_grid()
    assert y_pred is not None
    assert np.array_equal(y_pred, y), \
        f"scope='all' not working correctly.\nPred:\n{y_pred}\nExpected:\n{y}"


# ==============================================================================
# OPEN_CLOSE Property Tests
# ==============================================================================

def test_shrinking_open_close():
    """Test: apply(U) ⊆ U (closure only removes possibilities)."""
    x = np.array([
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0]
    ], dtype=int)

    U = init_top(x.shape[0], x.shape[1])
    U_copy = U.copy()

    closure = OPEN_CLOSE_Closure("test", {"mode": "open", "bg": 0})
    U_result = closure.apply(U, x)

    assert grid_subset_or_equal(U_result, U_copy), \
        "Shrinking violated: apply(U) should be subset of U"


def test_idempotence_open_close():
    """Test: ≤2-pass convergence for OPEN_CLOSE."""
    x = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=int)

    closure = OPEN_CLOSE_Closure("test", {"mode": "close", "bg": 0})
    U, stats = run_fixed_point([closure], x)

    assert stats["iters"] <= 2, \
        f"Expected ≤2 iterations, got {stats['iters']}"


def test_train_exactness_open():
    """Test: OPEN removes thin structures."""
    # Input: 2x2 solid block (too thin - will be completely eroded)
    x = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ], dtype=int)

    # Expected: OPEN on 2x2 block erodes everything (no pixel has all 4 neighbors as fg)
    # After ERODE: all pixels removed (each has some bg neighbors)
    # After DILATE: nothing to dilate
    y = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=int)

    closure = OPEN_CLOSE_Closure("test", {"mode": "open", "bg": 0})
    U, _ = run_fixed_point([closure], x)

    assert U.is_fully_determined()
    y_pred = U.to_grid()
    assert np.array_equal(y_pred, y), \
        f"OPEN test failed.\nInput:\n{x}\nPred:\n{y_pred}\nExpected:\n{y}"


def test_train_exactness_close():
    """Test: CLOSE fills single-pixel gap in larger shape."""
    # Input: ring with single pixel gap in center
    x = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)

    # Expected: Gap filled, edges removed by erosion
    # DILATE: fills the gap and expands to edges
    # ERODE: removes edge pixels, keeps filled core
    y = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=int)

    closure = OPEN_CLOSE_Closure("test", {"mode": "close", "bg": 0})
    U, _ = run_fixed_point([closure], x)

    assert U.is_fully_determined()
    y_pred = U.to_grid()
    assert np.array_equal(y_pred, y), \
        f"CLOSE test failed.\nInput:\n{x}\nPred:\n{y_pred}\nExpected:\n{y}"


# ==============================================================================
# AXIS_PROJECTION Property Tests
# ==============================================================================

def test_shrinking_axis_projection():
    """Test: apply(U) ⊆ U (closure only removes possibilities)."""
    # Single dot (1x1 object) - col projection fills entire column
    x = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=int)

    U = init_top(x.shape[0], x.shape[1])
    U_copy = U.copy()

    closure = AXIS_PROJECTION_Closure("test", {"axis": "col", "scope": "largest", "mode": "to_border", "bg": 0})
    U_result = closure.apply(U, x)

    assert grid_subset_or_equal(U_result, U_copy), \
        "Shrinking violated: apply(U) should be subset of U"

def test_idempotence_axis_projection():
    """Test: ≤2-pass convergence for AXIS_PROJECTION."""
    # Horizontal bar - row projection
    x = np.array([
        [0, 0, 0, 0],
        [0, 2, 2, 0],
        [0, 0, 0, 0]
    ], dtype=int)

    closure = AXIS_PROJECTION_Closure("test", {"axis": "row", "scope": "all", "mode": "to_border", "bg": 0})
    U, stats = run_fixed_point([closure], x)

    assert stats["iters"] <= 2, \
        f"Expected ≤2 iterations, got {stats['iters']}"

def test_train_exactness_axis_projection_col():
    """Test: Column projection fills entire column from object pixels."""
    # Single dot at (1,1)
    x = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=int)

    # Expected: entire column 1 filled with color 1
    y = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ], dtype=int)

    closure = AXIS_PROJECTION_Closure("test", {"axis": "col", "scope": "largest", "mode": "to_border", "bg": 0})
    U, _ = run_fixed_point([closure], x)

    assert U.is_fully_determined()
    y_pred = U.to_grid()
    assert np.array_equal(y_pred, y), \
        f"Column projection failed.\nInput:\n{x}\nPred:\n{y_pred}\nExpected:\n{y}"

def test_train_exactness_axis_projection_row():
    """Test: Row projection fills entire rows from horizontal bar."""
    # Horizontal bar at row 1
    x = np.array([
        [0, 0, 0, 0],
        [0, 2, 2, 0],
        [0, 0, 0, 0]
    ], dtype=int)

    # Expected: entire row 1 filled with color 2
    y = np.array([
        [0, 0, 0, 0],
        [2, 2, 2, 2],
        [0, 0, 0, 0]
    ], dtype=int)

    closure = AXIS_PROJECTION_Closure("test", {"axis": "row", "scope": "all", "mode": "to_border", "bg": 0})
    U, _ = run_fixed_point([closure], x)

    assert U.is_fully_determined()
    y_pred = U.to_grid()
    assert np.array_equal(y_pred, y), \
        f"Row projection failed.\nInput:\n{x}\nPred:\n{y_pred}\nExpected:\n{y}"

def test_axis_projection_scope_largest_vs_all():
    """Test: scope='largest' vs scope='all' on 2-object grid."""
    # Two dots: (0,0) color 1 (size 1), (2,2) color 2 (size 1)
    # Equal size → deterministic tie-break picks first by bbox position
    x = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 2]
    ], dtype=int)

    # scope="largest" should pick one (deterministic)
    closure_largest = AXIS_PROJECTION_Closure("test", {"axis": "col", "scope": "largest", "mode": "to_border", "bg": 0})
    U_largest, _ = run_fixed_point([closure_largest], x)

    # scope="all" should project both
    closure_all = AXIS_PROJECTION_Closure("test", {"axis": "col", "scope": "all", "mode": "to_border", "bg": 0})
    U_all, _ = run_fixed_point([closure_all], x)

    # Results should differ (largest picks one, all picks both)
    assert U_largest != U_all, \
        "scope='largest' vs scope='all' should produce different results on multi-object grid"


# ==============================================================================
# SYMMETRY_COMPLETION Property Tests
# ==============================================================================

def test_shrinking_symmetry_completion():
    """Test: apply(U) ⊆ U (closure only removes possibilities)."""
    # Half-shape (left side only)
    x = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0]
    ], dtype=int)

    U = init_top(x.shape[0], x.shape[1])
    U_copy = U.copy()

    closure = SYMMETRY_COMPLETION_Closure(
        "test",
        {"axis": "v", "scope": "global", "bg": 0}
    )
    U_result = closure.apply(U, x)

    assert grid_subset_or_equal(U_result, U_copy), \
        "Shrinking violated: apply(U) should be subset of U"


def test_idempotence_symmetry_completion():
    """Test: ≤2-pass convergence for SYMMETRY_COMPLETION."""
    # Asymmetric shape
    x = np.array([
        [0, 2, 0],
        [2, 2, 0],
        [0, 2, 0]
    ], dtype=int)

    closure = SYMMETRY_COMPLETION_Closure(
        "test",
        {"axis": "h", "scope": "global", "bg": 0}
    )
    U, stats = run_fixed_point([closure], x)

    assert stats["iters"] <= 2, \
        f"Expected ≤2 iterations, got {stats['iters']}"


def test_train_exactness_symmetry_vertical():
    """Test: Vertical reflection completes half-shape to full symmetric object."""
    # Input: half-shape on left
    x = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0]
    ], dtype=int)

    # Expected: mirrored vertically to create symmetric shape
    y = np.array([
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ], dtype=int)

    closure = SYMMETRY_COMPLETION_Closure(
        "test",
        {"axis": "v", "scope": "global", "bg": 0}
    )
    U, _ = run_fixed_point([closure], x)

    assert U.is_fully_determined()
    y_pred = U.to_grid()
    assert np.array_equal(y_pred, y), \
        f"Vertical symmetry failed.\nInput:\n{x}\nPred:\n{y_pred}\nExpected:\n{y}"


def test_train_exactness_symmetry_largest_only():
    """Test: scope='largest' reflects only largest component, leaves noise."""
    # Input: large component (color 1) + small noise (color 2)
    x = np.array([
        [1, 1, 0, 0, 2],
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0]
    ], dtype=int)

    # Expected: largest reflected (col 0->4, col 1->3), noise (2) preserved
    # Vertical reflection: column c mirrors to column W-1-c
    # For W=5: col 0->4, col 1->3, col 2->2 (center)
    # Union preserves original (including noise) over reflection
    y = np.array([
        [1, 1, 0, 1, 2],  # col 4 keeps noise (2), col 3 gets reflection of col 1
        [1, 0, 0, 0, 1],  # col 4 gets reflection of col 0
        [1, 1, 0, 1, 1]   # col 3,4 get reflections of cols 1,0
    ], dtype=int)

    closure = SYMMETRY_COMPLETION_Closure(
        "test",
        {"axis": "v", "scope": "largest", "bg": 0}
    )
    U, _ = run_fixed_point([closure], x)

    assert U.is_fully_determined()
    y_pred = U.to_grid()
    assert np.array_equal(y_pred, y), \
        f"scope='largest' failed.\nPred:\n{y_pred}\nExpected:\n{y}"


def test_train_exactness_symmetry_diagonal():
    """Test: Diagonal reflection on square grid."""
    # Input: upper-triangle pattern
    x = np.array([
        [0, 3, 3],
        [0, 0, 3],
        [0, 0, 0]
    ], dtype=int)

    # Expected: symmetric across main diagonal
    y = np.array([
        [0, 3, 3],
        [3, 0, 3],
        [3, 3, 0]
    ], dtype=int)

    closure = SYMMETRY_COMPLETION_Closure(
        "test",
        {"axis": "diag", "scope": "global", "bg": 0}
    )
    U, _ = run_fixed_point([closure], x)

    assert U.is_fully_determined()
    y_pred = U.to_grid()
    assert np.array_equal(y_pred, y), \
        f"Diagonal symmetry failed.\nPred:\n{y_pred}\nExpected:\n{y}"


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    run_all_tests()
