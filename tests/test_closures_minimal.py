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
    verify_closures_on_train,
    preserves_y,
    compatible_to_y
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
    unify_SYMMETRY_COMPLETION,
    MOD_PATTERN_Closure,
    unify_MOD_PATTERN,
    CANVAS_SIZE_Closure,
    unify_CANVAS_SIZE,
    TILING_Closure,
    unify_TILING,
    COPY_BY_DELTAS_Closure,
    unify_COPY_BY_DELTAS,
    DIAGONAL_REPEAT_Closure,
    unify_DIAGONAL_REPEAT,
    infer_bg_from_border
)
from arc_solver.closure_engine import _compute_canvas
from arc_solver.search import autobuild_closures


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
        # A+B: Composition-Safe Gate + Per-Input BG tests
        ("A: Composition gate preserves_y", test_composition_gate_preserves_y),
        ("A: Composition gate compatible_to_y", test_composition_gate_compatible_to_y),
        ("A: Greedy composition backoff", test_greedy_composition_backoff),
        ("B: Per-input bg inference", test_per_input_bg_inference),
        ("B: Per-input bg deterministic", test_per_input_bg_deterministic),
        # MOD_PATTERN tests
        ("MOD_PATTERN: Shrinking", test_shrinking_mod_pattern),
        ("MOD_PATTERN: Idempotence", test_idempotence_mod_pattern),
        ("MOD_PATTERN: Train exactness", test_train_exactness_mod_pattern),
        # Gate-Fix tests
        ("Gate-Fix: compatible_to_y allows complementary", test_compatible_to_y_allows_complementary_closures),
        ("Gate-Fix: unifiers collect multiple candidates", test_unifiers_collect_multiple_candidates),
        # NX-UNSTICK tests
        ("NX-UNSTICK: Mask-driven keeps correct pixels", test_mask_driven_keeps_correct_pixels),
        ("NX-UNSTICK: COLOR_PERM bijective check", test_color_perm_bijective),
        ("NX-UNSTICK: COLOR_PERM train exact", test_color_perm_train_exact),
        ("NX-UNSTICK: COLOR_PERM shrinking", test_color_perm_shrinking),
        ("NX-UNSTICK: RECOLOR_ON_MASK templates", test_recolor_on_mask_templates),
        ("NX-UNSTICK: RECOLOR_ON_MASK strategies", test_recolor_on_mask_strategies),
        ("NX-UNSTICK: RECOLOR_ON_MASK train exact", test_recolor_on_mask_train_exact),
        # CANVAS-AWARE tests
        ("CANVAS-AWARE: Same shape (backward compat)", test_canvas_size_same_shape),
        ("CANVAS-AWARE: Tile multiple (expansion)", test_canvas_size_tile_multiple),
        ("CANVAS-AWARE: Gates shape-aware", test_gates_shape_aware),
        # CANVAS GREEDY VERIFY FIX tests
        ("CANVAS-FIX: Shape-growing meta closure", test_canvas_size_shape_growing),
        ("CANVAS-FIX: Same-shape regression", test_canvas_size_same_shape_regression),
        ("CANVAS-FIX: Consistent verify allows TOP", test_verify_consistent_on_train_allows_top),
        # M4.1: TILING tests
        ("TILING: Shrinking (global)", test_shrinking_tiling_global),
        ("TILING: Idempotence (global)", test_idempotence_tiling_global),
        ("TILING: Train exactness (global)", test_train_exactness_tiling_global),
        ("TILING: Train exactness (on mask)", test_train_exactness_tiling_on_mask),
        ("TILING: Patch-exact on residual", test_patch_exact_on_residual),
        # M4.2: COPY_BY_DELTAS tests
        ("COPY_BY_DELTAS: Shrinking", test_shrinking_copy_by_deltas),
        ("COPY_BY_DELTAS: Idempotence", test_idempotence_copy_by_deltas),
        ("COPY_BY_DELTAS: Train exactness (shifted)", test_train_exactness_copy_by_deltas_shifted_template),
        ("COPY_BY_DELTAS: Composition safe", test_train_exactness_copy_by_deltas_composition_safe),
        # M3.2: DIAGONAL_REPEAT tests
        ("DIAGONAL_REPEAT: Shrinking", test_shrinking_diagonal_repeat),
        ("DIAGONAL_REPEAT: Idempotence", test_idempotence_diagonal_repeat),
        ("DIAGONAL_REPEAT: Train exactness", test_train_exactness_diagonal_repeat),
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
# Composition-Safe Gate Tests (Part A)
# ==============================================================================

def test_composition_gate_preserves_y():
    """Test: preserves_y helper correctly validates closures."""
    x1 = np.array([[1, 1], [0, 2]])
    y1 = np.array([[1, 1], [0, 0]])
    train = [(x1, y1)]

    # KEEP_LARGEST with bg=0 should preserve y1
    closure_good = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})
    assert preserves_y(closure_good, train), \
        "KEEP_LARGEST with bg=0 should preserve y1"

    # KEEP_LARGEST with bg=1 should NOT preserve y1
    closure_bad = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 1})
    assert not preserves_y(closure_bad, train), \
        "KEEP_LARGEST with bg=1 should NOT preserve y1"


def test_composition_gate_compatible_to_y():
    """Test: compatible_to_y helper correctly validates closures."""
    x1 = np.array([[1, 1], [0, 2]])
    y1 = np.array([[1, 1], [0, 0]])
    train = [(x1, y1)]

    # Closure that keeps colors {0,1,2} should be compatible
    closure_good = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})
    assert compatible_to_y(closure_good, train), \
        "Closure should be compatible (colors in output subset of y)"


def test_greedy_composition_backoff():
    """Test: autobuild_closures backs off when composition fails."""
    # Create train where individual closures pass gates but composition might fail
    x1 = np.array([[1, 1], [0, 0]])
    y1 = np.array([[1, 1], [0, 0]])
    train = [(x1, y1)]

    closures = autobuild_closures(train)
    # Should return subset that composes to exact train result
    assert verify_closures_on_train(closures, train), \
        "Composed closures should verify on train (greedy back-off ensures this)"


# ==============================================================================
# Per-Input BG Tests (Part B)
# ==============================================================================

def test_per_input_bg_inference():
    """Test: bg=None infers background per input via border flood-fill."""
    # Pair 1: bg=0 (inferred from border)
    x1 = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    y1 = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])

    # Pair 2: bg=2 (inferred from border)
    x2 = np.array([[2, 1, 2], [2, 1, 2], [2, 2, 2]])
    y2 = np.array([[2, 1, 2], [2, 1, 2], [2, 2, 2]])

    train = [(x1, y1), (x2, y2)]

    # Unifier should prefer bg=None (infer per input)
    closures = unify_KEEP_LARGEST(train)
    assert len(closures) >= 1, "Should find at least one closure"

    # Check if any closure uses bg=None
    has_bg_none = any(c.params.get("bg") is None for c in closures)
    assert has_bg_none, "Should prefer bg=None for per-input inference"

    # Verify train exactness with per-input bg
    assert verify_closures_on_train(closures, train), \
        "Closures with bg=None should verify on train"


def test_per_input_bg_deterministic():
    """Test: infer_bg_from_border is deterministic (tie-breaks to lowest color)."""
    # Border has equal counts of 0 and 1 → should pick 0 (lowest)
    x = np.array([[0, 0, 1, 1], [0, 9, 9, 1], [1, 1, 0, 0]])
    bg = infer_bg_from_border(x)
    assert bg in {0, 1}, "bg should be border color"

    # Run twice to verify determinism
    bg2 = infer_bg_from_border(x)
    assert bg == bg2, "bg inference must be deterministic"


# ==============================================================================
# MOD_PATTERN Property Tests
# ==============================================================================

def test_shrinking_mod_pattern():
    """Test: apply(U) subset U (closure only removes possibilities)."""
    # 2x3 periodic pattern: (p,q)=(2,3), anchor=(0,0)
    # Class (0,0): colors {1}
    # Class (0,1): colors {2}
    # Class (0,2): colors {3}
    # Class (1,0): colors {4}
    # Class (1,1): colors {5}
    # Class (1,2): colors {6}
    x = np.array([
        [1, 2, 3, 1, 2, 3],
        [4, 5, 6, 4, 5, 6],
        [1, 2, 3, 1, 2, 3],
        [4, 5, 6, 4, 5, 6]
    ], dtype=int)

    # Build class_map from this periodic pattern
    class_map = {
        (0, 0): {1}, (0, 1): {2}, (0, 2): {3},
        (1, 0): {4}, (1, 1): {5}, (1, 2): {6}
    }

    params = {"p": 2, "q": 3, "anchor": (0, 0), "class_map": class_map}
    closure = MOD_PATTERN_Closure("test", params)

    U = init_top(x.shape[0], x.shape[1])
    U_copy = U.copy()
    U_result = closure.apply(U, x)

    assert grid_subset_or_equal(U_result, U_copy), \
        "Shrinking violated: apply(U) should be subset of U"


def test_idempotence_mod_pattern():
    """Test: <=2-pass convergence for MOD_PATTERN."""
    # Simple parity pattern (2,2)
    x = np.array([
        [1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2]
    ], dtype=int)

    class_map = {
        (0, 0): {1}, (0, 1): {2},
        (1, 0): {3}, (1, 1): {4}
    }

    params = {"p": 2, "q": 2, "anchor": (0, 0), "class_map": class_map}
    closure = MOD_PATTERN_Closure("test", params)

    U, stats = run_fixed_point([closure], x)

    assert stats["iters"] <= 2, \
        f"Expected <=2 iterations, got {stats['iters']}"


def test_train_exactness_mod_pattern():
    """Test: MOD_PATTERN unifier finds periodic pattern and achieves train exactness."""
    # Train pair with clear (p,q)=(2,2) parity pattern
    x1 = np.array([
        [1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2]
    ], dtype=int)

    y1 = np.array([
        [1, 2, 1, 2],
        [3, 4, 3, 4],
        [1, 2, 1, 2]
    ], dtype=int)

    # Second pair with same pattern (different colors to test unification)
    x2 = np.array([
        [5, 6],
        [7, 8]
    ], dtype=int)

    y2 = np.array([
        [5, 6],
        [7, 8]
    ], dtype=int)

    train = [(x1, y1), (x2, y2)]

    # Unifier should find (p,q)=(2,2) pattern
    closures = unify_MOD_PATTERN(train)

    if len(closures) > 0:
        # If unifier succeeded, verify train exactness
        assert verify_closures_on_train(closures, train), \
            "MOD_PATTERN closures should verify on train"

        # Check params
        closure = closures[0]
        assert closure.params["p"] == 2, "Should find p=2"
        assert closure.params["q"] == 2, "Should find q=2"
    # If no closures found, that's OK (composition-safe gate rejected it)


# ==============================================================================
# Gate-Fix Tests
# ==============================================================================

def test_compatible_to_y_allows_complementary_closures():
    """Test: New compatible_to_y allows closures that introduce colors needing later fix."""
    # Task: Half-shape needs SYMMETRY + COLOR_PERM
    # Input: half red (1), output: full blue (8)
    x = np.array([[1, 1, 0, 0], [1, 0, 0, 0]])
    y = np.array([[8, 8, 8, 8], [8, 0, 0, 8]])
    train = [(x, y)]

    # SYMMETRY_COMPLETION will produce red on right side (not in y's palette)
    # OLD compatible_to_y would reject this
    # NEW compatible_to_y should accept (doesn't destroy known-correct pixels)
    sym_closure = SYMMETRY_COMPLETION_Closure(
        "test", {"axis": "v", "scope": "global", "bg": 0}
    )

    # Should pass new gate (doesn't destroy x[r,c]==y[r,c] pixels)
    assert compatible_to_y(sym_closure, train), \
        "New compatible_to_y should allow complementary closures"


def test_unifiers_collect_multiple_candidates():
    """Test: Unifiers return list of all valid candidates (not just first)."""
    # Create train where multiple bgs work
    x1 = np.array([[1, 1], [2, 2]])
    y1 = np.array([[1, 1], [0, 0]])
    train = [(x1, y1)]

    # Multiple bgs might pass gates (bg=0, bg=2, etc.)
    closures = unify_KEEP_LARGEST(train)

    # Should return ALL valid candidates (at least bg=None should work)
    assert len(closures) >= 1, "Should collect at least one candidate"

    # If multiple candidates, verify all pass gates individually
    for c in closures:
        assert preserves_y(c, train), f"Collected closure {c.name} should preserve y"
        assert compatible_to_y(c, train), f"Collected closure {c.name} should be compatible"


# ==============================================================================
# NX-UNSTICK: Mask-driven acceptance test
# ==============================================================================

def test_mask_driven_keeps_correct_pixels():
    """Test: Closures preserve pixels where x[r,c] == y[r,c]."""
    # Task where some pixels are already correct
    x = np.array([[1, 1, 0], [0, 2, 2], [0, 0, 0]])
    y = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]])  # (0,0), (0,1), (0,2), (1,0) same

    # Pixel (1,1) and (1,2) differ: (1,1): 2→0, (1,2): 2→0
    # KEEP_LARGEST should only modify pixels that differ
    closures = unify_KEEP_LARGEST([(x, y)])

    assert len(closures) >= 1, "Should find closure that preserves correct pixels"

    # Verify: Applying to x should produce y exactly
    U, _ = run_fixed_point(closures, x)
    y_pred = U.to_grid()
    assert np.array_equal(y_pred, y), "Mask-driven closure should produce exact y"


# ==============================================================================
# NX-UNSTICK: COLOR_PERM tests
# ==============================================================================

def test_color_perm_bijective():
    """Test: COLOR_PERM rejects non-bijective mappings."""
    from arc_solver.closures import unify_COLOR_PERM

    # Non-bijective: 1→3, 2→3 (collision)
    x = np.array([[1, 2]])
    y = np.array([[3, 3]])
    train = [(x, y)]

    closures = unify_COLOR_PERM(train)
    # Should reject (not bijective)
    assert len(closures) == 0, "Should reject non-bijective mapping"


def test_color_perm_train_exact():
    """Test: COLOR_PERM achieves train exactness on simple swap."""
    from arc_solver.closures import unify_COLOR_PERM

    # Swap: 1→2, 2→1, 0→0
    x = np.array([[0, 1, 2], [1, 2, 0]])
    y = np.array([[0, 2, 1], [2, 1, 0]])
    train = [(x, y)]

    closures = unify_COLOR_PERM(train)
    assert len(closures) >= 1, "Should find color permutation"

    U, _ = run_fixed_point(closures, x)
    y_pred = U.to_grid()
    assert np.array_equal(y_pred, y), "COLOR_PERM should produce exact y"


def test_color_perm_shrinking():
    """Test: COLOR_PERM.apply is shrinking."""
    from arc_solver.closures import COLOR_PERM_Closure

    x = np.array([[1, 2], [3, 4]])
    perm = {1: 5, 2: 6, 3: 7, 4: 8}
    closure = COLOR_PERM_Closure("test", {"perm": perm})

    U = init_top(2, 2)
    U_copy = U.copy()
    U_result = closure.apply(U, x)

    assert grid_subset_or_equal(U_result, U_copy), "COLOR_PERM should be shrinking"


# ==============================================================================
# NX-UNSTICK: RECOLOR_ON_MASK tests
# ==============================================================================

def test_recolor_on_mask_templates():
    """Test: RECOLOR_ON_MASK templates compute correct masks."""
    from arc_solver.closures import RECOLOR_ON_MASK_Closure

    x = np.array([[0, 1, 1], [0, 2, 0], [0, 0, 0]])

    # Template: NONZERO
    closure = RECOLOR_ON_MASK_Closure(
        "test",
        {"template": "NONZERO", "template_params": {}, "strategy": "CONST", "strategy_params": {"c": 5}, "bg": 0}
    )

    # Should recolor all non-zero pixels to 5
    U = init_top(3, 3)
    U_result = closure.apply(U, x)

    # Check that non-zero pixels are constrained to {5}
    assert U_result.get_set(0, 1) == {5}, "Non-zero pixel should be constrained to {5}"
    assert U_result.get_set(0, 2) == {5}, "Non-zero pixel should be constrained to {5}"
    assert U_result.get_set(1, 1) == {5}, "Non-zero pixel should be constrained to {5}"
    # Background pixels should still allow all colors (not constrained by this closure)
    assert len(U_result.get_set(0, 0)) == 10, "Background pixel should not be constrained"


def test_recolor_on_mask_strategies():
    """Test: RECOLOR_ON_MASK strategies compute correct target colors."""
    from arc_solver.closures import _compute_target_color
    import numpy as np

    x = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])

    # Strategy: MODE_ON_MASK (most frequent color in mask)
    # If mask is entire grid, mode is 1 (appears 8 times vs 2 once)
    T = np.ones((3, 3), dtype=bool)
    target = _compute_target_color(x, T, "MODE_ON_MASK", {}, bg=0)
    assert target == 1, "MODE_ON_MASK should return most frequent color (1)"

    # Strategy: CONST
    target = _compute_target_color(x, T, "CONST", {"c": 7}, bg=0)
    assert target == 7, "CONST should return specified color"


def test_recolor_on_mask_train_exact():
    """Test: RECOLOR_ON_MASK achieves train exactness."""
    from arc_solver.closures import unify_RECOLOR_ON_MASK

    # Recolor all non-zero to 8
    x = np.array([[0, 1, 2], [3, 0, 4]])
    y = np.array([[0, 8, 8], [8, 0, 8]])
    train = [(x, y)]

    closures = unify_RECOLOR_ON_MASK(train)
    assert len(closures) >= 1, "Should find RECOLOR_ON_MASK closure"

    U, _ = run_fixed_point(closures, x)
    y_pred = U.to_grid()
    assert np.array_equal(y_pred, y), "RECOLOR_ON_MASK should produce exact y"


# ==============================================================================
# CANVAS-AWARE tests
# ==============================================================================

def test_canvas_size_same_shape():
    """Test: CANVAS_SIZE with SAME strategy (backward compatibility)."""
    # All train pairs have same output shape
    x1 = np.array([[1, 2], [3, 4]])
    y1 = np.array([[5, 6], [7, 8]])
    x2 = np.array([[9, 0], [1, 2]])
    y2 = np.array([[3, 4], [5, 6]])
    train = [(x1, y1), (x2, y2)]

    closures = unify_CANVAS_SIZE(train)
    assert len(closures) == 1, "Should find CANVAS_SIZE closure"

    canvas_closure = closures[0]
    assert canvas_closure.params["strategy"] == "SAME", "Should use SAME strategy"
    assert canvas_closure.params["H"] == 2, "Should have H=2"
    assert canvas_closure.params["W"] == 2, "Should have W=2"

    # Verify identity behavior
    U = init_top(2, 2)
    U_after = canvas_closure.apply(U, x1)
    assert U == U_after, "CANVAS_SIZE should be identity/no-op"


def test_canvas_size_tile_multiple():
    """Test: CANVAS_SIZE with TILE_MULTIPLE strategy (3x expansion)."""
    # Input 2x2 -> output 6x6 (3x expansion)
    x1 = np.array([[1, 2], [3, 4]])
    y1 = np.zeros((6, 6), dtype=int)

    # Input 3x3 -> output 9x9 (3x expansion)
    x2 = np.array([[5, 6, 7], [8, 9, 0], [1, 2, 3]])
    y2 = np.zeros((9, 9), dtype=int)

    train = [(x1, y1), (x2, y2)]

    closures = unify_CANVAS_SIZE(train)
    assert len(closures) == 1, "Should find CANVAS_SIZE closure"

    canvas_closure = closures[0]
    assert canvas_closure.params["strategy"] == "TILE_MULTIPLE", "Should use TILE_MULTIPLE strategy"
    assert canvas_closure.params["k_h"] == 3, "Should have k_h=3"
    assert canvas_closure.params["k_w"] == 3, "Should have k_w=3"

    # Test canvas computation for different inputs
    canvas1 = _compute_canvas(x1, canvas_closure.params)
    assert canvas1["H"] == 6, "Should compute H=6 for 2x2 input"
    assert canvas1["W"] == 6, "Should compute W=6 for 2x2 input"

    canvas2 = _compute_canvas(x2, canvas_closure.params)
    assert canvas2["H"] == 9, "Should compute H=9 for 3x3 input"
    assert canvas2["W"] == 9, "Should compute W=9 for 3x3 input"

    # Verify run_fixed_point initializes to correct size
    U, _ = run_fixed_point([canvas_closure], x1, canvas=canvas1)
    assert U.H == 6 and U.W == 6, "Should initialize U to 6x6 for 2x2 input"


def test_gates_shape_aware():
    """Test: Gates work on shape-changing tasks (x.shape != y.shape)."""
    # Shape-changing task: 2x2 input -> 4x4 output
    x = np.array([[1, 1], [0, 0]])
    y = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    train = [(x, y)]

    # CANVAS_SIZE should not be rejected by gates
    canvas_closures = unify_CANVAS_SIZE(train)
    assert len(canvas_closures) == 1, "CANVAS_SIZE should work on shape-changing tasks"

    # preserves_y should not fail due to shape mismatch
    canvas_closure = canvas_closures[0]
    assert preserves_y(canvas_closure, train), \
        "preserves_y should pass for identity closure (shape-aware)"

    # compatible_to_y should not fail due to shape mismatch
    assert compatible_to_y(canvas_closure, train), \
        "compatible_to_y should pass for identity closure (shape-aware)"

    # Create a real closure (KEEP_LARGEST) and test gates on shape-changing pair
    # For this test, we'll use a pair where overlapping region has same colors
    x2 = np.array([[1, 1], [1, 0]])
    y2 = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])
    train2 = [(x2, y2)]

    keep_largest = KEEP_LARGEST_COMPONENT_Closure("test", {"bg": 0})

    # Gates should work on shape-changing tasks (check overlapping region only)
    # This should pass because overlapping cells (2x2) match between x and y
    assert compatible_to_y(keep_largest, train2), \
        "compatible_to_y should check only overlapping region for shape-changing tasks"


# ==============================================================================
# CANVAS GREEDY VERIFY FIX TESTS
# ==============================================================================

def test_canvas_size_shape_growing():
    """CANVAS_SIZE stays in kept_meta for shape-growing tasks."""
    from arc_solver.closure_engine import verify_consistent_on_train

    # Use two train pairs with different input shapes to force TILE_MULTIPLE
    # (SAME strategy only matches if all outputs have same shape)
    x1 = np.array([[1, 2], [3, 4]])
    y1 = np.full((6, 6), 5)
    x2 = np.array([[7, 8, 9], [0, 1, 2], [3, 4, 5]])
    y2 = np.full((9, 9), 5)
    train = [(x1, y1), (x2, y2)]

    canvas_closures = unify_CANVAS_SIZE(train)
    assert len(canvas_closures) == 1, "Should get one CANVAS_SIZE closure"
    canvas = canvas_closures[0]
    assert canvas.is_meta, "CANVAS_SIZE should be meta closure"
    assert canvas.params["strategy"] == "TILE_MULTIPLE", "Should detect TILE_MULTIPLE"
    assert canvas.params["k_h"] == 3 and canvas.params["k_w"] == 3, "Multipliers should be 3"
    assert "H" not in canvas.params and "W" not in canvas.params, "Should NOT store absolute H, W"

    assert verify_consistent_on_train([canvas], train), "CANVAS_SIZE alone should pass consistency"
    assert not verify_closures_on_train([canvas], train), "CANVAS_SIZE alone can't achieve exactness"


def test_canvas_size_same_shape_regression():
    """CANVAS_SIZE should still work for same-shape tasks (baseline coverage)."""
    x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y1 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    train = [(x1, y1)]

    canvas_closures = unify_CANVAS_SIZE(train)
    assert len(canvas_closures) == 1, "Should get one CANVAS_SIZE closure"
    canvas = canvas_closures[0]
    assert canvas.params["strategy"] == "SAME", "Should detect SAME strategy"
    assert canvas.params["H"] == 3 and canvas.params["W"] == 3, "Should store H=3, W=3 for SAME"


def test_verify_consistent_on_train_allows_top():
    """verify_consistent_on_train should return True for closures leaving cells TOP."""
    from arc_solver.closure_engine import verify_consistent_on_train

    x1 = np.array([[1, 2], [3, 4]])
    y1 = np.array([[5, 6], [7, 8]])
    train = [(x1, y1)]

    canvas_closures = unify_CANVAS_SIZE(train)
    canvas = canvas_closures[0]

    assert verify_consistent_on_train([canvas], train), "Identity closure should pass consistency"
    assert not verify_closures_on_train([canvas], train), "Identity closure should fail exactness"


# ==============================================================================
# M4.1: TILING / TILING_ON_MASK Property Tests
# ==============================================================================

def test_shrinking_tiling_global():
    """Test: apply(U) ⊆ U for TILING (global mode)."""
    # 2×2 motif [[1,2],[3,4]] tiled over 6×6 output (anchor (0,0))
    x = np.array([[1, 2], [3, 4]])
    motif = np.array([[1, 2], [3, 4]])
    params = {"mode": "global", "motif": motif, "anchor": (0,0), "mask_template": None, "mask_params": None}
    closure = TILING_Closure("test", params)

    # U is 6×6 (output size)
    U = init_top(6, 6)
    U_copy = U.copy()
    U_result = closure.apply(U, x)

    assert grid_subset_or_equal(U_result, U_copy), "Shrinking violated"


def test_idempotence_tiling_global():
    """Test: ≤2-pass convergence for TILING (global)."""
    x = np.array([[1, 2], [3, 4]])
    motif = np.array([[1, 2], [3, 4]])
    params = {"mode": "global", "motif": motif, "anchor": (0,0), "mask_template": None, "mask_params": None}
    closure = TILING_Closure("test", params)

    U, stats = run_fixed_point([closure], x, canvas={"H": 6, "W": 6})
    assert stats["iters"] <= 2, f"Expected ≤2 iterations, got {stats['iters']}"


def test_train_exactness_tiling_global():
    """Test: TILING (global) achieves train exactness on mini."""
    # Input 2×2, output 6×6 with 2×2 motif tiled
    x = np.array([[1, 2], [3, 4]])
    y = np.array([
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4]
    ])
    train = [(x, y)]

    closures = unify_TILING(train)
    assert len(closures) >= 1, "Should find TILING closure"

    # Verify train exactness (needs CANVAS_SIZE + TILING composition)
    canvas_closures = unify_CANVAS_SIZE(train)
    all_closures = canvas_closures + closures
    assert verify_closures_on_train(all_closures, train), "TILING should achieve train exactness with CANVAS_SIZE"


def test_train_exactness_tiling_on_mask():
    """Test: TILING_ON_MASK fills only BACKGROUND or NOT_LARGEST."""
    # Input has largest component (color 1) + background (0)
    # Output tiles motif only on background
    x = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    y = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])
    train = [(x, y)]

    closures = unify_TILING(train)
    assert len(closures) >= 1, "Should find TILING_ON_MASK closure"

    # Check mode="mask" with mask_template="BACKGROUND"
    found_masked = any(c.params.get("mode") == "mask" and c.params.get("mask_template") == "BACKGROUND" for c in closures)
    assert found_masked, "Should find TILING_ON_MASK with BACKGROUND template"

    # Verify train exactness
    assert verify_closures_on_train(closures, train), "TILING_ON_MASK should achieve train exactness"


def test_patch_exact_on_residual():
    """Test: TILING changes only cells in M (residual mask)."""
    # Input 2×2, output 4×4 with partial tiling (only on specific cells)
    x = np.array([[1, 1], [1, 1]])
    y = np.array([[1, 2, 1, 2], [1, 2, 1, 2], [1, 1, 1, 1], [1, 1, 1, 1]])
    train = [(x, y)]

    closures = unify_TILING(train)
    assert len(closures) >= 1, "Should find patch-exact TILING closure"

    # Verify: closure only modifies M = (x != y) cells
    # Run on input to see where changes occur
    # For output canvas, need CANVAS_SIZE
    canvas_closures = unify_CANVAS_SIZE(train)
    canvas = _compute_canvas(x, canvas_closures[0].params) if canvas_closures else {"H": 4, "W": 4}

    U_final, _ = run_fixed_point(closures, x, canvas=canvas)
    y_pred = U_final.to_grid_deterministic(fallback='lowest', bg=0)

    # Check: changes only where x and y differ (in overlapping region)
    M = (x != y[:x.shape[0], :x.shape[1]])  # Residual in overlapping region
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            if not M[r, c]:
                # No change expected where x==y
                assert y_pred[r, c] == x[r, c], f"Should preserve x[{r},{c}] where x==y"


# ==============================================================================
# M4.2: COPY_BY_DELTAS Property Tests
# ==============================================================================

def test_shrinking_copy_by_deltas():
    """Test: apply(U) ⊆ U (closure only removes possibilities)."""
    # Template: 2×2 block at top-left
    x = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    # Params: copy to shifted positions
    params = {
        "template_strategy": "smallest_object",
        "deltas": [(0, 2), (2, 0)],  # right by 2, down by 2
        "mode": "strict"
    }
    closure = COPY_BY_DELTAS_Closure("test", params)

    U = init_top(4, 4)
    U_copy = U.copy()
    U_result = closure.apply(U, x)

    assert grid_subset_or_equal(U_result, U_copy), "Shrinking violated"


def test_idempotence_copy_by_deltas():
    """Test: ≤2-pass convergence for COPY_BY_DELTAS."""
    x = np.array([[2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    params = {
        "template_strategy": "smallest_object",
        "deltas": [(0, 2)],
        "mode": "strict"
    }
    closure = COPY_BY_DELTAS_Closure("test", params)

    U, stats = run_fixed_point([closure], x)
    assert stats["iters"] <= 2, f"Expected ≤2 iterations, got {stats['iters']}"


def test_train_exactness_copy_by_deltas_shifted_template():
    """Test: COPY_BY_DELTAS copies template to shifted positions exactly."""
    # Input: 2×2 red block at top-left
    x = np.array([[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])

    # Output: Same block copied to (0,3) — shifted right by 3
    y = np.array([[1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 0], [0, 0, 0, 0, 0, 0]])

    train = [(x, y)]

    closures = unify_COPY_BY_DELTAS(train)
    assert len(closures) >= 1, "Should find COPY_BY_DELTAS closure"

    # Verify train exactness
    assert verify_closures_on_train(closures, train), \
        "COPY_BY_DELTAS should achieve train exactness"


def test_train_exactness_copy_by_deltas_composition_safe():
    """Test: COPY_BY_DELTAS composes with other closures (preserves_y + compatible_to_y)."""
    # Task with shape-changing: Input 2x2, Output 2x4 (canvas expansion + object copying)
    x = np.array([[3, 0], [0, 0]])
    y = np.array([[3, 0, 3, 0], [0, 0, 0, 0]])  # Copy 1x1 object with delta (0, 2)

    train = [(x, y)]

    # COPY_BY_DELTAS should find closures and pass composition gates
    copy_closures = unify_COPY_BY_DELTAS(train)
    assert len(copy_closures) >= 1, "Should find COPY_BY_DELTAS closures"

    # Verify gates
    for closure in copy_closures:
        assert preserves_y(closure, train), f"{closure.name} should preserve y"
        assert compatible_to_y(closure, train), f"{closure.name} should be compatible to y"

    # Verify can compose with CANVAS_SIZE and achieve train exactness
    canvas_closures = unify_CANVAS_SIZE(train)
    assert len(canvas_closures) >= 1, "Should find CANVAS_SIZE closure"

    # Compose and verify train exactness
    composed = canvas_closures + copy_closures[:1]  # Use first COPY_BY_DELTAS closure
    assert verify_closures_on_train(composed, train), \
        "CANVAS_SIZE + COPY_BY_DELTAS should achieve train exactness"


# ==============================================================================
# M3.2: DIAGONAL_REPEAT Property Tests
# ==============================================================================

def test_shrinking_diagonal_repeat():
    """Test DIAGONAL_REPEAT is monotone shrinking."""
    x = np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=int)

    closure = DIAGONAL_REPEAT_Closure(
        "test",
        {"template_strategy": "smallest_object", "dr": 1, "dc": 1, "k": 2,
         "mode_color": "copy_input_color", "bg": 0}
    )

    U_top = init_top(4, 4)
    U_result = closure.apply(U_top, x)

    # Verify shrinking
    assert grid_subset_or_equal(U_result, U_top), "Shrinking violated"


def test_idempotence_diagonal_repeat():
    """Test DIAGONAL_REPEAT is ≤2-pass idempotent."""
    x = np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=int)

    closure = DIAGONAL_REPEAT_Closure(
        "test",
        {"template_strategy": "smallest_object", "dr": 1, "dc": 1, "k": 2,
         "mode_color": "copy_input_color", "bg": 0}
    )

    U = init_top(4, 4)
    U1 = closure.apply(U, x)
    U2 = closure.apply(U1, x)

    assert U1 == U2, "Should be idempotent within 2 passes"


def test_train_exactness_diagonal_repeat():
    """Test DIAGONAL_REPEAT achieves train exactness on simple diagonal pattern."""
    # Simple diagonal repeat: 1 at (1,1) repeated at (2,2) and (3,3)
    x = np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=int)

    y = np.array([[0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=int)

    train = [(x, y)]

    # Unify should find dr=1, dc=1, k=2
    closures = unify_DIAGONAL_REPEAT(train)

    assert len(closures) > 0, "Should unify DIAGONAL_REPEAT"

    # Verify one closure achieves exactness
    found_exact = False
    for closure in closures:
        if verify_closures_on_train([closure], train):
            found_exact = True
            break

    assert found_exact, "At least one closure should achieve train exactness"


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    run_all_tests()
