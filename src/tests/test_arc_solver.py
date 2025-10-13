#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC Solver - Unit Tests
========================

Tests for the modular ARC solver implementation.
"""

import sys
sys.path.insert(0, '/Users/ravishq/code/arc-agi-1/src')

from arc_solver import (
    G, invariants, rot90, rot180, flip_h,
    ARCInstance, solve_instance, solve_with_beam
)
from arc_solver.operators.spatial import CROP_BBOX_NONZERO
from arc_solver.operators.color import COLOR_PERM

def test_invariants():
    """Test invariant detection."""
    g = G([[1,0,0],[1,2,0],[0,2,2]])
    inv = invariants(g, bg=0)

    assert inv.shape == (3,3), f"Expected shape (3,3), got {inv.shape}"
    assert inv.histogram[0] == 4, f"Expected 4 zeros, got {inv.histogram[0]}"
    assert inv.histogram[1] == 2, f"Expected 2 ones, got {inv.histogram[1]}"
    assert inv.histogram[2] == 3, f"Expected 3 twos, got {inv.histogram[2]}"
    assert inv.n_components == 2, f"Expected 2 components, got {inv.n_components}"
    assert inv.sym_rot90 == False, "Grid should not be rot90 symmetric"

    print("✓ test_invariants passed")

def test_symmetry_induction():
    """Test symmetry rule induction."""
    g = G([[1,0,0],[1,2,0],[0,2,2]])
    a = g
    b = rot90(a)

    inst = ARCInstance(
        "rot90_demo",
        [(a, b)],
        [G([[0,9,9],[0,0,9],[8,8,0]])],
        [rot90(G([[0,9,9],[0,0,9],[8,8,0]]))]
    )

    res = solve_instance(inst)

    assert res.rule is not None, "Rule should be found"
    assert res.rule.name == "ROT", f"Expected ROT, got {res.rule.name}"
    assert res.rule.params["k"] == 1, f"Expected k=1, got {res.rule.params['k']}"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"

    print("✓ test_symmetry_induction passed")

def test_crop_bbox():
    """Test crop-to-bbox rule induction."""
    x3 = G([[0,0,0,0],[0,5,5,0],[0,5,0,0]])
    y3 = x3[1:3,1:3]

    inst2 = ARCInstance(
        "crop_demo",
        [(x3, y3)],
        [G([[0,0,0],[0,7,0],[0,7,7]])],
        [G([[7,0],[7,7]])]
    )

    res2 = solve_instance(inst2)

    assert res2.rule is not None, "Rule should be found"
    assert res2.rule.name == "CROP_BBOX_NONZERO", f"Expected CROP_BBOX_NONZERO, got {res2.rule.name}"
    assert res2.acc_exact == 1.0, f"Expected 100% accuracy, got {res2.acc_exact}"

    print("✓ test_crop_bbox passed")

def test_demo_tasks():
    """Test mini demo tasks."""
    tasks = []

    # Symmetry rot180
    x1 = G([[0,1],[2,0]])
    y1 = rot180(x1)
    tasks.append(ARCInstance(
        "sym_rot180",
        [(x1, y1)],
        [G([[0,3],[4,0]])],
        [rot180(G([[0,3],[4,0]]))]
    ))

    # Flip horizontal
    x2 = G([[9,0,0],[0,9,0]])
    y2 = flip_h(x2)
    tasks.append(ARCInstance(
        "flip_h_demo",
        [(x2, y2)],
        [G([[5,0,0],[0,5,0]])],
        [flip_h(G([[5,0,0],[0,5,0]]))]
    ))

    # Crop bbox
    x3 = G([[0,0,0,0],[0,7,7,0],[0,7,0,0]])
    y3 = x3[1:3,1:3]
    tasks.append(ARCInstance(
        "crop_bbox_demo",
        [(x3, y3)],
        [G([[0,0,0],[0,8,0],[0,8,8]])],
        [G([[8,0],[8,8]])]
    ))

    # Run all
    total = 0
    exact = 0
    for inst in tasks:
        res = solve_instance(inst)
        total += len(inst.test_out)
        exact += int(res.acc_exact == 1.0) * len(inst.test_out)

        assert res.rule is not None, f"Task {inst.name}: Rule should be found"
        assert res.acc_exact == 1.0, f"Task {inst.name}: Expected 100% accuracy, got {res.acc_exact}"

    print(f"✓ test_demo_tasks passed ({exact}/{total} exact)")

def test_move_obj_rank():
    """Test MOVE_OBJ_RANK rule induction."""
    # MOVE largest by Δ=(+1,+2)
    x = G([[0,0,0,0,0],
           [0,1,1,0,0],
           [0,1,1,0,0],
           [0,0,0,0,0]])
    y = G([[0,0,0,0,0],
           [0,0,0,0,0],
           [0,0,0,1,1],
           [0,0,0,1,1]])

    inst = ARCInstance(
        "move_largest",
        train=[(x, y)],
        test_in=[G([[0,2,2,0,0],
                    [0,2,2,0,0],
                    [0,0,0,0,0],
                    [0,0,0,0,0]])],
        test_out=[G([[0,0,0,0,0],
                     [0,0,0,2,2],
                     [0,0,0,2,2],
                     [0,0,0,0,0]])]
    )

    res = solve_instance(inst)

    assert res.rule is not None, "Rule should be found"
    assert res.rule.name == "MOVE_OBJ_RANK", f"Expected MOVE_OBJ_RANK, got {res.rule.name}"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"
    assert res.receipts[0].residual == 0, f"Expected residual=0, got {res.receipts[0].residual}"

    print("✓ test_move_obj_rank passed")

def test_copy_obj_rank():
    """Test COPY_OBJ_RANK rule induction."""
    # COPY largest by Δ=(0,+3)
    x = G([[3,3,0,0,0,0],
           [3,3,0,0,0,0]])
    y = G([[3,3,0,3,3,0],
           [3,3,0,3,3,0]])

    inst = ARCInstance(
        "copy_largest",
        train=[(x, y)],
        test_in=[G([[4,4,0,0,0,0],
                    [4,4,0,0,0,0]])],
        test_out=[G([[4,4,0,4,4,0],
                     [4,4,0,4,4,0]])]
    )

    res = solve_instance(inst)

    assert res.rule is not None, "Rule should be found"
    assert res.rule.name == "COPY_OBJ_RANK", f"Expected COPY_OBJ_RANK, got {res.rule.name}"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"
    assert res.receipts[0].residual == 0, f"Expected residual=0, got {res.receipts[0].residual}"

    print("✓ test_copy_obj_rank passed")

def test_delta_enumeration():
    """Test Δ enumeration with multiple candidates."""
    # Create scenario where object moves among other objects
    # This can't be explained by simpler rules like ROT/FLIP

    # Train pair 1: Move largest object (size 4) right by 2
    x1 = G([[0,1,1,0,0,0],
            [0,1,1,0,0,0],
            [0,0,0,0,5,0]])  # Also has a small object
    y1 = G([[0,0,0,1,1,0],  # Delta (0,2) - moved right by 2
            [0,0,0,1,1,0],
            [0,0,0,0,5,0]])  # Small object stays

    # Train pair 2: Different colors, same delta (0,2)
    x2 = G([[0,2,2,0,0,0],
            [0,2,2,0,0,0],
            [0,0,0,0,7,0]])
    y2 = G([[0,0,0,2,2,0],  # Delta (0,2) - moved right by 2
            [0,0,0,2,2,0],
            [0,0,0,0,7,0]])

    inst = ARCInstance(
        "delta_enum_test",
        train=[(x1, y1), (x2, y2)],
        test_in=[G([[0,3,3,0,0,0],
                    [0,3,3,0,0,0],
                    [0,0,0,0,8,0]])],
        test_out=[G([[0,0,0,3,3,0],
                     [0,0,0,3,3,0],
                     [0,0,0,0,8,0]])]
    )

    res = solve_instance(inst)

    assert res.rule is not None, "Rule should be found with Δ enumeration"
    assert res.rule.name == "MOVE_OBJ_RANK", f"Expected MOVE_OBJ_RANK, got {res.rule.name}"
    assert res.rule.params["delta"] == (0, 2), f"Expected delta (0,2), got {res.rule.params['delta']}"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"
    assert res.receipts[0].residual == 0, f"Expected residual=0, got {res.receipts[0].residual}"

    print("✓ test_delta_enumeration passed")

def test_beam_rot_then_crop():
    """Test beam search: ROT90 → CROP_BBOX composition."""
    # Train: rotate 90° then crop to bbox
    a = G([[0,1,0],[0,0,1],[0,0,0]])
    b = CROP_BBOX_NONZERO(0)(rot90(a))

    # Test input
    test_in = G([[0,2,0],[0,0,2],[0,0,0]])
    test_out = CROP_BBOX_NONZERO(0)(rot90(test_in))

    inst = ARCInstance(
        "beam_rot_crop",
        train=[(a, b)],
        test_in=[test_in],
        test_out=[test_out]
    )

    res = solve_with_beam(inst, max_depth=2, beam_size=20)

    assert res.rule is not None, "Beam search should find solution"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"
    assert res.receipts[0].residual == 0, f"Expected residual=0, got {res.receipts[0].residual}"

    print("✓ test_beam_rot_then_crop passed")

def test_beam_versus_single():
    """Test that beam search can solve what single-step solver can."""
    # Use a task that single-step solver solves
    x = G([[0,1,0],[0,0,1],[0,0,0]])
    y = rot90(x)

    inst = ARCInstance(
        "beam_vs_single",
        train=[(x, y)],
        test_in=[G([[0,2,0],[0,0,2],[0,0,0]])],
        test_out=[rot90(G([[0,2,0],[0,0,2],[0,0,0]]))]
    )

    # Both should solve it
    res_single = solve_instance(inst)
    res_beam = solve_with_beam(inst, max_depth=2, beam_size=20)

    assert res_single.acc_exact == 1.0, "Single-step should solve"
    assert res_beam.acc_exact == 1.0, "Beam should also solve"
    assert res_beam.receipts[0].residual == 0, f"Expected residual=0"

    print("✓ test_beam_versus_single passed")

def test_beam_single_step_fallback():
    """Test beam search on single-step tasks (should still work)."""
    # Single-step rot90 task
    x = G([[0,1,0],[0,0,1],[0,0,0]])
    y = rot90(x)

    inst = ARCInstance(
        "beam_single_rot90",
        train=[(x, y)],
        test_in=[G([[0,2,0],[0,0,2],[0,0,0]])],
        test_out=[rot90(G([[0,2,0],[0,0,2],[0,0,0]]))]
    )

    res = solve_with_beam(inst, max_depth=2, beam_size=20)

    assert res.rule is not None, "Beam search should find single-step solution"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"
    assert res.receipts[0].residual == 0, f"Expected residual=0, got {res.receipts[0].residual}"

    print("✓ test_beam_single_step_fallback passed")

def test_tile_simple():
    """Test TILE rule induction."""
    # Tile 2x3
    x = G([[1,2],
           [3,4]])
    y = G([[1,2,1,2,1,2],
           [3,4,3,4,3,4],
           [1,2,1,2,1,2],
           [3,4,3,4,3,4]])

    inst = ARCInstance(
        "tile_2x3",
        train=[(x, y)],
        test_in=[G([[5,6],
                    [7,8]])],
        test_out=[G([[5,6,5,6,5,6],
                     [7,8,7,8,7,8],
                     [5,6,5,6,5,6],
                     [7,8,7,8,7,8]])]
    )

    res = solve_instance(inst)

    assert res.rule is not None, "TILE rule should be found"
    assert res.rule.name == "TILE", f"Expected TILE, got {res.rule.name}"
    assert res.rule.params["nx"] == 2, f"Expected nx=2, got {res.rule.params['nx']}"
    assert res.rule.params["ny"] == 3, f"Expected ny=3, got {res.rule.params['ny']}"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"
    assert res.receipts[0].residual == 0, f"Expected residual=0, got {res.receipts[0].residual}"

    print("✓ test_tile_simple passed")

def test_tile_const_pattern():
    """Test TILE_CONST_PATTERN rule induction (constant output pattern)."""
    # Output is always tiled [[5,6],[7,8]], sized to match input dimensions
    # This tests input-independent constant pattern tiling

    # Train pair 1: Input [[1,1],[0,0]] → Output [[5,6],[7,8]]
    # Input content irrelevant, only size matters
    x1 = G([[1,1],
            [0,0]])
    y1 = G([[5,6],
            [7,8]])

    # Train pair 2: Input [[2,2,2,2],[0,0,0,0]] → Output [[5,6,5,6],[7,8,7,8]]
    # Different input content, same motif pattern
    x2 = G([[2,2,2,2],
            [0,0,0,0]])
    y2 = G([[5,6,5,6],
            [7,8,7,8]])

    inst = ARCInstance(
        "tile_const_pattern",
        train=[(x1, y1), (x2, y2)],
        test_in=[G([[3,3,3,3,3,3],
                    [9,9,9,9,9,9]])],
        test_out=[G([[5,6,5,6,5,6],
                     [7,8,7,8,7,8]])]
    )

    res = solve_instance(inst)

    assert res.rule is not None, "TILE_CONST_PATTERN rule should be found"
    assert res.rule.name == "TILE_CONST_PATTERN", f"Expected TILE_CONST_PATTERN, got {res.rule.name}"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"
    assert res.receipts[0].residual == 0, f"Expected residual=0, got {res.receipts[0].residual}"

    print("✓ test_tile_const_pattern passed")

def test_draw_line_horizontal():
    """Test DRAW_LINE rule induction (horizontal line)."""
    # Draw horizontal line at row 1
    x = G([[0,0,0,0],
           [0,0,0,0],
           [0,0,0,0]])
    y = G([[0,0,0,0],
           [7,7,7,7],
           [0,0,0,0]])

    inst = ARCInstance(
        "draw_line_h",
        train=[(x, y)],
        test_in=[G([[0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0]])],
        test_out=[G([[0,0,0,0],
                     [7,7,7,7],
                     [0,0,0,0]])]
    )

    res = solve_instance(inst)

    assert res.rule is not None, "DRAW_LINE rule should be found"
    assert res.rule.name == "DRAW_LINE", f"Expected DRAW_LINE, got {res.rule.name}"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"
    assert res.receipts[0].residual == 0, f"Expected residual=0, got {res.receipts[0].residual}"

    print("✓ test_draw_line_horizontal passed")

def test_draw_line_vertical():
    """Test DRAW_LINE rule induction (vertical line)."""
    # Draw vertical line at col 2
    x = G([[0,0,0,0],
           [0,0,0,0],
           [0,0,0,0]])
    y = G([[0,0,5,0],
           [0,0,5,0],
           [0,0,5,0]])

    inst = ARCInstance(
        "draw_line_v",
        train=[(x, y)],
        test_in=[G([[0,0,0,0],
                    [0,0,0,0],
                    [0,0,0,0]])],
        test_out=[G([[0,0,5,0],
                     [0,0,5,0],
                     [0,0,5,0]])]
    )

    res = solve_instance(inst)

    assert res.rule is not None, "DRAW_LINE rule should be found"
    assert res.rule.name == "DRAW_LINE", f"Expected DRAW_LINE, got {res.rule.name}"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"
    assert res.receipts[0].residual == 0, f"Expected residual=0, got {res.receipts[0].residual}"

    print("✓ test_draw_line_vertical passed")

def test_parity_recolor():
    """Test RECOLOR_PARITY rule induction (checkerboard pattern)."""
    # Recolor even parity cells to color 9
    # Key: Same color (1) appears in both even and odd positions, mapping differently
    # COLOR_PERM cannot handle this (one color → two different values)
    x = G([[1,1,1,1],
           [2,2,2,2],
           [1,1,1,1]])
    # Even parity (r+c even) cells → 9, odd parity cells stay same
    # Color 1 at (0,0),(0,2),(1,1),(1,3),(2,0),(2,2) → stays at (0,1),(0,3),(2,1),(2,3)
    y = G([[9,1,9,1],
           [2,9,2,9],
           [9,1,9,1]])

    inst = ARCInstance(
        "parity_even",
        train=[(x, y)],
        test_in=[G([[3,3,3,3],
                    [4,4,4,4],
                    [3,3,3,3]])],
        test_out=[G([[9,3,9,3],
                     [4,9,4,9],
                     [9,3,9,3]])]
    )

    res = solve_instance(inst)

    assert res.rule is not None, "RECOLOR_PARITY rule should be found"
    assert res.rule.name == "RECOLOR_PARITY", f"Expected RECOLOR_PARITY, got {res.rule.name}"
    assert res.rule.params["parity"] == "even", f"Expected even parity, got {res.rule.params['parity']}"
    assert res.rule.params["color"] == 9, f"Expected color 9, got {res.rule.params['color']}"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"
    assert res.receipts[0].residual == 0, f"Expected residual=0, got {res.receipts[0].residual}"

    print("✓ test_parity_recolor passed")

def test_border_paint():
    """Test BORDER_PAINT rule induction."""
    # Paint border with color 9
    x = G([[1,2,3,4],
           [5,6,7,8],
           [9,1,2,3],
           [4,5,6,7]])
    # Border cells painted with 9, interior unchanged
    y = G([[9,9,9,9],
           [9,6,7,9],
           [9,1,2,9],
           [9,9,9,9]])

    inst = ARCInstance(
        "border_paint",
        train=[(x, y)],
        test_in=[G([[1,1,1,1],
                    [2,2,2,2],
                    [3,3,3,3],
                    [4,4,4,4]])],
        test_out=[G([[9,9,9,9],
                     [9,2,2,9],
                     [9,3,3,9],
                     [9,9,9,9]])]
    )

    res = solve_instance(inst)

    assert res.rule is not None, "BORDER_PAINT rule should be found"
    assert res.rule.name == "BORDER_PAINT", f"Expected BORDER_PAINT, got {res.rule.name}"
    assert res.rule.params["color"] == 9, f"Expected color 9, got {res.rule.params['color']}"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"
    assert res.receipts[0].residual == 0, f"Expected residual=0, got {res.receipts[0].residual}"

    print("✓ test_border_paint passed")

def test_copy_chain():
    """Test COPY_OBJ_RANK with repetition chains (k>1)."""
    # Simple test: Copy object 2 times with spacing
    # Original at (0,0), copies at (0,3), (0,6)
    x = G([[1,0,0,0,0,0,0,0,0],
           [1,0,0,0,0,0,0,0,0]])
    # k=2: Creates 2 additional copies at delta=(0,3), 2*delta=(0,6)
    y = G([[1,0,0,1,0,0,1,0,0],
           [1,0,0,1,0,0,1,0,0]])

    inst = ARCInstance(
        "copy_chain_k2",
        train=[(x, y)],
        test_in=[G([[2,0,0,0,0,0,0,0,0],
                    [2,0,0,0,0,0,0,0,0]])],
        test_out=[G([[2,0,0,2,0,0,2,0,0],
                     [2,0,0,2,0,0,2,0,0]])]
    )

    res = solve_instance(inst)

    assert res.rule is not None, "COPY_OBJ_RANK rule should be found"
    assert res.rule.name == "COPY_OBJ_RANK", f"Expected COPY_OBJ_RANK, got {res.rule.name}"
    assert res.rule.params["delta"] == (0, 3), f"Expected delta (0,3), got {res.rule.params['delta']}"
    assert res.rule.params["k"] == 2, f"Expected k=2, got {res.rule.params['k']}"
    assert res.acc_exact == 1.0, f"Expected 100% accuracy, got {res.acc_exact}"
    assert res.receipts[0].residual == 0, f"Expected residual=0, got {res.receipts[0].residual}"

    print("✓ test_copy_chain passed")

def run_all_tests():
    """Run all unit tests."""
    print("Running ARC Solver Unit Tests...")
    print("=" * 50)

    test_invariants()
    test_symmetry_induction()
    test_crop_bbox()
    test_demo_tasks()
    test_move_obj_rank()
    test_copy_obj_rank()
    test_delta_enumeration()

    print("\nRunning Tiling & Drawing Tests...")
    test_tile_simple()
    test_tile_const_pattern()
    test_draw_line_horizontal()
    test_draw_line_vertical()

    print("\nRunning Parity & Repetition Tests...")
    test_parity_recolor()
    test_border_paint()
    test_copy_chain()

    print("\nRunning Beam Search Tests...")
    test_beam_rot_then_crop()
    test_beam_versus_single()
    test_beam_single_step_fallback()

    print("=" * 50)
    print("All tests passed! ✅")

if __name__ == "__main__":
    run_all_tests()
