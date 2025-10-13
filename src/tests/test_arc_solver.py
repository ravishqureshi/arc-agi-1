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
    ARCInstance, solve_instance
)

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

def run_all_tests():
    """Run all unit tests."""
    print("Running ARC Solver Unit Tests...")
    print("=" * 50)

    test_invariants()
    test_symmetry_induction()
    test_crop_bbox()
    test_demo_tasks()

    print("=" * 50)
    print("All tests passed! ✅")

if __name__ == "__main__":
    run_all_tests()
