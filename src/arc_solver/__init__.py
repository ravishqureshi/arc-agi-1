"""
ARC Solver - Modular Architecture
==================================

A receipts-first ARC-AGI solver using Universe Intelligence principles.

Main components:
- core: Types, invariants, receipts, induction, solver
- operators: DSL operators (symmetry, spatial, masks, composition)

Quick start:
    >>> from arc_solver import G, solve_instance, ARCInstance
    >>> from arc_solver.core import rot180
    >>>
    >>> # Create task
    >>> x = G([[0,1],[2,0]])
    >>> y = rot180(x)
    >>> task = ARCInstance("demo", [(x,y)], [G([[0,3],[4,0]])], [rot180(G([[0,3],[4,0]]))])
    >>>
    >>> # Solve
    >>> result = solve_instance(task)
    >>> print(f"Rule: {result.rule.name}, Accuracy: {result.acc_exact}")
"""

# Core exports
from .core import (
    # Types
    Grid, Mask, ObjList, G, copy_grid, assert_grid, assert_mask,
    # Invariants
    Invariants, invariants, color_histogram, bbox_nonzero,
    connected_components, rot90, rot180, rot270, flip_h, flip_v, exact_equals,
    # Receipts
    Receipts, edit_counts, residual, generate_pce, verify_train_residuals_zero,
    # Induction
    Rule, CATALOG, induce_rule,
    # Solver
    ARCInstance, SolveResult, solve_instance, solve_with_beam,
)

# Operator exports
from .operators import (
    # Symmetry
    ROT, FLIP,
    # Spatial
    BBOX, CROP, CROP_BBOX_NONZERO,
    # Masks
    MASK_COLOR, MASK_NONZERO, KEEP, REMOVE,
    # Composition
    ON, SEQ,
)

__version__ = '1.0.0'

__all__ = [
    # Types
    'Grid', 'Mask', 'ObjList', 'G', 'copy_grid', 'assert_grid', 'assert_mask',
    # Invariants
    'Invariants', 'invariants', 'color_histogram', 'bbox_nonzero',
    'connected_components', 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v',
    'exact_equals',
    # Receipts
    'Receipts', 'edit_counts', 'residual', 'generate_pce',
    'verify_train_residuals_zero',
    # Induction
    'Rule', 'CATALOG', 'induce_rule',
    # Solver
    'ARCInstance', 'SolveResult', 'solve_instance', 'solve_with_beam',
    # Operators
    'ROT', 'FLIP', 'BBOX', 'CROP', 'CROP_BBOX_NONZERO',
    'MASK_COLOR', 'MASK_NONZERO', 'KEEP', 'REMOVE', 'ON', 'SEQ',
]
