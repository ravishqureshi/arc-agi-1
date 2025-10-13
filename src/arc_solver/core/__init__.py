"""ARC Solver - Core Architecture"""

from .types import Grid, Mask, ObjList, G, copy_grid, assert_grid, assert_mask
from .invariants import (
    Invariants,
    invariants,
    color_histogram,
    bbox_nonzero,
    connected_components,
    rot90,
    rot180,
    rot270,
    flip_h,
    flip_v,
    exact_equals,
)
from .receipts import (
    Receipts,
    edit_counts,
    residual,
    generate_pce,
    verify_train_residuals_zero,
)
from .induction import (
    Rule,
    induce_symmetry_rule,
    induce_crop_nonzero_rule,
    induce_keep_nonzero_rule,
    CATALOG,
    induce_rule,
)
from .solver import (
    ARCInstance,
    SolveResult,
    pce_for_rule,
    solve_instance,
)

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
    'Rule', 'induce_symmetry_rule', 'induce_crop_nonzero_rule',
    'induce_keep_nonzero_rule', 'CATALOG', 'induce_rule',
    # Solver
    'ARCInstance', 'SolveResult', 'pce_for_rule', 'solve_instance',
]
