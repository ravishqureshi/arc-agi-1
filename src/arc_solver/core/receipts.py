#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC Solver - Receipts & Verification
=====================================

Every transformation must provide mathematical receipts:
- Residual: Hamming distance to target (0 = exact match)
- Edit bill: (total_edits, boundary_edits, interior_edits)
- PCE: Proof-Carrying English explanation

Universe Intelligence discipline:
- Train residual must be 0 before predicting test
- Edit bills track where changes happen (edges write)
- PCE ties explanation to invariants
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .types import Grid

# =============================================================================
# Receipts Dataclass
# =============================================================================

@dataclass
class Receipts:
    """Mathematical proof that a transformation is correct."""
    residual: int                  # Hamming distance to target (0 = exact)
    edits_total: int               # # cells changed vs input
    edits_boundary: int            # # boundary cells changed
    edits_interior: int            # # interior cells changed
    pce: str                       # Proof-Carrying English explanation

# =============================================================================
# Edit Counting
# =============================================================================

def edit_counts(a: Grid, b: Grid) -> Tuple[int, int, int]:
    """
    Count edits between two grids.

    Returns:
        (total_edits, boundary_edits, interior_edits)

    Returns (-1, -1, -1) if shapes don't match (e.g., crop/resize).

    Universe Intelligence: "Edges write" - track where changes happen.
    """
    if a.shape != b.shape:
        # Shape mismatch (e.g., crop/resize operations)
        return -1, -1, -1

    diff = (a != b)
    total = int(diff.sum())

    if total == 0:
        return 0, 0, 0

    H, W = a.shape
    border = np.zeros_like(diff)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    boundary = int(np.logical_and(diff, border).sum())
    interior = total - boundary

    return total, boundary, interior

# =============================================================================
# Residual Computation
# =============================================================================

def residual(predicted: Grid, target: Grid) -> int:
    """
    Compute Hamming distance (# mismatched cells) between predicted and target.

    Returns:
        0 if exact match (residual = 0)
        N if N cells differ

    Universe Intelligence: residual must be 0 on train before predicting test.
    """
    if predicted.shape != target.shape:
        # Shape mismatch = infinite residual
        return predicted.size + target.size

    return int((predicted != target).sum())

# =============================================================================
# PCE Generation
# =============================================================================

def generate_pce(
    task_name: str,
    rule_name: str,
    params: dict,
    edits: Tuple[int, int, int]
) -> str:
    """
    Generate Proof-Carrying English explanation.

    Args:
        task_name: Task identifier
        rule_name: Name of transformation rule
        params: Parameters used
        edits: (total, boundary, interior) edit counts

    Returns:
        PCE string tying transformation to invariants
    """
    total, boundary, interior = edits

    if total == -1:
        # Shape-changing operation
        return f"[{task_name}] Using rule {rule_name}: transformed shape."
    else:
        return f"[{task_name}] Using rule {rule_name}: changed {total} cells (boundary {boundary}, interior {interior})."

# =============================================================================
# Verification
# =============================================================================

def verify_train_residuals_zero(
    train_pairs: list,
    program: callable
) -> Tuple[bool, list]:
    """
    Verify that program achieves residual=0 on all training pairs.

    Args:
        train_pairs: List of (input_grid, output_grid) tuples
        program: Transformation function Grid -> Grid

    Returns:
        (all_zero, residuals) where:
        - all_zero: True if all residuals are 0
        - residuals: List of residual values for each pair
    """
    residuals = []

    for inp, target in train_pairs:
        pred = program(inp)
        r = residual(pred, target)
        residuals.append(r)

    all_zero = all(r == 0 for r in residuals)

    return all_zero, residuals
