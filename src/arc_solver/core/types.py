#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC Solver - Type Definitions
==============================

Core types used throughout the ARC solver:
- Grid: 2D integer array (color values)
- Mask: 2D boolean array (selection mask)
- ObjList: List of connected components
"""

import numpy as np
from typing import List, Tuple

# =============================================================================
# Core Types
# =============================================================================

Grid = np.ndarray          # dtype=int, shape (H, W)
Mask = np.ndarray          # dtype=bool, same shape as Grid
ObjList = List[Tuple[int, List[Tuple[int,int]]]]  # [(color, [(r,c), ...]), ...]

# =============================================================================
# Type Utilities
# =============================================================================

def G(lst) -> Grid:
    """Helper to build a grid from nested lists."""
    return np.array(lst, dtype=int)

def copy_grid(g: Grid) -> Grid:
    """Create a copy of a grid."""
    return np.array(g, dtype=int)

def assert_grid(g: Grid):
    """Validate that g is a proper Grid."""
    assert isinstance(g, np.ndarray) and g.dtype == int and g.ndim == 2, \
        "Grid must be 2D int ndarray."

def assert_mask(m: Mask, shape: Tuple[int,int] = None):
    """Validate that m is a proper Mask."""
    assert isinstance(m, np.ndarray) and m.dtype == bool and m.ndim == 2, \
        "Mask must be 2D bool ndarray."
    if shape is not None:
        assert m.shape == shape, f"Mask shape {m.shape} doesn't match expected {shape}."
