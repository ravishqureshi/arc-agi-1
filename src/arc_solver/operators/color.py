#!/usr/bin/env python3
"""ARC Solver - Color Operators"""

from typing import Callable, Dict
import numpy as np
from ..core.types import Grid

def COLOR_PERM(mapping: Dict[int, int]) -> Callable[[Grid], Grid]:
    """
    Apply a color permutation (global color mapping).
    
    Args:
        mapping: Dict mapping source colors to target colors
                 e.g., {1: 3, 2: 4, 3: 1} means "all 1s become 3s, all 2s become 4s, etc."
    
    Returns:
        Function that applies the color permutation to a grid
    
    Example:
        >>> g = G([[1, 2], [2, 1]])
        >>> perm = COLOR_PERM({1: 3, 2: 4})
        >>> perm(g)
        array([[3, 4],
               [4, 3]])
    """
    def apply_perm(g: Grid) -> Grid:
        result = g.copy()
        for src_color, tgt_color in mapping.items():
            result[g == src_color] = tgt_color
        return result
    return apply_perm

def infer_color_mapping(input_grid: Grid, output_grid: Grid) -> Dict[int, int]:
    """
    Infer color mapping from a single input/output pair.
    
    Returns mapping dict if consistent, otherwise returns empty dict.
    """
    if input_grid.shape != output_grid.shape:
        return {}
    
    mapping = {}
    for src_color in np.unique(input_grid):
        # Find all positions with src_color in input
        positions = np.argwhere(input_grid == src_color)
        
        # Check what colors they map to in output
        tgt_colors = set(output_grid[tuple(p)] for p in positions)
        
        # Consistent mapping requires all instances of src_color → same tgt_color
        if len(tgt_colors) != 1:
            return {}  # Inconsistent mapping
        
        mapping[int(src_color)] = int(tgt_colors.pop())
    
    return mapping

def merge_mappings(m1: Dict[int, int], m2: Dict[int, int]) -> Dict[int, int]:
    """
    Merge two color mappings. Return empty dict if incompatible.
    """
    merged = m1.copy()
    for k, v in m2.items():
        if k in merged and merged[k] != v:
            return {}  # Conflict
        merged[k] = v
    return merged

def RECOLOR_CONST(new_color: int) -> Callable[[Grid], Grid]:
    """
    Recolor entire grid to a constant color.

    Note: This is typically used with ON(mask, RECOLOR_CONST(color))
    to recolor only a masked region.

    Args:
        new_color: Target color value

    Returns:
        Function that recolors the grid

    Example:
        >>> from ..operators.composition import ON
        >>> from ..operators.masks import MASK_OBJ_RANK
        >>> # Recolor largest object to color 9
        >>> prog = ON(MASK_OBJ_RANK(rank=0), RECOLOR_CONST(9))
        >>> result = prog(grid)
    """
    def f(z: Grid) -> Grid:
        out = z.copy()
        out[:, :] = new_color
        return out
    return f
