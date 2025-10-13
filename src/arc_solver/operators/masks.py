#!/usr/bin/env python3
"""ARC Solver - Mask Operators"""

import numpy as np
from typing import Callable
from ..core.types import Grid, Mask
from ..core.invariants import connected_components, rank_objects

def MASK_COLOR(c: int) -> Callable[[Grid], Mask]:
    """Create mask selecting all cells of color c."""
    def f(z: Grid):
        return (z == c)
    return f

def MASK_NONZERO(bg: int = 0) -> Callable[[Grid], Mask]:
    """Create mask selecting all non-background cells."""
    def f(z: Grid):
        return (z != bg)
    return f

def KEEP(mask_fn: Callable[[Grid], Mask]) -> Callable[[Grid], Grid]:
    """Keep only cells selected by mask (others → 0)."""
    def f(z: Grid):
        m = mask_fn(z)
        out = np.zeros_like(z)
        out[m] = z[m]
        return out
    return f

def REMOVE(mask_fn: Callable[[Grid], Mask]) -> Callable[[Grid], Grid]:
    """Remove cells selected by mask (set to 0)."""
    def f(z: Grid):
        m = mask_fn(z)
        out = z.copy()
        out[m] = 0
        return out
    return f

def MASK_OBJ_RANK(rank: int = 0, group: str = 'global', bg: int = 0) -> Callable[[Grid], Mask]:
    """
    Create mask selecting object at specified rank.

    Args:
        rank: Rank index (0 = largest/first, 1 = second largest, etc.)
        group: 'global' (rank across all objects) or 'per_color' (rank within each color)
        bg: Background color to ignore

    Returns:
        Function that creates mask for the ranked object(s)

    Example:
        >>> # Mask the largest component
        >>> mask_fn = MASK_OBJ_RANK(rank=0, group='global')
        >>> m = mask_fn(grid)

        >>> # Mask the largest component of each color
        >>> mask_fn = MASK_OBJ_RANK(rank=0, group='per_color')
        >>> m = mask_fn(grid)
    """
    def f(z: Grid) -> Mask:
        objs = connected_components(z, bg)
        m = np.zeros_like(z, dtype=bool)

        if not objs:
            return m

        ranks = rank_objects(objs, group=group, by='size')

        if group == 'global':
            order = ranks["global"]
            if rank < len(order):
                # Mask pixels of the object at this rank
                obj = objs[order[rank]]
                for (r, c) in obj.pixels:
                    m[r, c] = True

        else:  # per_color
            # For each color, mask the object at specified rank
            for color, order in ranks.items():
                if rank < len(order):
                    obj = objs[order[rank]]
                    for (r, c) in obj.pixels:
                        m[r, c] = True

        return m

    return f

def PARITY_MASK(parity: str = 'even', anchor: tuple = (0, 0)) -> Callable[[Grid], Mask]:
    """
    Create checkerboard-style parity mask.

    Args:
        parity: 'even' or 'odd' - selects alternating cells
        anchor: (r, c) anchor position for parity calculation

    Returns:
        Function that creates parity mask

    Example:
        >>> # Select even parity cells (checkerboard pattern)
        >>> mask_fn = PARITY_MASK(parity='even', anchor=(0,0))
        >>> m = mask_fn(grid)
        >>> # m[r,c] = True if (r+c - anchor_r - anchor_c) is even
    """
    assert parity in ('even', 'odd'), f"parity must be 'even' or 'odd', got {parity}"
    ar, ac = anchor

    def f(z: Grid) -> Mask:
        H, W = z.shape
        rr, cc = np.indices((H, W))
        # Parity mask: (r+c - ar - ac) is even
        parity_mask = ((rr + cc - ar - ac) & 1) == 0
        if parity == 'odd':
            parity_mask = ~parity_mask
        return parity_mask

    return f

def RECOLOR_PARITY(color: int, parity: str = 'even', anchor: tuple = (0, 0)) -> Callable[[Grid], Grid]:
    """
    Recolor cells matching parity pattern to a constant color.

    Args:
        color: Target color for parity cells
        parity: 'even' or 'odd' - selects alternating cells
        anchor: (r, c) anchor position for parity calculation

    Returns:
        Function that recolors parity cells

    Example:
        >>> # Recolor even parity cells to color 9
        >>> prog = RECOLOR_PARITY(color=9, parity='even', anchor=(0,0))
        >>> output = prog(input_grid)
    """
    mask_fn = PARITY_MASK(parity, anchor)

    def f(z: Grid) -> Grid:
        out = z.copy()
        m = mask_fn(z)
        out[m] = color
        return out

    return f
