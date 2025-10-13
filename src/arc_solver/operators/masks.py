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
