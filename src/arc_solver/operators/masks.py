#!/usr/bin/env python3
"""ARC Solver - Mask Operators"""

import numpy as np
from typing import Callable
from ..core.types import Grid, Mask

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
