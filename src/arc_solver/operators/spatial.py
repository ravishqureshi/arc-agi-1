#!/usr/bin/env python3
"""ARC Solver - Spatial Operators"""

from typing import Callable, Tuple
from ..core.types import Grid
from ..core.invariants import bbox_nonzero

def BBOX(bg: int = 0) -> Callable[[Grid], Tuple[int, int, int, int]]:
    """Get bounding box of non-background content."""
    def f(z: Grid):
        return bbox_nonzero(z, bg)
    return f

def CROP(rect_fn: Callable[[Grid], Tuple[int, int, int, int]]) -> Callable[[Grid], Grid]:
    """Crop grid to rectangle defined by rect_fn."""
    def f(z: Grid):
        r0, c0, r1, c1 = rect_fn(z)
        return z[r0:r1+1, c0:c1+1]
    return f

def CROP_BBOX_NONZERO(bg: int = 0) -> Callable[[Grid], Grid]:
    """Convenience: Crop to bounding box of non-background content."""
    return CROP(BBOX(bg))
