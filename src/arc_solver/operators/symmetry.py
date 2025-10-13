#!/usr/bin/env python3
"""ARC Solver - Symmetry Operators"""

from typing import Callable
from ..core.types import Grid
from ..core.invariants import rot90, rot180, rot270, flip_h, flip_v

def ROT(k: int) -> Callable[[Grid], Grid]:
    """Rotate grid by k*90 degrees counterclockwise."""
    assert k in (0, 1, 2, 3), f"k must be in {{0,1,2,3}}, got {k}"
    if k == 0:
        return lambda z: z
    elif k == 1:
        return rot90
    elif k == 2:
        return rot180
    else:
        return rot270

def FLIP(axis: str) -> Callable[[Grid], Grid]:
    """Flip grid along specified axis."""
    assert axis in ('h', 'v'), f"axis must be 'h' or 'v', got {axis}"
    return flip_h if axis == 'h' else flip_v
