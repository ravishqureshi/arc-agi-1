#!/usr/bin/env python3
"""ARC Solver - Composition Operators"""

from typing import Callable
from ..core.types import Grid, Mask

def ON(mask_fn: Callable[[Grid], Mask], prog: Callable[[Grid], Grid]) -> Callable[[Grid], Grid]:
    """Apply prog only on masked region; outside unchanged."""
    def f(z: Grid):
        m = mask_fn(z)
        sub = prog(z.copy())
        out = z.copy()
        out[m] = sub[m]
        return out
    return f

def SEQ(p1: Callable[[Grid], Grid], p2: Callable[[Grid], Grid]) -> Callable[[Grid], Grid]:
    """Sequential composition: apply p1, then p2."""
    return lambda z: p2(p1(z))
