#!/usr/bin/env python3
"""ARC Solver - Tiling Operators"""

import numpy as np
from typing import Callable
from ..core.types import Grid

def TILE(nx: int, ny: int) -> Callable[[Grid], Grid]:
    """
    Tile the input grid nx times vertically, ny times horizontally.

    Args:
        nx: Number of vertical repetitions
        ny: Number of horizontal repetitions

    Returns:
        Function that tiles input grid

    Example:
        >>> TILE(2, 3)(G([[1,2],[3,4]]))
        # Returns:
        # [[1,2,1,2,1,2],
        #  [3,4,3,4,3,4],
        #  [1,2,1,2,1,2],
        #  [3,4,3,4,3,4]]
    """
    assert nx >= 1 and ny >= 1, "Tile counts must be >= 1"

    def f(z: Grid) -> Grid:
        return np.tile(z, (nx, ny))

    return f

def TILE_SUBGRID(r0: int, c0: int, r1: int, c1: int, nx: int, ny: int) -> Callable[[Grid], Grid]:
    """
    Extract subgrid and tile it.

    Args:
        r0, c0, r1, c1: Bounding box of subgrid to extract
        nx, ny: Tile repetitions

    Returns:
        Function that extracts subgrid and tiles it
    """
    assert nx >= 1 and ny >= 1
    assert r0 <= r1 and c0 <= c1

    def f(z: Grid) -> Grid:
        subgrid = z[r0:r1+1, c0:c1+1]
        return np.tile(subgrid, (nx, ny))

    return f
