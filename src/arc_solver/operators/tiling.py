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

def TILE_CONST_PATTERN(motif: Grid) -> Callable[[Grid], Grid]:
    """
    Tile a constant pattern to match input dimensions.

    This handles tasks where the output is always a tiled constant pattern,
    independent of the input content (input only determines output dimensions).

    Args:
        motif: The constant pattern to tile

    Returns:
        Function that tiles the motif to match input grid dimensions

    Example:
        >>> # Output is always tiled [[5,6],[7,8]], sized to match input
        >>> prog = TILE_CONST_PATTERN(G([[5,6],[7,8]]))
        >>> prog(G([[0]]))        # Returns [[5,6],[7,8]]
        >>> prog(G([[0,0,0,0]]))  # Returns [[5,6,5,6],[7,8,7,8]]
    """
    motif_h, motif_w = motif.shape

    def f(z: Grid) -> Grid:
        H, W = z.shape
        # Calculate how many repetitions needed to cover target dimensions
        nx = (H + motif_h - 1) // motif_h
        ny = (W + motif_w - 1) // motif_w
        # Tile and crop to exact size
        tiled = np.tile(motif, (nx, ny))
        return tiled[:H, :W]

    return f

def extract_motif(grid: Grid) -> Grid:
    """
    Extract the minimal repeating motif from a tiled grid.

    Tries all possible motif sizes starting from smallest, and returns
    the first motif that perfectly tiles to reproduce the grid.

    Args:
        grid: Grid to extract motif from

    Returns:
        The minimal repeating motif, or the full grid if no repetition found

    Example:
        >>> extract_motif(G([[5,6,5,6],[7,8,7,8]]))  # Returns [[5,6],[7,8]]
        >>> extract_motif(G([[1,2,3]]))              # Returns [[1,2,3]] (no repetition)
    """
    H, W = grid.shape

    # Try all divisor combinations of the dimensions
    for mh in range(1, H + 1):
        if H % mh != 0:
            continue
        for mw in range(1, W + 1):
            if W % mw != 0:
                continue

            # Extract candidate motif
            motif = grid[:mh, :mw]

            # Check if tiling this motif reproduces the grid
            nx, ny = H // mh, W // mw
            tiled = np.tile(motif, (nx, ny))

            if np.array_equal(tiled, grid):
                return motif

    # No repetition found, return full grid
    return grid
