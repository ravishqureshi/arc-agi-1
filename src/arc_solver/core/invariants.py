#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC Solver - Invariant Engine
==============================

Precompute invariants for each grid:
- Size, color histogram
- Connected components (with area/centroid)
- Bounding boxes
- Symmetry groups (rot/flip)
- Line/segment detectors (future)
- Periodicity (future)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
from collections import deque

from .types import Grid, ObjList, assert_grid

# =============================================================================
# Invariants Dataclass
# =============================================================================

@dataclass
class Invariants:
    """Precomputed invariant properties of a grid."""
    shape: Tuple[int,int]
    histogram: Dict[int,int]
    bbox: Tuple[int,int,int,int]  # (r0,c0,r1,c1)
    n_components: int
    # Symmetry flags (exact)
    sym_rot90: bool
    sym_rot180: bool
    sym_rot270: bool
    sym_flip_h: bool
    sym_flip_v: bool

# =============================================================================
# Histogram & Statistics
# =============================================================================

def color_histogram(g: Grid) -> Dict[int,int]:
    """Count occurrences of each color in grid."""
    vals, cnt = np.unique(g, return_counts=True)
    return {int(v): int(c) for v,c in zip(vals,cnt)}

# =============================================================================
# Bounding Box
# =============================================================================

def bbox_nonzero(g: Grid, bg: int=0) -> Tuple[int,int,int,int]:
    """
    Return bounding box (r0, c0, r1, c1) of all non-background pixels.
    If all background, returns full grid bounds.
    """
    idx = np.argwhere(g != bg)
    if idx.size == 0:
        return (0, 0, g.shape[0]-1, g.shape[1]-1)
    r0, c0 = idx.min(axis=0)
    r1, c1 = idx.max(axis=0)
    return (int(r0), int(c0), int(r1), int(c1))

# =============================================================================
# Connected Components
# =============================================================================

def connected_components(g: Grid, bg: int=0) -> ObjList:
    """
    Return list of connected components (4-connectivity).
    Each component: (color, [(r,c), ...])
    """
    H, W = g.shape
    vis = np.zeros_like(g, dtype=bool)
    comps = []

    for r in range(H):
        for c in range(W):
            if g[r,c] != bg and not vis[r,c]:
                col = g[r,c]
                q = deque([(r,c)])
                vis[r,c] = True
                pix = [(r,c)]

                while q:
                    rr, cc = q.popleft()
                    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                        nr, nc = rr+dr, cc+dc
                        if 0 <= nr < H and 0 <= nc < W and not vis[nr,nc] and g[nr,nc] == col:
                            vis[nr,nc] = True
                            q.append((nr,nc))
                            pix.append((nr,nc))

                comps.append((col, pix))

    return comps

# =============================================================================
# Symmetry Detection
# =============================================================================

def rot90(g: Grid) -> Grid:
    """Rotate grid 90 degrees counterclockwise."""
    return np.rot90(g, k=1)

def rot180(g: Grid) -> Grid:
    """Rotate grid 180 degrees."""
    return np.rot90(g, k=2)

def rot270(g: Grid) -> Grid:
    """Rotate grid 270 degrees counterclockwise (= 90 clockwise)."""
    return np.rot90(g, k=3)

def flip_h(g: Grid) -> Grid:
    """Flip grid horizontally (left-right)."""
    return np.fliplr(g)

def flip_v(g: Grid) -> Grid:
    """Flip grid vertically (up-down)."""
    return np.flipud(g)

def exact_equals(a: Grid, b: Grid) -> bool:
    """Check if two grids are exactly equal."""
    return a.shape == b.shape and np.array_equal(a, b)

# =============================================================================
# Main Invariants Computation
# =============================================================================

def invariants(g: Grid, bg: int=0) -> Invariants:
    """
    Compute all invariants for grid g.

    Args:
        g: Input grid
        bg: Background color (default 0)

    Returns:
        Invariants object with all precomputed properties
    """
    assert_grid(g)

    r0, c0, r1, c1 = bbox_nonzero(g, bg)
    comps = connected_components(g, bg)

    return Invariants(
        shape=g.shape,
        histogram=color_histogram(g),
        bbox=(r0, c0, r1, c1),
        n_components=len(comps),
        sym_rot90=exact_equals(rot90(g), g),
        sym_rot180=exact_equals(rot180(g), g),
        sym_rot270=exact_equals(rot270(g), g),
        sym_flip_h=exact_equals(flip_h(g), g),
        sym_flip_v=exact_equals(flip_v(g), g),
    )
