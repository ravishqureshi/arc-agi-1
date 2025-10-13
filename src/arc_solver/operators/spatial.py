#!/usr/bin/env python3
"""ARC Solver - Spatial Operators"""

from typing import Callable, Tuple
from ..core.types import Grid
from ..core.invariants import bbox_nonzero, connected_components, rank_objects

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

def in_bounds(r: int, c: int, H: int, W: int) -> bool:
    """Check if coordinates (r, c) are within grid bounds."""
    return 0 <= r < H and 0 <= c < W

def MOVE_OBJ_RANK(rank: int, delta: Tuple[int, int], group: str = 'global',
                  clear: bool = True, bg: int = 0) -> Callable[[Grid], Grid]:
    """
    Find the rank-th object(s) (global or per-color largest), translate by delta=(dr,dc).
    If clear=True, erase source pixels (move). Else, leave source (copy/paste).

    Args:
        rank: Object rank (0=largest, 1=second largest, etc.)
        delta: Translation offset (dr, dc)
        group: 'global' ranks all objects by size, 'per_color' ranks within each color
        clear: If True, erase source (MOVE). If False, keep source (COPY).
        bg: Background color to use when erasing

    Returns:
        Callable that applies the transformation to a grid
    """
    dr, dc = delta
    def f(z: Grid) -> Grid:
        objs = connected_components(z, bg)
        out = z.copy()
        if not objs:
            return out
        ranks = rank_objects(objs, group=group)
        idx_list = []
        if group == 'global':
            ord_list = ranks["global"]
            if rank < len(ord_list):
                idx_list = [ord_list[rank]]
        else:  # per_color: apply to each color's rank-th object
            for _, ord_list in ranks.items():
                if rank < len(ord_list):
                    idx_list.append(ord_list[rank])
        for idx in idx_list:
            obj = objs[idx]
            # compute new positions; discard any pixel falling out of bounds
            new_pix = []
            for (r, c) in obj.pixels:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc, *z.shape):
                    new_pix.append((nr, nc))
            # clear source if move
            if clear:
                for (r, c) in obj.pixels:
                    out[r, c] = bg
            # paste
            for (nr, nc) in new_pix:
                out[nr, nc] = obj.color
        return out
    return f

def COPY_OBJ_RANK(rank: int, delta: Tuple[int, int], group: str = 'global',
                  bg: int = 0) -> Callable[[Grid], Grid]:
    """
    Copy the rank-th object(s) by delta, keeping the source.
    Equivalent to MOVE_OBJ_RANK with clear=False.

    Args:
        rank: Object rank (0=largest, 1=second largest, etc.)
        delta: Translation offset (dr, dc)
        group: 'global' ranks all objects by size, 'per_color' ranks within each color
        bg: Background color (for consistency with MOVE)

    Returns:
        Callable that applies the transformation to a grid
    """
    return MOVE_OBJ_RANK(rank, delta, group=group, clear=False, bg=bg)
