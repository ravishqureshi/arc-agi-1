"""
Operator functions for ARC Solver.

Each operator is a function that returns a Grid → Grid callable.
"""

import numpy as np
from typing import Callable, Dict, Tuple, Optional, List
from collections import deque
from .types import Grid
from .utils import inb, components


# ==============================================================================
# Symmetry operators
# ==============================================================================

def rot90(g): return np.rot90(g, 1)
def rot180(g): return np.rot90(g, 2)
def rot270(g): return np.rot90(g, 3)
def flip_h(g): return np.fliplr(g)
def flip_v(g): return np.flipud(g)


def ROT(k: int) -> Callable:
    """Rotation operator (k * 90 degrees)."""
    assert k in (0, 1, 2, 3)
    return (lambda z: z) if k == 0 else (rot90 if k == 1 else (rot180 if k == 2 else rot270))


def FLIP(axis: str) -> Callable:
    """Flip operator (horizontal or vertical)."""
    assert axis in ('h', 'v')
    return flip_h if axis == 'h' else flip_v


# ==============================================================================
# Cropping operators
# ==============================================================================

def CROP_BBOX_NONZERO(bg: int = 0) -> Callable:
    """Crop to bounding box of non-background cells."""
    from .utils import bbox_nonzero

    def f(z: Grid):
        r0, c0, r1, c1 = bbox_nonzero(z, bg=bg)
        return z[r0:r1 + 1, c0:c1 + 1]
    return f


def KEEP_NONZERO(bg: int = 0) -> Callable:
    """Keep only non-background cells."""
    def f(z: Grid):
        m = (z != bg)
        out = np.zeros_like(z)
        out[m] = z[m]
        return out
    return f


def KEEP_LARGEST_COMPONENT(bg: int = 0) -> Callable:
    """
    Keep only the largest connected component (by pixel count).
    All other cells become background.
    """
    def f(z: Grid):
        objs = components(z, bg=bg)
        if not objs:
            return np.full_like(z, bg)

        # Find largest by size
        largest = max(objs, key=lambda o: o.size)

        # Create output with only largest component
        out = np.full_like(z, bg)
        for (r, c) in largest.pixels:
            out[r, c] = largest.color

        return out
    return f


# ==============================================================================
# Color operators
# ==============================================================================

def COLOR_PERM(mapping: Dict[int, int]) -> Callable:
    """Color permutation (remap colors)."""
    def f(z: Grid):
        out = z.copy()
        for c, yc in mapping.items():
            out[z == c] = yc
        return out
    return f


# ==============================================================================
# Parity operators
# ==============================================================================

def PARITY_MASK(parity: str = 'even', anchor: Tuple[int, int] = (0, 0)) -> Callable:
    """Generate parity mask."""
    assert parity in ('even', 'odd')
    ar, ac = anchor

    def f(z: Grid):
        H, W = z.shape
        rr, cc = np.indices((H, W))
        m = ((rr + cc - ar - ac) & 1) == 0
        return m if parity == 'even' else ~m
    return f


def RECOLOR_PARITY_CONST(color: int, parity: str = 'even', anchor: Tuple[int, int] = (0, 0)) -> Callable:
    """Recolor cells matching parity pattern."""
    mfn = PARITY_MASK(parity, anchor)

    def f(z: Grid):
        out = z.copy()
        m = mfn(z)
        out[m] = color
        return out
    return f


# ==============================================================================
# Tiling operators
# ==============================================================================

def extract_motif(y: Grid) -> Optional[Tuple[int, int, Grid]]:
    """
    Extract repeating tile motif if grid is perfectly tiled.
    Returns (h, w, motif) or None.
    """
    H, W = y.shape
    for h in range(1, H + 1):
        if H % h != 0:
            continue
        for w in range(1, W + 1):
            if W % w != 0:
                continue
            motif = y[:h, :w]
            if np.array_equal(np.tile(motif, (H // h, W // w)), y):
                return h, w, motif
    return None


def REPEAT_TILE(motif: Grid) -> Callable:
    """Repeat tile to fill grid."""
    mh, mw = motif.shape

    def f(z: Grid):
        H, W = z.shape
        tiled = np.tile(motif, ((H + mh - 1) // mh, (W + mw - 1) // mw))
        return tiled[:H, :W]
    return f


# ==============================================================================
# Morphology operators
# ==============================================================================

def fill_holes_in_bbox(sub: Grid, color: int, bg: int = 0) -> Grid:
    """Fill holes within a bounding box."""
    H, W = sub.shape
    bgmask = (sub == bg)
    reach = np.zeros_like(bgmask, dtype=bool)
    q = deque()

    # Flood from borders
    for c in range(W):
        if bgmask[0, c]:
            reach[0, c] = True
            q.append((0, c))
        if bgmask[H - 1, c]:
            reach[H - 1, c] = True
            q.append((H - 1, c))

    for r in range(H):
        if bgmask[r, 0]:
            reach[r, 0] = True
            q.append((r, 0))
        if bgmask[r, W - 1]:
            reach[r, W - 1] = True
            q.append((r, W - 1))

    while q:
        r, c = q.popleft()
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (not reach[nr, nc]) and bgmask[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))

    holes = np.logical_and(bgmask, ~reach)
    out = sub.copy()
    out[holes] = color
    return out


def HOLE_FILL_ALL(bg: int = 0) -> Callable:
    """Fill holes in all objects."""
    def f(z: Grid):
        out = z.copy()
        for o in components(z, bg=bg):
            r0, c0, r1, c1 = o.bbox
            sub = out[r0:r1 + 1, c0:c1 + 1]
            out[r0:r1 + 1, c0:c1 + 1] = fill_holes_in_bbox(sub, o.color, bg)
        return out
    return f


# ==============================================================================
# Object manipulation operators
# ==============================================================================

def COPY_OBJ_RANK_BY_DELTAS(rank: int, deltas: List[Tuple[int, int]], group: str = 'global', bg: int = 0) -> Callable:
    """Copy object(s) by delta offsets."""
    from .utils import rank_by_size

    def f(z: Grid):
        H, W = z.shape
        objs = components(z, bg=bg)
        out = z.copy()
        if not objs:
            return out

        ranks = rank_by_size(objs, group)
        bases = [objs[ranks["global"][rank]]] if group == 'global' else \
                [objs[ord_list[rank]] for _, ord_list in rank_by_size(objs, 'per_color').items()]

        for base in bases:
            for dr, dc in deltas:
                for (r, c) in base.pixels:
                    nr, nc = r + dr, c + dc
                    if inb(nr, nc, H, W):
                        out[nr, nc] = base.color
        return out
    return f
