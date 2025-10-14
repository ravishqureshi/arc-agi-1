"""
Utility functions for ARC Solver.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Dict
from .types import Grid, Obj


def G(lst) -> Grid:
    """Convert list to Grid (numpy array)."""
    return np.array(lst, dtype=int)


def equal(a: Grid, b: Grid) -> bool:
    """Check if two grids are exactly equal."""
    return a.shape == b.shape and np.array_equal(a, b)


def residual(a: Grid, b: Grid) -> int:
    """
    Compute residual (number of differing cells).
    If shapes don't match, return max size.
    """
    if a.shape != b.shape:
        return int(max(a.size, b.size))
    return int((a != b).sum())


def inb(r: int, c: int, H: int, W: int) -> bool:
    """Check if (r, c) is within bounds."""
    return 0 <= r < H and 0 <= c < W


def bbox_nonzero(g: Grid, bg: int = 0) -> Tuple[int, int, int, int]:
    """
    Get bounding box of non-background cells.
    Returns: (r0, c0, r1, c1)
    """
    idx = np.argwhere(g != bg)
    if idx.size == 0:
        return (0, 0, g.shape[0] - 1, g.shape[1] - 1)
    r0, c0 = idx.min(axis=0)
    r1, c1 = idx.max(axis=0)
    return (int(r0), int(c0), int(r1), int(c1))


def components(g: Grid, bg: int = 0) -> List[Obj]:
    """
    Extract connected components (4-connected).
    Returns list of Obj with color, pixels, size, bbox, centroid.
    """
    H, W = g.shape
    vis = np.zeros_like(g, dtype=bool)
    out = []

    for r in range(H):
        for c in range(W):
            if g[r, c] != bg and not vis[r, c]:
                col = g[r, c]
                q = deque([(r, c)])
                vis[r, c] = True
                pix = [(r, c)]

                while q:
                    rr, cc = q.popleft()
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nr, nc = rr + dr, cc + dc
                        if inb(nr, nc, H, W) and (not vis[nr, nc]) and g[nr, nc] == col:
                            vis[nr, nc] = True
                            q.append((nr, nc))
                            pix.append((nr, nc))

                rs, cs = zip(*pix)
                out.append(Obj(
                    int(col), pix, len(pix),
                    (min(rs), min(cs), max(rs), max(cs)),
                    (float(np.mean(rs)), float(np.mean(cs)))
                ))

    return out


def rank_by_size(objs: List[Obj], group: str = 'global') -> Dict:
    """
    Rank objects by size.

    Args:
        objs: List of objects
        group: 'global' or 'per_color'

    Returns:
        - 'global': {"global": [sorted indices]}
        - 'per_color': {color: [sorted indices for that color]}
    """
    if group == 'global':
        ord_ = np.argsort([-o.size for o in objs])
        return {"global": [int(i) for i in ord_]}

    elif group == 'per_color':
        per = {}
        for c in sorted(set(o.color for o in objs)):
            idx = [i for i, o in enumerate(objs) if o.color == c]
            ord_ = sorted(idx, key=lambda i: -objs[i].size)
            per[int(c)] = [int(i) for i in ord_]
        return per

    else:
        raise ValueError(f"Invalid group: {group}")
