"""
Inducer functions for ARC Solver.

Each inducer learns operator parameters from training pairs and verifies exact fit.
Returns list of operators with train residual=0.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
from .types import Operator, Grid
from .utils import equal, components
from .operators import (
    COLOR_PERM, ROT, FLIP,
    CROP_BBOX_NONZERO, KEEP_NONZERO,
    RECOLOR_PARITY_CONST, PARITY_MASK,
    REPEAT_TILE, extract_motif,
    HOLE_FILL_ALL,
    COPY_OBJ_RANK_BY_DELTAS
)


# ==============================================================================
# Color permutation inducer
# ==============================================================================

def learn_color_perm_mapping(train) -> Optional[Dict[int, int]]:
    """
    Learn color mapping from train pairs.
    Returns None if no consistent mapping exists.
    """
    m = {}
    for x, y in train:
        if x.shape != y.shape:
            return None
        vals = np.unique(x)
        for c in vals:
            mask = (x == c)
            if not np.any(mask):
                continue
            ys, counts = np.unique(y[mask], return_counts=True)
            yc = int(ys[np.argmax(counts)])
            if c in m and m[c] != yc:
                return None
            m[c] = yc
    return m


def induce_COLOR_PERM(train) -> List[Operator]:
    """Learn color permutation."""
    m = learn_color_perm_mapping(train)
    if m is None:
        return []
    P = COLOR_PERM(m)
    return [Operator("COLOR_PERM", {"map": m}, P, f"Color perm {m}")] \
        if all(equal(P(x), y) for x, y in train) else []


# ==============================================================================
# Symmetry inducers
# ==============================================================================

def induce_ROT_FLIP(train) -> List[Operator]:
    """Try all rotations and flips."""
    out = []
    for k in (0, 1, 2, 3):
        P = ROT(k)
        if all(equal(P(x), y) for x, y in train):
            out.append(Operator("ROT", {"k": k}, P, f"Rotate {k * 90}° (exact)"))
    for a in ('h', 'v'):
        P = FLIP(a)
        if all(equal(P(x), y) for x, y in train):
            out.append(Operator("FLIP", {"axis": a}, P, f"Flip {'horizontal' if a == 'h' else 'vertical'} (exact)"))
    return out


# ==============================================================================
# Crop/Keep inducers
# ==============================================================================

def induce_CROP_KEEP(train, bg=0) -> List[Operator]:
    """Induce crop and keep operators."""
    out = []
    P = CROP_BBOX_NONZERO(bg)
    if all(equal(P(x), y) for x, y in train):
        out.append(Operator("CROP_BBOX_NONZERO", {"bg": bg}, P, "Crop bbox non-zero"))
    P2 = KEEP_NONZERO(bg)
    if all(equal(P2(x), y) for x, y in train):
        out.append(Operator("KEEP_NONZERO", {"bg": bg}, P2, "Keep non-zero"))
    return out


# ==============================================================================
# Parity inducer
# ==============================================================================

def induce_PARITY_CONST(train) -> List[Operator]:
    """Induce parity recoloring."""
    out = []
    if not all(x.shape == y.shape for x, y in train):
        return out

    for parity in ('even', 'odd'):
        x0, y0 = train[0]
        for ar in (0, 1):
            for ac in (0, 1):
                m = PARITY_MASK(parity, (ar, ac))(x0)
                if not np.any(m):
                    continue
                col = int(Counter(y0[m].tolist()).most_common(1)[0][0])
                P = RECOLOR_PARITY_CONST(col, parity, (ar, ac))
                if all(equal(P(x), y) for x, y in train):
                    out.append(Operator("RECOLOR_PARITY_CONST",
                                      {"color": col, "parity": parity, "anchor": (ar, ac)},
                                      P, f"Recolor {parity} parity→{col}, anchor {(ar, ac)}"))
    return out


# ==============================================================================
# Tiling inducer
# ==============================================================================

def induce_TILING_AND_MASK(train) -> List[Operator]:
    """Induce tiling operators."""
    out = []
    m0 = extract_motif(train[0][1])
    if m0:
        h, w, motif = m0
        P = REPEAT_TILE(motif)
        if all(equal(P(x), y) for x, y in train):
            out.append(Operator("REPEAT_TILE", {"h": h, "w": w}, P, f"Tile motif {h}x{w}"))
    return out


# ==============================================================================
# Hole fill inducer
# ==============================================================================

def induce_HOLE_FILL(train, bg=0) -> List[Operator]:
    """Induce hole filling."""
    P = HOLE_FILL_ALL(bg)
    return [Operator("HOLE_FILL_ALL", {"bg": bg}, P, "Fill enclosed holes per object")] \
        if all(equal(P(x), y) for x, y in train) else []


# ==============================================================================
# Object copy inducer
# ==============================================================================

def enumerate_deltas_for_pair(x: Grid, y: Grid, bg=0) -> List[Tuple[int, int]]:
    """Enumerate centroid deltas between objects in x and y."""
    deltas = set()
    X = components(x, bg)
    Y = components(y, bg)
    if not X or not Y:
        return []
    for xi, xo in enumerate(X):
        cand = [yj for yj, yo in enumerate(Y) if yo.color == xo.color and yo.size == xo.size]
        for yj in cand:
            dr = int(round(Y[yj].centroid[0] - xo.centroid[0]))
            dc = int(round(Y[yj].centroid[1] - xo.centroid[1]))
            deltas.add((dr, dc))
    return sorted(deltas)


def induce_COPY_BY_DELTAS(train, rank: int = -1, group: str = 'global', bg: int = 0) -> List[Operator]:
    """Induce object copying by delta offsets."""
    delta_sets = []
    for x, y in train:
        ds = enumerate_deltas_for_pair(x, y, bg)
        if not ds:
            return []
        delta_sets.append(set(ds))
    common = set.intersection(*delta_sets)
    out = []
    for Δ in sorted(common):
        P = COPY_OBJ_RANK_BY_DELTAS(rank, [Δ], group, bg)
        if all(equal(P(x), y) for x, y in train):
            out.append(Operator("COPY_OBJ_RANK_BY_DELTAS",
                              {"rank": rank, "group": group, "deltas": [Δ], "bg": bg},
                              P, f"Copy rank={rank} {group} object by Δ={Δ}"))
    return out
