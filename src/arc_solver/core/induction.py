#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARC Solver - Induction Routines"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from .types import Grid
from .invariants import exact_equals, connected_components, rank_objects
from ..operators.symmetry import ROT, FLIP
from ..operators.spatial import CROP, BBOX, CROP_BBOX_NONZERO, MOVE_OBJ_RANK, COPY_OBJ_RANK
from ..operators.masks import KEEP, MASK_NONZERO, MASK_OBJ_RANK
from ..operators.color import COLOR_PERM, infer_color_mapping, merge_mappings
import numpy as np
from collections import Counter

@dataclass
class Rule:
    """A transformation rule with parameters."""
    name: str
    params: dict
    prog: Callable[[Grid], Grid]

def induce_symmetry_rule(train: List[Tuple[Grid, Grid]]) -> Optional[Rule]:
    """Try pure symmetry transforms that exactly map all train pairs."""
    candidates = [
        ("ROT", {"k": 0}, ROT(0)),
        ("ROT", {"k": 1}, ROT(1)),
        ("ROT", {"k": 2}, ROT(2)),
        ("ROT", {"k": 3}, ROT(3)),
        ("FLIP", {"axis": "h"}, FLIP('h')),
        ("FLIP", {"axis": "v"}, FLIP('v')),
    ]
    for name, params, prog in candidates:
        ok = True
        for x, y in train:
            if not exact_equals(prog(x), y):
                ok = False
                break
        if ok:
            return Rule(name, params, prog)
    return None

def induce_crop_nonzero_rule(train: List[Tuple[Grid, Grid]], bg: int = 0) -> Optional[Rule]:
    """Try crop-to-bbox of nonzero (bg) content."""
    prog = CROP(BBOX(bg))
    ok = all(exact_equals(prog(x), y) for x, y in train)
    if ok:
        return Rule("CROP_BBOX_NONZERO", {"bg": bg}, prog)
    return None

def induce_keep_nonzero_rule(train: List[Tuple[Grid, Grid]], bg: int = 0) -> Optional[Rule]:
    """Try keep-nonzero (remove background) rule."""
    prog = KEEP(MASK_NONZERO(bg))
    ok = all(exact_equals(prog(x), y) for x, y in train)
    return Rule("KEEP_NONZERO", {"bg": bg}, prog) if ok else None

def induce_color_perm_rule(train: List[Tuple[Grid, Grid]]) -> Optional[Rule]:
    """
    Learn a global color permutation from training pairs.

    Strategy:
    1. Infer color mapping from first train pair
    2. Verify that same mapping works for ALL train pairs (residual=0)
    3. Return rule if consistent, None otherwise

    This is NOT hard coding - we learn from TRAINING pairs, apply to TEST inputs.
    """
    if not train:
        return None

    # Infer mapping from first train pair
    x0, y0 = train[0]
    mapping = infer_color_mapping(x0, y0)

    if not mapping:
        return None

    # Build the transformation
    prog = COLOR_PERM(mapping)

    # Verify: Does this mapping achieve residual=0 on ALL train pairs?
    for x, y in train:
        if not exact_equals(prog(x), y):
            return None

    # Success - this mapping works for entire training set
    return Rule("COLOR_PERM", {"mapping": mapping}, prog)

def induce_recolor_obj_rank(train: List[Tuple[Grid, Grid]], rank: int = 0, group: str = 'global', bg: int = 0) -> Optional[Rule]:
    """
    Learn object-rank recolor rule from training pairs.

    Hypothesis:
    - global: Output == input except rank-th object recolored to single target color
    - per_color: Output == input except rank-th object of EACH color recolored
                 (each source color → potentially different target color)

    Args:
        train: Training pairs
        rank: Object rank (0 = largest, 1 = second largest, etc.)
        group: 'global' (rank across all) or 'per_color' (rank within each color)
        bg: Background color

    Returns:
        Rule if pattern matches all train pairs with residual=0, else None
    """
    if not train:
        return None

    # Shape must match for this rule to apply
    for x, y in train:
        if x.shape != y.shape:
            return None

    if group == 'global':
        # GLOBAL: Single target color for the ranked object
        target_color = None
        for x, y in train:
            mask_fn = MASK_OBJ_RANK(rank=rank, group=group, bg=bg)
            m = mask_fn(x)

            if not np.any(m):
                return None  # No such object at this rank

            # Check colors inside mask in output (should be uniform)
            vals = y[m]
            if vals.size == 0:
                return None

            # Most common color in masked region
            c = int(Counter(vals.tolist()).most_common(1)[0][0])

            if target_color is None:
                target_color = c
            elif target_color != c:
                return None  # Inconsistent target color across train pairs

        # Build program: recolor only masked area to target_color
        def recolor_prog_global(z: Grid) -> Grid:
            mask_fn = MASK_OBJ_RANK(rank=rank, group=group, bg=bg)
            m = mask_fn(z)
            out = z.copy()
            out[m] = target_color
            return out

        # Verify exact match on ALL train pairs (residual=0)
        for x, y in train:
            if not exact_equals(recolor_prog_global(x), y):
                return None

        return Rule("RECOLOR_OBJ_RANK", {
            "rank": rank,
            "group": group,
            "bg": bg,
            "color": target_color
        }, recolor_prog_global)

    else:  # group == 'per_color'
        # PER_COLOR: Build mapping {src_color → tgt_color} for each color's ranked object
        color_map = {}

        for x, y in train:
            # Get objects and their per-color ranks
            objs = connected_components(x, bg)
            if not objs:
                return None

            ranks = rank_objects(objs, group='per_color', by='size')

            # For each color, check what the rank-th object gets recolored to
            for src_color, order in ranks.items():
                if rank >= len(order):
                    continue  # This color doesn't have object at this rank

                obj_idx = order[rank]
                obj = objs[obj_idx]

                # Get target color from output
                # All pixels of this object should map to same target
                tgt_colors = set()
                for r, c in obj.pixels:
                    tgt_colors.add(int(y[r, c]))

                if len(tgt_colors) != 1:
                    return None  # Non-uniform target for this object

                tgt_color = tgt_colors.pop()

                # Check consistency across train pairs
                if src_color in color_map:
                    if color_map[src_color] != tgt_color:
                        return None  # Inconsistent mapping across train pairs
                else:
                    color_map[src_color] = tgt_color

        if not color_map:
            return None

        # Build program: recolor each color's ranked object per the mapping
        def recolor_prog_per_color(z: Grid) -> Grid:
            out = z.copy()
            objs = connected_components(z, bg)
            if not objs:
                return out

            ranks = rank_objects(objs, group='per_color', by='size')

            for src_color, order in ranks.items():
                if rank >= len(order):
                    continue
                if src_color not in color_map:
                    continue  # No mapping learned for this color

                obj_idx = order[rank]
                obj = objs[obj_idx]
                tgt_color = color_map[src_color]

                for r, c in obj.pixels:
                    out[r, c] = tgt_color

            return out

        # Verify exact match on ALL train pairs (residual=0)
        for x, y in train:
            if not exact_equals(recolor_prog_per_color(x), y):
                return None

        return Rule("RECOLOR_OBJ_RANK", {
            "rank": rank,
            "group": group,
            "bg": bg,
            "color_map": color_map
        }, recolor_prog_per_color)

def induce_keep_obj_topk(train: List[Tuple[Grid, Grid]], k: int = 1, bg: int = 0) -> Optional[Rule]:
    """
    Learn keep-top-k-objects rule from training pairs.

    Hypothesis: Output == only top-k (global rank by size) objects kept, others zeroed.

    Args:
        train: Training pairs
        k: Number of top objects to keep
        bg: Background color

    Returns:
        Rule if pattern matches all train pairs with residual=0, else None
    """
    if not train:
        return None

    # Check if rule applies (shape must remain same)
    for x, y in train:
        if x.shape != y.shape:
            return None

    def keep_prog(z: Grid) -> Grid:
        objs = connected_components(z, bg)
        if not objs:
            return np.zeros_like(z)

        ranks = rank_objects(objs, group='global', by='size')
        order = ranks["global"]

        m = np.zeros_like(z, dtype=bool)
        for rnk, idx in enumerate(order):
            if rnk < k:
                for (r, c) in objs[idx].pixels:
                    m[r, c] = True

        out = np.zeros_like(z)
        out[m] = z[m]
        return out

    # Verify exact match on ALL train pairs
    for x, y in train:
        if not exact_equals(keep_prog(x), y):
            return None

    return Rule("KEEP_OBJ_TOPK", {"k": k, "bg": bg}, keep_prog)

def shape_mask(objs: List, idx: int, H: int, W: int) -> np.ndarray:
    """Create boolean mask for object at given index."""
    m = np.zeros((H, W), dtype=bool)
    for (r, c) in objs[idx].pixels:
        m[r, c] = True
    return m

def shift_mask(m: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """Shift a boolean mask by (dr, dc), clipping at borders."""
    H, W = m.shape
    out = np.zeros_like(m)
    # shift by copying individual True cells; clip at borders
    rr, cc = np.where(m)
    nr = rr + dr
    nc = cc + dc
    keep = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
    out[nr[keep], nc[keep]] = True
    return out

def induce_move_or_copy_obj_rank(train: List[Tuple[Grid, Grid]],
                                 group: str = 'global', rank: int = 0, bg: int = 0) -> Optional[Rule]:
    """
    Infer a single Δ=(dr,dc) that maps the rank-th object(s) from X to Y
    on **all** train pairs, either as MOVE (clear=True) or COPY (clear=False).

    Strategy (with Δ enumeration):
    1. Extract rank-th object mask in X
    2. Enumerate ALL valid deltas: for each Y object with same color & size,
       check if shape matches when X's mask is shifted by Δ
    3. Find common deltas across ALL train pairs (set intersection)
    4. Try common deltas in priority order (shortest manhattan distance first)
    5. Test MOVE; if fails, test COPY; accept only if residual=0 for all train pairs

    Args:
        train: Training pairs
        group: 'global' ranks all objects by size, 'per_color' ranks within each color
        rank: Object rank (0 = largest, 1 = second largest, etc.)
        bg: Background color

    Returns:
        Rule if pattern matches all train pairs with residual=0, else None
    """
    if not train:
        return None

    # Collect ALL valid deltas per train pair
    all_deltas_per_pair = []

    for x, y in train:
        # Check shape consistency - MOVE/COPY requires same shape
        if x.shape != y.shape:
            return None

        H, W = x.shape
        Xobjs = connected_components(x, bg)
        if not Xobjs:
            return None
        ranks_x = rank_objects(Xobjs, group=group)
        idxs = []
        if group == 'global':
            ord_list = ranks_x["global"]
            if rank >= len(ord_list):
                return None
            idxs = [ord_list[rank]]
        else:  # per_color: we require identical Δ for each color's rank-th object
            # we'll just use the first color occurrence to infer Δ per pair
            for _, ord_list in ranks_x.items():
                if rank < len(ord_list):
                    idxs = [ord_list[rank]]
                    break
            if not idxs:
                return None

        xi = idxs[0]
        xmask = shape_mask(Xobjs, xi, H, W)
        c = Xobjs[xi].color
        sz = Xobjs[xi].size

        # Find ALL candidates in Y: same color & size
        Yobjs = connected_components(y, bg)
        cand = [j for j, o in enumerate(Yobjs) if o.color == c and o.size == sz]
        if not cand:
            return None

        # Enumerate ALL valid deltas for this train pair
        # (instead of taking first match)
        valid_deltas_this_pair = []
        for j in cand:
            dx = int(round(Yobjs[j].centroid[0] - Xobjs[xi].centroid[0]))
            dy = int(round(Yobjs[j].centroid[1] - Xobjs[xi].centroid[1]))
            if (dx, dy) == (0, 0):
                continue  # Skip identity transform
            # Shape check: shifted xmask should match y's mask
            ymask = shape_mask(Yobjs, j, H, W)
            if np.array_equal(shift_mask(xmask, dx, dy), ymask):
                valid_deltas_this_pair.append((dx, dy))

        if not valid_deltas_this_pair:
            return None  # No valid deltas for this train pair

        all_deltas_per_pair.append(set(valid_deltas_this_pair))

    # Find common deltas across ALL train pairs (set intersection)
    common_deltas = set.intersection(*all_deltas_per_pair)

    if not common_deltas:
        return None  # No delta works for all train pairs

    # Try common deltas in priority order: shortest manhattan distance first
    # This prioritizes simpler transformations
    sorted_deltas = sorted(common_deltas, key=lambda d: abs(d[0]) + abs(d[1]))

    for delta in sorted_deltas:
        # Test MOVE then COPY
        move_prog = MOVE_OBJ_RANK(rank, delta, group=group, clear=True, bg=bg)
        if all(exact_equals(move_prog(x), y) for x, y in train):
            return Rule("MOVE_OBJ_RANK", {
                "rank": rank,
                "group": group,
                "delta": delta,
                "clear": True,
                "bg": bg
            }, move_prog)

        copy_prog = COPY_OBJ_RANK(rank, delta, group=group, bg=bg)
        if all(exact_equals(copy_prog(x), y) for x, y in train):
            return Rule("COPY_OBJ_RANK", {
                "rank": rank,
                "group": group,
                "delta": delta,
                "clear": False,
                "bg": bg
            }, copy_prog)

    return None  # No delta worked with MOVE or COPY

# Try rules in order of simplicity (Occam's razor)
CATALOG = [
    induce_symmetry_rule,
    induce_color_perm_rule,
    induce_crop_nonzero_rule,
    induce_keep_nonzero_rule,
    # Object rank rules (try common patterns)
    lambda train: induce_recolor_obj_rank(train, rank=0, group='global', bg=0),
    lambda train: induce_recolor_obj_rank(train, rank=0, group='per_color', bg=0),
    lambda train: induce_keep_obj_topk(train, k=1, bg=0),
    lambda train: induce_keep_obj_topk(train, k=2, bg=0),
    # Object move/copy rules
    lambda train: induce_move_or_copy_obj_rank(train, rank=0, group='global', bg=0),
    lambda train: induce_move_or_copy_obj_rank(train, rank=0, group='per_color', bg=0),
]

def induce_rule(train: List[Tuple[Grid, Grid]]) -> Optional[Rule]:
    """Try all induction routines in catalog order."""
    for induce_fn in CATALOG:
        rule = induce_fn(train)
        if rule is not None:
            return rule
    return None
