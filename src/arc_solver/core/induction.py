#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARC Solver - Induction Routines"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from .types import Grid
from .invariants import exact_equals, connected_components, rank_objects
from ..operators.symmetry import ROT, FLIP
from ..operators.spatial import CROP, BBOX, CROP_BBOX_NONZERO, MOVE_OBJ_RANK, COPY_OBJ_RANK
from ..operators.masks import KEEP, MASK_NONZERO, MASK_OBJ_RANK, PARITY_MASK, RECOLOR_PARITY
from ..operators.color import COLOR_PERM, infer_color_mapping, merge_mappings
from ..operators.tiling import TILE, TILE_SUBGRID
from ..operators.drawing import DRAW_LINE, DRAW_BOX, FLOOD_FILL
import numpy as np
from collections import Counter

@dataclass
class Rule:
    """A transformation rule with parameters."""
    name: str
    params: dict
    prog: Callable[[Grid], Grid]
    pce: str = ""  # Proof-Carrying English explanation

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
        # Try different repetition counts (k=1 for single, k>1 for chains)
        for k in range(1, 6):
            # Test MOVE then COPY with repetition count k
            move_prog = MOVE_OBJ_RANK(rank, delta, group=group, clear=True, bg=bg, k=k)
            if all(exact_equals(move_prog(x), y) for x, y in train):
                pce = f"Move {rank}-th {group} by Δ={delta}"
                if k > 1:
                    pce += f" (chain k={k})"
                return Rule("MOVE_OBJ_RANK", {
                    "rank": rank,
                    "group": group,
                    "delta": delta,
                    "clear": True,
                    "bg": bg,
                    "k": k
                }, move_prog, pce)

            copy_prog = COPY_OBJ_RANK(rank, delta, group=group, bg=bg, k=k)
            if all(exact_equals(copy_prog(x), y) for x, y in train):
                pce = f"Copy {rank}-th {group} by Δ={delta}"
                if k > 1:
                    pce += f" (chain k={k})"
                return Rule("COPY_OBJ_RANK", {
                    "rank": rank,
                    "group": group,
                    "delta": delta,
                    "clear": False,
                    "bg": bg,
                    "k": k
                }, copy_prog, pce)

    return None  # No delta worked with MOVE or COPY

def induce_tile_rule(train: List[Tuple[Grid, Grid]]) -> Optional[Rule]:
    """
    Learn tiling parameters from training pairs.

    Strategy:
    1. Check if output == tile(input, nx, ny) for some nx, ny
    2. Check if output == tile(subgrid, nx, ny) for some subgrid
    3. Find params that work for ALL train pairs (observer=observed)

    Returns:
        Rule if pattern matches, else None
    """
    if not train:
        return None

    # Try simple tiling (tile entire input)
    for nx in range(1, 6):
        for ny in range(1, 6):
            if nx == 1 and ny == 1:
                continue  # Identity

            prog = TILE(nx, ny)
            if all(exact_equals(prog(x), y) for x, y in train):
                return Rule("TILE", {"nx": nx, "ny": ny}, prog,
                           f"Tile input {nx}×{ny} times")

    # Try tiling subgrid (extract pattern from input and tile it)
    # Only check first train pair for pattern extraction
    if train:
        x0, y0 = train[0]
        h_out, w_out = y0.shape

        # Try different subgrid sizes
        for h_pat in range(1, min(x0.shape[0] + 1, 10)):
            for w_pat in range(1, min(x0.shape[1] + 1, 10)):
                if h_out % h_pat == 0 and w_out % w_pat == 0:
                    nx = h_out // h_pat
                    ny = w_out // w_pat

                    if nx == 1 and ny == 1:
                        continue

                    # Try extracting pattern from different positions
                    for r0 in range(max(1, x0.shape[0] - h_pat + 1)):
                        for c0 in range(max(1, x0.shape[1] - w_pat + 1)):
                            r1 = r0 + h_pat - 1
                            c1 = c0 + w_pat - 1

                            prog = TILE_SUBGRID(r0, c0, r1, c1, nx, ny)

                            # Check if works for all train pairs
                            try:
                                if all(exact_equals(prog(x), y) for x, y in train):
                                    return Rule("TILE_SUBGRID",
                                              {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "nx": nx, "ny": ny},
                                              prog,
                                              f"Tile subgrid [{r0}:{r1+1}, {c0}:{c1+1}] {nx}×{ny} times")
                            except:
                                continue

    return None

def induce_draw_line_rule(train: List[Tuple[Grid, Grid]], bg: int = 0) -> Optional[Rule]:
    """
    Learn line drawing parameters from training pairs.

    Strategy:
    1. Find cells that changed from input to output
    2. Check if they form a line
    3. Learn line endpoints and color
    4. Verify across ALL train pairs

    Returns:
        Rule if pattern matches, else None
    """
    if not train:
        return None

    # Must have same shape
    for x, y in train:
        if x.shape != y.shape:
            return None

    # Detect line parameters from first train pair
    x0, y0 = train[0]
    diff = (x0 != y0)

    if not np.any(diff):
        return None

    changed_pixels = np.argwhere(diff)
    if len(changed_pixels) < 2:  # Need at least 2 pixels for a line
        return None

    # Get line color (most common color in changed pixels)
    colors = [int(y0[r, c]) for r, c in changed_pixels]
    line_color = int(Counter(colors).most_common(1)[0][0])

    # Try to find line endpoints
    # Simple heuristic: use extremes of changed pixels
    rows = changed_pixels[:, 0]
    cols = changed_pixels[:, 1]

    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    # Check if this is a horizontal or vertical line
    is_horizontal = len(np.unique(rows)) == 1
    is_vertical = len(np.unique(cols)) == 1

    if not (is_horizontal or is_vertical):
        # Diagonal or complex line - skip for now
        return None

    prog = DRAW_LINE(r0, c0, r1, c1, line_color)

    # Verify on all train pairs
    if all(exact_equals(prog(x), y) for x, y in train):
        return Rule("DRAW_LINE",
                   {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "color": line_color},
                   prog,
                   f"Draw line from ({r0},{c0}) to ({r1},{c1}) with color {line_color}")

    return None

def induce_parity_recolor(train: List[Tuple[Grid, Grid]]) -> Optional[Rule]:
    """
    Learn parity recoloring from training pairs.

    Hypothesis: Output == input except parity cells recolored to constant color.
    Try both 'even' and 'odd' parity with different anchors.

    Strategy:
    1. Infer parity (even/odd) and anchor from first pair
    2. Detect target color (most common in parity cells)
    3. Verify across ALL train pairs (residual=0)

    Returns:
        Rule if pattern matches, else None
    """
    if not train:
        return None

    # Must have same shape
    for x, y in train:
        if x.shape != y.shape:
            return None

    # Try both parity types
    for parity in ('even', 'odd'):
        # Try minimal anchor variations (0 or 1 shifts)
        for ar in range(2):
            for ac in range(2):
                # Infer color from first pair
                x0, y0 = train[0]
                mask_fn = PARITY_MASK(parity, (ar, ac))
                m = mask_fn(x0)

                if not np.any(m):
                    continue

                # Get target color (most common in parity cells of output)
                vals = y0[m]
                if vals.size == 0:
                    continue

                target_color = int(Counter(vals.tolist()).most_common(1)[0][0])

                # Build program and verify on all pairs
                prog = RECOLOR_PARITY(target_color, parity, (ar, ac))

                if all(exact_equals(prog(x), y) for x, y in train):
                    return Rule("RECOLOR_PARITY",
                              {"color": target_color, "parity": parity, "anchor": (ar, ac)},
                              prog,
                              f"Recolor {parity} parity cells to {target_color} (anchor={(ar,ac)})")

    return None

# Try rules in order of simplicity (Occam's razor)
# Principle: Try more specific/constrained operations before general ones
CATALOG = [
    induce_symmetry_rule,
    induce_color_perm_rule,
    induce_crop_nonzero_rule,
    induce_keep_nonzero_rule,
    induce_parity_recolor,  # Parity-based recoloring (checkerboard patterns)
    # Object rank rules (try common patterns) - More specific than tiling
    lambda train: induce_recolor_obj_rank(train, rank=0, group='global', bg=0),
    lambda train: induce_recolor_obj_rank(train, rank=0, group='per_color', bg=0),
    lambda train: induce_keep_obj_topk(train, k=1, bg=0),
    lambda train: induce_keep_obj_topk(train, k=2, bg=0),
    # Object move/copy rules - More specific than tiling
    lambda train: induce_move_or_copy_obj_rank(train, rank=0, group='global', bg=0),
    lambda train: induce_move_or_copy_obj_rank(train, rank=0, group='per_color', bg=0),
    # Tiling and drawing operators - More general, try after object operations
    induce_tile_rule,
    induce_draw_line_rule,
]

def induce_rule(train: List[Tuple[Grid, Grid]]) -> Optional[Rule]:
    """Try all induction routines in catalog order."""
    for induce_fn in CATALOG:
        rule = induce_fn(train)
        if rule is not None:
            return rule
    return None

# =============================================================================
# Beam Search Compatible Induction (returns List[Rule])
# =============================================================================

def induce_symmetry(train: List[Tuple[Grid, Grid]]) -> List[Rule]:
    """Try all symmetry transforms that exactly map train pairs. Returns list of ALL matches."""
    candidates = [
        ("ROT", {"k": 0}, ROT(0), "Identity (0°)"),
        ("ROT", {"k": 1}, ROT(1), "Rotate 90°"),
        ("ROT", {"k": 2}, ROT(2), "Rotate 180°"),
        ("ROT", {"k": 3}, ROT(3), "Rotate 270°"),
        ("FLIP", {"axis": "h"}, FLIP('h'), "Flip horizontal"),
        ("FLIP", {"axis": "v"}, FLIP('v'), "Flip vertical"),
    ]
    out = []
    for name, params, prog, pce_text in candidates:
        if all(exact_equals(prog(x), y) for x, y in train):
            out.append(Rule(name, params, prog, pce_text + " (exact on train)"))
    return out

def induce_crop_bbox(train: List[Tuple[Grid, Grid]], bg: int = 0) -> List[Rule]:
    """Try crop-to-bbox of nonzero content."""
    prog = CROP(BBOX(bg))
    if all(exact_equals(prog(x), y) for x, y in train):
        return [Rule("CROP_BBOX_NONZERO", {"bg": bg}, prog, "Crop to bbox of non-zero")]
    return []

def induce_keep_nonzero(train: List[Tuple[Grid, Grid]], bg: int = 0) -> List[Rule]:
    """Try keep-nonzero (remove background) rule."""
    prog = KEEP(MASK_NONZERO(bg))
    if all(exact_equals(prog(x), y) for x, y in train):
        return [Rule("KEEP_NONZERO", {"bg": bg}, prog, "Keep non-zero")]
    return []

def induce_color_perm(train: List[Tuple[Grid, Grid]]) -> List[Rule]:
    """Learn global color permutation from training pairs."""
    if not train:
        return []

    x0, y0 = train[0]
    mapping = infer_color_mapping(x0, y0)

    if not mapping:
        return []

    prog = COLOR_PERM(mapping)

    if all(exact_equals(prog(x), y) for x, y in train):
        pairs = ', '.join(f"{k}→{v}" for k, v in sorted(mapping.items()))
        return [Rule("COLOR_PERM", {"mapping": mapping}, prog, f"Color permutation {pairs}")]
    return []

def induce_recolor_obj_rank_beam(train: List[Tuple[Grid, Grid]], rank: int = 0, group: str = 'global', bg: int = 0) -> List[Rule]:
    """Learn object-rank recolor rule. Beam-compatible version."""
    if not train:
        return []

    # Shape must match
    for x, y in train:
        if x.shape != y.shape:
            return []

    if group == 'global':
        target_color = None
        for x, y in train:
            mask_fn = MASK_OBJ_RANK(rank=rank, group=group, bg=bg)
            m = mask_fn(x)
            if not np.any(m):
                return []
            vals = y[m]
            if vals.size == 0:
                return []
            c = int(Counter(vals.tolist()).most_common(1)[0][0])
            if target_color is None:
                target_color = c
            elif target_color != c:
                return []

        def recolor_prog(z: Grid) -> Grid:
            m = MASK_OBJ_RANK(rank=rank, group=group, bg=bg)(z)
            out = z.copy()
            out[m] = target_color
            return out

        if all(exact_equals(recolor_prog(x), y) for x, y in train):
            scope = "globally" if group == 'global' else "per color"
            target = "largest" if rank == 0 else f"rank {rank}"
            return [Rule("RECOLOR_OBJ_RANK", {"rank": rank, "group": group, "color": target_color, "bg": bg},
                        recolor_prog, f"Recolor {target} {scope}→{target_color}")]
    else:
        # per_color logic (similar to existing)
        color_map = {}
        for x, y in train:
            objs = connected_components(x, bg)
            if not objs:
                return []
            ranks = rank_objects(objs, group='per_color', by='size')
            for src_color, order in ranks.items():
                if rank >= len(order):
                    continue
                obj_idx = order[rank]
                obj = objs[obj_idx]
                tgt_colors = set()
                for r, c in obj.pixels:
                    tgt_colors.add(int(y[r, c]))
                if len(tgt_colors) != 1:
                    return []
                tgt_color = tgt_colors.pop()
                if src_color in color_map:
                    if color_map[src_color] != tgt_color:
                        return []
                else:
                    color_map[src_color] = tgt_color

        if not color_map:
            return []

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
                    continue
                obj_idx = order[rank]
                obj = objs[obj_idx]
                tgt_color = color_map[src_color]
                for r, c in obj.pixels:
                    out[r, c] = tgt_color
            return out

        if all(exact_equals(recolor_prog_per_color(x), y) for x, y in train):
            target = "largest" if rank == 0 else f"rank {rank}"
            mappings = ', '.join(f"{k}→{v}" for k, v in sorted(color_map.items()))
            return [Rule("RECOLOR_OBJ_RANK", {"rank": rank, "group": group, "bg": bg, "color_map": color_map},
                        recolor_prog_per_color, f"Recolor {target} per color: {mappings}")]
    return []

def induce_keep_topk(train: List[Tuple[Grid, Grid]], k: int = 2, bg: int = 0) -> List[Rule]:
    """Keep top-k objects by size."""
    for x, y in train:
        if x.shape != y.shape:
            return []

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

    if all(exact_equals(keep_prog(x), y) for x, y in train):
        return [Rule("KEEP_OBJ_TOPK", {"k": k, "bg": bg}, keep_prog, f"Keep top-{k} components")]
    return []

def induce_move_copy(train: List[Tuple[Grid, Grid]], rank: int = 0, group: str = 'global', bg: int = 0) -> List[Rule]:
    """Infer MOVE/COPY operations with Δ enumeration. Beam-compatible version."""
    if not train:
        return []

    # Collect ALL valid deltas per train pair
    all_deltas_per_pair = []

    for x, y in train:
        if x.shape != y.shape:
            return []

        H, W = x.shape
        Xobjs = connected_components(x, bg)
        if not Xobjs:
            return []
        ranks_x = rank_objects(Xobjs, group=group)
        idxs = []
        if group == 'global':
            ord_list = ranks_x["global"]
            if rank >= len(ord_list):
                return []
            idxs = [ord_list[rank]]
        else:
            for _, ord_list in ranks_x.items():
                if rank < len(ord_list):
                    idxs = [ord_list[rank]]
                    break
            if not idxs:
                return []

        xi = idxs[0]
        xmask = shape_mask(Xobjs, xi, H, W)
        c = Xobjs[xi].color
        sz = Xobjs[xi].size

        Yobjs = connected_components(y, bg)
        cand = [j for j, o in enumerate(Yobjs) if o.color == c and o.size == sz]
        if not cand:
            return []

        valid_deltas_this_pair = []
        for j in cand:
            dx = int(round(Yobjs[j].centroid[0] - Xobjs[xi].centroid[0]))
            dy = int(round(Yobjs[j].centroid[1] - Xobjs[xi].centroid[1]))
            if (dx, dy) == (0, 0):
                continue
            ymask = shape_mask(Yobjs, j, H, W)
            if np.array_equal(shift_mask(xmask, dx, dy), ymask):
                valid_deltas_this_pair.append((dx, dy))

        if not valid_deltas_this_pair:
            return []

        all_deltas_per_pair.append(set(valid_deltas_this_pair))

    common_deltas = set.intersection(*all_deltas_per_pair)
    if not common_deltas:
        return []

    sorted_deltas = sorted(common_deltas, key=lambda d: abs(d[0]) + abs(d[1]))

    out = []
    for delta in sorted_deltas:
        # Try different repetition counts (k=1 for single, k>1 for chains)
        for k in range(1, 6):
            move_prog = MOVE_OBJ_RANK(rank, delta, group=group, clear=True, bg=bg, k=k)
            if all(exact_equals(move_prog(x), y) for x, y in train):
                target = "largest" if rank == 0 else f"rank {rank}"
                pce = f"Move {target} {group} by Δ={delta}"
                if k > 1:
                    pce += f" (chain k={k})"
                out.append(Rule("MOVE_OBJ_RANK",
                              {"rank": rank, "group": group, "delta": delta, "clear": True, "bg": bg, "k": k},
                              move_prog, pce))

            copy_prog = COPY_OBJ_RANK(rank, delta, group=group, bg=bg, k=k)
            if all(exact_equals(copy_prog(x), y) for x, y in train):
                target = "largest" if rank == 0 else f"rank {rank}"
                pce = f"Copy {target} {group} by Δ={delta}"
                if k > 1:
                    pce += f" (chain k={k})"
                out.append(Rule("COPY_OBJ_RANK",
                              {"rank": rank, "group": group, "delta": delta, "clear": False, "bg": bg, "k": k},
                              copy_prog, pce))

    return out

def induce_tile(train: List[Tuple[Grid, Grid]]) -> List[Rule]:
    """Learn tiling parameters. Beam-compatible version."""
    if not train:
        return []

    rules = []

    # Try simple tiling (tile entire input)
    for nx in range(1, 6):
        for ny in range(1, 6):
            if nx == 1 and ny == 1:
                continue  # Identity

            prog = TILE(nx, ny)
            if all(exact_equals(prog(x), y) for x, y in train):
                rules.append(Rule("TILE", {"nx": nx, "ny": ny}, prog,
                                f"Tile input {nx}×{ny} times"))

    # Try tiling subgrid (extract pattern from input and tile it)
    if train:
        x0, y0 = train[0]
        h_out, w_out = y0.shape

        # Try different subgrid sizes (limit search to avoid explosion)
        for h_pat in range(1, min(x0.shape[0] + 1, 6)):
            for w_pat in range(1, min(x0.shape[1] + 1, 6)):
                if h_out % h_pat == 0 and w_out % w_pat == 0:
                    nx = h_out // h_pat
                    ny = w_out // w_pat

                    if nx == 1 and ny == 1:
                        continue

                    # Try extracting pattern from different positions (limit search)
                    for r0 in range(min(3, max(1, x0.shape[0] - h_pat + 1))):
                        for c0 in range(min(3, max(1, x0.shape[1] - w_pat + 1))):
                            r1 = r0 + h_pat - 1
                            c1 = c0 + w_pat - 1

                            prog = TILE_SUBGRID(r0, c0, r1, c1, nx, ny)

                            # Check if works for all train pairs
                            try:
                                if all(exact_equals(prog(x), y) for x, y in train):
                                    rules.append(Rule("TILE_SUBGRID",
                                                    {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "nx": nx, "ny": ny},
                                                    prog,
                                                    f"Tile subgrid [{r0}:{r1+1}, {c0}:{c1+1}] {nx}×{ny} times"))
                            except:
                                continue

    return rules

def induce_draw_line(train: List[Tuple[Grid, Grid]], bg: int = 0) -> List[Rule]:
    """Learn line drawing parameters. Beam-compatible version."""
    if not train:
        return []

    # Must have same shape
    for x, y in train:
        if x.shape != y.shape:
            return []

    # Detect line parameters from first train pair
    x0, y0 = train[0]
    diff = (x0 != y0)

    if not np.any(diff):
        return []

    changed_pixels = np.argwhere(diff)
    if len(changed_pixels) < 2:  # Need at least 2 pixels for a line
        return []

    # Get line color (most common color in changed pixels)
    colors = [int(y0[r, c]) for r, c in changed_pixels]
    line_color = int(Counter(colors).most_common(1)[0][0])

    # Try to find line endpoints
    rows = changed_pixels[:, 0]
    cols = changed_pixels[:, 1]

    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    # Check if this is a horizontal or vertical line
    is_horizontal = len(np.unique(rows)) == 1
    is_vertical = len(np.unique(cols)) == 1

    if not (is_horizontal or is_vertical):
        # Diagonal or complex line - skip for now
        return []

    prog = DRAW_LINE(r0, c0, r1, c1, line_color)

    # Verify on all train pairs
    if all(exact_equals(prog(x), y) for x, y in train):
        return [Rule("DRAW_LINE",
                   {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "color": line_color},
                   prog,
                   f"Draw line from ({r0},{c0}) to ({r1},{c1}) with color {line_color}")]

    return []

def induce_parity_beam(train: List[Tuple[Grid, Grid]]) -> List[Rule]:
    """
    Learn parity recoloring from training pairs. Beam-compatible version.
    Returns all matching parity patterns.
    """
    if not train:
        return []

    # Must have same shape
    for x, y in train:
        if x.shape != y.shape:
            return []

    rules = []
    # Try both parity types
    for parity in ('even', 'odd'):
        # Try minimal anchor variations (0 or 1 shifts)
        for ar in range(2):
            for ac in range(2):
                # Infer color from first pair
                x0, y0 = train[0]
                mask_fn = PARITY_MASK(parity, (ar, ac))
                m = mask_fn(x0)

                if not np.any(m):
                    continue

                # Get target color (most common in parity cells of output)
                vals = y0[m]
                if vals.size == 0:
                    continue

                target_color = int(Counter(vals.tolist()).most_common(1)[0][0])

                # Build program and verify on all pairs
                prog = RECOLOR_PARITY(target_color, parity, (ar, ac))

                if all(exact_equals(prog(x), y) for x, y in train):
                    rules.append(Rule("RECOLOR_PARITY",
                              {"color": target_color, "parity": parity, "anchor": (ar, ac)},
                              prog,
                              f"Recolor {parity} parity cells to {target_color} (anchor={(ar,ac)})"))

    return rules
