"""
Utility functions for ARC Solver.
"""

import numpy as np
import json
import hashlib
from collections import deque, Counter
from typing import List, Tuple, Dict
from pathlib import Path
from datetime import datetime
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


def bbox_nonzero(g: Grid, *, bg: int) -> Tuple[int, int, int, int]:
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


def components(g: Grid, *, bg: int) -> List[Obj]:
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


# ==============================================================================
# Hash functions for receipts
# ==============================================================================

def task_sha(train_pairs: List[Tuple[Grid, Grid]]) -> str:
    """
    Compute SHA-256 hash of train pairs for task identification.

    Args:
        train_pairs: List of (input, output) grid pairs

    Returns:
        Hex string of SHA-256 hash
    """
    payload = {"train": [{"in": x.tolist(), "out": y.tolist()} for x, y in train_pairs]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def program_sha(ops: List) -> str:
    """
    Compute SHA-256 hash of operator sequence.

    Args:
        ops: List of Operator objects

    Returns:
        Hex string of SHA-256 hash
    """
    if not ops:
        return hashlib.sha256(b"[]").hexdigest()

    # Convert params to JSON-serializable format
    payload = []
    for op in ops:
        params_serializable = {}
        for k, v in op.params.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                params_serializable[k] = v
            elif isinstance(v, dict):
                params_serializable[k] = {str(kk): vv for kk, vv in v.items()}
            elif isinstance(v, (list, tuple)):
                params_serializable[k] = list(v)
            else:
                params_serializable[k] = str(v)
        payload.append({"name": op.name, "params": params_serializable})

    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def closure_set_sha(closures: List) -> str:
    """
    Compute SHA-256 hash of closure sequence.

    Args:
        closures: List of Closure objects

    Returns:
        Hex string of SHA-256 hash
    """
    if not closures:
        return hashlib.sha256(b"[]").hexdigest()

    # Convert to JSON-serializable format
    payload = []
    for closure in closures:
        params_serializable = {}
        for k, v in closure.params.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                params_serializable[k] = v
            elif isinstance(v, dict):
                params_serializable[k] = {str(kk): vv for kk, vv in v.items()}
            elif isinstance(v, (list, tuple)):
                params_serializable[k] = list(v)
            else:
                params_serializable[k] = str(v)
        payload.append({"name": closure.name, "params": params_serializable})

    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ==============================================================================
# Invariant computation for receipts
# ==============================================================================

def compute_palette_delta(x: Grid, y: Grid) -> Dict:
    """
    Compute palette invariants between input and output.

    Args:
        x: Input grid
        y: Output grid

    Returns:
        {"preserved": bool, "delta": {color: count_delta}}
    """
    x_counts = Counter(x.flatten().tolist())
    y_counts = Counter(y.flatten().tolist())

    all_colors = set(x_counts.keys()) | set(y_counts.keys())
    delta = {int(c): y_counts.get(c, 0) - x_counts.get(c, 0) for c in all_colors}

    # Preserved if all colors in input are in output (count can change)
    preserved = set(x_counts.keys()).issubset(set(y_counts.keys()))

    return {"preserved": preserved, "delta": delta}


def compute_component_delta(x: Grid, y: Grid, *, bg: int) -> Dict:
    """
    Compute component invariants between input and output.

    Args:
        x: Input grid
        y: Output grid
        bg: Background color

    Returns:
        {"count_delta": int, "largest_kept": bool, "size_largest_x": int, "size_largest_y": int}
    """
    x_objs = components(x, bg=bg)
    y_objs = components(y, bg=bg)

    count_delta = len(y_objs) - len(x_objs)
    size_largest_x = max([o.size for o in x_objs], default=0)
    size_largest_y = max([o.size for o in y_objs], default=0)

    # Check if largest component from x is kept in y
    largest_kept = False
    if x_objs and y_objs:
        largest_x = max(x_objs, key=lambda o: o.size)
        # Check if any y component has same color and size as largest_x
        largest_kept = any(o.color == largest_x.color and o.size == largest_x.size for o in y_objs)

    return {
        "count_delta": count_delta,
        "largest_kept": largest_kept,
        "size_largest_x": size_largest_x,
        "size_largest_y": size_largest_y
    }


# ==============================================================================
# Receipt logging
# ==============================================================================

def log_receipt(record: Dict, out_dir: str = None) -> None:
    """
    Write receipt record to JSONL file.

    Args:
        record: Dictionary with receipt data
        out_dir: Output directory (default: runs/YYYY-MM-DD)
    """
    if out_dir is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_dir = f"runs/{date_str}"

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    receipt_path = Path(out_dir) / "receipts.jsonl"

    with open(receipt_path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
