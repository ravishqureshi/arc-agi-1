#!/usr/bin/env python3
"""ARC Solver - Drawing Operators"""

import numpy as np
from typing import Callable, Tuple
from ..core.types import Grid

def DRAW_LINE(r0: int, c0: int, r1: int, c1: int, color: int) -> Callable[[Grid], Grid]:
    """
    Draw a line from (r0, c0) to (r1, c1) with given color.

    Uses Bresenham's line algorithm for diagonal lines.

    Args:
        r0, c0: Start coordinates
        r1, c1: End coordinates
        color: Color to draw the line

    Returns:
        Function that draws line on input grid
    """
    def f(z: Grid) -> Grid:
        out = z.copy()
        H, W = out.shape

        # Bresenham's line algorithm
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        r, c = r0, c0
        while True:
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = color

            if r == r1 and c == c1:
                break

            err2 = 2 * err
            if err2 > -dc:
                err -= dc
                r += sr
            if err2 < dr:
                err += dr
                c += sc

        return out

    return f

def DRAW_BOX(r0: int, c0: int, r1: int, c1: int, color: int, filled: bool = False) -> Callable[[Grid], Grid]:
    """
    Draw a rectangle (box) from (r0, c0) to (r1, c1).

    Args:
        r0, c0: Top-left corner
        r1, c1: Bottom-right corner
        color: Color to draw
        filled: If True, fill interior; if False, only draw border

    Returns:
        Function that draws box on input grid
    """
    def f(z: Grid) -> Grid:
        out = z.copy()
        H, W = out.shape

        # Clip to grid bounds
        r0_clip = max(0, min(r0, H - 1))
        r1_clip = max(0, min(r1, H - 1))
        c0_clip = max(0, min(c0, W - 1))
        c1_clip = max(0, min(c1, W - 1))

        if filled:
            # Fill entire rectangle
            out[r0_clip:r1_clip+1, c0_clip:c1_clip+1] = color
        else:
            # Draw border only
            out[r0_clip:r1_clip+1, c0_clip] = color  # Left edge
            out[r0_clip:r1_clip+1, c1_clip] = color  # Right edge
            out[r0_clip, c0_clip:c1_clip+1] = color  # Top edge
            out[r1_clip, c0_clip:c1_clip+1] = color  # Bottom edge

        return out

    return f

def FLOOD_FILL(start_r: int, start_c: int, new_color: int) -> Callable[[Grid], Grid]:
    """
    Flood fill from starting position with new color.

    Fills all connected cells of the same color as start position.

    Args:
        start_r, start_c: Starting position
        new_color: Color to fill with

    Returns:
        Function that performs flood fill
    """
    def f(z: Grid) -> Grid:
        out = z.copy()
        H, W = out.shape

        if not (0 <= start_r < H and 0 <= start_c < W):
            return out

        old_color = int(out[start_r, start_c])
        if old_color == new_color:
            return out

        # BFS flood fill
        from collections import deque
        queue = deque([(start_r, start_c)])
        visited = set()

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            if not (0 <= r < H and 0 <= c < W):
                continue
            if out[r, c] != old_color:
                continue

            visited.add((r, c))
            out[r, c] = new_color

            # Add 4-connected neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                queue.append((r + dr, c + dc))

        return out

    return f
