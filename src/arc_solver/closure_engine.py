"""
Fixed-Point Closure Engine for ARC Solver.

Based on Master Operator paradigm from arc_agi_master_operator.md:
- Set-valued grids (each cell = set of allowable colors)
- Closures = monotone, shrinking, idempotent functions
- Fixed-point iteration guaranteed by Tarski theorem
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from .types import Grid


# Maximum iterations for fixed-point convergence (safety net; Tarski guarantees finite)
DEFAULT_MAX_ITERS = 100


# ==============================================================================
# Set-Valued Grid Representation
# ==============================================================================

class SetValuedGrid:
    """
    Grid where each cell contains a set of allowable colors {0-9}.

    Implementation: H×W array of 10-bit integers (bit i = color i is possible).
    Bit 0 = color 0 allowed, bit 1 = color 1 allowed, ..., bit 9 = color 9 allowed.
    """

    def __init__(self, H: int, W: int, init_mask: int = 0x3FF):
        """
        Initialize set-valued grid.

        Args:
            H, W: Grid dimensions
            init_mask: Initial mask for all cells (default: 0x3FF = all colors {0-9})
        """
        self.H = H
        self.W = W
        # Use uint16 to hold 10 bits
        self.data = np.full((H, W), init_mask, dtype=np.uint16)

    def copy(self) -> 'SetValuedGrid':
        """Deep copy of set-valued grid."""
        U = SetValuedGrid(self.H, self.W, 0)
        U.data = self.data.copy()
        return U

    def get_set(self, r: int, c: int) -> set:
        """Get set of allowed colors at cell (r, c)."""
        mask = int(self.data[r, c])
        return {i for i in range(10) if mask & (1 << i)}

    def set_mask(self, r: int, c: int, mask: int):
        """Set mask at cell (r, c)."""
        self.data[r, c] = mask

    def intersect(self, r: int, c: int, mask: int):
        """Intersect cell (r, c) with mask (monotone shrinking)."""
        self.data[r, c] &= mask

    def is_singleton(self, r: int, c: int) -> bool:
        """Check if cell (r, c) is a singleton (exactly one color)."""
        mask = self.data[r, c]
        return mask != 0 and (mask & (mask - 1)) == 0

    def is_empty(self, r: int, c: int) -> bool:
        """Check if cell (r, c) is empty (no colors allowed)."""
        return self.data[r, c] == 0

    def to_grid(self) -> Optional[Grid]:
        """
        Convert to deterministic grid if all cells are singletons.
        Returns None if any cell is multi-valued or empty.
        """
        if not self.is_fully_determined():
            return None

        result = np.zeros((self.H, self.W), dtype=int)
        for r in range(self.H):
            for c in range(self.W):
                mask = self.data[r, c]
                if mask == 0:
                    return None
                # Find the single bit set
                for color in range(10):
                    if mask & (1 << color):
                        result[r, c] = color
                        break
        return result

    def to_grid_deterministic(self, *, fallback: str = 'lowest', bg: int) -> Grid:
        """
        Convert to deterministic grid, breaking ties deterministically.

        Args:
            fallback: 'lowest' (pick lowest color from multi-valued cells). Currently only 'lowest' is implemented.
            bg: Background color to use for empty cells (REQUIRED keyword-only, no default)

        Returns:
            Grid with deterministic choice for each cell
        """
        result = np.zeros((self.H, self.W), dtype=int)
        for r in range(self.H):
            for c in range(self.W):
                mask = self.data[r, c]
                if mask == 0:
                    result[r, c] = bg  # Empty → background (bg must be provided by caller)
                else:
                    # Pick lowest color from set
                    for color in range(10):
                        if mask & (1 << color):
                            result[r, c] = color
                            break
        return result

    def is_fully_determined(self) -> bool:
        """Check if all cells are singletons."""
        return np.all((self.data != 0) & ((self.data & (self.data - 1)) == 0))

    def count_multi_valued_cells(self) -> int:
        """Count cells with multiple colors allowed."""
        count = 0
        for r in range(self.H):
            for c in range(self.W):
                mask = self.data[r, c]
                if mask != 0 and (mask & (mask - 1)) != 0:  # Not singleton
                    count += 1
        return count

    def __eq__(self, other: 'SetValuedGrid') -> bool:
        """Check if two set-valued grids are equal."""
        if self.H != other.H or self.W != other.W:
            return False
        return np.array_equal(self.data, other.data)


# ==============================================================================
# Helper Functions
# ==============================================================================

def color_to_mask(color: int) -> int:
    """Convert single color to 10-bit mask."""
    return 1 << color

def set_to_mask(colors: set) -> int:
    """Convert set of colors to 10-bit mask."""
    mask = 0
    for c in colors:
        mask |= (1 << c)
    return mask

def init_top(H: int, W: int) -> SetValuedGrid:
    """Initialize ⊤ (all cells allow all colors {0-9})."""
    return SetValuedGrid(H, W, init_mask=0x3FF)

def init_from_grid(g: Grid) -> SetValuedGrid:
    """Initialize set-valued grid from deterministic grid (all singletons)."""
    H, W = g.shape
    U = SetValuedGrid(H, W, 0)
    for r in range(H):
        for c in range(W):
            U.set_mask(r, c, color_to_mask(int(g[r, c])))
    return U


# ==============================================================================
# Closure Base Class
# ==============================================================================

@dataclass
class Closure:
    """
    Base class for closures (monotone, shrinking, idempotent functions).

    A closure F: SetValuedGrid → SetValuedGrid must satisfy:
    1. Monotone: U ⊆ V → F(U) ⊆ F(V)
    2. Shrinking: F(U) ⊆ U
    3. Idempotent: F(F(U)) = F(U)
    """
    name: str
    params: Dict
    is_meta: bool = False  # True for metadata-only closures (e.g., CANVAS_SIZE)

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        """
        Apply closure to set-valued grid.

        Args:
            U: Current set-valued grid
            x_input: Original input grid (for context)

        Returns:
            New set-valued grid (monotone shrinking from U)
        """
        raise NotImplementedError("Subclass must implement apply()")


# ==============================================================================
# Fixed-Point Iterator
# ==============================================================================

def run_fixed_point(closures: List[Closure],
                    x_input: Grid,
                    *,
                    canvas: Optional[Dict] = None,
                    max_iters: int = DEFAULT_MAX_ITERS) -> Tuple[SetValuedGrid, Dict]:
    """
    Run fixed-point iteration until convergence.

    CANVAS-AWARE: Can now initialize U to output shape (not just input shape).

    Args:
        closures: List of closures to apply
        x_input: Input grid
        canvas: Optional dict with "H" and "W" keys for output shape
                If None, defaults to x_input.shape (backward compatible)
        max_iters: Maximum iterations

    Returns:
        (U_final, stats) where stats = {"iters": N, "cells_multi": M}
    """
    # CANVAS-AWARE: Use canvas size if provided, else default to input shape
    if canvas:
        H, W = canvas["H"], canvas["W"]
    else:
        H, W = x_input.shape

    U = init_top(H, W)

    for iteration in range(max_iters):
        U_prev = U.copy()

        # Apply all closures in sequence
        for closure in closures:
            U = closure.apply(U, x_input)

        # Check convergence
        if U == U_prev:
            stats = {
                "iters": iteration + 1,
                "cells_multi": U.count_multi_valued_cells()
            }
            return U, stats

    # Max iterations reached
    stats = {
        "iters": max_iters,
        "cells_multi": U.count_multi_valued_cells()
    }
    return U, stats


# ==============================================================================
# Train Verification
# ==============================================================================

def verify_closures_on_train(closures: List[Closure],
                              train: List[Tuple[Grid, Grid]]) -> bool:
    """
    Verify that closures produce exact output on ALL train pairs.

    CANVAS-AWARE: Extracts canvas from CANVAS_SIZE closure if present.

    Args:
        closures: List of closures
        train: List of (input, output) pairs

    Returns:
        True if all train pairs converge to singleton(expected_output)
    """
    # CANVAS-AWARE: Extract canvas strategy from CANVAS_SIZE closure
    canvas_params = None
    for closure in closures:
        if closure.name.startswith("CANVAS_SIZE"):
            canvas_params = closure.params
            break

    for x, y in train:
        # Compute canvas per-input for TILE_MULTIPLE strategy
        canvas = _compute_canvas(x, canvas_params) if canvas_params else None

        U_final, _ = run_fixed_point(closures, x, canvas=canvas)

        # Check if U_final is fully determined
        if not U_final.is_fully_determined():
            return False

        # Check if U_final equals expected output
        y_pred = U_final.to_grid()
        if y_pred is None or not np.array_equal(y_pred, y):
            return False

    return True


def verify_consistent_on_train(closures: List[Closure],
                                train: List[Tuple[Grid, Grid]]) -> bool:
    """
    Verify closures don't create contradictions with train outputs.

    Unlike verify_closures_on_train, this does NOT require U to be fully determined.
    Only checks:
    1. No cell is empty (contradiction)
    2. For cells where x[r,c] == y[r,c], closure doesn't remove y's color

    Args:
        closures: List of closures to verify
        train: List of (input, output) pairs

    Returns:
        True if no contradictions detected (allows TOP cells)
    """
    # Extract canvas params if present
    canvas_params = None
    for closure in closures:
        if closure.name.startswith("CANVAS_SIZE"):
            canvas_params = closure.params
            break

    for x, y in train:
        # Compute canvas per-input
        canvas = _compute_canvas(x, canvas_params) if canvas_params else None

        # Run fixed point
        U_final, _ = run_fixed_point(closures, x, canvas=canvas)

        # Check for contradictions (not full determination)
        # CANVAS-AWARE: Iterate over U_final bounds, check only cells that exist in y
        for r in range(U_final.H):
            for c in range(U_final.W):
                # Skip cells outside y bounds (canvas may be larger than y)
                if r >= y.shape[0] or c >= y.shape[1]:
                    continue  # Cells outside y are allowed to be multi-valued

                allowed = U_final.get_set(r, c)
                if len(allowed) == 0:
                    return False  # Empty cell = contradiction

                # If x and y agree on this cell, y's color must still be allowed
                if r < x.shape[0] and c < x.shape[1]:
                    if int(x[r, c]) == int(y[r, c]):
                        if int(y[r, c]) not in allowed:
                            return False  # Removed correct color

    return True  # No contradictions (may have TOP cells)


def _compute_canvas(x_input: Grid, canvas_params: Dict) -> Dict:
    """
    Compute canvas size for a given input based on strategy.

    Args:
        x_input: Input grid
        canvas_params: CANVAS_SIZE closure params with "strategy": "TILE_MULTIPLE", "k_h", "k_w"

    Returns:
        Dict with "H" and "W" keys for output shape
    """
    strategy = canvas_params["strategy"]

    if strategy == "TILE_MULTIPLE":
        # Per-input canvas: H_out = k_h * H_in, W_out = k_w * W_in
        k_h = canvas_params["k_h"]
        k_w = canvas_params["k_w"]
        H_in, W_in = x_input.shape
        return {"H": k_h * H_in, "W": k_w * W_in}
    else:
        # Unknown strategy - fallback to input shape (backward compatible)
        return {"H": x_input.shape[0], "W": x_input.shape[1]}


def preserves_y(closure: Closure, train: List[Tuple[Grid, Grid]]) -> bool:
    """
    Verify closure preserves singleton(y) for all train pairs.

    A closure is composition-safe if applying it to the expected output
    doesn't change the output. This ensures the closure won't interfere
    with other closures in the composition.

    CANVAS-AWARE: For shape-changing tasks (x.shape != y.shape), only checks
    cells in the overlapping region that exist in both x and y. Cells outside
    x_input bounds are not processed by closures, so they're allowed to differ.

    Args:
        closure: Closure to test
        train: List of (input, output) pairs

    Returns:
        True if closure preserves y's colors in cells that closures can process
    """
    for x, y in train:
        # Build U_y with y.shape (not x.shape) - CANVAS-AWARE
        U_y = init_from_grid(y)
        U_y_after = closure.apply(U_y, x)

        # CANVAS-AWARE: Only check cells within x_input bounds
        # Closures only process cells within x_input shape; cells outside remain unchanged
        H_check = min(x.shape[0], y.shape[0], U_y_after.H)
        W_check = min(x.shape[1], y.shape[1], U_y_after.W)

        for r in range(H_check):
            for c in range(W_check):
                # Check if this cell's color set was modified
                if U_y.get_set(r, c) != U_y_after.get_set(r, c):
                    return False

    return True


def compatible_to_y(closure: Closure, train: List[Tuple[Grid, Grid]]) -> bool:
    """
    Non-destructive on known-correct pixels.
    For cells where x[r,c] == y[r,c], applying closure to singleton(x) must retain that color.
    Does not enforce global palette subset - final composition checked by verify_closures_on_train.

    CANVAS-AWARE: Now works with shape-changing tasks (x.shape != y.shape).
    Only checks cells that exist in BOTH x and y (overlapping region).
    """
    for x, y in train:
        Ux = init_from_grid(x)
        Ux2 = closure.apply(Ux, x)

        # CANVAS-AWARE: Check only overlapping region
        H_min = min(x.shape[0], y.shape[0])
        W_min = min(x.shape[1], y.shape[1])

        for r in range(H_min):
            for c in range(W_min):
                if int(x[r,c]) == int(y[r,c]):
                    if int(y[r,c]) not in Ux2.get_set(r, c):
                        return False
    return True
