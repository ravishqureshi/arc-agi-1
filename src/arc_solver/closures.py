"""
Closure implementations for ARC Solver.

Each closure:
1. Inherits from Closure base class
2. Implements apply(U, x_input) - monotone & shrinking
3. Has corresponding unifier function to learn params from train pairs
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
from .types import Grid, Operator
from .utils import equal, components
from .closure_engine import Closure, SetValuedGrid, color_to_mask, set_to_mask


def infer_bg_from_border(x: np.ndarray) -> int:
    """
    Infer background color from border pixels (deterministic majority vote).

    Takes the most common color on the border (top row, bottom row, left col, right col).
    Deterministic tie-break: if multiple colors have same count, choose lowest color.

    Args:
        x: Input grid (H×W numpy array)

    Returns:
        Background color (int 0-9)
    """
    H, W = x.shape
    border_pixels = []

    # Top & bottom rows
    border_pixels.extend(x[0, :])
    border_pixels.extend(x[H-1, :])

    # Left & right cols (excluding corners already counted)
    if H > 2:
        border_pixels.extend(x[1:H-1, 0])
        border_pixels.extend(x[1:H-1, W-1])

    # Majority vote (deterministic tie-break: lowest color)
    counts = Counter(int(p) for p in border_pixels)
    # Sort by (-count, color) to pick highest count, then lowest color on ties
    bg = min(counts.keys(), key=lambda c: (-counts[c], c))
    return int(bg)


# ==============================================================================
# BASE: INPUT_IDENTITY - Constrains based on input grid
# ==============================================================================

class INPUT_IDENTITY_Closure(Closure):
    """
    Base closure that constrains each cell to only allow colors present in input at that position.

    For tasks where output = transform(input), this provides the base constraint.
    """

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        U_new = U.copy()

        # CANVAS-AWARE: Only process cells within x_input bounds
        H_in, W_in = x_input.shape

        for r in range(U.H):
            for c in range(U.W):
                # Skip cells outside x_input bounds (leave unchanged in U)
                if r >= H_in or c >= W_in:
                    continue

                input_color = int(x_input[r, c])
                input_mask = color_to_mask(input_color)
                U_new.intersect(r, c, input_mask)
        return U_new


# ==============================================================================
# B1: KEEP_LARGEST_COMPONENT
# ==============================================================================

class KEEP_LARGEST_COMPONENT_Closure(Closure):
    """
    Keep only largest connected component.

    Closure behavior:
    - For cells NOT in largest component of x_input: intersect with {bg}
    - For cells IN largest component: intersect with {input_color}
    """

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        # Infer bg per-input if params["bg"] is None
        bg = self.params["bg"]
        if bg is None:
            bg = infer_bg_from_border(x_input)
        objs = components(x_input, bg=bg)

        if not objs:
            # No components - everything becomes background
            U_new = U.copy()
            bg_mask = color_to_mask(bg)
            for r in range(U.H):
                for c in range(U.W):
                    U_new.intersect(r, c, bg_mask)
            return U_new

        # Find largest component (deterministic tie-breaking by bbox position)
        largest = max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))

        # Build set of pixels in largest component
        largest_pixels = set(largest.pixels)

        # Shrink U
        U_new = U.copy()
        bg_mask = color_to_mask(bg)

        # CANVAS-AWARE: Only process cells within x_input bounds
        H_in, W_in = x_input.shape

        for r in range(U.H):
            for c in range(U.W):
                # Skip cells outside x_input bounds (leave unchanged in U)
                if r >= H_in or c >= W_in:
                    continue

                if (r, c) not in largest_pixels:
                    # Not in largest → must be background
                    U_new.intersect(r, c, bg_mask)
                else:
                    # In largest → preserve input color
                    input_color = int(x_input[r, c])
                    input_mask = color_to_mask(input_color)
                    U_new.intersect(r, c, input_mask)

        return U_new


def unify_KEEP_LARGEST(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    """
    Unifier for KEEP_LARGEST_COMPONENT.

    Uses composition-safe gate (preserves_y + compatible_to_y).
    Prefers bg=None (per-input inference) over explicit bg values.

    Returns:
        List of KEEP_LARGEST closures (0..N candidates)
        Empty list if no bg value works
    """
    from .closure_engine import preserves_y, compatible_to_y

    valid = []

    # Try bg=None first (per-input inference)
    candidate_none = KEEP_LARGEST_COMPONENT_Closure(
        "KEEP_LARGEST_COMPONENT[bg=None]",
        {"bg": None}
    )
    if preserves_y(candidate_none, train) and compatible_to_y(candidate_none, train):
        valid.append(candidate_none)

    # Fallback: enumerate explicit bgs
    for bg in range(10):
        candidate = KEEP_LARGEST_COMPONENT_Closure(
            f"KEEP_LARGEST_COMPONENT[bg={bg}]",
            {"bg": bg}
        )
        if preserves_y(candidate, train) and compatible_to_y(candidate, train):
            valid.append(candidate)

    return valid


# ==============================================================================
# B2: OUTLINE_OBJECTS
# ==============================================================================

class OUTLINE_OBJECTS_Closure(Closure):
    """
    Outline (outer, thickness=1) closure.

    params = {"mode": "outer", "scope": "largest"|"all", "bg": int}

    Law: Object pixels adjacent (4-connected) to background form the outline.
         Outline pixels retain object color; interior pixels forced to {bg}.
    """

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        # Fail loudly if required params not provided
        mode = self.params["mode"]
        scope = self.params["scope"]
        bg = self.params["bg"]
        if bg is None:
            bg = infer_bg_from_border(x_input)

        if mode != "outer":
            raise ValueError(f"Only mode='outer' is implemented, got '{mode}'")

        objs = components(x_input, bg=bg)

        if not objs:
            # No components - everything becomes background
            U_new = U.copy()
            bg_mask = color_to_mask(bg)
            for r in range(U.H):
                for c in range(U.W):
                    U_new.intersect(r, c, bg_mask)
            return U_new

        # Select components based on scope
        if scope == "largest":
            # Choose largest component (deterministic tie-breaking)
            selected_objs = [max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))]
        elif scope == "all":
            selected_objs = objs
        else:
            raise ValueError(f"Invalid scope: {scope}")

        # Build outline pixels set for selected components
        outline_pixels = set()
        object_pixels = set()
        color_map = {}  # (r, c) -> object_color

        H, W = x_input.shape

        for obj in selected_objs:
            # Track all pixels of this object
            for r, c in obj.pixels:
                object_pixels.add((r, c))
                color_map[(r, c)] = obj.color

            # Find outline: pixels with at least one 4-neighbor = bg
            for r, c in obj.pixels:
                is_outline = False
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    # Check if neighbor is out of bounds or background
                    if not (0 <= nr < H and 0 <= nc < W):
                        is_outline = True
                        break
                    if x_input[nr, nc] == bg:
                        is_outline = True
                        break

                if is_outline:
                    outline_pixels.add((r, c))

        # Build masks
        U_new = U.copy()
        bg_mask = color_to_mask(bg)

        # CANVAS-AWARE: Only process cells within x_input bounds
        H_in, W_in = x_input.shape

        for r in range(U.H):
            for c in range(U.W):
                # Skip cells outside x_input bounds (leave unchanged in U)
                if r >= H_in or c >= W_in:
                    continue

                if (r, c) in outline_pixels:
                    # Outline pixel: keep object color
                    obj_color = color_map[(r, c)]
                    obj_mask = color_to_mask(obj_color)
                    U_new.intersect(r, c, obj_mask)
                elif (r, c) in object_pixels:
                    # Interior object pixel (not outline): force to bg
                    U_new.intersect(r, c, bg_mask)
                else:
                    # Background pixel: force to bg
                    U_new.intersect(r, c, bg_mask)

        return U_new


def unify_OUTLINE_OBJECTS(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    """
    Unifier for OUTLINE_OBJECTS.

    Uses composition-safe gate (preserves_y + compatible_to_y).
    Prefers bg=None (per-input inference) over explicit bg values.

    Enumerates:
    - mode="outer" (fixed for M1)
    - scope ∈ {"largest", "all"}
    - bg: Try None first, then {0..9}

    Returns:
        List of OUTLINE_OBJECTS closures (0..N candidates)
        Empty list if no params work
    """
    from .closure_engine import preserves_y, compatible_to_y

    valid = []

    # Enumerate parameters
    mode = "outer"  # Fixed for M1
    scopes = ["largest", "all"]

    # Try bg=None first for each scope
    for scope in scopes:
        candidate_none = OUTLINE_OBJECTS_Closure(
            f"OUTLINE_OBJECTS[mode={mode},scope={scope},bg=None]",
            {"mode": mode, "scope": scope, "bg": None}
        )
        if preserves_y(candidate_none, train) and compatible_to_y(candidate_none, train):
            valid.append(candidate_none)

    # Fallback: enumerate explicit bgs
    for scope in scopes:
        for bg in range(10):
            candidate = OUTLINE_OBJECTS_Closure(
                f"OUTLINE_OBJECTS[mode={mode},scope={scope},bg={bg}]",
                {"mode": mode, "scope": scope, "bg": bg}
            )
            if preserves_y(candidate, train) and compatible_to_y(candidate, train):
                valid.append(candidate)

    return valid


# ==============================================================================
# Morphology Helpers (k=1, 4-connected)
# ==============================================================================

def erode_k1_4conn(mask: np.ndarray) -> np.ndarray:
    """
    Erode binary mask by k=1 with 4-connected structuring element.

    For each pixel: keep True only if all 4-neighbors are also True.
    Out-of-bounds treated as background (False).
    """
    H, W = mask.shape
    result = np.zeros_like(mask, dtype=bool)

    for r in range(H):
        for c in range(W):
            if not mask[r, c]:
                continue

            # Check all 4-neighbors (up, down, left, right)
            keep = True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                # Out-of-bounds treated as background
                if not (0 <= nr < H and 0 <= nc < W):
                    keep = False
                    break
                if not mask[nr, nc]:
                    keep = False
                    break

            result[r, c] = keep

    return result


def dilate_k1_4conn(mask: np.ndarray) -> np.ndarray:
    """
    Dilate binary mask by k=1 with 4-connected structuring element.

    For each pixel: set True if any 4-neighbor is True.
    """
    H, W = mask.shape
    result = mask.copy()

    for r in range(H):
        for c in range(W):
            if mask[r, c]:
                # Spread to all 4-neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        result[nr, nc] = True

    return result


# ==============================================================================
# M2.1: OPEN_CLOSE
# ==============================================================================

class OPEN_CLOSE_Closure(Closure):
    """
    Morphological opening/closing (k=1, 4-connected).

    params = {"mode": "open" | "close", "bg": int}

    Laws:
    - OPEN = ERODE then DILATE (removes small protrusions)
    - CLOSE = DILATE then ERODE (fills small gaps)
    """

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        # Fail loudly if required params not provided
        mode = self.params["mode"]
        bg = self.params["bg"]
        if bg is None:
            bg = infer_bg_from_border(x_input)

        if mode not in ["open", "close"]:
            raise ValueError(f"Invalid mode: {mode}. Expected 'open' or 'close'.")

        # Build binary foreground mask
        fg_mask = (x_input != bg)

        # Compute morphological operation
        if mode == "open":
            # OPEN = ERODE then DILATE
            eroded = erode_k1_4conn(fg_mask)
            y_star = dilate_k1_4conn(eroded)
        else:  # mode == "close"
            # CLOSE = DILATE then ERODE
            dilated = dilate_k1_4conn(fg_mask)
            y_star = erode_k1_4conn(dilated)

        # Build per-cell allowed set from y_star
        U_new = U.copy()
        bg_mask = color_to_mask(bg)

        # CANVAS-AWARE: Only process cells within x_input bounds
        H_in, W_in = x_input.shape

        for r in range(U.H):
            for c in range(U.W):
                # Skip cells outside x_input bounds (leave unchanged in U)
                if r >= H_in or c >= W_in:
                    continue

                if y_star[r, c]:
                    # Foreground pixel in morphology result
                    if x_input[r, c] != bg:
                        # Already foreground in input: keep input color
                        obj_color = int(x_input[r, c])
                        obj_mask = color_to_mask(obj_color)
                        U_new.intersect(r, c, obj_mask)
                    else:
                        # Background in input but foreground in morphology: infer color from neighbors
                        # Find nearest foreground neighbor color
                        neighbor_color = None
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < x_input.shape[0] and 0 <= nc < x_input.shape[1]:
                                if x_input[nr, nc] != bg:
                                    neighbor_color = int(x_input[nr, nc])
                                    break
                        # Fail loudly if color cannot be inferred
                        if neighbor_color is None:
                            raise ValueError(
                                f"OPEN_CLOSE: Cannot infer color for morphology result at ({r}, {c}). "
                                f"Pixel is foreground in morphology but has no foreground neighbors. "
                                f"This indicates malformed input or incorrect bg parameter."
                            )
                        obj_mask = color_to_mask(neighbor_color)
                        U_new.intersect(r, c, obj_mask)
                else:
                    # Background pixel: force to {bg}
                    U_new.intersect(r, c, bg_mask)

        return U_new


def unify_OPEN_CLOSE(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    """
    Unifier for OPEN_CLOSE.

    Uses composition-safe gate (preserves_y + compatible_to_y).
    Prefers bg=None (per-input inference) over explicit bg values.

    Enumerates:
    - mode ∈ {"open", "close"}
    - bg: Try None first, then {0..9}

    Returns:
        List of OPEN_CLOSE closures (0..N candidates)
        Empty list if no params work
    """
    from .closure_engine import preserves_y, compatible_to_y

    valid = []

    # Enumerate parameters
    modes = ["open", "close"]

    # Try bg=None first for each mode
    for mode in modes:
        candidate_none = OPEN_CLOSE_Closure(
            f"OPEN_CLOSE[mode={mode},bg=None]",
            {"mode": mode, "bg": None}
        )
        if preserves_y(candidate_none, train) and compatible_to_y(candidate_none, train):
            valid.append(candidate_none)

    # Fallback: enumerate explicit bgs
    for mode in modes:
        for bg in range(10):
            candidate = OPEN_CLOSE_Closure(
                f"OPEN_CLOSE[mode={mode},bg={bg}]",
                {"mode": mode, "bg": bg}
            )
            if preserves_y(candidate, train) and compatible_to_y(candidate, train):
                valid.append(candidate)

    return valid


# ==============================================================================
# M2.2: AXIS_PROJECTION
# ==============================================================================

class AXIS_PROJECTION_Closure(Closure):
    """
    Axis projection: extend object pixels along row/col to border.

    params = {"axis": "row"|"col", "scope": "largest"|"all", "mode": "to_border", "bg": int}

    Law: From each object pixel, extend along axis to image border with object color.
    """

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        # Fail loudly if required params not provided
        axis = self.params["axis"]
        scope = self.params["scope"]
        mode = self.params["mode"]
        bg = self.params["bg"]
        if bg is None:
            bg = infer_bg_from_border(x_input)

        if axis not in ["row", "col"]:
            raise ValueError(f"Invalid axis: {axis}. Expected 'row' or 'col'.")
        if scope not in ["largest", "all"]:
            raise ValueError(f"Invalid scope: {scope}. Expected 'largest' or 'all'.")
        if mode != "to_border":
            raise ValueError(f"Only mode='to_border' is implemented, got '{mode}'")

        objs = components(x_input, bg=bg)

        if not objs:
            # No components - everything becomes background
            U_new = U.copy()
            bg_mask = color_to_mask(bg)
            for r in range(U.H):
                for c in range(U.W):
                    U_new.intersect(r, c, bg_mask)
            return U_new

        # Select components based on scope
        if scope == "largest":
            # Choose largest component (deterministic tie-breaking)
            selected_objs = [max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))]
        elif scope == "all":
            selected_objs = objs
        else:
            raise ValueError(f"Invalid scope: {scope}")

        # Build projection: for each object pixel, extend along axis to border
        H, W = x_input.shape
        projected_pixels = {}  # (r, c) -> color

        for obj in selected_objs:
            for r, c in obj.pixels:
                if axis == "row":
                    # Paint entire row with object color
                    for col in range(W):
                        projected_pixels[(r, col)] = obj.color
                elif axis == "col":
                    # Paint entire column with object color
                    for row in range(H):
                        projected_pixels[(row, c)] = obj.color

        # Build masks: projected pixels allow object color, others allow bg only
        U_new = U.copy()
        bg_mask = color_to_mask(bg)

        # CANVAS-AWARE: Process all cells (projection extends beyond input)
        # But check projected_pixels which are computed from x_input
        for r in range(U.H):
            for c in range(U.W):
                if (r, c) in projected_pixels:
                    # Projected pixel: allow object color
                    obj_color = projected_pixels[(r, c)]
                    obj_mask = color_to_mask(obj_color)
                    U_new.intersect(r, c, obj_mask)
                else:
                    # Non-projected pixel: force to bg
                    U_new.intersect(r, c, bg_mask)

        return U_new


def unify_AXIS_PROJECTION(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    """
    Unifier for AXIS_PROJECTION.

    Uses composition-safe gate (preserves_y + compatible_to_y).
    Prefers bg=None (per-input inference) over explicit bg values.

    Enumerates:
    - axis ∈ {"row", "col"}
    - scope ∈ {"largest", "all"}
    - mode = "to_border" (fixed)
    - bg: Try None first, then {0..9}

    Returns:
        List of AXIS_PROJECTION closures (0..N candidates)
        Empty list if no params work on ALL train pairs
    """
    from .closure_engine import preserves_y, compatible_to_y

    valid = []

    # Enumerate parameters
    axes = ["row", "col"]
    scopes = ["largest", "all"]
    mode = "to_border"  # Fixed for M2.2

    # Try bg=None first for each axis/scope combo
    for axis in axes:
        for scope in scopes:
            candidate_none = AXIS_PROJECTION_Closure(
                f"AXIS_PROJECTION[axis={axis},scope={scope},mode={mode},bg=None]",
                {"axis": axis, "scope": scope, "mode": mode, "bg": None}
            )
            if preserves_y(candidate_none, train) and compatible_to_y(candidate_none, train):
                valid.append(candidate_none)

    # Fallback: enumerate explicit bgs
    for axis in axes:
        for scope in scopes:
            for bg in range(10):
                candidate = AXIS_PROJECTION_Closure(
                    f"AXIS_PROJECTION[axis={axis},scope={scope},mode={mode},bg={bg}]",
                    {"axis": axis, "scope": scope, "mode": mode, "bg": bg}
                )
                if preserves_y(candidate, train) and compatible_to_y(candidate, train):
                    valid.append(candidate)

    return valid


# ==============================================================================
# M2.3: SYMMETRY_COMPLETION
# ==============================================================================

class SYMMETRY_COMPLETION_Closure(Closure):
    """
    Symmetry completion: reflect content and union with original.

    params = {
        "axis": "v" | "h" | "diag" | "anti",
        "scope": "global" | "largest" | "per_object",
        "bg": int  # background color
    }

    Law: y* = x ∪ reflect_axis_scope(x, axis, scope, bg)
         U' = U ∩ one_hot(y*)
    """

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        """
        Compute reflection union; intersect U with result masks (only clear bits).

        Returns:
            U' where U'[r,c] = U[r,c] ∩ {color_allowed_at_(r,c)_in_y*}
        """
        # Fail loudly if required params not provided
        axis = self.params["axis"]
        scope = self.params["scope"]
        bg = self.params["bg"]
        if bg is None:
            bg = infer_bg_from_border(x_input)

        if axis not in ["v", "h", "diag", "anti"]:
            raise ValueError(f"Invalid axis: {axis}. Expected 'v', 'h', 'diag', or 'anti'.")
        if scope not in ["global", "largest", "per_object"]:
            raise ValueError(f"Invalid scope: {scope}. Expected 'global', 'largest', or 'per_object'.")

        # Compute reflected version
        R = _reflect_axis_scope(x_input, axis, scope, bg)

        # Compute union: prioritize original over reflection
        y_star = np.where(x_input != bg, x_input, R)

        # Build per-cell allowed set from y_star
        U_new = U.copy()
        bg_mask = color_to_mask(bg)

        # CANVAS-AWARE: Only process cells within y_star (x_input) bounds
        H_in, W_in = y_star.shape

        for r in range(U.H):
            for c in range(U.W):
                # Skip cells outside y_star bounds (leave unchanged in U)
                if r >= H_in or c >= W_in:
                    continue

                color_at_rc = int(y_star[r, c])
                allowed_mask = color_to_mask(color_at_rc)
                U_new.intersect(r, c, allowed_mask)

        return U_new


def _reflect_axis_scope(x: Grid, axis: str, scope: str, bg: int) -> Grid:
    """
    Reflect grid across axis with given scope.

    Args:
        x: Input grid
        axis: "v" | "h" | "diag" | "anti"
        scope: "global" | "largest" | "per_object"
        bg: Background color

    Returns:
        Reflected grid R
    """
    H, W = x.shape

    # Validate diagonal axes on non-square grids
    if axis in ["diag", "anti"] and H != W:
        raise ValueError(f"Diagonal reflection (axis={axis}) requires square grid, got shape ({H}, {W})")

    if scope == "global":
        return _reflect_global(x, axis, bg)
    elif scope == "largest":
        return _reflect_largest(x, axis, bg)
    elif scope == "per_object":
        return _reflect_per_object(x, axis, bg)
    else:
        raise ValueError(f"Invalid scope: {scope}")


def _reflect_global(x: Grid, axis: str, bg: int) -> Grid:
    """Reflect entire grid across axis."""
    H, W = x.shape
    R = np.full_like(x, bg)

    if axis == "v":
        # Vertical reflection (across vertical center line)
        for r in range(H):
            for c in range(W):
                c_mirror = W - 1 - c
                R[r, c_mirror] = x[r, c]
    elif axis == "h":
        # Horizontal reflection (across horizontal center line)
        for r in range(H):
            for c in range(W):
                r_mirror = H - 1 - r
                R[r_mirror, c] = x[r, c]
    elif axis == "diag":
        # Main diagonal reflection (top-left to bottom-right)
        for r in range(H):
            for c in range(W):
                R[c, r] = x[r, c]
    elif axis == "anti":
        # Anti-diagonal reflection (top-right to bottom-left)
        for r in range(H):
            for c in range(W):
                r_mirror = W - 1 - c
                c_mirror = H - 1 - r
                R[r_mirror, c_mirror] = x[r, c]

    return R


def _reflect_largest(x: Grid, axis: str, bg: int) -> Grid:
    """Reflect only the largest component."""
    objs = components(x, bg=bg)

    if not objs:
        # No components - return background
        return np.full_like(x, bg)

    # Find largest component (deterministic tie-breaking)
    largest = max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))

    # Create a grid with only the largest component
    x_largest = np.full_like(x, bg)
    for r, c in largest.pixels:
        x_largest[r, c] = x[r, c]

    # Reflect the largest component
    R_largest = _reflect_global(x_largest, axis, bg)

    return R_largest


def _reflect_per_object(x: Grid, axis: str, bg: int) -> Grid:
    """Reflect each object around its own center."""
    objs = components(x, bg=bg)

    if not objs:
        return np.full_like(x, bg)

    R = np.full_like(x, bg)
    H, W = x.shape

    for obj in objs:
        # Get bounding box
        r0, c0, r1, c1 = obj.bbox
        bbox_h = r1 - r0 + 1
        bbox_w = c1 - c0 + 1

        # For diagonal reflection, skip non-square objects
        if axis in ["diag", "anti"] and bbox_h != bbox_w:
            # Skip this object for diagonal reflection (non-square bbox)
            continue

        # Extract object in bbox
        obj_grid = np.full((bbox_h, bbox_w), bg, dtype=x.dtype)
        for r, c in obj.pixels:
            obj_grid[r - r0, c - c0] = x[r, c]

        # Reflect object within its bbox
        obj_reflected = _reflect_global(obj_grid, axis, bg)

        # Place reflected object back, preserving original pixels
        for r_local in range(bbox_h):
            for c_local in range(bbox_w):
                r_global = r_local + r0
                c_global = c_local + c0
                if 0 <= r_global < H and 0 <= c_global < W:
                    if obj_reflected[r_local, c_local] != bg:
                        # Only write if target is currently bg (don't overwrite original)
                        if R[r_global, c_global] == bg:
                            R[r_global, c_global] = obj_reflected[r_local, c_local]

    return R


def unify_SYMMETRY_COMPLETION(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    """
    Unifier for SYMMETRY_COMPLETION.

    Uses composition-safe gate (preserves_y + compatible_to_y).
    Prefers bg=None (per-input inference) over explicit bg values.

    Enumerates:
    - axis ∈ {"v", "h", "diag", "anti"}
    - scope ∈ {"global", "largest", "per_object"}
    - bg: Try None first, then {0..9}

    Returns:
        List of SYMMETRY_COMPLETION closures (0..N candidates)
        Empty list if no params work on ALL train pairs
    """
    from .closure_engine import preserves_y, compatible_to_y

    valid = []

    # Check if any train input is non-square
    has_nonsquare = any(x.shape[0] != x.shape[1] for x, _ in train)

    # Enumerate parameters
    if has_nonsquare:
        # Skip diagonal axes for non-square grids
        axes = ["v", "h"]
    else:
        axes = ["v", "h", "diag", "anti"]

    scopes = ["global", "largest", "per_object"]

    # Try bg=None first for each axis/scope combo
    for axis in axes:
        for scope in scopes:
            candidate_none = SYMMETRY_COMPLETION_Closure(
                f"SYMMETRY_COMPLETION[axis={axis},scope={scope},bg=None]",
                {"axis": axis, "scope": scope, "bg": None}
            )
            if preserves_y(candidate_none, train) and compatible_to_y(candidate_none, train):
                valid.append(candidate_none)

    # Fallback: enumerate explicit bgs
    for axis in axes:
        for scope in scopes:
            for bg in range(10):
                candidate = SYMMETRY_COMPLETION_Closure(
                    f"SYMMETRY_COMPLETION[axis={axis},scope={scope},bg={bg}]",
                    {"axis": axis, "scope": scope, "bg": bg}
                )
                if preserves_y(candidate, train) and compatible_to_y(candidate, train):
                    valid.append(candidate)

    return valid


# ==============================================================================
# M3.1: MOD_PATTERN
# ==============================================================================

class MOD_PATTERN_Closure(Closure):
    """
    Periodic pattern closure via modulo arithmetic.

    params = {
        "p": int,              # row period (2-6)
        "q": int,              # col period (2-6)
        "anchor": (int, int),  # anchor point (ar, ac)
        "class_map": Dict[Tuple[int,int], Set[int]]  # (i,j) -> allowed colors
    }

    Law: For cell (r,c), congruence class is ((r-ar) mod p, (c-ac) mod q).
         Cell (r,c) may only contain colors in class_map[(i,j)] where i,j are the class indices.
    """

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        """
        Apply periodic mask constraint (intersect only).

        Algorithm:
        1. Extract params: p, q, anchor (ar, ac), class_map
        2. For each cell (r,c):
           - Compute class indices: i = (r - ar) % p, j = (c - ac) % q
           - Allowed colors: class_map[(i,j)]
           - Build mask from allowed colors
           - Intersect U[r,c] with mask
        3. Return U_new (deterministic, monotone, shrinking)

        Contract:
        - Monotone & shrinking: U' = U & mask (only clear bits)
        - Deterministic: no RNG, no I/O, no wall-clock
        - <=2-pass idempotent on typical U (class masks are deterministic from params)
        - Masks derived from params only (not from x_input colors, just grid shape)
        - Does NOT use y; y only used in unifier for verification
        """
        p = self.params["p"]
        q = self.params["q"]
        ar, ac = self.params["anchor"]
        class_map = self.params["class_map"]

        H_in, W_in = x_input.shape
        U_new = U.copy()

        # CANVAS-AWARE: Loop through U dimensions, check x_input bounds
        for r in range(U.H):
            for c in range(U.W):
                # Skip cells outside x_input bounds (leave unchanged in U)
                if r >= H_in or c >= W_in:
                    continue  # MOD_PATTERN operates on x_input grid only

                # Compute congruence class
                i = (r - ar) % p
                j = (c - ac) % q

                # Get allowed colors for this class
                if (i, j) in class_map:
                    allowed_colors = class_map[(i, j)]
                else:
                    # Fallback: if class not in map, allow all colors (no constraint)
                    allowed_colors = set(range(10))

                # Build bitmask from allowed colors
                allowed_mask = set_to_mask(allowed_colors)

                # Intersect (only clear bits)
                U_new.intersect(r, c, allowed_mask)

        return U_new


def unify_MOD_PATTERN(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    """
    Unifier for MOD_PATTERN (composition-safe).

    Algorithm:
    1. Enumerate candidate anchors:
       - (0, 0) - grid origin
       - bbox corners of first train input (4 points)
       - quadrant origins: (0, W//2), (H//2, 0), (H//2, W//2)

    2. For each anchor, enumerate small periods:
       - p in range(2, 7)  # row period
       - q in range(2, 7)  # col period

    3. Build class_map from INPUT only:
       - For each train pair (x, y):
         - Compute congruence classes for x
         - For each class (i,j), collect colors that appear in x at positions with that class
       - Intersect class color sets across all train pairs (observer = observed)

    4. Composition-safe gates:
       - Check preserves_y(candidate, train): apply(singleton(y)) == singleton(y)
       - Check compatible_to_y(candidate, train): apply(singleton(x)) colors subset y colors
       - If both pass, collect candidate

    5. Return list of all valid candidates

    Returns:
        List of MOD_PATTERN closures (0..N candidates)
        Empty list if no (p,q,anchor) works on ALL train pairs
    """
    from .closure_engine import preserves_y, compatible_to_y

    valid = []

    # Step 1: Generate candidate anchors
    x0, y0 = train[0]
    H, W = x0.shape

    anchors = {(0, 0)}  # Grid origin

    # Add bbox corners from first train input (non-bg pixels)
    # Try all possible bg candidates
    for bg_candidate in range(10):
        objs = components(x0, bg=bg_candidate)
        if objs:
            # Get overall bbox
            r_min = min(o.bbox[0] for o in objs)
            c_min = min(o.bbox[1] for o in objs)
            r_max = max(o.bbox[2] for o in objs)
            c_max = max(o.bbox[3] for o in objs)

            anchors.add((r_min, c_min))
            anchors.add((r_min, c_max))
            anchors.add((r_max, c_min))
            anchors.add((r_max, c_max))

    # Add quadrant origins
    anchors.add((0, W // 2))
    anchors.add((H // 2, 0))
    anchors.add((H // 2, W // 2))

    # Step 2: Enumerate (p, q) in small range
    for ar, ac in sorted(anchors):
        for p in range(2, 7):
            for q in range(2, 7):
                # Step 3: Build class_map from INPUT
                # For each class (i,j), track allowed colors across ALL train pairs
                class_colors_per_pair = []

                for x, y in train:
                    H_cur, W_cur = x.shape
                    class_colors = {}  # (i,j) -> Set[int]

                    for r in range(H_cur):
                        for c in range(W_cur):
                            i = (r - ar) % p
                            j = (c - ac) % q

                            if (i, j) not in class_colors:
                                class_colors[(i, j)] = set()

                            # Add color from INPUT x
                            class_colors[(i, j)].add(int(x[r, c]))

                    class_colors_per_pair.append(class_colors)

                # Intersect class color sets across all pairs (unified params)
                # Start with first pair's class_colors
                class_map = {}
                for key in class_colors_per_pair[0]:
                    class_map[key] = class_colors_per_pair[0][key].copy()

                for pair_colors in class_colors_per_pair[1:]:
                    # Intersect each class
                    for (i, j) in list(class_map.keys()):
                        if (i, j) in pair_colors:
                            class_map[(i, j)] &= pair_colors[(i, j)]
                        else:
                            class_map[(i, j)] = set()  # No overlap

                # Skip if any class has empty allowed colors
                if any(len(colors) == 0 for colors in class_map.values()):
                    continue

                # Step 4: Build candidate closure
                params = {
                    "p": p,
                    "q": q,
                    "anchor": (ar, ac),
                    "class_map": class_map
                }

                candidate = MOD_PATTERN_Closure(
                    f"MOD_PATTERN[p={p},q={q},anchor=({ar},{ac})]",
                    params
                )

                # Step 5: Composition-safe gates
                if preserves_y(candidate, train) and compatible_to_y(candidate, train):
                    valid.append(candidate)

    return valid


# ==============================================================================
# M4.1: TILING / TILING_ON_MASK
# ==============================================================================

class TILING_Closure(Closure):
    """
    Tile motif across output canvas (global or masked).

    params = {
        "mode": "global" | "mask",
        "motif": np.ndarray[h,w],       # colors derived from x
        "anchor": Tuple[int,int],       # (ar, ac)
        "mask_template": Optional[str], # for mode="mask": "ALL" | "NONZERO" | "BACKGROUND" | "NOT_LARGEST" | "PARITY" | "STRIPES"
        "mask_params": Optional[Dict],  # e.g., {"anchor2": (0,1)} for PARITY, {"axis": "row", "k": 2} for STRIPES
    }

    Law: Tile motif across output lattice (H_out × W_out).
         If mode="mask", restrict writes to input-only mask.
         Intersect only (U & motif-bitmask at each covered cell).
    """

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        """
        Tile motif over output lattice (U.H × U.W).

        Algorithm:
        1. Extract params: mode, motif, anchor, mask_template, mask_params
        2. If mode="mask", compute mask_template(x_input)
        3. For each cell (r,c) in U:
           - Compute motif position: (mr, mc) = ((r - ar) % h, (c - ac) % w)
           - If mode="global" OR (mode="mask" AND mask[r,c]):
             - Intersect U[r,c] with {motif[mr, mc]}
           - Else: no change
        4. Return U_new (deterministic, monotone, shrinking)

        Contract:
        - Monotone & shrinking: U' = U & mask (only clear bits)
        - Deterministic: no RNG, no I/O, no wall-clock
        - ≤2-pass idempotent
        - Output-canvas aware: operates on U.H × U.W (from fixed-point engine)
        - Masks are input-only: computed from x_input only
        """
        mode = self.params["mode"]
        motif = self.params["motif"]
        ar, ac = self.params["anchor"]
        mask_template = self.params.get("mask_template")
        mask_params = self.params.get("mask_params", {})

        h, w = motif.shape
        U_new = U.copy()

        # Compute mask if mode="mask"
        if mode == "mask":
            # Compute mask from x_input (input-only)
            mask = _compute_tiling_mask(x_input, mask_template, mask_params)
        else:
            # mode="global": no mask (all cells allowed)
            mask = None

        # Tile motif across output canvas (U.H × U.W)
        for r in range(U.H):
            for c in range(U.W):
                # Check if this cell should be written with motif
                if mode == "mask":
                    # Only write motif if within x_input bounds and mask is True
                    if r >= x_input.shape[0] or c >= x_input.shape[1]:
                        continue  # Outside input bounds, skip
                    if not mask[r, c]:
                        # Mask is False: preserve input color
                        if r < x_input.shape[0] and c < x_input.shape[1]:
                            input_color = int(x_input[r, c])
                            input_mask = color_to_mask(input_color)
                            U_new.intersect(r, c, input_mask)
                        continue
                # mode="global": write to all cells

                # Compute motif position (with wrapping)
                mr = (r - ar) % h
                mc = (c - ac) % w

                # Get motif color
                motif_color = int(motif[mr, mc])
                motif_mask = color_to_mask(motif_color)

                # Intersect (only clear bits)
                U_new.intersect(r, c, motif_mask)

        return U_new


def _compute_tiling_mask(x: Grid, template: str, params: dict) -> np.ndarray:
    """
    Compute boolean mask for TILING_ON_MASK.

    Templates:
    - ALL: mask = all True
    - NONZERO: mask = (x != bg)
    - BACKGROUND: mask = (x == bg)
    - NOT_LARGEST: mask = (not in largest component)
    - PARITY: mask = checkerboard pattern from anchor2
    - STRIPES: mask = repeating stripes along axis

    Returns:
        Boolean mask (H×W array)
    """
    H, W = x.shape

    if template == "ALL":
        return np.ones((H, W), dtype=bool)

    elif template == "NONZERO":
        bg = params.get("bg", 0)
        return (x != bg)

    elif template == "BACKGROUND":
        bg = params.get("bg", 0)
        return (x == bg)

    elif template == "NOT_LARGEST":
        bg = params.get("bg", 0)
        objs = components(x, bg=bg)
        if not objs:
            # No components - all pixels are background
            return np.ones((H, W), dtype=bool)

        # Find largest component
        largest = max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))
        largest_pixels = set(largest.pixels)

        # Mask is True where NOT in largest
        mask = np.ones((H, W), dtype=bool)
        for r, c in largest_pixels:
            mask[r, c] = False

        return mask

    elif template == "PARITY":
        # Checkerboard pattern from anchor2
        ar2, ac2 = params.get("anchor2", (0, 0))
        mask = np.zeros((H, W), dtype=bool)
        for r in range(H):
            for c in range(W):
                # Parity: (r + c) % 2 == (ar2 + ac2) % 2
                if (r + c) % 2 == (ar2 + ac2) % 2:
                    mask[r, c] = True
        return mask

    elif template == "STRIPES":
        # Repeating stripes along axis
        axis = params.get("axis", "row")
        k = params.get("k", 2)
        mask = np.zeros((H, W), dtype=bool)

        if axis == "row":
            # Horizontal stripes (vary by row)
            for r in range(H):
                if (r // k) % 2 == 0:
                    mask[r, :] = True
        elif axis == "col":
            # Vertical stripes (vary by column)
            for c in range(W):
                if (c // k) % 2 == 0:
                    mask[:, c] = True

        return mask

    else:
        raise ValueError(f"Unknown mask template: {template}")


def unify_TILING(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    """
    Unifier for TILING / TILING_ON_MASK (composition-safe).

    Algorithm:
    1. For each train pair, compute residual M = (x != y) in overlapping region
    2. Candidate enumeration:
       - anchors ∈ {(0,0), bbox_corners(x), quadrant_origins(x)}
       - Infer motif (h,w) from y's periodicity or M's structure
       - Derive motif colors from x (Option A: canvas mapper φ; Option B: mode color per tile)
       - mode="global" and mode="mask" with mask_template ∈ {ALL, NONZERO, BACKGROUND, NOT_LARGEST, PARITY, STRIPES}
    3. For each candidate:
       - Check patch-exact on M, preserves_y, compatible_to_y
       - If all checks pass, append candidate
    4. Return list of all passing candidates (may be 0..N)

    Returns:
        List of TILING closures (0..N candidates)
        Empty list if no params work on ALL train pairs
    """
    from .closure_engine import preserves_y, compatible_to_y, run_fixed_point, _compute_canvas

    valid = []

    # Step 1: Enumerate anchors
    x0, y0 = train[0]
    H_in, W_in = x0.shape
    H_out, W_out = y0.shape

    anchors = {(0, 0)}  # Grid origin

    # Add bbox corners from first train input
    for bg_candidate in range(10):
        objs = components(x0, bg=bg_candidate)
        if objs:
            r_min = min(o.bbox[0] for o in objs)
            c_min = min(o.bbox[1] for o in objs)
            r_max = max(o.bbox[2] for o in objs)
            c_max = max(o.bbox[3] for o in objs)

            anchors.add((r_min, c_min))
            anchors.add((r_min, c_max))
            anchors.add((r_max, c_min))
            anchors.add((r_max, c_max))

    # Add quadrant origins
    anchors.add((0, W_in // 2))
    anchors.add((H_in // 2, 0))
    anchors.add((H_in // 2, W_in // 2))

    # Limit to max 8 anchors
    anchors = sorted(anchors)[:8]

    # Step 2: Enumerate motif sizes (small range)
    # Use fixed constant to avoid depending on output dimensions
    MAX_MOTIF_SIZE = 6
    motif_sizes = []
    for h in range(1, MAX_MOTIF_SIZE + 1):
        for w in range(1, MAX_MOTIF_SIZE + 1):
            motif_sizes.append((h, w))

    # Step 3: Enumerate modes and templates
    modes_and_templates = [("global", None, {})]
    for template in ["ALL", "NONZERO", "BACKGROUND", "NOT_LARGEST"]:
        modes_and_templates.append(("mask", template, {}))
    # PARITY with anchor2 variants
    for ar2, ac2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        modes_and_templates.append(("mask", "PARITY", {"anchor2": (ar2, ac2)}))
    # STRIPES with axis and k variants
    for axis in ["row", "col"]:
        for k in [2, 3]:
            modes_and_templates.append(("mask", "STRIPES", {"axis": axis, "k": k}))

    # Step 4: Enumerate candidates
    for ar, ac in anchors:
        for h, w in motif_sizes:
            # Enumerate modes
            for mode, mask_template, mask_params in modes_and_templates:
                # Derive motif based on mode
                motif = np.zeros((h, w), dtype=int)

                if mode == "global":
                    # For global mode: derive from input x0
                    for mr in range(h):
                        for mc in range(w):
                            r_in = (ar + mr) % H_in
                            c_in = (ac + mc) % W_in
                            motif[mr, mc] = int(x0[r_in, c_in])
                else:
                    # For mask mode: infer from INPUT x0 on the mask
                    # Derive motif colors from x (input-only), use y only for verification
                    try:
                        mask = _compute_tiling_mask(x0, mask_template, mask_params)
                        # Collect colors from x0 (INPUT) where mask is True
                        colors_on_mask = []
                        for r in range(min(H_in, mask.shape[0])):
                            for c in range(min(W_in, mask.shape[1])):
                                if mask[r, c]:
                                    colors_on_mask.append(int(x0[r, c]))

                        if not colors_on_mask:
                            continue  # Empty mask, skip

                        # Use mode color from x0 on mask (deterministic tie-breaking)
                        from collections import Counter
                        counts = Counter(colors_on_mask)
                        mode_color = min(counts.keys(), key=lambda c: (-counts[c], c))

                        # Fill motif with mode color
                        motif.fill(mode_color)
                    except:
                        continue  # Skip if mask computation fails
                # Build candidate params
                params = {
                    "mode": mode,
                    "motif": motif,
                    "anchor": (ar, ac),
                    "mask_template": mask_template,
                    "mask_params": mask_params
                }

                candidate = TILING_Closure(
                    f"TILING[mode={mode},anchor=({ar},{ac}),h={h},w={w},template={mask_template}]",
                    params
                )

                # Step 5: Composition-safe gates
                try:
                    if not (preserves_y(candidate, train) and compatible_to_y(candidate, train)):
                        continue

                    # Additional check: patch-exact on M
                    # For efficiency, only check patch-exactness if gates pass
                    patch_exact = True
                    for x, y in train:
                        # Compute canvas
                        canvas_candidates = unify_CANVAS_SIZE(train)
                        canvas_params = canvas_candidates[0].params if canvas_candidates else None
                        canvas = _compute_canvas(x, canvas_params) if canvas_params else {"H": y.shape[0], "W": y.shape[1]}

                        # Run fixed-point with CANVAS_SIZE + candidate
                        closures_to_test = canvas_candidates + [candidate] if canvas_candidates else [candidate]
                        U_final, _ = run_fixed_point(closures_to_test, x, canvas=canvas)

                        # Check if fully determined
                        if not U_final.is_fully_determined():
                            patch_exact = False
                            break

                        y_pred = U_final.to_grid_deterministic(fallback='lowest', bg=0)

                        # Check exact match with y (train exactness)
                        if not np.array_equal(y_pred, y):
                            patch_exact = False
                            break

                    if patch_exact:
                        valid.append(candidate)
                        # Early exit after finding enough good candidates
                        # (but allow more for complex cases)
                        if len(valid) >= 10:
                            return valid
                except Exception:
                    # Skip candidates that fail gates or patch-exact check
                    continue

    return valid


# ==============================================================================
# NX-UNSTICK: COLOR_PERM
# ==============================================================================

class COLOR_PERM_Closure(Closure):
    """
    Global color permutation.

    params = {"perm": Dict[int, int]}  # bijective color mapping

    Law: y[r,c] = π(x[r,c]) for all cells, where π is the permutation
    """

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        """
        Apply color permutation (intersect only).

        For each cell (r,c):
        - Get x_color = x_input[r,c]
        - If x_color in perm: target = perm[x_color]
        - Intersect U[r,c] with {target}

        Returns U_new (monotone, shrinking, deterministic)
        """
        perm = self.params["perm"]
        U_new = U.copy()

        # CANVAS-AWARE: Only process cells within x_input bounds
        H_in, W_in = x_input.shape

        for r in range(U.H):
            for c in range(U.W):
                # Skip cells outside x_input bounds (leave unchanged in U)
                if r >= H_in or c >= W_in:
                    continue

                x_color = int(x_input[r, c])
                if x_color in perm:
                    target = perm[x_color]
                    target_mask = color_to_mask(target)
                    U_new.intersect(r, c, target_mask)
                # If x_color not in perm, leave U unchanged (identity on that color)

        return U_new


def unify_COLOR_PERM(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    """
    Unifier for COLOR_PERM (composition-safe).

    Algorithm:
    1. Check for shape mismatches (skip shape-changing tasks)
    2. Extract color mappings from all train pairs
    3. Ensure bijective (no collisions)
    4. Build candidate closure
    5. Check preserves_y + compatible_to_y
    6. Return [candidate] if valid, else []

    Returns:
        List with 0 or 1 COLOR_PERM closure
        Empty list if not bijective or gates fail or shapes don't match
    """
    from .closure_engine import preserves_y, compatible_to_y

    # SHAPE GUARD: Skip if any train pair has different shapes
    for x, y in train:
        if x.shape != y.shape:
            return []  # Shape-changing task, skip COLOR_PERM

    # Step 1: Build color mapping from all train pairs
    color_map = {}  # x_color -> y_color

    for x, y in train:
        H, W = x.shape
        for r in range(H):
            for c in range(W):
                x_c = int(x[r, c])
                y_c = int(y[r, c])

                if x_c in color_map:
                    # Check consistency
                    if color_map[x_c] != y_c:
                        return []  # Contradiction
                else:
                    color_map[x_c] = y_c

    # Step 2: Ensure bijective (no two x_colors map to same y_color)
    used_y_colors = list(color_map.values())
    if len(set(used_y_colors)) != len(used_y_colors):
        return []  # Not bijective

    # Step 3: Build candidate closure
    candidate = COLOR_PERM_Closure(
        f"COLOR_PERM[perm={color_map}]",
        {"perm": color_map}
    )

    # Step 4: Check composition-safe gates
    if preserves_y(candidate, train) and compatible_to_y(candidate, train):
        return [candidate]

    return []


# ==============================================================================
# NX-UNSTICK: RECOLOR_ON_MASK
# ==============================================================================

class RECOLOR_ON_MASK_Closure(Closure):
    """
    Recolor pixels matching template to target strategy.

    params = {
        "template": str,      # "NONZERO"|"BACKGROUND"|"NOT_LARGEST"
        "template_params": dict,
        "strategy": str,      # "CONST"|"MODE_ON_MASK"
        "strategy_params": dict,
        "bg": int | None
    }

    Law: For pixels where template(x) is True, constrain to strategy(x, template(x))
    """

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        """
        Apply recolor constraint (intersect only).

        Algorithm:
        1. Compute template mask T from x_input
        2. Compute target color from strategy + x_input
        3. For each cell (r,c):
           - If T[r,c]: intersect U[r,c] with {target_color}
           - Else: no change

        Returns U_new (monotone, shrinking, deterministic)
        """
        template = self.params["template"]
        template_params = self.params.get("template_params", {})
        strategy = self.params["strategy"]
        strategy_params = self.params.get("strategy_params", {})
        bg = self.params.get("bg")

        # Infer bg if None
        if bg is None:
            bg = infer_bg_from_border(x_input)

        # Compute template mask
        T = _compute_template_mask(x_input, template, template_params, bg)

        # Compute target color
        target_color = _compute_target_color(x_input, T, strategy, strategy_params, bg)

        # Apply mask
        U_new = U.copy()
        target_mask = color_to_mask(target_color)

        # CANVAS-AWARE: Only process cells within x_input/T bounds
        H_in, W_in = x_input.shape

        for r in range(U.H):
            for c in range(U.W):
                # Skip cells outside x_input bounds (leave unchanged in U)
                if r >= H_in or c >= W_in:
                    continue

                if T[r, c]:
                    U_new.intersect(r, c, target_mask)

        return U_new


def _compute_template_mask(x: Grid, template: str, params: dict, bg: int) -> np.ndarray:
    """
    Compute boolean mask for template.

    Templates:
    - NONZERO: mask = (x != bg)
    - BACKGROUND: mask = (x == bg)
    - NOT_LARGEST: mask = (not in largest component)

    Returns:
        Boolean mask (H×W array)
    """
    H, W = x.shape

    if template == "NONZERO":
        return (x != bg)

    elif template == "BACKGROUND":
        return (x == bg)

    elif template == "NOT_LARGEST":
        objs = components(x, bg=bg)
        if not objs:
            # No components - all pixels are background
            return np.ones((H, W), dtype=bool)

        # Find largest component
        largest = max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))
        largest_pixels = set(largest.pixels)

        # Mask is True where NOT in largest
        mask = np.ones((H, W), dtype=bool)
        for r, c in largest_pixels:
            mask[r, c] = False

        return mask

    else:
        raise ValueError(f"Unknown template: {template}")


def _compute_target_color(x: Grid, T: np.ndarray, strategy: str, params: dict, bg: int) -> int:
    """
    Compute target color for recoloring.

    Strategies:
    - CONST: target = params["c"]
    - MODE_ON_MASK: target = most frequent color in x where T is True

    Returns:
        Target color (int 0-9)
    """
    if strategy == "CONST":
        return int(params["c"])

    elif strategy == "MODE_ON_MASK":
        # Get all colors where mask is True
        colors_on_mask = []
        H, W = x.shape
        for r in range(H):
            for c in range(W):
                if T[r, c]:
                    colors_on_mask.append(int(x[r, c]))

        if not colors_on_mask:
            # Empty mask - fallback to bg
            return bg

        # Mode with deterministic tie-breaking (lowest color)
        counts = Counter(colors_on_mask)
        mode_color = min(counts.keys(), key=lambda c: (-counts[c], c))
        return int(mode_color)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def unify_RECOLOR_ON_MASK(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    """
    Unifier for RECOLOR_ON_MASK (composition-safe).

    Algorithm:
    1. Check for shape mismatches (skip shape-changing tasks)
    2. Enumerate templates: NONZERO, BACKGROUND, NOT_LARGEST
    3. For each template, check if template(x) matches residual mask M=(x!=y)
    4. If yes, enumerate strategies: CONST(0-9), MODE_ON_MASK
    5. Check if strategy produces exactly y's colors on M
    6. Build candidate; check preserves_y + compatible_to_y
    7. Return list of valid closures

    Returns:
        List of RECOLOR_ON_MASK closures (0..N candidates)
        Empty list if no template/strategy works or shapes don't match
    """
    from .closure_engine import preserves_y, compatible_to_y

    # SHAPE GUARD: Skip if any train pair has different shapes
    for x, y in train:
        if x.shape != y.shape:
            return []  # Shape-changing task, skip RECOLOR_ON_MASK

    valid = []

    # Enumerate templates
    templates = [
        ("NONZERO", {}),
        ("BACKGROUND", {}),
        ("NOT_LARGEST", {})
    ]

    # Try bg=None first, then explicit bgs
    bg_candidates = [None] + list(range(10))

    for bg_cand in bg_candidates:
        for template_name, template_params in templates:
            # Check if template matches residual mask for ALL train pairs
            template_matches = True
            inferred_bg = bg_cand  # Will be set per-input if None

            for x, y in train:
                # Infer bg if None
                if bg_cand is None:
                    inferred_bg = infer_bg_from_border(x)
                else:
                    inferred_bg = bg_cand

                # Compute template mask
                try:
                    T = _compute_template_mask(x, template_name, template_params, inferred_bg)
                except:
                    template_matches = False
                    break

                # Compute residual mask
                M = (x != y)

                # Check if template matches residual
                if not np.array_equal(T, M):
                    template_matches = False
                    break

            if not template_matches:
                continue

            # Template matches! Now enumerate strategies
            strategies = [("MODE_ON_MASK", {})]
            for c in range(10):
                strategies.append(("CONST", {"c": c}))

            for strategy_name, strategy_params in strategies:
                # Check if strategy produces correct colors for ALL train pairs
                strategy_works = True

                for x, y in train:
                    # Infer bg if None
                    if bg_cand is None:
                        inferred_bg = infer_bg_from_border(x)
                    else:
                        inferred_bg = bg_cand

                    # Compute template mask
                    T = _compute_template_mask(x, template_name, template_params, inferred_bg)

                    # Compute target color
                    try:
                        target_color = _compute_target_color(x, T, strategy_name, strategy_params, inferred_bg)
                    except:
                        strategy_works = False
                        break

                    # Check if target_color matches y on mask
                    H, W = x.shape
                    for r in range(H):
                        for c in range(W):
                            if T[r, c]:
                                if int(y[r, c]) != target_color:
                                    strategy_works = False
                                    break
                        if not strategy_works:
                            break

                    if not strategy_works:
                        break

                if not strategy_works:
                    continue

                # Build candidate closure
                candidate = RECOLOR_ON_MASK_Closure(
                    f"RECOLOR_ON_MASK[template={template_name},strategy={strategy_name},bg={bg_cand}]",
                    {
                        "template": template_name,
                        "template_params": template_params,
                        "strategy": strategy_name,
                        "strategy_params": strategy_params,
                        "bg": bg_cand
                    }
                )

                # Check composition-safe gates
                if preserves_y(candidate, train) and compatible_to_y(candidate, train):
                    valid.append(candidate)

    return valid


# ==============================================================================
# CANVAS-AWARE: CANVAS_SIZE
# ==============================================================================

class CANVAS_SIZE_Closure(Closure):
    """
    Canvas size inference closure (metadata-only, no-op on apply).

    params = {
        "strategy": str,    # "TILE_MULTIPLE" (only strategy)
        "k_h": int,         # row multiplier for TILE_MULTIPLE
        "k_w": int          # col multiplier for TILE_MULTIPLE
    }

    Law: All train outputs satisfy H_out = k_h * H_in, W_out = k_w * W_in
         where k_h and k_w are constant across all train pairs.
    """

    def __init__(self, name: str, params: Dict):
        super().__init__(name, params, is_meta=True)

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        """
        Identity/no-op - this closure only carries metadata for the engine.

        Returns U unchanged (trivially monotone, shrinking, idempotent).
        """
        return U


def unify_CANVAS_SIZE(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    """
    Unifier for CANVAS_SIZE (infer output shape from train pairs).

    Strategy: TILE_MULTIPLE only
    - H_out = k_h * H_in, W_out = k_w * W_in (constant k_h, k_w across all train pairs)
    - Stores only multipliers (k_h, k_w) - parametric and generalizable
    - Same-shape tasks are handled by k_h=k_w=1

    Returns:
        List with 0 or 1 CANVAS_SIZE closure
        Empty list if TILE_MULTIPLE doesn't fit or gates fail
    """
    from .closure_engine import preserves_y, compatible_to_y

    if not train:
        return []

    # TILE_MULTIPLE: H_out = k_h * H_in, W_out = k_w * W_in
    # Check if all pairs satisfy constant multiplier
    multipliers = []
    for x, y in train:
        H_in, W_in = x.shape
        H_out, W_out = y.shape
        if H_in == 0 or W_in == 0:
            multipliers.append(None)
            continue
        # Check if H_out is integer multiple of H_in
        if H_out % H_in == 0 and W_out % W_in == 0:
            k_h = H_out // H_in
            k_w = W_out // W_in
            multipliers.append((k_h, k_w))
        else:
            multipliers.append(None)

    # Check if all pairs have same non-None multiplier
    if all(m is not None for m in multipliers) and len(set(multipliers)) == 1:
        k_h, k_w = multipliers[0]
        # H, W are computed per-input by _compute_canvas (closure_engine.py)
        candidate = CANVAS_SIZE_Closure(
            f"CANVAS_SIZE[strategy=TILE_MULTIPLE,k_h={k_h},k_w={k_w}]",
            {"strategy": "TILE_MULTIPLE", "k_h": k_h, "k_w": k_w}
        )
        if preserves_y(candidate, train) and compatible_to_y(candidate, train):
            return [candidate]

    # No strategy fits
    return []


# ==============================================================================
# M4.2: COPY_BY_DELTAS (Stub - to be implemented)
# ==============================================================================

class COPY_BY_DELTAS_Closure(Closure):
    """
    Copy template to shifted positions (stub for M4.2).

    params = {
        "template_strategy": str,  # "smallest_object" | "largest_object" | etc.
        "deltas": List[Tuple[int,int]],  # list of (dr, dc) shifts
        "mode": str  # "strict" | "union"
    }

    Law: Copy template T (derived from x) to positions T+delta for each delta in deltas.
    """

    def apply(self, U: SetValuedGrid, x_input: Grid) -> SetValuedGrid:
        """Stub implementation - returns U unchanged."""
        return U


def unify_COPY_BY_DELTAS(train: List[Tuple[Grid, Grid]]) -> List[Closure]:
    """
    Unifier for COPY_BY_DELTAS (stub for M4.2).

    Returns:
        Empty list (not yet implemented)
    """
    return []
