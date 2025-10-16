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
        for r in range(U.H):
            for c in range(U.W):
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

        for r in range(U.H):
            for c in range(U.W):
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
        List of KEEP_LARGEST closures (usually 0 or 1)
        Empty list if no bg value works
    """
    from .closure_engine import preserves_y, compatible_to_y

    # Try bg=None first (per-input inference)
    candidate_none = KEEP_LARGEST_COMPONENT_Closure(
        "KEEP_LARGEST_COMPONENT[bg=None]",
        {"bg": None}
    )
    if preserves_y(candidate_none, train) and compatible_to_y(candidate_none, train):
        return [candidate_none]

    # Fallback: enumerate explicit bgs
    for bg in range(10):
        candidate = KEEP_LARGEST_COMPONENT_Closure(
            f"KEEP_LARGEST_COMPONENT[bg={bg}]",
            {"bg": bg}
        )
        if preserves_y(candidate, train) and compatible_to_y(candidate, train):
            return [candidate]

    return []


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

        for r in range(U.H):
            for c in range(U.W):
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
        List of OUTLINE_OBJECTS closures (usually 0 or 1)
        Empty list if no params work
    """
    from .closure_engine import preserves_y, compatible_to_y

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
            return [candidate_none]

    # Fallback: enumerate explicit bgs
    for scope in scopes:
        for bg in range(10):
            candidate = OUTLINE_OBJECTS_Closure(
                f"OUTLINE_OBJECTS[mode={mode},scope={scope},bg={bg}]",
                {"mode": mode, "scope": scope, "bg": bg}
            )
            if preserves_y(candidate, train) and compatible_to_y(candidate, train):
                return [candidate]

    return []


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

        for r in range(U.H):
            for c in range(U.W):
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
        List of OPEN_CLOSE closures (usually 0 or 1)
        Empty list if no params work
    """
    from .closure_engine import preserves_y, compatible_to_y

    # Enumerate parameters
    modes = ["open", "close"]

    # Try bg=None first for each mode
    for mode in modes:
        candidate_none = OPEN_CLOSE_Closure(
            f"OPEN_CLOSE[mode={mode},bg=None]",
            {"mode": mode, "bg": None}
        )
        if preserves_y(candidate_none, train) and compatible_to_y(candidate_none, train):
            return [candidate_none]

    # Fallback: enumerate explicit bgs
    for mode in modes:
        for bg in range(10):
            candidate = OPEN_CLOSE_Closure(
                f"OPEN_CLOSE[mode={mode},bg={bg}]",
                {"mode": mode, "bg": bg}
            )
            if preserves_y(candidate, train) and compatible_to_y(candidate, train):
                return [candidate]

    return []


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
        List of AXIS_PROJECTION closures (usually 0 or 1)
        Empty list if no params work on ALL train pairs
    """
    from .closure_engine import preserves_y, compatible_to_y

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
                return [candidate_none]

    # Fallback: enumerate explicit bgs
    for axis in axes:
        for scope in scopes:
            for bg in range(10):
                candidate = AXIS_PROJECTION_Closure(
                    f"AXIS_PROJECTION[axis={axis},scope={scope},mode={mode},bg={bg}]",
                    {"axis": axis, "scope": scope, "mode": mode, "bg": bg}
                )
                if preserves_y(candidate, train) and compatible_to_y(candidate, train):
                    return [candidate]

    return []


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

        for r in range(U.H):
            for c in range(U.W):
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
        List of SYMMETRY_COMPLETION closures (usually 0 or 1)
        Empty list if no params work on ALL train pairs
    """
    from .closure_engine import preserves_y, compatible_to_y

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
                return [candidate_none]

    # Fallback: enumerate explicit bgs
    for axis in axes:
        for scope in scopes:
            for bg in range(10):
                candidate = SYMMETRY_COMPLETION_Closure(
                    f"SYMMETRY_COMPLETION[axis={axis},scope={scope},bg={bg}]",
                    {"axis": axis, "scope": scope, "bg": bg}
                )
                if preserves_y(candidate, train) and compatible_to_y(candidate, train):
                    return [candidate]

    return []