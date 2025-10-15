"""
Closure implementations for ARC Solver.

Each closure:
1. Inherits from Closure base class
2. Implements apply(U, x_input) - monotone & shrinking
3. Has corresponding unifier function to learn params from train pairs
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .types import Grid, Operator
from .utils import equal, components
from .closure_engine import Closure, SetValuedGrid, color_to_mask, set_to_mask


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
        # Fail loudly if bg not provided (unifier MUST set it)
        bg = self.params["bg"]
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

    Tries all possible bg values {0-9} and returns closures for those that verify.

    Returns:
        List of KEEP_LARGEST closures (one per valid bg value)
        Empty list if no bg value works
    """
    from .closure_engine import run_fixed_point, verify_closures_on_train

    valid_closures = []

    # Try all possible background colors
    for bg in range(10):
        candidate = KEEP_LARGEST_COMPONENT_Closure(
            f"KEEP_LARGEST_COMPONENT[bg={bg}]",
            {"bg": bg}
        )
        if verify_closures_on_train([candidate], train):
            valid_closures.append(candidate)

    return valid_closures


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

    Enumerates:
    - mode="outer" (fixed for M1)
    - scope ∈ {"largest", "all"}
    - bg ∈ {0..9}

    For each candidate:
    - Build closure
    - Call verify_closures_on_train([candidate], train)
    - If exact on all pairs, collect candidate

    Returns:
        List of OUTLINE_OBJECTS closures (usually 0 or 1)
        Empty list if no params work
    """
    from .closure_engine import verify_closures_on_train

    valid_closures = []

    # Enumerate parameters
    mode = "outer"  # Fixed for M1
    scopes = ["largest", "all"]
    bgs = range(10)

    for scope in scopes:
        for bg in bgs:
            candidate = OUTLINE_OBJECTS_Closure(
                f"OUTLINE_OBJECTS[mode={mode},scope={scope},bg={bg}]",
                {"mode": mode, "scope": scope, "bg": bg}
            )
            if verify_closures_on_train([candidate], train):
                valid_closures.append(candidate)

    return valid_closures


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

    Enumerates:
    - mode ∈ {"open", "close"}
    - bg ∈ {0..9}

    For each candidate:
    - Build closure
    - Call verify_closures_on_train([candidate], train)
    - If exact on all pairs, collect candidate

    Returns:
        List of OPEN_CLOSE closures (usually 0 or 1)
        Empty list if no params work
    """
    from .closure_engine import verify_closures_on_train

    valid_closures = []

    # Enumerate parameters
    modes = ["open", "close"]
    bgs = range(10)

    for mode in modes:
        for bg in bgs:
            candidate = OPEN_CLOSE_Closure(
                f"OPEN_CLOSE[mode={mode},bg={bg}]",
                {"mode": mode, "bg": bg}
            )
            if verify_closures_on_train([candidate], train):
                valid_closures.append(candidate)

    return valid_closures
