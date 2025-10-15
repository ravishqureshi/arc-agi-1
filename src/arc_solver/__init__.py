"""
ARC Solver - Universe Intelligence Approach

Modular implementation of inducer-based operator discovery for ARC-AGI.
"""

from .types import Grid, Operator, ARCInstance, Node
from .utils import (
    G, equal, residual, inb,
    bbox_nonzero, components, rank_by_size,
    task_sha, program_sha, closure_set_sha,
    compute_palette_delta, compute_component_delta,
    log_receipt
)
from .operators import (
    ROT, FLIP, COLOR_PERM,
    CROP_BBOX_NONZERO, KEEP_NONZERO, KEEP_LARGEST_COMPONENT,
    RECOLOR_PARITY_CONST, PARITY_MASK,
    REPEAT_TILE, extract_motif,
    HOLE_FILL_ALL, fill_holes_in_bbox,
    COPY_OBJ_RANK_BY_DELTAS
)
from .inducers import (
    induce_COLOR_PERM,
    induce_ROT_FLIP,
    induce_CROP_KEEP,
    induce_KEEP_LARGEST,
    induce_PARITY_CONST,
    induce_TILING_AND_MASK,
    induce_HOLE_FILL,
    induce_COPY_BY_DELTAS
)
from .search import (
    autobuild_operators,
    beam_search,
    solve_with_beam,
    solve_with_closures
)
from .closure_engine import (
    SetValuedGrid, Closure, run_fixed_point,
    verify_closures_on_train, init_top, init_from_grid
)
from .closures import (
    KEEP_LARGEST_COMPONENT_Closure,
    unify_KEEP_LARGEST
)

__all__ = [
    # Types
    'Grid', 'Operator', 'ARCInstance', 'Node',

    # Utils
    'G', 'equal', 'residual', 'inb',
    'bbox_nonzero', 'components', 'rank_by_size',
    'task_sha', 'program_sha', 'closure_set_sha',
    'compute_palette_delta', 'compute_component_delta',
    'log_receipt',

    # Operators
    'ROT', 'FLIP', 'COLOR_PERM',
    'CROP_BBOX_NONZERO', 'KEEP_NONZERO', 'KEEP_LARGEST_COMPONENT',
    'RECOLOR_PARITY_CONST', 'PARITY_MASK',
    'REPEAT_TILE', 'extract_motif',
    'HOLE_FILL_ALL', 'fill_holes_in_bbox',
    'COPY_OBJ_RANK_BY_DELTAS',

    # Inducers
    'induce_COLOR_PERM',
    'induce_ROT_FLIP',
    'induce_CROP_KEEP',
    'induce_KEEP_LARGEST',
    'induce_PARITY_CONST',
    'induce_TILING_AND_MASK',
    'induce_HOLE_FILL',
    'induce_COPY_BY_DELTAS',

    # Search
    'autobuild_operators',
    'beam_search',
    'solve_with_beam',
    'solve_with_closures',

    # Closure Engine
    'SetValuedGrid', 'Closure', 'run_fixed_point',
    'verify_closures_on_train', 'init_top', 'init_from_grid',

    # Closures
    'KEEP_LARGEST_COMPONENT_Closure',
    'unify_KEEP_LARGEST',
]
