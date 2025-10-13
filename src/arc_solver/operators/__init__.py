"""ARC Solver - DSL Operators"""

from .symmetry import ROT, FLIP
from .spatial import BBOX, CROP, CROP_BBOX_NONZERO
from .masks import MASK_COLOR, MASK_NONZERO, KEEP, REMOVE
from .composition import ON, SEQ

__all__ = [
    # Symmetry
    'ROT', 'FLIP',
    # Spatial
    'BBOX', 'CROP', 'CROP_BBOX_NONZERO',
    # Masks
    'MASK_COLOR', 'MASK_NONZERO', 'KEEP', 'REMOVE',
    # Composition
    'ON', 'SEQ',
]
