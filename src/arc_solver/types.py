"""
Type definitions and dataclasses for ARC Solver.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable

# Grid type
Grid = np.ndarray


@dataclass
class Obj:
    """Object representation from connected components."""
    color: int
    pixels: List[Tuple[int, int]]
    size: int
    bbox: Tuple[int, int, int, int]  # (r0, c0, r1, c1)
    centroid: Tuple[float, float]  # (r_mean, c_mean)


@dataclass
class Operator:
    """
    Operator = (name, params, prog, pce)

    - name: operator name (e.g., "COLOR_PERM")
    - params: dictionary of parameters
    - prog: Grid → Grid function
    - pce: human-readable description
    """
    name: str
    params: Dict
    prog: Callable[[Grid], Grid]
    pce: str


@dataclass
class ARCInstance:
    """ARC task instance."""
    name: str
    train: List[Tuple[Grid, Grid]]  # Training pairs
    test_in: List[Grid]  # Test inputs
    test_out: List[Grid]  # Test outputs (for validation)


@dataclass
class Node:
    """Beam search node."""
    prog: Callable[[Grid], Grid]
    steps: List[Operator]
    res_list: List[int]  # Residuals for each train pair
    total: int  # Sum of residuals
    depth: int
