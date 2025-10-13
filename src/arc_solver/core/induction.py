#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARC Solver - Induction Routines"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from .types import Grid
from .invariants import exact_equals
from ..operators.symmetry import ROT, FLIP
from ..operators.spatial import CROP, BBOX, CROP_BBOX_NONZERO
from ..operators.masks import KEEP, MASK_NONZERO

@dataclass
class Rule:
    """A transformation rule with parameters."""
    name: str
    params: dict
    prog: Callable[[Grid], Grid]

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

# Try rules in order of simplicity (Occam's razor)
CATALOG = [
    induce_symmetry_rule,
    induce_crop_nonzero_rule,
    induce_keep_nonzero_rule,
]

def induce_rule(train: List[Tuple[Grid, Grid]]) -> Optional[Rule]:
    """Try all induction routines in catalog order."""
    for induce_fn in CATALOG:
        rule = induce_fn(train)
        if rule is not None:
            return rule
    return None
