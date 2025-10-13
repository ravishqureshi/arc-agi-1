#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARC Solver - Main Solver Harness"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .types import Grid
from .invariants import exact_equals
from .receipts import Receipts, edit_counts, residual as compute_residual
from .induction import Rule, CATALOG

@dataclass
class ARCInstance:
    """ARC task instance with train/test splits."""
    name: str
    train: List[Tuple[Grid, Grid]]
    test_in: List[Grid]
    test_out: List[Grid]

@dataclass
class SolveResult:
    """Result of solving an ARC instance."""
    name: str
    rule: Optional[Rule]
    preds: List[Grid]
    receipts: List[Receipts]
    acc_exact: float

def pce_for_rule(rule: Rule) -> str:
    """Generate Proof-Carrying English explanation for a rule."""
    if rule is None:
        return "No rule matched."
    if rule.name == "ROT":
        degrees = rule.params['k'] * 90
        return f"Rotate grid by {degrees} degrees (exact symmetry fit on train)."
    if rule.name == "FLIP":
        axis_name = 'horizontal' if rule.params['axis'] == 'h' else 'vertical'
        return f"Flip grid along {axis_name} axis."
    if rule.name == "COLOR_PERM":
        mapping = rule.params['mapping']
        pairs = ', '.join(f"{k}→{v}" for k, v in sorted(mapping.items()))
        return f"Color permutation: {pairs} (learned from train, residual=0)."
    if rule.name == "RECOLOR_OBJ_RANK":
        grp = rule.params['group']
        r = rule.params['rank']
        target = "largest" if r == 0 else f"rank {r}"

        if grp == 'global':
            col = rule.params['color']
            return f"Recolor the {target} component globally to color {col} (verified on train: residual=0)."
        else:  # per_color
            color_map = rule.params['color_map']
            mappings = ', '.join(f"{k}→{v}" for k, v in sorted(color_map.items()))
            return f"Recolor the {target} component of each color: {mappings} (verified on train: residual=0)."
    if rule.name == "KEEP_OBJ_TOPK":
        k = rule.params['k']
        return f"Keep only the top-{k} component{'s' if k > 1 else ''} by size; zero others (verified on train: residual=0)."
    if rule.name == "MOVE_OBJ_RANK":
        grp = rule.params['group']
        r = rule.params['rank']
        delta = rule.params['delta']
        target = "largest" if r == 0 else f"rank {r}"
        scope = "globally" if grp == 'global' else "for each color"
        return f"MOVE the {target} object {scope} by Δ={delta} (verified on train: residual=0)."
    if rule.name == "COPY_OBJ_RANK":
        grp = rule.params['group']
        r = rule.params['rank']
        delta = rule.params['delta']
        target = "largest" if r == 0 else f"rank {r}"
        scope = "globally" if grp == 'global' else "for each color"
        return f"COPY the {target} object {scope} by Δ={delta} (verified on train: residual=0)."
    if rule.name == "CROP_BBOX_NONZERO":
        return "Crop to bounding box of non-zero content (edge writes define the present)."
    if rule.name == "KEEP_NONZERO":
        return "Keep only non-zero cells; set background elsewhere."
    return f"{rule.name}: {rule.params}"

def solve_instance(inst: ARCInstance) -> SolveResult:
    """Solve an ARC instance using induction + verification + receipts."""
    # 1. Induction: Try catalog
    rule = None
    for induce_fn in CATALOG:
        r = induce_fn(inst.train)
        if r is not None:
            rule = r
            break
    preds = []
    recs = []
    # 2. Verification: Ensure residual=0 on train
    if rule:
        ok_train = all(exact_equals(rule.prog(x), y) for x, y in inst.train)
        if not ok_train:
            rule = None  # Safety: reject if train doesn't fit exactly
    # 3. Prediction + Receipts
    if rule:
        pce = pce_for_rule(rule)
        for i, x in enumerate(inst.test_in):
            yhat = rule.prog(x)
            res = compute_residual(yhat, inst.test_out[i])
            edits_total, edits_boundary, edits_interior = edit_counts(x, yhat)
            recs.append(Receipts(res, edits_total, edits_boundary, edits_interior,
                                 f"[{inst.name} test#{i}] {pce}"))
            preds.append(yhat)
    else:
        for i, x in enumerate(inst.test_in):
            preds.append(x.copy())
            res = compute_residual(preds[-1], inst.test_out[i])
            recs.append(Receipts(res, 0, 0, 0, f"[{inst.name} test#{i}] No rule"))
    acc = float(np.mean([int(exact_equals(p, inst.test_out[i])) for i,p in enumerate(preds)]))
    return SolveResult(inst.name, rule, preds, recs, acc)
