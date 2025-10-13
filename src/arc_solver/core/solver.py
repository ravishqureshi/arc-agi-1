#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ARC Solver - Main Solver Harness"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from .types import Grid
from .invariants import exact_equals
from .receipts import Receipts, edit_counts, residual as compute_residual
from .induction import (
    Rule, CATALOG,
    # Beam-compatible induction functions
    induce_symmetry, induce_crop_bbox, induce_keep_nonzero,
    induce_color_perm, induce_recolor_obj_rank_beam, induce_keep_topk,
    induce_move_copy
)

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

# =============================================================================
# Beam Search for Multi-Step Composition
# =============================================================================

@dataclass
class Node:
    """Beam search node representing a partial program."""
    prog: Callable[[Grid], Grid]
    steps: List[Rule]         # for PCE
    res_per_pair: List[int]   # residuals on train pairs
    total_res: int
    length: int

def compose(p: Callable[[Grid], Grid], q: Callable[[Grid], Grid]) -> Callable[[Grid], Grid]:
    """Compose two programs: p then q."""
    return lambda z: q(p(z))

def apply_prog_to_pairs(pairs: List[Tuple[Grid, Grid]], prog: Callable[[Grid], Grid]) -> List[int]:
    """Apply program to train pairs and return residuals."""
    return [compute_residual(prog(x), y) for x, y in pairs]

def candidate_rules(train: List[Tuple[Grid, Grid]]) -> List[Rule]:
    """
    Generate all candidate primitive rules for beam search.

    Strategy: Generate a universal set of primitive operations, not just those
    that match the train exactly. The beam search will figure out which
    compositions work.
    """
    from ..operators.symmetry import ROT, FLIP
    from ..operators.spatial import CROP, BBOX
    from ..operators.masks import KEEP, MASK_NONZERO

    rules = []

    # Universal primitives (always include, regardless of train)
    # Symmetries
    for k in [1, 2, 3]:  # Skip identity (k=0)
        prog = ROT(k)
        rules.append(Rule("ROT", {"k": k}, prog, f"Rotate {k*90}°"))

    for axis in ['h', 'v']:
        prog = FLIP(axis)
        axis_name = 'horizontal' if axis == 'h' else 'vertical'
        rules.append(Rule("FLIP", {"axis": axis}, prog, f"Flip {axis_name}"))

    # Crop bbox
    prog = CROP(BBOX(0))
    rules.append(Rule("CROP_BBOX_NONZERO", {"bg": 0}, prog, "Crop to bbox"))

    # Keep nonzero
    prog = KEEP(MASK_NONZERO(0))
    rules.append(Rule("KEEP_NONZERO", {"bg": 0}, prog, "Keep non-zero"))

    # Train-specific primitives (only if they match)
    rules += induce_color_perm(train)
    rules += induce_recolor_obj_rank_beam(train, rank=0, group='global')
    rules += induce_recolor_obj_rank_beam(train, rank=0, group='per_color')
    rules += induce_keep_topk(train, k=2)
    rules += induce_move_copy(train, rank=0, group='global')
    rules += induce_move_copy(train, rank=0, group='per_color')

    return rules

def beam_search(inst: ARCInstance, max_depth: int = 4, beam_size: int = 50) -> Optional[List[Rule]]:
    """
    Beam search for multi-step compositions.

    Strategy:
    - Start with identity program
    - At each depth, try composing each beam node with each candidate primitive
    - Prune if residual increases on ANY train pair
    - Keep only beam_size best nodes (sorted by total residual)
    - Return immediately when residual=0 on all train pairs

    Args:
        inst: ARC instance with train/test splits
        max_depth: Maximum composition depth
        beam_size: Maximum beam width

    Returns:
        List of rules that compose to solve the task, or None if not found
    """
    train = inst.train

    # Initial node: identity
    id_prog = lambda z: z
    init_res = apply_prog_to_pairs(train, id_prog)
    beam = [Node(id_prog, [], init_res, sum(init_res), 0)]

    for depth in range(max_depth):
        new_nodes = []

        # Generate candidate primitive rules once per depth
        rules = candidate_rules(train)

        for node in beam:
            base_prog = node.prog
            base_res = node.res_per_pair
            base_total = node.total_res

            for r in rules:
                # Compose: base → r
                comp_prog = compose(base_prog, r.prog)
                res = apply_prog_to_pairs(train, comp_prog)

                # Prune if any residual increases
                if any(res[i] > base_res[i] for i in range(len(res))):
                    continue

                tot = sum(res)

                # Keep nodes that strictly reduce total residual
                if tot < base_total:
                    new_nodes.append(Node(comp_prog, node.steps + [r], res, tot, node.length + 1))

                    # Found perfect solution
                    if tot == 0:
                        return node.steps + [r]

        if not new_nodes:
            break

        # Beam prune: keep best beam_size nodes
        new_nodes.sort(key=lambda nd: (nd.total_res, nd.length))
        beam = new_nodes[:beam_size]

    return None

def pce_for_steps(steps: List[Rule]) -> str:
    """Generate PCE for a sequence of rule steps."""
    if not steps:
        return "Identity (no change)."
    return " → ".join(r.pce for r in steps)

def solve_with_beam(inst: ARCInstance, max_depth: int = 4, beam_size: int = 50) -> SolveResult:
    """
    Solve ARC instance using beam search for multi-step compositions.

    Args:
        inst: ARC instance
        max_depth: Maximum composition depth
        beam_size: Beam width

    Returns:
        SolveResult with predictions and receipts
    """
    steps = beam_search(inst, max_depth=max_depth, beam_size=beam_size)

    if steps is None:
        # No solution found - return identity
        preds = []
        recs = []
        for i, x in enumerate(inst.test_in):
            preds.append(x.copy())
            res = compute_residual(x, inst.test_out[i])
            recs.append(Receipts(res, 0, 0, 0, f"[{inst.name} test#{i}] No solution found"))
        acc = float(np.mean([int(exact_equals(p, inst.test_out[i])) for i, p in enumerate(preds)]))
        return SolveResult(inst.name, None, preds, recs, acc)

    # Verify train receipts (sanity check)
    for x, y in inst.train:
        yhat = x.copy()
        for r in steps:
            yhat = r.prog(yhat)
        if not exact_equals(yhat, y):
            raise AssertionError(f"Train residual must be 0, but got {compute_residual(yhat, y)}")

    # Predict test + generate receipts
    preds = []
    recs = []
    pce = pce_for_steps(steps)

    for i, x in enumerate(inst.test_in):
        yhat = x.copy()
        for r in steps:
            yhat = r.prog(yhat)

        ygt = inst.test_out[i]
        res = compute_residual(yhat, ygt)
        edits_total, edits_boundary, edits_interior = edit_counts(x, yhat)

        recs.append(Receipts(res, edits_total, edits_boundary, edits_interior,
                            f"[{inst.name} test#{i}] {pce}"))
        preds.append(yhat)

    acc = float(np.mean([int(exact_equals(p, inst.test_out[i])) for i, p in enumerate(preds)]))

    # Create a combined rule for the result
    if len(steps) == 1:
        combined_rule = steps[0]
    else:
        # Multi-step composition - create combined rule
        def multi_prog(z: Grid) -> Grid:
            out = z.copy()
            for r in steps:
                out = r.prog(out)
            return out
        combined_rule = Rule("COMPOSED", {"steps": [r.name for r in steps]}, multi_prog, pce)

    return SolveResult(inst.name, combined_rule, preds, recs, acc)
