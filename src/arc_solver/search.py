"""
Search and composition logic for ARC Solver.
"""

from typing import List, Tuple, Optional, Callable
from .types import Operator, ARCInstance, Node, Grid
from .utils import residual, equal
from .inducers import (
    induce_COLOR_PERM,
    induce_ROT_FLIP,
    induce_CROP_KEEP,
    induce_PARITY_CONST,
    induce_TILING_AND_MASK,
    induce_HOLE_FILL,
    induce_COPY_BY_DELTAS
)


def autobuild_operators(train) -> List[Operator]:
    """
    Try all inducers and return operators with train residual=0.

    This is where we add new inducers as we expand coverage.
    """
    ops = []
    ops += induce_COLOR_PERM(train)
    ops += induce_ROT_FLIP(train)
    ops += induce_CROP_KEEP(train, bg=0)
    ops += induce_PARITY_CONST(train)
    ops += induce_TILING_AND_MASK(train)
    ops += induce_HOLE_FILL(train, bg=0)
    ops += induce_COPY_BY_DELTAS(train, rank=-1, group='global', bg=0)
    ops += induce_COPY_BY_DELTAS(train, rank=-1, group='per_color', bg=0)
    return ops


def compose(p: Callable, q: Callable) -> Callable:
    """Compose two functions: (p; q)(z) = q(p(z))."""
    return lambda z: q(p(z))


def apply_prog_to_pairs(pairs, prog) -> List[int]:
    """Apply program to train pairs and compute residuals."""
    return [residual(prog(x), y) for x, y in pairs]


def beam_search(inst: ARCInstance, max_depth=6, beam_size=160) -> Optional[List[Operator]]:
    """
    Beam search with residual=0 pruning discipline.

    Args:
        inst: ARC instance with train pairs
        max_depth: maximum composition depth
        beam_size: beam width

    Returns:
        List of operators if train residual=0, else None
    """
    train = inst.train
    id_prog = lambda z: z
    init_res = apply_prog_to_pairs(train, id_prog)
    beam = [Node(id_prog, [], init_res, sum(init_res), 0)]

    for _ in range(max_depth):
        new = []
        rules = autobuild_operators(train)
        for node in beam:
            for r in rules:
                comp = compose(node.prog, r.prog)
                res = apply_prog_to_pairs(train, comp)
                # Receipts-first prune: any residual increase → drop
                if any(res[i] > node.res_list[i] for i in range(len(res))):
                    continue
                tot = sum(res)
                if tot < node.total:
                    new.append(Node(comp, node.steps + [r], res, tot, node.depth + 1))
                    if tot == 0:
                        return node.steps + [r]
        if not new:
            break
        new.sort(key=lambda nd: (nd.total, nd.depth))
        beam = new[:beam_size]
    return None


def solve_with_beam(inst: ARCInstance, max_depth=6, beam_size=160):
    """
    Solve ARC instance with beam search.

    Returns:
        (steps, preds, residuals)
        - steps: List of operators if solved, else None
        - preds: Predictions on test inputs
        - residuals: Residuals for test outputs
    """
    steps = beam_search(inst, max_depth=max_depth, beam_size=beam_size)
    if steps is None:
        return None, [], []

    # Verify train
    for x, y in inst.train:
        yhat = x.copy()
        for op in steps:
            yhat = op.prog(yhat)
        assert equal(yhat, y), "Train residual must be 0."

    # Predict test
    preds = []
    for x in inst.test_in:
        yhat = x.copy()
        for op in steps:
            yhat = op.prog(yhat)
        preds.append(yhat)

    residuals = [residual(preds[i], inst.test_out[i]) if i < len(inst.test_out) else -1
                 for i in range(len(preds))]

    return steps, preds, residuals
