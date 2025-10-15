"""
Search and composition logic for ARC Solver.
"""

import time
from typing import List, Tuple, Optional, Callable, Dict
from .types import Operator, ARCInstance, Node, Grid
from .utils import residual, equal, compute_palette_delta, compute_component_delta
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


def autobuild_operators(train) -> List[Operator]:
    """
    Try all inducers and return operators with train residual=0.

    Ordered by cost (cheap → expensive) per IMPLEMENTATION_PLAN.md.
    """
    ops = []
    # 1-4: Ultra cheap and often decisive
    ops += induce_COLOR_PERM(train)
    ops += induce_ROT_FLIP(train)
    ops += induce_PARITY_CONST(train)
    ops += induce_CROP_KEEP(train, bg=0)
    # 5: B1 - Keep largest component
    ops += induce_KEEP_LARGEST(train, bg=0)
    # Later: More expensive operators
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


def beam_search(inst: ARCInstance, max_depth=6, beam_size=160) -> Tuple[Optional[List[Operator]], Dict]:
    """
    Beam search with residual=0 pruning discipline.

    Args:
        inst: ARC instance with train pairs
        max_depth: maximum composition depth
        beam_size: beam width

    Returns:
        (operators, beam_stats)
        - operators: List of operators if train residual=0, else None
        - beam_stats: Dict with expanded, pruned, depth counts
    """
    train = inst.train
    id_prog = lambda z: z
    init_res = apply_prog_to_pairs(train, id_prog)
    beam = [Node(id_prog, [], init_res, sum(init_res), 0)]

    expanded = 0
    pruned = 0
    final_depth = 0

    for depth in range(max_depth):
        new = []
        rules = autobuild_operators(train)
        for node in beam:
            for r in rules:
                expanded += 1
                comp = compose(node.prog, r.prog)
                res = apply_prog_to_pairs(train, comp)
                # Receipts-first prune: any residual increase → drop
                if any(res[i] > node.res_list[i] for i in range(len(res))):
                    pruned += 1
                    continue
                tot = sum(res)
                if tot < node.total:
                    new.append(Node(comp, node.steps + [r], res, tot, node.depth + 1))
                    if tot == 0:
                        final_depth = depth + 1
                        beam_stats = {"expanded": expanded, "pruned": pruned, "depth": final_depth}
                        return node.steps + [r], beam_stats
                else:
                    pruned += 1
        if not new:
            break
        new.sort(key=lambda nd: (nd.total, nd.depth))
        beam = new[:beam_size]
        final_depth = depth + 1

    beam_stats = {"expanded": expanded, "pruned": pruned, "depth": final_depth}
    return None, beam_stats


def solve_with_beam(inst: ARCInstance, max_depth=6, beam_size=160):
    """
    Solve ARC instance with beam search.

    Returns:
        (steps, preds, residuals, metadata)
        - steps: List of operators if solved, else None
        - preds: Predictions on test inputs
        - residuals: Residuals for test outputs
        - metadata: Dict with beam stats, timing, invariants
    """
    t_start = time.time()

    # Run beam search
    t_beam_start = time.time()
    steps, beam_stats = beam_search(inst, max_depth=max_depth, beam_size=beam_size)
    t_beam_end = time.time()

    # Compute train residuals
    if steps:
        residuals_per_pair = []
        for x, y in inst.train:
            yhat = x.copy()
            for op in steps:
                yhat = op.prog(yhat)
            residuals_per_pair.append(residual(yhat, y))
        total_residual = sum(residuals_per_pair)
    else:
        residuals_per_pair = [residual(x, y) for x, y in inst.train]
        total_residual = sum(residuals_per_pair)

    # Compute invariants across all train pairs
    palette_deltas = []
    component_deltas = []
    for x, y in inst.train:
        palette_deltas.append(compute_palette_delta(x, y))
        component_deltas.append(compute_component_delta(x, y, bg=0))

    # Aggregate invariants
    palette_preserved = all(pd["preserved"] for pd in palette_deltas)
    # Sum all color deltas
    merged_palette_delta = {}
    for pd in palette_deltas:
        for color, delta in pd["delta"].items():
            merged_palette_delta[color] = merged_palette_delta.get(color, 0) + delta

    # Aggregate component invariants
    component_count_deltas = [cd["count_delta"] for cd in component_deltas]
    largest_kept_all = all(cd["largest_kept"] for cd in component_deltas)

    palette_invariants = {
        "preserved": palette_preserved,
        "delta": merged_palette_delta
    }

    component_invariants = {
        "largest_kept": largest_kept_all,
        "count_delta": component_count_deltas[0] if len(set(component_count_deltas)) == 1 else "varies"
    }

    # Timing
    t_end = time.time()
    timing_ms = {
        "beam": int((t_beam_end - t_beam_start) * 1000),
        "total": int((t_end - t_start) * 1000)
    }

    # Build metadata
    metadata = {
        "beam": beam_stats,
        "timing_ms": timing_ms,
        "total_residual": total_residual,
        "residuals_per_pair": residuals_per_pair,
        "palette_invariants": palette_invariants,
        "component_invariants": component_invariants
    }

    if steps is None:
        return None, [], [], metadata

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

    test_residuals = [residual(preds[i], inst.test_out[i]) if i < len(inst.test_out) else -1
                      for i in range(len(preds))]

    return steps, preds, test_residuals, metadata


# ==============================================================================
# Closure-based solver (Master Operator paradigm)
# ==============================================================================

def autobuild_closures(train):
    """
    Try all unifiers and return closures that produce exact output on train.

    Ordered by cost (cheap → expensive).
    """
    from .closures import unify_KEEP_LARGEST, unify_OUTLINE_OBJECTS

    closures = []
    # B1: KEEP_LARGEST_COMPONENT
    closures += unify_KEEP_LARGEST(train)
    # B2: OUTLINE_OBJECTS
    closures += unify_OUTLINE_OBJECTS(train)
    # TODO: Add more closure families as they're implemented

    return closures


def solve_with_closures(inst: ARCInstance):
    """
    Solve ARC instance with fixed-point closure engine.

    Returns:
        (closures, preds, residuals, metadata)
        - closures: List of closures if solved, else None
        - preds: Predictions on test inputs
        - residuals: Residuals for test outputs
        - metadata: Dict with fp stats, timing, invariants
    """
    from .closure_engine import run_fixed_point

    t_start = time.time()

    # Try all unifiers
    t_unify_start = time.time()
    closures = autobuild_closures(inst.train)
    t_unify_end = time.time()

    if not closures:
        # No closures found
        metadata = {
            "fp": {"iters": 0, "cells_multi": -1},
            "timing_ms": {"unify": 0, "total": int((time.time() - t_start) * 1000)},
            "total_residual": sum(residual(x, y) for x, y in inst.train),
            "residuals_per_pair": [residual(x, y) for x, y in inst.train],
            "palette_invariants": {},
            "component_invariants": {}
        }
        return None, [], [], metadata

    # Verify closures on train
    from .closure_engine import verify_closures_on_train

    t_fp_start = time.time()
    train_ok = verify_closures_on_train(closures, inst.train)
    t_fp_end = time.time()

    # Get fp_iters_list for metadata (re-run to collect stats)
    fp_iters_list = []
    if train_ok:
        for x, y in inst.train:
            U_final, fp_stats = run_fixed_point(closures, x)
            fp_iters_list.append(fp_stats["iters"])

    if not train_ok:
        # Closures don't solve train exactly
        metadata = {
            "fp": {"iters": max(fp_iters_list) if fp_iters_list else 0, "cells_multi": -1},
            "timing_ms": {
                "unify": int((t_unify_end - t_unify_start) * 1000),
                "fp": int((t_fp_end - t_fp_start) * 1000),
                "total": int((time.time() - t_start) * 1000)
            },
            "total_residual": sum(residual(x, y) for x, y in inst.train),
            "residuals_per_pair": [residual(x, y) for x, y in inst.train],
            "palette_invariants": {},
            "component_invariants": {}
        }
        return None, [], [], metadata

    # Predict on test
    preds = []
    test_fp_iters = []

    for x_test in inst.test_in:
        U_final, fp_stats = run_fixed_point(closures, x_test)
        test_fp_iters.append(fp_stats["iters"])

        # Extract bg from first closure that defines it
        # (Not all closures have bg - e.g., OUTLINE, MOD_PATTERN, etc.)
        bg = None
        for closure in closures:
            if "bg" in closure.params:
                bg = closure.params["bg"]
                break

        # If no closure defines bg, fail loudly
        if bg is None:
            raise ValueError(f"No closure defines 'bg' parameter. Closures: {[c.name for c in closures]}")

        y_pred = U_final.to_grid_deterministic(fallback='lowest', bg=bg)
        preds.append(y_pred)

    test_residuals = [residual(preds[i], inst.test_out[i]) if i < len(inst.test_out) else -1
                      for i in range(len(preds))]

    # Compute invariants
    palette_deltas = []
    component_deltas = []

    # Extract bg from closures (safe extraction)
    bg = None
    for closure in closures:
        if "bg" in closure.params:
            bg = closure.params["bg"]
            break
    if bg is None:
        raise ValueError(f"No closure defines 'bg' parameter for invariants. Closures: {[c.name for c in closures]}")

    for x, y in inst.train:
        palette_deltas.append(compute_palette_delta(x, y))
        component_deltas.append(compute_component_delta(x, y, bg=bg))

    palette_preserved = all(pd["preserved"] for pd in palette_deltas)
    merged_palette_delta = {}
    for pd in palette_deltas:
        for color, delta in pd["delta"].items():
            merged_palette_delta[color] = merged_palette_delta.get(color, 0) + delta

    component_count_deltas = [cd["count_delta"] for cd in component_deltas]
    largest_kept_all = all(cd["largest_kept"] for cd in component_deltas)

    palette_invariants = {
        "preserved": palette_preserved,
        "delta": merged_palette_delta
    }

    component_invariants = {
        "largest_kept": largest_kept_all,
        "count_delta": component_count_deltas[0] if len(set(component_count_deltas)) == 1 else "varies"
    }

    # Build metadata
    t_end = time.time()
    metadata = {
        "fp": {
            "iters": max(fp_iters_list + test_fp_iters),
            "cells_multi": 0  # All singletons if we got here
        },
        "timing_ms": {
            "unify": int((t_unify_end - t_unify_start) * 1000),
            "fp": int((t_fp_end - t_fp_start) * 1000),
            "total": int((t_end - t_start) * 1000)
        },
        "total_residual": 0,  # Train exactness guaranteed
        "residuals_per_pair": [0] * len(inst.train),
        "palette_invariants": palette_invariants,
        "component_invariants": component_invariants
    }

    return closures, preds, test_residuals, metadata
