!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-UI STARTER — Invariant Engine + Core DSL (Symmetry & Masks)
===============================================================

For the other AI: implement these **exact interfaces** first.
This file is a runnable, receipts-first starter with specs, stubs,
and unit tests (using simple asserts). Expand operator coverage after
these pass.

GOAL (phase 1)
--------------
1) Invariant Engine:
   - grid size, color histogram
   - connected components, bounding boxes
   - symmetry detection (rot90/180/270, flip_h, flip_v) that *exactly*
     maps source→target on train pairs when applicable.

2) Core DSL (v0):
   Types:
     Grid (np.ndarray int), Mask (bool array same shape)
   Ops:
     ROT(k∈{0,1,2,3}), FLIP(axis∈{'h','v'}),
     BBOX(bg=0), CROP(rect),
     MASK_COLOR(c), MASK_NONZERO(bg=0),
     KEEP(mask), REMOVE(mask),
     ON(mask, prog)  # apply prog only on masked region
     SEQ(prog1, prog2)  # composition
   Induction helpers for:
     - symmetry-only programs
     - mask-based crop/recolor skeletons

3) Receipts:
   - residual (train/test exactness)
   - edit bill (#edits total / boundary / interior)
   - PCE line (proof-carrying English) tied to the chosen invariants.

HOW TO RUN
----------
    python arc_ui_starter.py

No external deps beyond numpy.

NEXT STEPS AFTER PHASE 1 PASSES
-------------------------------
- Add color permutation, object-rank masks, component recolor, tiling, drawing.
- Add search/beam to compose ops with receipts.
- Keep this receipts discipline: residual==0 on train before predicting test.

"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict
import numpy as np
from collections import Counter, deque

# =============================================================================
# SECTION 0 — Types and Utilities
# =============================================================================

Grid = np.ndarray          # dtype=int, shape (H, W)
Mask = np.ndarray          # dtype=bool, same shape as Grid

def G(lst) -> Grid:
    """Helper to build a grid from nested lists."""
    return np.array(lst, dtype=int)

def copy_grid(g: Grid) -> Grid:
    return np.array(g, dtype=int)

def assert_grid(g: Grid):
    assert isinstance(g, np.ndarray) and g.dtype == int and g.ndim == 2, "Grid must be 2D int ndarray."

# =============================================================================
# SECTION 1 — Invariant Engine (v0)
# =============================================================================

@dataclass

class Invariants:
    shape: Tuple[int,int]
    histogram: Dict[int,int]
    bbox: Tuple[int,int,int,int]  # (r0,c0,r1,c1)
    n_components: int
    # symmetry flags (exact)
    sym_rot90: bool
    sym_rot180: bool
    sym_rot270: bool
    sym_flip_h: bool
    sym_flip_v: bool

def color_histogram(g: Grid) -> Dict[int,int]:
    vals, cnt = np.unique(g, return_counts=True)
    return {int(v): int(c) for v,c in zip(vals,cnt)}

def bbox_nonzero(g: Grid, bg: int=0) -> Tuple[int,int,int,int]:
    idx = np.argwhere(g != bg)
    if idx.size == 0:
        return (0,0,g.shape[0]-1,g.shape[1]-1)
    r0,c0 = idx.min(axis=0); r1,c1 = idx.max(axis=0)
    return (int(r0),int(c0),int(r1),int(c1))

def connected_components(g: Grid, bg: int=0) -> List[Tuple[int, List[Tuple[int,int]]]]:
    H,W = g.shape
    vis = np.zeros_like(g, dtype=bool)
    comps=[]
    for r in range(H):
        for c in range(W):
            if g[r,c] != bg and not vis[r,c]:
                col = g[r,c]
                q = deque([(r,c)]); vis[r,c]=True
                pix = [(r,c)]
                while q:
                    rr,cc = q.popleft()
                    for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nr,nc = rr+dr, cc+dc
                        if 0<=nr<H and 0<=nc<W and (not vis[nr,nc]) and g[nr,nc]==col:
                            vis[nr,nc]=True
                            q.append((nr,nc))
                            pix.append((nr,nc))
                comps.append((col, pix))
    return comps

# — Exact symmetry checks (these are the ones we prototype first) —

def rot90(g: Grid)  -> Grid: return np.rot90(g, k=1)
def rot180(g: Grid) -> Grid: return np.rot90(g, k=2)
def rot270(g: Grid) -> Grid: return np.rot90(g, k=3)
def flip_h(g: Grid) -> Grid:  return np.fliplr(g)
def flip_v(g: Grid) -> Grid:  return np.flipud(g)

def exact_equals(a: Grid, b: Grid) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)

def invariants(g: Grid, bg: int=0) -> Invariants:
    assert_grid(g)
    r0,c0,r1,c1 = bbox_nonzero(g, bg)
    comps = connected_components(g, bg)
    return Invariants(
        shape = g.shape,
        histogram = color_histogram(g),
        bbox = (r0,c0,r1,c1),
        n_components = len(comps),
        sym_rot90  = exact_equals(rot90(g),  g),
        sym_rot180 = exact_equals(rot180(g), g),
        sym_rot270 = exact_equals(rot270(g), g),
        sym_flip_h = exact_equals(flip_h(g),  g),
        sym_flip_v = exact_equals(flip_v(g),  g),
    )

# =============================================================================
# SECTION 2 — Receipts helpers
# =============================================================================

@dataclass

class Receipts:
    residual: int                  # Hamming distance to target (0 = exact)
    edits_total: int               # # cells changed vs input
    edits_boundary: int
    edits_interior: int
    pce: str                       # proof-carrying explanation (tie to invariants)

def edit_counts(a: Grid, b: Grid) -> Tuple[int,int,int]:
    assert a.shape == b.shape
    diff = (a != b)
    total = int(diff.sum())
    H,W = a.shape
    border = np.zeros_like(diff)
    border[0,:]=True; border[-1,:]=True; border[:,0]=True; border[:,-1]=True
    boundary = int(np.logical_and(diff, border).sum())
    interior = total - boundary
    return total, boundary, interior

# =============================================================================
# SECTION 3 — Core DSL (v0): symmetry + mask ops + composition
# =============================================================================

# Prog = Callable[[Grid], Grid]

def ROT(k: int) -> Callable[[Grid], Grid]:
    """k in {0,1,2,3} → 0°, 90°, 180°, 270°"""
    assert k in (0,1,2,3)
    if k==0: return lambda z: z
    if k==1: return rot90
    if k==2: return rot180
    return rot270

def FLIP(axis: str) -> Callable[[Grid], Grid]:
    """axis in {'h','v'}"""
    assert axis in ('h','v')
    return flip_h if axis=='h' else flip_v

def BBOX(bg: int=0) -> Callable[[Grid], Tuple[int,int,int,int]]:
    def f(z: Grid):
        return bbox_nonzero(z, bg)
    return f

def CROP(rect_fn: Callable[[Grid], Tuple[int,int,int,int]]) -> Callable[[Grid], Grid]:
    def f(z: Grid):
        r0,c0,r1,c1 = rect_fn(z)
        return z[r0:r1+1, c0:c1+1]
    return f

def MASK_COLOR(c: int) -> Callable[[Grid], Mask]:
    def f(z: Grid):
        return (z == c)
    return f

def MASK_NONZERO(bg: int=0) -> Callable[[Grid], Mask]:
    def f(z: Grid):
        return (z != bg)
    return f

def KEEP(mask_fn: Callable[[Grid], Mask]) -> Callable[[Grid], Grid]:
    def f(z: Grid):
        m = mask_fn(z)
        out = np.zeros_like(z)
        out[m] = z[m]
        return out
    return f

def REMOVE(mask_fn: Callable[[Grid], Mask]) -> Callable[[Grid], Grid]:
    def f(z: Grid):
        m = mask_fn(z)
        out = z.copy()
        out[m] = 0
        return out
    return f

def ON(mask_fn: Callable[[Grid], Mask], prog: Callable[[Grid], Grid]) -> Callable[[Grid], Grid]:
    """Apply prog only on mask; outside unchanged."""
    def f(z: Grid):
        m = mask_fn(z)
        sub = prog(z.copy())
        out = z.copy()
        out[m] = sub[m]
        return out
    return f

def SEQ(p1: Callable[[Grid], Grid], p2: Callable[[Grid], Grid]) -> Callable[[Grid], Grid]:
    return lambda z: p2(p1(z))

# =============================================================================
# SECTION 4 — Induction routines for symmetry/mask programs
# =============================================================================

@dataclass

class Rule:
    name: str
    params: dict
    prog: Callable[[Grid], Grid]

def induce_symmetry_rule(train: List[Tuple[Grid,Grid]]) -> Optional[Rule]:
    """Try pure symmetry transforms that exactly map all train pairs."""
    candidates = [
        ("ROT", {"k":0}, ROT(0)),
        ("ROT", {"k":1}, ROT(1)),
        ("ROT", {"k":2}, ROT(2)),
        ("ROT", {"k":3}, ROT(3)),
        ("FLIP", {"axis":"h"}, FLIP('h')),
        ("FLIP", {"axis":"v"}, FLIP('v')),
    ]
    for name, params, prog in candidates:
        ok = True
        for x,y in train:
            if not exact_equals(prog(x), y):
                ok = False; break
        if ok:
            return Rule(name, params, prog)
    return None

def induce_crop_nonzero_rule(train: List[Tuple[Grid,Grid]], bg: int=0) -> Optional[Rule]:
    """Try crop-to-bbox of nonzero (bg) content."""
    prog = CROP(BBOX(bg))
    ok = all(exact_equals(prog(x), y) for x,y in train)
    if ok:
        return Rule("CROP_BBOX_NONZERO", {"bg":bg}, prog)
    return None

# (Example) Induce keep-nonzero or remove-nonzero
def induce_keep_nonzero_rule(train: List[Tuple[Grid,Grid]], bg: int=0) -> Optional[Rule]:
    prog = KEEP(MASK_NONZERO(bg))
    ok = all(exact_equals(prog(x), y) for x,y in train)
    return Rule("KEEP_NONZERO", {"bg":bg}, prog) if ok else None

# Try rules in order of simplicity (Occam)
CATALOG = [
    induce_symmetry_rule,
    induce_crop_nonzero_rule,
    induce_keep_nonzero_rule,
]

# =============================================================================
# SECTION 5 — Solver harness with receipts + PCE
# =============================================================================

@dataclass

class ARCInstance:
    name: str
    train: List[Tuple[Grid,Grid]]
    test_in: List[Grid]
    test_out: List[Grid]

@dataclass

class SolveResult:
    name: str
    rule: Optional[Rule]
    preds: List[Grid]
    receipts: List[Receipts]
    acc_exact: float

def pce_for_rule(rule: Rule) -> str:
    if rule is None:
        return "No rule matched."
    if http://rule.name == "ROT":
        return f"Rotate grid by {rule.params['k']*90} degrees (exact symmetry fit on train)."
    if http://rule.name == "FLIP":
        return f"Flip grid along {'horizontal' if rule.params['axis']=='h' else 'vertical'} axis."
    if http://rule.name == "CROP_BBOX_NONZERO":
        return "Crop to bounding box of non-zero content (edge writes define the present)."
    if http://rule.name == "KEEP_NONZERO":
        return "Keep only non-zero cells; set background elsewhere."
    return f"{http://rule.name}: {rule.params}"

def solve_instance(inst: ARCInstance) -> SolveResult:
    # Induce
    rule = None
    for induce in CATALOG:
        r = induce(inst.train)
        if r is not None:
            rule = r; break
    preds=[]; recs=[]
    if rule:
        # Verify residual=0 on train (receipts); else discard
        ok_train = all(exact_equals(rule.prog(x), y) for x,y in inst.train)
        if not ok_train:
            rule = None  # safety
    # Predict + receipts
    if rule:
        pce = pce_for_rule(rule)
        for i, x in enumerate(inst.test_in):
            yhat = rule.prog(x)
            residual = int(np.sum(yhat != inst.test_out[i]))
            edits_total, edits_boundary, edits_interior = edit_counts(x, yhat)
            recs.append(Receipts(residual, edits_total, edits_boundary, edits_interior,
                                 f"[{http://inst.name} test#{i}] {pce}"))
            preds.append(yhat)
    else:
        for i, x in enumerate(inst.test_in):
            preds.append(x.copy())
            residual = int(np.sum(preds[-1] != inst.test_out[i]))
            recs.append(Receipts(residual, 0, 0, 0, f"[{http://inst.name} test#{i}] No rule"))
    acc = float(np.mean([int(exact_equals(p, inst.test_out[i])) for i,p in enumerate(preds)]))
    return SolveResult(http://inst.name, rule, preds, recs, acc)

# =============================================================================
# SECTION 6 — Unit tests / mini demo
# =============================================================================

def _unit_tests():
    # 1) invariants basics
    g = G([[1,0,0],[1,2,0],[0,2,2]])
    inv = invariants(g, bg=0)
    assert inv.shape == (3,3)
    assert inv.histogram[0] == 5
    assert inv.n_components == 2
    assert inv.sym_rot90 == False

    # 2) symmetry induction
    a = g
    b = rot90(a)
    inst = ARCInstance("rot90_demo", [(a,b)], [G([[0,9,9],[0,0,9],[8,8,0]])], [rot90(G([[0,9,9],[0,0,9],[8,8,0]]))])
    res = solve_instance(inst)
    assert res.rule is not None and http://res.rule.name == "ROT" and res.rule.params["k"]==1
    assert res.acc_exact == 1.0

    # 3) crop bbox
    x3 = G([[0,0,0,0],[0,5,5,0],[0,5,0,0]])
    y3 = x3[1:3,1:3]
    inst2 = ARCInstance("crop_demo", [(x3,y3)], [G([[0,0,0],[0,7,0],[0,7,7]])], [G([[7,0],[7,7]])])
    res2 = solve_instance(inst2)
    assert res2.rule is not None and http://res2.rule.name == "CROP_BBOX_NONZERO"
    assert res2.acc_exact == 1.0

    print("All unit tests passed.")

def _mini_press_demo():
    tasks=[]

    # symmetry rot180
    x1 = G([[0,1],[2,0]]); y1 = rot180(x1)
    tasks.append(ARCInstance("sym_rot180", [(x1,y1)], [G([[0,3],[4,0]])], [rot180(G([[0,3],[4,0]]))]))

    # flip_h
    x2 = G([[9,0,0],[0,9,0]]); y2 = flip_h(x2)
    tasks.append(ARCInstance("flip_h_demo", [(x2,y2)], [G([[5,0,0],[0,5,0]])], [flip_h(G([[5,0,0],[0,5,0]]))]))

    # crop bbox
    x3 = G([[0,0,0,0],[0,7,7,0],[0,7,0,0]]); y3 = x3[1:3,1:3]
    tasks.append(ARCInstance("crop_bbox_demo", [(x3,y3)], [G([[0,0,0],[0,8,0],[0,8,8]])], [G([[8,0],[8,8]])]))

    # run
    total=0; exact=0
    for inst in tasks:
        res = solve_instance(inst)
        print(f"\n[{http://inst.name}] rule={http://res.rule.name if res.rule else 'None'}  acc={res.acc_exact:.2f}")
        for i, rc in enumerate(res.receipts):
            print(f"  test#{i}: residual={rc.residual} edits={rc.edits_total} (boundary {rc.edits_boundary} interior {rc.edits_interior})")
            print("  PCE:", rc.pce)
        total += len(inst.test_out); exact += int(res.acc_exact==1.0)*len(inst.test_out)
    print(f"\nPRESS DEMO: {exact}/{total} exact.")

# =============================================================================
# SECTION 7 — Build Guidance for the Other AI (step-by-step)
# =============================================================================

GUIDE = r"""
IMPLEMENTATION GUIDE (Phase 1)
==============================

1) Keep this file structure and interfaces. Do not change function signatures.

2) Complete invariants:
   - Add periodicity checks (optional): detect repeating motif horizontally/vertically.
   - Add symmetry group summary: which transforms keep the grid invariant.

3) Expand DSL gradually:
   - SHIFT(dx,dy): wrap/no-wrap variants.
   - MASK_RECT(r0,c0,r1,c1)
   - RECOLOR(mapping: dict[int->int])
   - ON(mask, prog) already provided; ensure it composes well with ROT/FLIP/CROP.

4) Induction routines:
   - For symmetry: verify exact equality source→target on all train pairs.
   - For crop: detect that each output equals crop_bbox(input).
   - For keep/remove masks: verify exact mapping.

5) Receipts discipline:
   - Before predicting test, verify train residuals are all zero.
   - For each prediction, compute edits_total/boundary/interior and print PCE.

6) Unit tests:
   - Keep `_unit_tests()` passing.
   - Add tests as you add ops (SHIFT, RECOLOR, MASK_RECT).

7) Next milestone (after Phase 1 passes):
   - Add color permutation induction.
   - Add component-rank masks and recolor by rank.
   - Add a tiny beam search over SEQ compositions depth ≤ 3 with pruning (residual must drop or stay 0 on train at each step).

8) Always keep the receipts-first contract: train residual == 0, then predict test with PCE.
"""

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    _unit_tests()
    _mini_press_demo()
    print(GUIDE)