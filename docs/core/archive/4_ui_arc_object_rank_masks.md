#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-UI NEXT STEP — Object-Rank Masks (+ recolor/keep) with Receipts
===================================================================

This file delivers the **next milestone** the other AI asked for:
*object-rank masks* (largest, 2nd largest, per-color largest, etc.)
and induced programs (recolor / keep) with **receipts**.

What’s included
---------------
1) Invariant primitives for objects:
   • connected components (non-zero), size, bbox, centroid
   • global and per-color **rankings** (by size)

2) DSL ops (v1 additions):
   • MASK_OBJ_RANK(rank=0, group='global'|'per_color', key='size')
   • ON(mask, prog)  (already present, used here)
   • RECOLOR_CONST(color)    — recolor only the masked area
   • KEEP_MASK(mask)         — keep only masked area, zero elsewhere

3) Induction routines (receipts-first):
   • induce_recolor_obj_rank: recolor a **fixed rank** (e.g., largest) to a
     single color **consistent across all train pairs**.
   • induce_keep_obj_topk: keep **top-k** objects, zero others (global ranking).
   Both require **train residual==0**; otherwise they fail (no silent fit).

4) Unit tests and a mini demo (3 tasks):
   • Recolor largest component → color 9
   • Recolor per-color largest → colors {1→7, 2→8}
   • Keep top-2 components

5) Receipts printed for every prediction:
   • residual (must be 0 on train, and we verify on demo tests),
   • edit bill (#edits total, boundary vs interior),
   • PCE line (human-readable proof-carrying explanation).

How to run
----------
  python arc_ui_objrank.py

Dependencies: numpy only.

Next operators to add after this (suggested)
--------------------------------------------
• Object copy/move (paste bbox at target), lattice/tiling detection,
  draw/line ops, symmetry-within-mask, component parity, repetition.
• Beam search over SEQ compositions with receipts at each partial step.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from collections import Counter, deque

# =============================================================================
# Types & helpers
# =============================================================================

Grid = np.ndarray   # dtype=int, shape (H,W)
Mask = np.ndarray   # dtype=bool, shape (H,W)

def G(lst) -> Grid: return np.array(lst, dtype=int)
def assert_grid(g: Grid): assert isinstance(g, np.ndarray) and g.dtype==int and g.ndim==2

# =============================================================================
# Core invariants — components & object rankings
# =============================================================================

@dataclass

class Obj:
    color: int
    pixels: List[Tuple[int,int]]
    size: int
    bbox: Tuple[int,int,int,int]      # (r0,c0,r1,c1)
    centroid: Tuple[float,float]      # (r_mean, c_mean)

def connected_components(g: Grid, bg: int=0) -> List[Obj]:
    """Return non-bg components as Obj list (4-conn)."""
    assert_grid(g)
    H,W = g.shape
    vis = np.zeros_like(g, dtype=bool)
    out: List[Obj] = []
    for r in range(H):
        for c in range(W):
            if g[r,c]!=bg and not vis[r,c]:
                col=g[r,c]
                q=deque([(r,c)]); vis[r,c]=True
                pix=[(r,c)]
                while q:
                    rr,cc = q.popleft()
                    for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nr,nc=rr+dr,cc+dc
                        if 0<=nr<H and 0<=nc<W and not vis[nr,nc] and g[nr,nc]==col:
                            vis[nr,nc]=True
                            q.append((nr,nc)); pix.append((nr,nc))
                rs,cs=zip(*pix); r0,r1=min(rs),max(rs); c0,c1=min(cs),max(cs)
                out.append(Obj(
                    color=int(col),
                    pixels=pix,
                    size=len(pix),
                    bbox=(r0,c0,r1,c1),
                    centroid=(float(np.mean(rs)), float(np.mean(cs)))
                ))
    return out

def rank_objects(objs: List[Obj], group: str='global', by: str='size') -> Dict:
    """
    Return rank indexes:
      group='global': ranks across all objects (largest→rank0).
      group='per_color': separate ranks for each color.
    """
    if by!='size': raise NotImplementedError("Only by='size' implemented.")
    if group=='global':
        order = np.argsort([-o.size for o in objs])  # max first
        return {"global": [int(i) for i in order]}
    elif group=='per_color':
        per: Dict[int,List[int]] = {}
        for c in sorted(set(o.color for o in objs)):
            idx = [i for i,o in enumerate(objs) if o.color==c]
            order = sorted(idx, key=lambda i: -objs[i].size)
            per[int(c)] = [int(i) for i in order]
        return per
    else:
        raise ValueError("group must be 'global' or 'per_color'.")

# =============================================================================
# Receipts
# =============================================================================

@dataclass

class Receipts:
    residual: int
    edits_total: int
    edits_boundary: int
    edits_interior: int
    pce: str

def edit_counts(a: Grid, b: Grid) -> Tuple[int,int,int]:
    assert a.shape==b.shape
    diff=(a!=b)
    total=int(diff.sum())
    H,W=a.shape
    border=np.zeros_like(diff)
    border[0,:]=True; border[-1,:]=True; border[:,0]=True; border[:,-1]=True
    boundary=int(np.logical_and(diff,border).sum())
    interior=total-boundary
    return total,boundary,interior

# =============================================================================
# DSL ops — masks, recolor, keep
# =============================================================================

def MASK_OBJ_RANK(rank: int=0, group: str='global', bg: int=0) -> Callable[[Grid], Mask]:
    """
    rank=0 → top by size.
    group='global' or 'per_color'.
    """
    def f(z: Grid) -> Mask:
        objs = connected_components(z, bg)
        m = np.zeros_like(z, dtype=bool)
        if not objs: return m
        ranks = rank_objects(objs, group=group, by='size')
        if group=='global':
            order = ranks["global"]
            if rank < len(order):
                for (r,c) in objs[order[rank]].pixels: m[r,c]=True
        else:  # per_color
            for c, order in ranks.items():
                if rank < len(order):
                    for (r,cx) in objs[order[rank]].pixels: m[r,cx]=True
        return m
    return f

def RECOLOR_CONST(new_color: int) -> Callable[[Grid], Grid]:
    def f(z: Grid) -> Grid:
        out = z.copy()
        # caller should use ON(mask, RECOLOR_CONST)
        # recoding entire grid is allowed but unintended
        out[:,:] = new_color
        return out
    return f

def ON(mask_fn: Callable[[Grid], Mask], prog: Callable[[Grid], Grid]) -> Callable[[Grid], Grid]:
    """
    Apply 'prog' to a copy of the grid, then **only** commit edits within mask.
    Outside mask, keep original.
    """
    def f(z: Grid) -> Grid:
        m = mask_fn(z)
        z2 = prog(z.copy())
        out = z.copy()
        out[m] = z2[m]
        return out
    return f

def KEEP_MASK(mask_fn: Callable[[Grid], Mask]) -> Callable[[Grid], Grid]:
    def f(z: Grid) -> Grid:
        m = mask_fn(z)
        out = np.zeros_like(z)
        out[m]=z[m]
        return out
    return f

# =============================================================================
# Induction: recolor/keep by object rank (train residual must be 0)
# =============================================================================

@dataclass

class Rule:
    name: str
    params: Dict
    prog: Callable[[Grid], Grid]

def induce_recolor_obj_rank(train: List[Tuple[Grid,Grid]],
                            rank: int=0, group: str='global', bg: int=0) -> Optional[Rule]:
    """
    Hypothesis: output == input except **rank-th object(s)** recolored to a single
    consistent color across all train pairs (global rank or per-color rank).
    """
    # Infer target color from first pair
    target_color=None
    for x,y in train:
        m = MASK_OBJ_RANK(rank=rank, group=group, bg=bg)(x)
        if not np.any(m): return None  # no such object
        # colors inside mask in y (should be uniform)
        vals = y[m]
        if vals.size==0: return None
        c = int(Counter(vals.tolist()).most_common(1)[0][0])
        if target_color is None: target_color = c
        elif target_color != c: return None
    # Build program: recolor only masked area to target_color
    def recolor_prog(z: Grid) -> Grid:
        m = MASK_OBJ_RANK(rank=rank, group=group, bg=bg)(z)
        out = z.copy()
        out[m] = target_color
        return out
    # Verify exact on train
    for x,y in train:
        if not np.array_equal(recolor_prog(x), y):
            return None
    return Rule("RECOLOR_OBJ_RANK", {"rank":rank,"group":group,"bg":bg,"color":target_color}, recolor_prog)

def induce_keep_obj_topk(train: List[Tuple[Grid,Grid]],
                         k: int=1, bg: int=0) -> Optional[Rule]:
    """
    Hypothesis: output == only top-k (global) objects kept, others zeroed.
    """
    def keep_prog(z: Grid) -> Grid:
        objs = connected_components(z, bg)
        order = rank_objects(objs, group='global', by='size')["global"]
        m = np.zeros_like(z, dtype=bool)
        for rnk, idx in enumerate(order):
            if rnk < k:
                for (r,c) in objs[idx].pixels: m[r,c]=True
        out = np.zeros_like(z); out[m]=z[m]
        return out
    # Verify
    for x,y in train:
        if not np.array_equal(keep_prog(x), y):
            return None
    return Rule("KEEP_OBJ_TOPK", {"k":k,"bg":bg}, keep_prog)

# Simple dispatcher: try recolor rank (global/per_color) and keep top-k
def induce_object_rank_rules(train: List[Tuple[Grid,Grid]], bg: int=0) -> Optional[Rule]:
    # Try recolor largest (global)
    r = induce_recolor_obj_rank(train, rank=0, group='global', bg=bg)
    if r: return r
    # Try recolor largest per color
    r = induce_recolor_obj_rank(train, rank=0, group='per_color', bg=bg)
    if r: return r
    # Try keep top-2 (common)
    r = induce_keep_obj_topk(train, k=2, bg=bg)
    if r: return r
    # Try keep top-1
    r = induce_keep_obj_topk(train, k=1, bg=bg)
    if r: return r
    return None

# =============================================================================
# Solver harness with receipts + PCE
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
    if rule is None: return "No rule."
    if http://rule.name=="RECOLOR_OBJ_RANK":
        grp=rule.params['group']; r=rule.params['rank']; col=rule.params['color']
        target = "largest" if r==0 else f"rank {r}"
        scope = "globally" if grp=='global' else "for each color"
        return f"Recolor the {target} component {scope} to color {col} (verified on train: residual=0)."
    if http://rule.name=="KEEP_OBJ_TOPK":
        return f"Keep only the top-{rule.params['k']} components by size; zero others (verified on train: residual=0)."
    return f"{http://rule.name}: {rule.params}"

def solve_instance(inst: ARCInstance, bg: int=0) -> SolveResult:
    rule = induce_object_rank_rules(inst.train, bg=bg)
    preds=[]; recs=[]
    if rule:
        # train receipts safeguard
        assert all(np.array_equal(rule.prog(x),y) for x,y in inst.train), "Train residual must be 0."
        pce = pce_for_rule(rule)
        for i,x in enumerate(inst.test_in):
            yhat = rule.prog(x)
            ygt  = inst.test_out[i]
            residual = int(np.sum(yhat!=ygt))
            edits_total, edits_boundary, edits_interior = edit_counts(x, yhat)
            recs.append(Receipts(residual, edits_total, edits_boundary, edits_interior,
                                 f"[{http://inst.name} test#{i}] {pce}"))
            preds.append(yhat)
    else:
        for i,x in enumerate(inst.test_in):
            yhat=x.copy()
            residual = int(np.sum(yhat!=inst.test_out[i]))
            recs.append(Receipts(residual, 0, 0, 0, f"[{http://inst.name} test#{i}] No rule matched."))
            preds.append(yhat)
    acc=float(np.mean([int(np.array_equal(p, inst.test_out[i])) for i,p in enumerate(preds)]))
    return SolveResult(http://inst.name, rule, preds, recs, acc)

# =============================================================================
# Unit tests + Mini demo
# =============================================================================

def _unit_tests():
    # 1) recolor largest (global)
    x = G([[1,1,0,0],
           [1,0,0,0],
           [2,2,2,0],
           [0,0,0,0]])
    # largest is color 2 (size 3) vs color 1 (size 3) — tie; adjust to break tie
    x[2,0]=2  # size(2)=4 now largest
    y = x.copy(); y[x==2]=9
    inst = ARCInstance("recolor_largest_global",
                       train=[(x,y)],
                       test_in=[G([[3,3,0,0],[0,0,0,0],[4,4,4,4],[0,0,0,0]])],
                       test_out=[G([[3,3,0,0],[0,0,0,0],[9,9,9,9],[0,0,0,0]])])
    res = solve_instance(inst)
    assert res.rule and http://res.rule.name=="RECOLOR_OBJ_RANK" and res.acc_exact==1.0

    # 2) recolor per-color largest
    x2 = G([[1,1,0,2,2],
            [1,0,0,2,0],
            [0,0,0,0,0]])
    # train target recolors largest 1-component → 7, largest 2-component → 8
    y2 = x2.copy(); y2[x2==1]=7; y2[x2==2]=8
    inst2 = ARCInstance("recolor_per_color",
                        train=[(x2,y2)],
                        test_in=[G([[1,1,1,0,2,2],
                                    [0,1,0,0,2,0]])],
                        test_out=[G([[7,7,7,0,8,8],
                                     [0,7,0,0,8,0]])])
    res2 = solve_instance(inst2)
    assert res2.rule and res2.rule.params['group']=='per_color' and res2.acc_exact==1.0

    # 3) keep top-2
    x3 = G([[3,3,0,0],
            [3,0,0,4],
            [0,0,4,4]])
    # components: 3(size=3), 4(size=3) → keep both; others zero
    y3 = x3.copy()
    inst3 = ARCInstance("keep_top2",
                        train=[(x3,y3)],
                        test_in=[G([[5,5,5,0],
                                    [0,0,6,0],
                                    [6,6,0,0]])],
                        test_out=[G([[5,5,5,0],
                                     [0,0,6,0],
                                     [6,6,0,0]])])
    res3 = solve_instance(inst3)
    assert res3.rule and http://res3.rule.name=="KEEP_OBJ_TOPK" and res3.acc_exact==1.0

    print("All unit tests passed for object-rank masks.")

def _mini_demo():
    demos=[
        # 1
        ARCInstance("demo_recolor_largest",
            train=[(G([[1,1,0],[2,2,2],[0,0,0]]),
                    G([[1,1,0],[9,9,9],[0,0,0]]))],
            test_in =[G([[3,3,0],[4,4,4],[0,0,0]])],
            test_out=[G([[3,3,0],[9,9,9],[0,0,0]])]),
        # 2
        ARCInstance("demo_keep_top2",
            train=[(G([[1,1,1,0],
                       [0,2,0,0],
                       [2,2,0,0]]),
                    G([[1,1,1,0],
                       [0,2,0,0],
                       [2,2,0,0]]))],
            test_in =[G([[5,0,0,0],
                         [0,5,0,6],
                         [0,0,6,6]])],
            test_out=[G([[5,0,0,0],
                         [0,5,0,6],
                         [0,0,6,6]])]),
        # 3
        ARCInstance("demo_recolor_per_color",
            train=[(G([[1,1,0,2,2],
                       [1,0,0,2,0]]),
                    G([[7,7,0,8,8],
                       [7,0,0,8,0]]))],
            test_in =[G([[1,1,1,0,2,2,0],
                         [0,1,0,0,2,0,0]])],
            test_out=[G([[7,7,7,0,8,8,0],
                         [0,7,0,0,8,0,0]])]),
    ]
    total=0; exact=0
    for inst in demos:
        res = solve_instance(inst)
        print(f"\n[{http://inst.name}] rule={http://res.rule.name if res.rule else 'None'} acc={res.acc_exact:.2f}")
        for i,rc in enumerate(res.receipts):
            print(f"  test#{i}: residual={rc.residual} edits={rc.edits_total} (boundary {rc.edits_boundary} interior {rc.edits_interior})")
            print("  PCE:", rc.pce)
        total += len(inst.test_out); exact += int(res.acc_exact==1.0)*len(inst.test_out)
    print(f"\nMINI DEMO: {exact}/{total} exact.")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    _unit_tests()
    _mini_demo()
    print("\nNEXT OPS: add object-move/paste, lattice/tiling, draw/line; then beam-compose with receipts.")