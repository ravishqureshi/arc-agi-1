#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-UI NEXT: Object MOVE / PASTE via bbox/centroid (with receipts)
==================================================================

Universe-functioning brief
--------------------------
• Inside settles → we only accept a rule when **train residual = 0**.
• Edges write → the MOVE/PASTE bill is measured as edited cells (boundary vs interior).
• Observer = observed → the *same* parameters (rank, Δ) must fit **all** train pairs.

What this delivers
------------------
1) Object primitives (already in prior step): connected components, size, bbox, centroid,
   global and per-color ranking.

2) DSL ops (new):
   • MOVE_OBJ_RANK(rank, Δ=(dr,dc), group='global'|'per_color', clear=True, bg=0)
     – Find the rank-th object mask; translate it by Δ; if clear=True, erase source.
   • COPY_OBJ_RANK(rank, Δ, group, bg=0)
     – Same as MOVE but **preserve** source (paste-only).

3) Induction (receipts-first):
   • induce_move_or_copy_obj_rank(train, group, bg)
     – For each train pair: pick rank=0 by default (largest), identify source obj in X.
     – In Y, find object with **same color and size** and whose **shape matches** when
       X’s mask is shifted by Δ. Require that the inferred **Δ** is **identical** across
       all train pairs (observer=observed). Verify the MOVE (or COPY) program gives
       **exact** Y on all train pairs. Otherwise, fail.

4) Receipts printed for every prediction:
   – residual (# mismatched cells) — must be 0 on train; shown on test.
   – edit bill (total, boundary, interior).
   – PCE (Proof-Carrying English): human-readable, tied to invariants.

Run
---
    python arc_ui_move_paste.py

Dependencies: numpy only.

Next steps (after this passes)
------------------------------
• Object copy-paste between masks/quadrants, collision handling,
  candidate-Δ enumeration over all same-color/size targets (when multiple),
  then integrate into beam composition with receipts at each step.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from collections import Counter, deque

# =============================================================================
# Types & helpers
# =============================================================================

Grid = np.ndarray
Mask = np.ndarray

def G(lst) -> Grid: return np.array(lst, dtype=int)
def assert_grid(g: Grid): assert isinstance(g, np.ndarray) and g.dtype==int and g.ndim==2

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
# Object primitives & ranking
# =============================================================================

@dataclass

class Obj:
    color: int
    pixels: List[Tuple[int,int]]
    size: int
    bbox: Tuple[int,int,int,int]      # (r0,c0,r1,c1)
    centroid: Tuple[float,float]

def connected_components(g: Grid, bg: int=0) -> List[Obj]:
    assert_grid(g)
    H,W=g.shape
    vis=np.zeros_like(g,dtype=bool)
    out: List[Obj]=[]
    for r in range(H):
        for c in range(W):
            if g[r,c]!=bg and not vis[r,c]:
                col=g[r,c]
                q=deque([(r,c)]); vis[r,c]=True
                pix=[(r,c)]
                while q:
                    rr,cc=q.popleft()
                    for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nr,nc=rr+dr,cc+dc
                        if 0<=nr<H and 0<=nc<W and not vis[nr,nc] and g[nr,nc]==col:
                            vis[nr,nc]=True; q.append((nr,nc)); pix.append((nr,nc))
                rs,cs=zip(*pix); r0,r1=min(rs),max(rs); c0,c1=min(cs),max(cs)
                out.append(Obj(
                    color=int(col),
                    pixels=pix,
                    size=len(pix),
                    bbox=(r0,c0,r1,c1),
                    centroid=(float(np.mean(rs)), float(np.mean(cs)))
                ))
    return out

def rank_objects(objs: List[Obj], group: str='global') -> Dict:
    if group=='global':
        order = np.argsort([-o.size for o in objs])
        return {"global":[int(i) for i in order]}
    elif group=='per_color':
        per: Dict[int,List[int]]={}
        for c in sorted(set(o.color for o in objs)):
            idx=[i for i,o in enumerate(objs) if o.color==c]
            order=sorted(idx, key=lambda i: -objs[i].size)
            per[int(c)]=[int(i) for i in order]
        return per
    else:
        raise ValueError("group must be 'global' or 'per_color'.")

# =============================================================================
# DSL ops — MOVE / COPY by object rank
# =============================================================================

def in_bounds(r:int,c:int,H:int,W:int)->bool: return 0<=r<H and 0<=c<W

def MOVE_OBJ_RANK(rank:int, delta:Tuple[int,int], group:str='global', clear:bool=True, bg:int=0)\
        -> Callable[[Grid], Grid]:
    """
    Find the rank-th object(s) (global or per-color largest), translate by delta=(dr,dc).
    If clear=True, erase source pixels (move). Else, leave source (copy/paste).
    """
    dr,dc = delta
    def f(z: Grid) -> Grid:
        objs = connected_components(z, bg)
        out = z.copy()
        if not objs: return out
        ranks = rank_objects(objs, group=group)
        idx_list=[]
        if group=='global':
            ord_list = ranks["global"]
            if rank < len(ord_list): idx_list=[ord_list[rank]]
        else:  # per_color: apply to each color’s rank-th object
            for _, ord_list in ranks.items():
                if rank < len(ord_list): idx_list.append(ord_list[rank])
        for idx in idx_list:
            obj=objs[idx]
            # compute new positions; discard any pixel falling out of bounds
            new_pix=[]
            for (r,c) in obj.pixels:
                nr, nc = r+dr, c+dc
                if in_bounds(nr,nc,*z.shape): new_pix.append((nr,nc))
            # clear source if move
            if clear:
                for (r,c) in obj.pixels: out[r,c]=bg
            # paste
            for (nr,nc) in new_pix: out[nr,nc]=obj.color
        return out
    return f

def COPY_OBJ_RANK(rank:int, delta:Tuple[int,int], group:str='global', bg:int=0)\
        -> Callable[[Grid], Grid]:
    return MOVE_OBJ_RANK(rank, delta, group=group, clear=False, bg=bg)

# =============================================================================
# Induction: learn Δ from train by centroid (and shape check)
# =============================================================================

@dataclass

class Rule:
    name: str
    params: Dict
    prog: Callable[[Grid], Grid]

def shape_mask(objs: List[Obj], idx:int, H:int, W:int)->Mask:
    m=np.zeros((H,W),dtype=bool)
    for (r,c) in objs[idx].pixels: m[r,c]=True
    return m

def shift_mask(m: Mask, dr:int, dc:int) -> Mask:
    H,W=m.shape
    out=np.zeros_like(m)
    # shift by copying individual True cells; clip at borders
    rr,cc=np.where(m)
    nr=rr+dr; nc=cc+dc
    keep = (nr>=0)&(nr<H)&(nc>=0)&(nc<W)
    out[nr[keep], nc[keep]] = True
    return out

def induce_move_or_copy_obj_rank(train: List[Tuple[Grid,Grid]],
                                 group:str='global', rank:int=0, bg:int=0) -> Optional[Rule]:
    """
    Infer a single Δ=(dr,dc) that maps the rank-th object(s) from X to Y
    on **all** train pairs, either as MOVE (clear=True) or COPY (clear=False).
    We:
      1) Extract rank-th object mask in X,
      2) Among Y’s objects of same color & size, pick the one whose mask matches
         X’s mask shifted by Δ,
      3) Require the same Δ for all train pairs,
      4) Test MOVE; if fails, test COPY; accept only if residual=0 for all train pairs.
    """
    deltas=[]
    for x,y in train:
        H,W=x.shape
        Xobjs=connected_components(x,bg)
        if not Xobjs: return None
        ranks=rank_objects(Xobjs, group=group)
        idxs=[]
        if group=='global':
            ord_list=ranks["global"]
            if rank >= len(ord_list): return None
            idxs=[ord_list[rank]]
        else:
            # per_color: we require identical Δ for each color’s rank-th object;
            # we’ll just use the first color occurrence to infer Δ per pair
            for _, ord_list in ranks.items():
                if rank < len(ord_list):
                    idxs=[ord_list[rank]]; break
            if not idxs: return None
        xi=idxs[0]
        xmask=shape_mask(Xobjs, xi, H,W)
        c=Xobjs[xi].color; sz=Xobjs[xi].size
        # find candidate in Y: same color & size
        Yobjs=connected_components(y,bg)
        cand=[j for j,o in enumerate(Yobjs) if o.color==c and o.size==sz]
        if not cand: return None
        # pick Δ by centroid difference from first such candidate
        found=None
        for j in cand:
            dx = int(round(Yobjs[j].centroid[0] - Xobjs[xi].centroid[0]))
            dy = int(round(Yobjs[j].centroid[1] - Xobjs[xi].centroid[1]))
            # shape check: shifted xmask should match y's mask
            ymask=shape_mask(Yobjs, j, H,W)
            if np.array_equal(shift_mask(xmask, dx, dy), ymask):
                found=(dx,dy); break
        if found is None: return None
        deltas.append(found)
    # Require identical Δ across all pairs
    if any(d != deltas[0] for d in deltas): return None
    Δ = deltas[0]

    # Test MOVE then COPY
    move_prog = MOVE_OBJ_RANK(rank, Δ, group=group, clear=True, bg=bg)
    if all(np.array_equal(move_prog(x), y) for x,y in train):
        return Rule("MOVE_OBJ_RANK", {"rank":rank,"group":group,"delta":Δ,"clear":True,"bg":bg}, move_prog)
    copy_prog = COPY_OBJ_RANK(rank, Δ, group=group, bg=bg)
    if all(np.array_equal(copy_prog(x), y) for x,y in train):
        return Rule("COPY_OBJ_RANK", {"rank":rank,"group":group,"delta":Δ,"clear":False,"bg":bg}, copy_prog)
    return None

# =============================================================================
# Solver harness with receipts + PCE
# =============================================================================

@dataclass

class ARCInstance:
    name: str
    train: List[Tuple[Grid,Grid]]
    test_in: List[Grid]
    test_out: List[Grid]]

@dataclass

class Receipts:
    residual: int
    edits_total: int
    edits_boundary: int
    edits_interior: int
    pce: str

@dataclass

class SolveResult:
    name: str
    rule: Optional[Rule]
    preds: List[Grid]
    receipts: List[Receipts]
    acc_exact: float

def pce_for_rule(rule: Rule) -> str:
    if rule is None: return "No rule."
    d=rule.params['delta']; r=rule.params['rank']; grp=rule.params['group']
    kind="MOVE" if rule.params['clear'] else "COPY"
    rank_txt="largest" if r==0 else f"rank {r}"
    scope="globally" if grp=='global' else "for each color"
    return f"{kind} the {rank_txt} object {scope} by Δ={d} (verified on train: residual=0)."

def solve_instance(inst: ARCInstance, group: str='global', rank: int=0, bg: int=0) -> SolveResult:
    rule = induce_move_or_copy_obj_rank(inst.train, group=group, rank=rank, bg=bg)
    preds=[]; recs=[]
    if rule:
        # train receipts safety
        assert all(np.array_equal(rule.prog(x),y) for x,y in inst.train), "Train residual must be 0."
        pce=pce_for_rule(rule)
        for i,x in enumerate(inst.test_in):
            yhat = rule.prog(x); ygt=inst.test_out[i]
            residual = int(np.sum(yhat!=ygt))
            eT,eB,eI = edit_counts(x,yhat)
            recs.append(Receipts(residual,eT,eB,eI,f"[{http://inst.name} test#{i}] {pce}"))
            preds.append(yhat)
    else:
        for i,x in enumerate(inst.test_in):
            yhat=x.copy(); ygt=inst.test_out[i]
            residual=int(np.sum(yhat!=ygt))
            recs.append(Receipts(residual,0,0,0,f"[{http://inst.name} test#{i}] No rule matched."))
            preds.append(yhat)
    acc=float(np.mean([int(np.array_equal(p, inst.test_out[i])) for i,p in enumerate(preds)]))
    return SolveResult(http://inst.name, rule, preds, recs, acc)

# =============================================================================
# Unit tests & mini demo
# =============================================================================

def _unit_tests():
    # MOVE largest by Δ=(+1,+2)
    x = G([[0,0,0,0,0],
           [0,1,1,0,0],
           [0,1,1,0,0],
           [0,0,0,0,0]])
    y = x.copy()
    # erase source; paste moved block at rows+1, cols+2
    y[1:3,1:3]=0; y[2:4,3:5]=1
    inst = ARCInstance("move_largest",
                       train=[(x,y)],
                       test_in=[G([[0,2,2,0,0],
                                   [0,2,2,0,0],
                                   [0,0,0,0,0],
                                   [0,0,0,0,0]])],
                       test_out=[G([[0,0,0,2,2],
                                    [0,0,0,2,2],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0]])])
    res = solve_instance(inst, group='global', rank=0, bg=0)
    assert res.rule and http://res.rule.name=="MOVE_OBJ_RANK" and res.acc_exact==1.0

    # COPY largest by Δ=(0,+3)
    x2=G([[3,3,0,0,0,0],
          [3,3,0,0,0,0]])
    y2=x2.copy(); y2[:,3:5]=x2[:,:2]  # paste at +3 cols, keep source
    inst2=ARCInstance("copy_largest",
                      train=[(x2,y2)],
                      test_in=[G([[4,4,0,0,0,0],
                                  [4,4,0,0,0,0]])],
                      test_out=[G([[4,4,0,4,4,0],
                                   [4,4,0,4,4,0]])])
    res2 = solve_instance(inst2, group='global', rank=0, bg=0)
    assert res2.rule and http://res2.rule.name=="COPY_OBJ_RANK" and res2.acc_exact==1.0

    print("All unit tests passed for MOVE/COPY.")

def _mini_demo():
    demos=[
        # MOVE example
        ARCInstance("demo_move",
            train=[(G([[0,0,0,0,0],
                       [0,7,7,0,0],
                       [0,7,7,0,0],
                       [0,0,0,0,0]]),
                    G([[0,0,0,0,0],
                       [0,0,0,0,0],
                       [0,0,0,7,7],
                       [0,0,0,7,7]]))],
            test_in=[G([[0,0,0,0,0],
                        [0,5,5,0,0],
                        [0,5,5,0,0],
                        [0,0,0,0,0]])],
            test_out=[G([[0,0,0,0,0],
                         [0,0,0,0,0],
                         [0,0,0,5,5],
                         [0,0,0,5,5]])]),
        # COPY example
        ARCInstance("demo_copy",
            train=[(G([[0,9,9,0,0,0],
                       [0,9,9,0,0,0]]),
                    G([[0,9,9,0,9,9,0],
                       [0,9,9,0,9,9,0]]))],
            test_in=[G([[0,2,2,0,0,0],
                        [0,2,2,0,0,0]])],
            test_out=[G([[0,2,2,0,2,2,0],
                         [0,2,2,0,2,2,0]])]),
        # MOVE per-color largest (two colors)
        ARCInstance("demo_move_per_color",
            train=[(G([[1,1,0,2,2],
                       [1,0,0,2,0],
                       [0,0,0,0,0]]),
                    G([[0,0,1,1,0],
                       [0,0,1,0,0],
                       [0,2,2,2,0]]))],
            test_in=[G([[3,3,0,4,4],
                        [3,0,0,4,0],
                        [0,0,0,0,0]])],
            test_out=[G([[0,0,3,3,0],
                         [0,0,3,0,0],
                         [0,4,4,4,0]])]),
    ]
    total=0; exact=0
    for inst in demos:
        res = solve_instance(inst, group='per_color' if "per_color" in http://inst.name else 'global', rank=0, bg=0)
        print(f"\n[{http://inst.name}] rule={http://res.rule.name if res.rule else 'None'} acc={res.acc_exact:.2f}")
        for i, rc in enumerate(res.receipts):
            print(f"  test#{i}: residual={rc.residual} edits={rc.edits_total} (boundary {rc.edits_boundary} interior {rc.edits_interior})")
            print(f"  PCE: {rc.pce}")
        total += len(inst.test_out); exact += int(res.acc_exact==1.0)*len(inst.test_out)
    print(f"\nMINI DEMO: {exact}/{total} exact.")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    _unit_tests()
    _mini_demo()
    print("\nNEXT: enumerate candidate Δ from all same-color/size targets; "
          "add collision handling & paste-between-regions; then integrate into "
          "beam composition with receipts at each step.")