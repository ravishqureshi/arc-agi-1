#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI–ARC PRESS DEMO — Smart, Receipts-First Abstraction & Reasoning
==================================================================

What this is
------------
A clean, runnable **press/demo harness** that:
  • Solves a *suite* of ARC-style tasks with a **small, transparent rule catalog**,
  • **Verifies** every learned rule on train pairs (residual = 0),
  • **Explains** each prediction with Proof-Carrying English (PCE),
  • Reports **receipts** (exact residuals, edit bills, boundary/interior counts),
  • Prints a compact “press summary” that addresses the common question:
      “Great demo — but ARC has hundreds of tasks; how does this generalize?”

Honest scope
------------
• This *does not* claim we solved the full official ARC benchmark.
• It demonstrates **Universe Intelligence (UI)** methodology:
    learn invariants from train → verify on train → predict test **with receipts**.
• The catalog is intentionally small; add more rules to scale coverage.

Run
---
    python ui_arc_press_demo.py

Dependencies: numpy only.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional
import numpy as np
from collections import Counter, deque
from time import perf_counter

# -----------------------
# Grid helpers & transforms
# -----------------------
Grid = np.ndarray  # dtype=int

def G(lst): return np.array(lst, dtype=int)
def rot90(g):  return np.rot90(g, k=1)
def rot180(g): return np.rot90(g, k=2)
def rot270(g): return np.rot90(g, k=3)
def flip_h(g): return np.fliplr(g)
def flip_v(g): return np.flipud(g)

TRANSFORMS = [
    ("rot90",  rot90),
    ("rot180", rot180),
    ("rot270", rot270),
    ("flip_h", flip_h),
    ("flip_v", flip_v),
    ("id",     lambda x: x),
]

# -----------------------
# Components & bbox
# -----------------------
def bbox_nonzero(g: Grid, bg: int=0):
    idx = np.argwhere(g != bg)
    if idx.size == 0:
        return 0,0,g.shape[0]-1,g.shape[1]-1
    r0,c0 = idx.min(axis=0)
    r1,c1 = idx.max(axis=0)
    return int(r0),int(c0),int(r1),int(c1)

def crop_bbox(g: Grid, bg: int=0) -> Grid:
    r0,c0,r1,c1 = bbox_nonzero(g, bg)
    return g[r0:r1+1, c0:c1+1]

def connected_components(g: Grid, bg: int=0):
    H,W = g.shape
    vis = np.zeros_like(g, dtype=bool)
    comps=[]
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
                            vis[nr,nc]=True
                            q.append((nr,nc)); pix.append((nr,nc))
                comps.append((col,pix))
    return comps

# -----------------------
# Receipts
# -----------------------
@dataclass

class Receipts:
    residual: int
    edits_total: int
    edits_boundary: int
    edits_interior: int
    explanation: str

def edits_between(a: Grid, b: Grid):
    assert a.shape==b.shape
    diff=(a!=b)
    total=int(diff.sum())
    H,W=a.shape
    border=np.zeros_like(diff)
    border[0,:]=True; border[-1,:]=True; border[:,0]=True; border[:,-1]=True
    boundary=int(np.logical_and(diff,border).sum())
    interior=total-boundary
    return total,boundary,interior

# -----------------------
# Rule catalog (induce → verify → apply)
# -----------------------
@dataclass

class Rule:
    name: str
    params: dict
    f: Callable[[Grid], Grid]

def r_symmetry(train):
    for name,tf in TRANSFORMS:
        if all(tf(x).shape==y.shape and np.array_equal(tf(x),y) for x,y in train):
            return Rule(f"symmetry:{name}",{},tf)
    return None

def r_color_perm(train):
    mapping={}
    for x,y in train:
        vals=np.unique(x)
        for c in vals:
            mask=(x==c)
            yvals,counts=np.unique(y[mask],return_counts=True)
            yc=int(yvals[np.argmax(counts)])
            if c in mapping and mapping[c]!=yc: return None
            mapping[c]=yc
    def f(z):
        out=z.copy()
        for c,yc in mapping.items(): out[z==c]=yc
        return out
    if all(np.array_equal(f(x),y) for x,y in train):
        return Rule("color_perm",{"map":mapping},f)
    return None

def r_crop_bbox(train, bg=0):
    if all(np.array_equal(crop_bbox(x,bg),y) for x,y in train):
        return Rule("crop_bbox",{"bg":bg},lambda z: crop_bbox(z,bg))
    return None

def r_move_bbox_to_origin(train, bg=0):
    def move(z):
        r0,c0,r1,c1=bbox_nonzero(z,bg)
        sub=z[r0:r1+1,c0:c1+1]
        out=np.full_like(z,bg); out[:sub.shape[0],:sub.shape[1]]=sub
        return out
    if all(np.array_equal(move(x),y) for x,y in train):
        return Rule("move_bbox_to_origin",{"bg":bg},move)
    return None

def r_mirror_left_to_right(train):
    def mirror_lr(z):
        H,W=z.shape
        assert W%2==0
        mid=W//2
        out=z.copy()
        out[:,mid:]=np.fliplr(out[:,:mid])
        return out
    try:
        if all(np.array_equal(mirror_lr(x),y) for x,y in train):
            return Rule("mirror_left_to_right",{},mirror_lr)
    except AssertionError:
        return None
    return None

def r_component_size_recolor(train):
    mapping={}
    for x,y in train:
        comps=connected_components(x)
        sizes=[len(p) for _,p in comps]
        order=np.argsort(-np.array(sizes))
        for rk,idx in enumerate(order):
            c,pix=comps[idx]
            ys=[int(y[r,cc]) for r,cc in pix]
            if not ys: continue
            new=Counter(ys).most_common(1)[0][0]
            key=(c,rk)
            if key in mapping and mapping[key]!=new: return None
            mapping[key]=new
    def f(z):
        out=z.copy()
        comps=connected_components(z)
        sizes=[len(p) for _,p in comps]
        order=np.argsort(-np.array(sizes))
        for rk,idx in enumerate(order):
            c,pix=comps[idx]
            key=(c,rk)
            if key in mapping:
                for r,cc in pix: out[r,cc]=mapping[key]
        return out
    if all(np.array_equal(f(x),y) for x,y in train):
        return Rule("component_size_recolor",{"map":mapping},f)
    return None

CATALOG=[r_symmetry,r_color_perm,r_crop_bbox,r_move_bbox_to_origin,r_mirror_left_to_right,r_component_size_recolor]

# -----------------------
# Solver harness
# -----------------------
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
    accuracy: float

def solve_instance(inst: ARCInstance) -> SolveResult:
    rule=None
    for induce in CATALOG:
        r=induce(inst.train)
        if r is not None:
            rule=r; break
    preds=[]; recs=[]
    if rule:
        for i,x in enumerate(inst.test_in):
            yhat=rule.f(x)
            residual=int(np.sum(yhat!=inst.test_out[i]))
            total,bound,interior=edits_between(x,yhat)
            expl=f"[{http://inst.name} #{i}] {http://rule.name}: edits={total} (boundary {bound}, interior {interior}); residual={residual}."
            recs.append(Receipts(residual,total,bound,interior,expl))
            preds.append(yhat)
    else:
        for i,x in enumerate(inst.test_in):
            preds.append(x.copy())
            recs.append(Receipts(residual=int(np.sum(x!=inst.test_out[i])),edits_total=0,edits_boundary=0,edits_interior=0,explanation="No rule matched."))
    acc=float(np.mean([int(np.array_equal(p,inst.test_out[i])) for i,p in enumerate(preds)]))
    return SolveResult(http://inst.name,rule,preds,recs,acc)

# -----------------------
# Demo suite (10 curated tasks; easily extendable)
# -----------------------
def suite():
    tasks=[]

    # 1) color perm
    train=[(G([[1,1,0],[2,2,0]]),G([[3,3,0],[4,4,0]])),
           (G([[2,1,0],[2,1,0]]),G([[4,3,0],[4,3,0]]))]
    test_in=[G([[1,2,2],[1,0,0]])]
    test_out=[G([[3,4,4],[3,0,0]])]
    tasks.append(ARCInstance("color_perm",train,test_in,test_out))

    # 2) rot90
    a=G([[1,0,0],[1,2,0],[0,2,2]]); b=rot90(a)
    tasks.append(ARCInstance("rot90",[(a,b)],[G([[0,9,9],[0,0,9],[8,8,0]])],[rot90(G([[0,9,9],[0,0,9],[8,8,0]]))]))

    # 3) crop bbox
    x3=G([[0,0,0,0],[0,5,5,0],[0,5,0,0]]); y3=crop_bbox(x3)
    tasks.append(ARCInstance("crop_bbox",[(x3,y3)],[G([[0,0,0],[0,7,0],[0,7,7]])],[crop_bbox(G([[0,0,0],[0,7,0],[0,7,7]]))]))

    # 4) move_bbox_to_origin
    x4=G([[0,0,0],[0,2,2],[0,2,0]]); y4=G([[2,2,0],[2,0,0],[0,0,0]])
    tasks.append(ARCInstance("move_bbox_to_origin",[(x4,y4)],[G([[0,0,0,0],[0,0,3,3],[0,0,3,0]])],[G([[3,3,0,0],[3,0,0,0],[0,0,0,0]])]))

    # 5) mirror_left_to_right (even width)
    x5=G([[1,0,0,0],[0,1,0,0]]); y5=G([[1,0,0,1],[0,1,1,0]])
    tasks.append(ARCInstance("mirror_left_to_right",[(x5,y5)],[G([[2,0,0,0],[0,2,0,0]])],[G([[2,0,0,2],[0,2,2,0]])]))

    # 6) symmetry: flip_h
    x6=G([[1,0,0],[0,1,0]]); y6=flip_h(x6)
    tasks.append(ARCInstance("flip_h",[(x6,y6)],[G([[9,0,0],[0,9,0]])],[flip_h(G([[9,0,0],[0,9,0]]))]))

    # 7) symmetry: rot180
    x7=G([[0,1],[2,0]]); y7=rot180(x7)
    tasks.append(ARCInstance("rot180",[(x7,y7)],[G([[0,3],[4,0]])],[rot180(G([[0,3],[4,0]]))]))

    # 8) component_size_recolor (largest component → color 9)
    x8=G([[1,1,0],[2,0,2],[2,0,2]])
    y8=x8.copy(); y8[x8==2]=9  # assume '2' is largest component in train
    tasks.append(ARCInstance("component_size_recolor",[(x8,y8)],[G([[3,3,0],[4,0,4],[4,0,4]])],[G([[3,3,0],[9,0,9],[9,0,9]])]))

    # 9) symmetry: rot270
    x9=G([[0,5,5],[0,0,5]]); y9=rot270(x9)
    tasks.append(ARCInstance("rot270",[(x9,y9)],[G([[0,6,6],[0,0,6]])],[rot270(G([[0,6,6],[0,0,6]]))]))

    # 10) color perm 2
    train10=[(G([[1,2],[2,1]]),G([[7,8],[8,7]]))]
    tasks.append(ARCInstance("color_perm_2",train10,[G([[2,1],[1,2]])],[G([[8,7],[7,8]])]))

    return tasks

# -----------------------
# Press demo runner
# -----------------------
def run_press_demo():
    tasks=suite()
    total=0; correct=0; solved=0
    t0=perf_counter()
    for inst in tasks:
        res=solve_instance(inst)
        total += len(inst.test_out)
        correct += int(res.accuracy==1.0)*len(inst.test_out)
        if res.rule: solved += 1
        print(f"\n[{http://inst.name}] rule={http://res.rule.name if res.rule else 'None'}  acc={res.accuracy:.2f}")
        for i,rc in enumerate(res.receipts):
            print(f"  test#{i}: {rc.explanation}")
    t1=perf_counter()
    print("\n=== PRESS SUMMARY ================================")
    print(f"Tasks: {len(tasks)} | Solved with rule: {solved} | Exact test accuracy (per task): {correct}/{total}")
    print(f"End-to-end time: {(t1-t0):.3f}s")
    print("Method: learn invariant → verify on train (residual=0) → predict test with receipts.")
    print("Every line above is traceable: rule name, edits, boundary/interior split, residual.")
    print("To scale coverage: extend rule catalog (tiling, repetition, affine, object arithmetic, masking).")
    print("This is a *receipts-first* ARC approach — not a black-box guesser.")

if __name__ == "__main__":
    run_press_demo()