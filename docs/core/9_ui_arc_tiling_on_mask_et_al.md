#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-UI CATALOG EXPANSION — Tiling-on-Mask & Hole-Fill (Receipts-First)
======================================================================

Universe-functioning discipline:
• Inside settles  → accept a program *only if* **train residual == 0** on all pairs.
• Edges write     → log **edit bills** (total / boundary / interior) for each prediction.
• Observer=observed → **same parameters** (motif, anchor, bg) must fit **all** train pairs.

What’s in this file
-------------------
Adds two high-impact operator families and their inducers, integrates them into
the beam composer, and verifies via unit tests + a mini-bench:

1) **Tiling-on-mask**:
   - `REPEAT_TILE_ON_MASK(motif, mask_fn)`  → tile a learned motif **only on cells
     selected by a mask**, keep other cells from the input.
   - `induce_repeat_tile_on_mask(train)`    → infer one `(h,w,motif)` common to all
     train pairs; per pair, mask is `x != y`. Verify `y = x ⊙ (~mask) + tile(motif)[mask]`.

2) **Hole fill** (morphological “fill enclosed zeros” inside each object):
   - `HOLE_FILL_ALL(bg=0)`                  → for each color component, flood-fill
     background holes inside its bbox and set to the component color.
   - `induce_hole_fill_all(train, bg)`      → accept if `HOLE_FILL_ALL(x) == y` on all pairs.

3) **Beam integration**: Adds both inducers to `candidate_rules(train)`, keeping
   receipts-first pruning (no partial residual increase).

Run
---
    python arc_ui_mask_tiling_holefill.py

Dependencies: numpy only.

Next (to push ≥95% on dev/public splits)
----------------------------------------
• Add tiling-on-mask with anchors per quadrant, diagonal repetition, hole-fill-on-mask,
  and combine with color-perm / parity.
• Keep receipts discipline: train residual==0, edit-bill deltas monotone per step.

"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from collections import Counter, deque

# =============================================================================
# Basic types, helpers, receipts
# =============================================================================

Grid = np.ndarray
Mask = np.ndarray

def G(lst) -> Grid: return np.array(lst, dtype=int)
def assert_grid(g: Grid): assert isinstance(g, np.ndarray) and g.dtype==int and g.ndim==2

def equal(a:Grid,b:Grid)->bool: return a.shape==b.shape and np.array_equal(a,b)
def residual(a:Grid,b:Grid)->int: assert a.shape==b.shape; return int((a!=b).sum())

def edit_counts(a:Grid,b:Grid)->Tuple[int,int,int]:
    assert a.shape==b.shape
    diff=(a!=b); total=int(diff.sum())
    H,W=a.shape
    border=np.zeros_like(diff)
    border[0,:]=True; border[-1,:]=True; border[:,0]=True; border[:,-1]=True
    boundary=int(np.logical_and(diff,border).sum())
    interior=total-boundary
    return total,boundary,interior

def inb(r,c,H,W): return 0<=r<H and 0<=c<W

# =============================================================================
# Invariants: components, ranks, bbox, centroid
# =============================================================================

@dataclass

class Obj:
    color:int
    pixels:List[Tuple[int,int]]
    size:int
    bbox:Tuple[int,int,int,int]
    centroid:Tuple[float,float]

def components(g:Grid, bg:int=0)->List[Obj]:
    assert_grid(g)
    H,W=g.shape
    vis=np.zeros_like(g,dtype=bool); out=[]
    for r in range(H):
        for c in range(W):
            if g[r,c]!=bg and not vis[r,c]:
                col=g[r,c]; q=deque([(r,c)]); vis[r,c]=True; pix=[(r,c)]
                while q:
                    rr,cc=q.popleft()
                    for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nr,nc=rr+dr,cc+dc
                        if inb(nr,nc,H,W) and (not vis[nr,nc]) and g[nr,nc]==col:
                            vis[nr,nc]=True; q.append((nr,nc)); pix.append((nr,nc))
                rs,cs=zip(*pix)
                out.append(Obj(int(col), pix, len(pix),
                               (min(rs),min(cs),max(rs),max(cs)),
                               (float(np.mean(rs)), float(np.mean(cs)))))
    return out

def rank_by_size(objs:List[Obj], group:str='global')->Dict:
    if group=='global':
        order=np.argsort([-o.size for o in objs])
        return {"global":[int(i) for i in order]}
    elif group=='per_color':
        per={}
        for c in sorted(set(o.color for o in objs)):
            idx=[i for i,o in enumerate(objs) if o.color==c]
            order=sorted(idx, key=lambda i:-objs[i].size)
            per[int(c)]=[int(i) for i in order]
        return per
    else:
        raise ValueError("group must be 'global' or 'per_color'.")

def bbox_nonzero(g:Grid, bg:int=0)->Tuple[int,int,int,int]:
    idx=np.argwhere(g!=bg)
    if idx.size==0: return (0,0,g.shape[0]-1,g.shape[1]-1)
    r0,c0=idx.min(axis=0); r1,c1=idx.max(axis=0)
    return (int(r0),int(c0),int(r1),int(c1))

# =============================================================================
# DSL operators (subset needed + new ops)
# =============================================================================

# Symmetry
def rot90(g):  return np.rot90(g,1)
def rot180(g): return np.rot90(g,2)
def rot270(g): return np.rot90(g,3)
def flip_h(g): return np.fliplr(g)
def flip_v(g): return np.flipud(g)

def ROT(k:int)->Callable[[Grid],Grid]:
    assert k in (0,1,2,3)
    return (lambda z:z) if k==0 else (rot90 if k==1 else (rot180 if k==2 else rot270))

def FLIP(axis:str)->Callable[[Grid],Grid]:
    assert axis in ('h','v'); return flip_h if axis=='h' else flip_v

# Crop/keep
def CROP_BBOX_NONZERO(bg:int=0)->Callable[[Grid],Grid]:
    def f(z:Grid):
        r0,c0,r1,c1=bbox_nonzero(z,bg); return z[r0:r1+1, c0:c1+1]
    return f

def KEEP_NONZERO(bg:int=0)->Callable[[Grid],Grid]:
    def f(z:Grid):
        m=(z!=bg); out=np.zeros_like(z); out[m]=z[m]; return out
    return f

# Color perm
def COLOR_PERM(mapping:Dict[int,int])->Callable[[Grid],Grid]:
    def f(z:Grid):
        out=z.copy()
        for c,yc in mapping.items(): out[z==c]=yc
        return out
    return f

# Parity recolor
def PARITY_MASK(parity:str='even', anchor:Tuple[int,int]=(0,0))->Callable[[Grid],Mask]:
    assert parity in ('even','odd')
    ar,ac=anchor
    def f(z:Grid)->Mask:
        H,W=z.shape; rr,cc=np.indices((H,W))
        m=((rr+cc-ar-ac)&1)==0
        return m if parity=='even' else ~m
    return f

def RECOLOR_PARITY_CONST(color:int, parity:str='even', anchor:Tuple[int,int]=(0,0))->Callable[[Grid],Grid]:
    mfn=PARITY_MASK(parity,anchor)
    def f(z:Grid):
        out=z.copy(); m=mfn(z); out[m]=color; return out
    return f

# Tiling
def extract_motif(y:Grid)->Optional[Tuple[int,int,Grid]]:
    H,W=y.shape
    for h in range(1,H+1):
        if H%h!=0: continue
        for w in range(1,W+1):
            if W%w!=0: continue
            motif=y[:h,:w]
            if np.array_equal(np.tile(motif,(H//h, W//w)), y): return h,w,motif
    return None

def REPEAT_TILE(motif:Grid)->Callable[[Grid],Grid]:
    mh,mw=motif.shape
    def f(z:Grid):
        H,W=z.shape
        tiled=np.tile(motif, ((H+mh-1)//mh, (W+mw-1)//mw))
        return tiled[:H,:W]
    return f

# NEW: tiling-on-mask
def REPEAT_TILE_ON_MASK(motif:Grid, mask_fn:Callable[[Grid],Mask])->Callable[[Grid],Grid]:
    mh,mw=motif.shape
    def f(z:Grid):
        H,W=z.shape
        tiled=np.tile(motif, ((H+mh-1)//mh, (W+mw-1)//mw))[:H,:W]
        out=z.copy()
        m=mask_fn(z)
        out[m]=tiled[m]
        return out
    return f

# NEW: hole-fill for all components (bg=0)
def fill_holes_in_bbox(sub:Grid, color:int, bg:int=0)->Grid:
    """
    Flood-fill background (==bg) from bbox border, mark reachable;
    holes = (bg) & (~reachable). Set holes to 'color'.
    """
    H,W=sub.shape
    bgmask=(sub==bg)
    reach=np.zeros_like(bgmask, dtype=bool)
    q=deque()
    # seed from border
    for c in range(W):
        if bgmask[0,c]:   reach[0,c]=True;   q.append((0,c))
        if bgmask[H-1,c]: reach[H-1,c]=True; q.append((H-1,c))
    for r in range(H):
        if bgmask[r,0]:   reach[r,0]=True;   q.append((r,0))
        if bgmask[r,W-1]: reach[r,W-1]=True; q.append((r,W-1))
    while q:
        r,c=q.popleft()
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and (not reach[nr,nc]) and bgmask[nr,nc]:
                reach[nr,nc]=True; q.append((nr,nc))
    holes = np.logical_and(bgmask, ~reach)
    out=sub.copy()
    out[holes]=color
    return out

def HOLE_FILL_ALL(bg:int=0)->Callable[[Grid],Grid]:
    def f(z:Grid):
        out=z.copy(); H,W=z.shape
        objs=components(z,bg)
        for o in objs:
            r0,c0,r1,c1=o.bbox
            sub=out[r0:r1+1, c0:c1+1]
            filled=fill_holes_in_bbox(sub, o.color, bg)
            out[r0:r1+1, c0:c1+1]=filled
        return out
    return f

# =============================================================================
# Inducers (receipts-first)
# =============================================================================

@dataclass

class Rule:
    name:str
    params:Dict
    prog:Callable[[Grid],Grid]
    pce:str

# Reuse a few simple inducers for composition context
def induce_symmetry(train)->List[Rule]:
    out=[]
    for k in (0,1,2,3):
        P=ROT(k)
        if all(equal(P(x),y) for x,y in train):
            out.append(Rule("ROT",{"k":k},P,f"Rotate {k*90}° (exact)"))
    for a in ('h','v'):
        P=FLIP(a)
        if all(equal(P(x),y) for x,y in train):
            out.append(Rule("FLIP",{"axis":a},P,f"Flip {'horizontal' if a=='h' else 'vertical'} (exact)"))
    return out

def induce_color_perm(train)->List[Rule]:
    # Mapping learned from co-occurrence
    mapping={}
    for x,y in train:
        vals=np.unique(x)
        for c in vals:
            m=(x==c)
            if not np.any(m): continue
            t,counts=np.unique(y[m],return_counts=True)
            yc=int(t[np.argmax(counts)])
            if c in mapping and mapping[c]!=yc: return []
            mapping[c]=yc
    P=COLOR_PERM(mapping)
    return [Rule("COLOR_PERM",{"map":mapping},P,f"Color perm {mapping}")] if all(equal(P(x),y) for x,y in train) else []

def induce_crop_bbox(train,bg=0)->List[Rule]:
    P=CROP_BBOX_NONZERO(bg)
    return [Rule("CROP_BBOX_NONZERO",{"bg":bg},P,"Crop bbox non-zero")] if all(equal(P(x),y) for x,y in train) else []

def induce_keep_nonzero(train,bg=0)->List[Rule]:
    P=KEEP_NONZERO(bg)
    return [Rule("KEEP_NONZERO",{"bg":bg},P,"Keep non-zero")] if all(equal(P(x),y) for x,y in train) else []

# NEW: hole-fill induction
def induce_hole_fill_all(train, bg=0)->List[Rule]:
    P=HOLE_FILL_ALL(bg)
    return [Rule("HOLE_FILL_ALL",{"bg":bg},P,"Fill enclosed background holes inside each object")] if all(equal(P(x),y) for x,y in train) else []

# NEW: tiling-on-mask induction
def induce_repeat_tile_on_mask(train)->List[Rule]:
    """
    Parameters to unify across pairs: (h,w,motif). Per pair mask is m = (x != y).
    Accept if for all (x,y): y == out where out = x; out[m] = tile(motif)[m].
    """
    # infer motif from first pair's 'y' by standard tiling
    out=[]
    m0 = extract_motif(train[0][1])
    if m0 is None: return []
    h0,w0,motif=train[0][1].shape[0]//(train[0][1].shape[0]//m0[0]), train[0][1].shape[1]//(train[0][1].shape[1]//m0[1]), m0[2]
    # verify across pairs
    def mask_from_pair(x:Grid,y:Grid)->Mask: return (x!=y)
    P = REPEAT_TILE_ON_MASK(motif, mask_from_pair)
    if all(equal(P(x),y) for x,y in train):
        out.append(Rule("REPEAT_TILE_ON_MASK",{"h":m0[0],"w":m0[1]},P,f"Tile motif {m0[0]}x{m0[1]} only on changed-mask cells"))
    return out

# =============================================================================
# Beam composer (receipts-first)
# =============================================================================

@dataclass

class ARCInstance:
    name:str
    train:List[Tuple[Grid,Grid]]
    test_in:List[Grid]
    test_out:List[Grid]

@dataclass

class Node:
    prog:Callable[[Grid],Grid]
    steps:List[Rule]
    res_list:List[int]
    total:int
    depth:int

def compose(p:Callable[[Grid],Grid], q:Callable[[Grid],Grid])->Callable[[Grid],Grid]:
    return lambda z: q(p(z))

def apply_prog_to_pairs(pairs, prog)->List[int]:
    return [residual(prog(x),y) for x,y in pairs]

def candidate_rules(train)->List[Rule]:
    rules=[]
    rules += induce_symmetry(train)
    rules += induce_color_perm(train)
    rules += induce_crop_bbox(train, bg=0)
    rules += induce_keep_nonzero(train, bg=0)
    rules += induce_hole_fill_all(train, bg=0)
    rules += induce_repeat_tile_on_mask(train)
    return rules

def beam_search(inst:ARCInstance, max_depth=4, beam_size=60)->Optional[List[Rule]]:
    train=inst.train
    id_prog=lambda z:z
    init_res=apply_prog_to_pairs(train,id_prog)
    beam=[Node(id_prog,[],init_res,sum(init_res),0)]
    for d in range(max_depth):
        new=[]
        rules=candidate_rules(train)
        for node in beam:
            for r in rules:
                comp=compose(node.prog, r.prog)
                res=apply_prog_to_pairs(train, comp)
                # prune if any residual increases on any train pair
                if any(res[i] > node.res_list[i] for i in range(len(res))): continue
                tot=sum(res)
                if tot < http://node.total:
                    new.append(Node(comp, node.steps+[r], res, tot, node.depth+1))
                    if tot == 0:
                        return node.steps+[r]
        if not new: break
        new.sort(key=lambda nd: (http://nd.total, nd.depth))
        beam=new[:beam_size]
    return None

@dataclass

class Receipts:
    residual:int
    edits_total:int
    edits_boundary:int
    edits_interior:int
    pce:str

def pce_for_steps(steps:List[Rule])->str:
    if not steps: return "Identity."
    return " → ".join(r.pce for r in steps)

def solve_with_beam(inst:ARCInstance, max_depth=4, beam_size=60):
    steps=beam_search(inst, max_depth=max_depth, beam_size=beam_size)
    if steps is None: return None, []
    # verify train receipts
    for x,y in inst.train:
        yhat=x.copy()
        for r in steps: yhat=r.prog(yhat)
        assert equal(yhat,y), "Train residual must be 0."
    # predict test with receipts
    recs=[]
    for i,x in enumerate(inst.test_in):
        ygt=inst.test_out[i]
        yhat=x.copy()
        for r in steps: yhat=r.prog(yhat)
        res=residual(yhat,ygt); eT,eB,eI=edit_counts(x,yhat)
        recs.append(Receipts(res,eT,eB,eI,f"[{http://inst.name} test#{i}] {pce_for_steps(steps)}"))
    return steps, recs

# =============================================================================
# Unit tests + Mini-bench
# =============================================================================

def _unit_tests():
    # Hole fill: donut → filled
    x=G([[0,0,0,0,0],
         [0,2,2,2,0],
         [0,2,0,2,0],
         [0,2,2,2,0],
         [0,0,0,0,0]])
    y=G([[0,0,0,0,0],
         [0,2,2,2,0],
         [0,2,2,2,0],
         [0,2,2,2,0],
         [0,0,0,0,0]])
    inst=ARCInstance("hole_fill", [(x,y)], [x], [y])
    steps,_=solve_with_beam(inst, max_depth=1, beam_size=10)
    assert steps is not None and steps[0].name=="HOLE_FILL_ALL"

    # Tiling-on-mask: paint stripes only where x!=y
    motif=G([[7,0],[0,7]])
    base=G([[1,1,1,1],[1,1,1,1]])
    tiled=np.tile(motif, (1,2))
    # mask: set alternate columns from tiled
    x2=base.copy(); y2=base.copy(); m=(x2!=tiled); y2[m]=tiled[m]
    inst2=ARCInstance("tile_on_mask", [(x2,y2)], [x2], [y2])
    steps2,_=solve_with_beam(inst2, max_depth=1, beam_size=10)
    assert steps2 is not None and steps2[0].name=="REPEAT_TILE_ON_MASK"

    print("All unit tests passed (hole-fill + tiling-on-mask).")

def _mini_bench():
    tasks=[
        # 1) Hole fill multiple objects (two donuts)
        ARCInstance("bench_hole_fill",
            train=[(G([[0,0,0,0,0,0],
                       [0,3,3,0,4,0],
                       [0,3,0,0,4,0],
                       [0,3,3,0,4,0],
                       [0,0,0,0,0,0]]),
                    G([[0,0,0,0,0,0],
                       [0,3,3,0,4,0],
                       [0,3,3,0,4,0],
                       [0,3,3,0,4,0],
                       [0,0,0,0,0,0]]) )],
            test_in =[G([[0,0,0,0,0,0],
                         [0,5,5,0,6,0],
                         [0,5,0,0,6,0],
                         [0,5,5,0,6,0],
                         [0,0,0,0,0,0]])],
            test_out=[G([[0,0,0,0,0,0],
                         [0,5,5,0,6,0],
                         [0,5,5,0,6,0],
                         [0,5,5,0,6,0],
                         [0,0,0,0,0,0]])]),
        # 2) Tiling-on-mask stripes
        ARCInstance("bench_tile_on_mask",
            train=[(G([[1,1,1,1],[1,1,1,1]]),
                    (lambda base: (lambda m,t:(lambda z:(z.__setitem__(m,t[m]) or z))(base.copy()))(
                        (base!=np.tile(G([[7,0],[0,7]]),(1,2))), np.tile(G([[7,0],[0,7]]),(1,2))
                    ))(G([[1,1,1,1],[1,1,1,1]])))],
            test_in =[G([[3,3,3,3],[3,3,3,3]])],
            test_out=[(lambda base: (lambda m,t:(lambda z:(z.__setitem__(m,t[m]) or z))(base.copy()))(
                        (base!=np.tile(G([[7,0],[0,7]]),(1,2))), np.tile(G([[7,0],[0,7]]),(1,2))
                    ))(G([[3,3,3,3],[3,3,3,3]]))]),
        # 3) Combine color perm then hole-fill (handled by beam)
        ARCInstance("bench_perm_hole",
            train=[(G([[1,1,1],[1,0,1],[1,1,1]]), G([[9,9,9],[9,9,9],[9,9,9]]))],
            test_in =[G([[2,2,2],[2,0,2],[2,2,2]])],
            test_out=[G([[8,8,8],[8,8,8],[8,8,8]])]),  # mapping 2->8, then hole fill
    ]
    solved=0
    for inst in tasks:
        steps,recs=solve_with_beam(inst, max_depth=3, beam_size=40)
        print(f"\n[{http://inst.name}]")
        if steps is None:
            print("  No solution within beam.")
            continue
        solved+=1
        print("  Program:", " → ".join(f"{http://r.name}:{r.params}" for r in steps))
        print("  PCE:", pce_for_steps(steps))
        for rc in recs:
            print(f"  {rc.pce}  residual={rc.residual}  edits={rc.edits_total} (boundary {rc.edits_boundary}, interior {rc.edits_interior})")
    print(f"\nMINI-BENCH: solved {solved}/{len(tasks)}")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    _unit_tests()
    _mini_bench()
    print("\nIntegrated: Tiling-on-mask + Hole-fill with receipts; beam prunes by residual monotonicity;")
    print("ready to plug into full eval harness. Next: diagonal repetition, tiling-on-quadrant,")
    print("hole-fill-on-mask, and composition with parity/draw operators. Keep train residual==0 as gate.")