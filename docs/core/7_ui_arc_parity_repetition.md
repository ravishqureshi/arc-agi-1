#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-UI COMPLETION — Parity & Repetition Catalog + Beam + Receipts
==================================================================

Full universe-functioning step:
• Inside settles → accept only programs with **train residual = 0**.
• Edges write → log **edit bills** (total / boundary / interior).
• Observer = observed → **same parameters** must fit **all** train pairs.

What’s new in this completion step
----------------------------------
1) **Parity** operators (checkerboard-style):
   • PARITY_MASK(even|odd, anchor=(r0,c0))
   • RECOLOR_PARITY_CONST(color, parity)  —— recolor parity cells to single color
   • INDUCE: detect that y = x except parity cells switched to one constant color.

2) **Repetition** operators:
   • REPEAT_TILE(motif_rect)        —— tile a learned motif across the whole grid
   • REPEAT_OBJECT_CHAIN(rank, Δ, k, clear/preserve, group) —— copy/move the rank-th
     object along Δ, k-1 times (handles collisions by “target wins”).
   • INDUCE:
       – tile: find minimal motif that tiles to y; verify on all train pairs;
       – object-chain: infer Δ, k from train by centroid & shape checks; unify across pairs.

3) **Beam search** updated: integrates the new inducers; prunes any partial that
   increases residual on any pair; stops at residual=0 with **PCE** and **receipts**.

Run
---
    python arc_ui_completion.py

Dependencies: numpy only.

Next steps (evaluation)
-----------------------
• Add 30–60 more curated tasks; measure accuracy; iterate on failing categories.
• Run public ARC-AGI split in an offline harness with the same receipts discipline.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from collections import Counter, deque
from time import perf_counter

# =============================================================================
# Types, helpers, receipts
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
                        if 0<=nr<H and 0<=nc<W and not vis[nr,nc] and g[nr,nc]==col:
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
    else: raise ValueError

def bbox_nonzero(g:Grid, bg:int=0)->Tuple[int,int,int,int]:
    idx=np.argwhere(g!=bg)
    if idx.size==0: return (0,0,g.shape[0]-1,g.shape[1]-1)
    r0,c0=idx.min(axis=0); r1,c1=idx.max(axis=0)
    return (int(r0),int(c0),int(r1),int(c1))

# =============================================================================
# DSL: basic ops (symmetry, crop, keep, color perm)
# =============================================================================

def rot90(g):  return np.rot90(g,1)
def rot180(g): return np.rot90(g,2)
def rot270(g): return np.rot90(g,3)
def flip_h(g): return np.fliplr(g)
def flip_v(g): return np.flipud(g)

def ROT(k:int)->Callable[[Grid],Grid]:
    assert k in (0,1,2,3)
    if k==0: return lambda z:z
    if k==1: return rot90
    if k==2: return rot180
    return rot270

def FLIP(axis:str)->Callable[[Grid],Grid]:
    assert axis in ('h','v'); return flip_h if axis=='h' else flip_v

def CROP_BBOX_NONZERO(bg:int=0)->Callable[[Grid],Grid]:
    def f(z:Grid):
        r0,c0,r1,c1=bbox_nonzero(z,bg); return z[r0:r1+1, c0:c1+1]
    return f

def KEEP_NONZERO(bg:int=0)->Callable[[Grid],Grid]:
    def f(z:Grid):
        m=(z!=bg); out=np.zeros_like(z); out[m]=z[m]; return out
    return f

def COLOR_PERM(mapping:Dict[int,int])->Callable[[Grid],Grid]:
    def f(z:Grid):
        out=z.copy()
        for c,yc in mapping.items(): out[z==c]=yc
        return out
    return f

# =============================================================================
# Parity operators
# =============================================================================

def PARITY_MASK(parity:str='even', anchor:Tuple[int,int]=(0,0))->Callable[[Grid],Mask]:
    assert parity in ('even','odd')
    ar,ac=anchor
    def f(z:Grid)->Mask:
        H,W=z.shape
        rr,cc=np.indices((H,W))
        parity_mask=((rr+cc- ar - ac) & 1)==0  # even if sum matches anchor parity
        if parity=='odd': parity_mask = ~parity_mask
        return parity_mask
    return f

def RECOLOR_PARITY_CONST(color:int, parity:str='even', anchor:Tuple[int,int]=(0,0))->Callable[[Grid],Grid]:
    mask_fn=PARITY_MASK(parity, anchor)
    def f(z:Grid)->Grid:
        out=z.copy(); m=mask_fn(z); out[m]=color; return out
    return f

# =============================================================================
# Repetition — tiling motif & object chains
# =============================================================================

def extract_motif(y:Grid)->Optional[Tuple[int,int,Grid]]:
    """
    Try to find the smallest motif (h,w) that tiles exactly to y.
    Return (h,w,motif) or None if no tiling found.
    """
    H,W=y.shape
    for h in range(1,H+1):
        if H%h!=0: continue
        for w in range(1,W+1):
            if W%w!=0: continue
            motif=y[:h,:w]
            tiled=np.tile(motif, (H//h, W//w))
            if np.array_equal(tiled, y):
                return h,w,motif
    return None

def REPEAT_TILE(motif:Grid)->Callable[[Grid],Grid]:
    mh,mw=motif.shape
    def f(z:Grid)->Grid:
        H,W=z.shape
        tiled=np.tile(motif, ( (H+mh-1)//mh, (W+mw-1)//mw ))
        return tiled[:H,:W]
    return f

def inb(r,c,H,W): return 0<=r<H and 0<=c<W

def MOVE_OR_COPY_OBJ_RANK(rank:int, delta:Tuple[int,int], k:int=1, group:str='global', clear:bool=True, bg:int=0)\
        -> Callable[[Grid],Grid]:
    dr,dc=delta
    def f(z:Grid)->Grid:
        H,W=z.shape; objs=components(z,bg); out=z.copy()
        if not objs: return out
        ranks=rank_by_size(objs,group)
        idxs=[]
        if group=='global':
            ord_list=ranks["global"]; 
            if rank<len(ord_list): idxs=[ord_list[rank]]
        else:
            for _,ord_list in ranks.items():
                if rank<len(ord_list): idxs.append(ord_list[rank])
        # clear source for move
        if clear:
            for idx in idxs:
                for (r,c) in objs[idx].pixels: out[r,c]=bg
        # paste chain
        for idx in idxs:
            col=objs[idx].color
            base=objs[idx].pixels
            for t in range(1, k+1):
                for (r,c) in base:
                    nr,nc=r+dr*t, c+dc*t
                    if inb(nr,nc,H,W): out[nr,nc]=col  # target wins
        return out
    return f

# =============================================================================
# Induction utilities
# =============================================================================

@dataclass

class Rule:
    name:str
    params:Dict
    prog:Callable[[Grid],Grid]
    pce:str

# Symmetry
def induce_symmetry(train)->List[Rule]:
    out=[]; cand=[("ROT",{"k":k},ROT(k),f"Rotate {k*90}°") for k in (0,1,2,3)]
    cand += [("FLIP",{"axis":a},FLIP(a),f"Flip {'h' if a=='h' else 'v'}") for a in ('h','v')]
    for name,params,prog,txt in cand:
        if all(equal(prog(x),y) for x,y in train):
            out.append(Rule(name,params,prog,txt+" (exact on train)"))
    return out

# Crop/keep
def induce_crop_bbox(train,bg=0)->List[Rule]:
    prog=CROP_BBOX_NONZERO(bg); 
    return [Rule("CROP_BBOX_NONZERO",{"bg":bg},prog,"Crop to bbox non-zero")] if all(equal(prog(x),y) for x,y in train) else []

def induce_keep_nonzero(train,bg=0)->List[Rule]:
    prog=KEEP_NONZERO(bg)
    return [Rule("KEEP_NONZERO",{"bg":bg},prog,"Keep non-zero")] if all(equal(prog(x),y) for x,y in train) else []

# Color perm
def learn_color_perm_mapping(train)->Optional[Dict[int,int]]:
    mapping={}
    for x,y in train:
        vals=np.unique(x)
        for c in vals:
            mask=(x==c)
            if not np.any(mask): continue
            tgt,counts=np.unique(y[mask],return_counts=True)
            yc=int(tgt[np.argmax(counts)])
            if c in mapping and mapping[c]!=yc: return None
            mapping[c]=yc
    return mapping

def induce_color_perm(train)->List[Rule]:
    mapping=learn_color_perm_mapping(train)
    if mapping is None: return []
    prog=COLOR_PERM(mapping)
    return [Rule("COLOR_PERM",{"map":mapping},prog,f"Color perm {mapping}")] if all(equal(prog(x),y) for x,y in train) else []

# Parity recolor
def induce_parity_recolor_const(train)->List[Rule]:
    """
    Detect y = x except parity cells recolored to one constant color.
    Try parity∈{even,odd}, anchor inferred from first pair by minimization.
    """
    out=[]
    for parity in ('even','odd'):
        # infer color and anchor from first pair by minimizing non-const deviations
        x0,y0=train[0]
        H,W=x0.shape; best=None
        for ar in range(2):  # try minimal anchors (0 or 1 shifts)
            for ac in range(2):
                m=PARITY_MASK(parity,(ar,ac))(x0)
                vals=y0[m]
                if vals.size==0: continue
                col=int(Counter(vals.tolist()).most_common(1)[0][0])
                # verify across all pairs
                prog=RECOLOR_PARITY_CONST(col, parity, (ar,ac))
                if all(equal(prog(x),y) for x,y in train):
                    out.append(Rule("RECOLOR_PARITY_CONST",{"color":col,"parity":parity,"anchor":(ar,ac)},
                                    prog,f"Recolor {parity} parity to {col} (anchor {(ar,ac)})"))
    return out

# Repeat tile
def induce_repeat_tile(train)->List[Rule]:
    """
    If y tiles a motif that also matches input shape (in ARC many times y is tiling-independent of x),
    we accept the tiling program if it matches y exactly on train.
    """
    out=[]
    # infer motif from first pair’s output and verify equal motif across pairs
    hwmotif=[]
    for _,y in train:
        m=extract_motif(y)
        if m is None: return []
        hwmotif.append(m)
    # require same motif dims and content
    h0,w0,m0=hwmotif[0]
    if any(h!=h0 or w!=w0 or not np.array_equal(m,m0) for h,w,m in hwmotif): return []
    prog=REPEAT_TILE(m0)
    if all(equal(prog(x),y) for x,y in train):
        out.append(Rule("REPEAT_TILE",{"h":h0,"w":w0},prog,f"Tile motif {h0}x{w0} across grid"))
    return out

# Repeat object chain
def enumerate_deltas_for_pair(x:Grid, y:Grid, bg=0)->List[Tuple[int,int]]:
    deltas=set()
    X=components(x,bg); Y=components(y,bg)
    if not X or not Y: return []
    H,W=x.shape
    def mask(o:Obj):
        m=np.zeros((H,W),dtype=bool)
        for r,c in o.pixels: m[r,c]=True
        return m
    for xi,xo in enumerate(X):
        cand=[yj for yj,yo in enumerate(Y) if yo.color==xo.color and yo.size==xo.size]
        for yj in cand:
            dx=int(round(Y[yj].centroid[0]-xo.centroid[0]))
            dy=int(round(Y[yj].centroid[1]-xo.centroid[1]))
            # accept as a base delta (we'll unify across pairs and learn k separately)
            deltas.add((dx,dy))
    return sorted(deltas)

def induce_repeat_object_chain(train, rank=0, group='global', bg=0)->List[Rule]:
    """
    Infer Δ common across pairs; infer k per pair as # copies - 1; unify k across pairs.
    Accept MOVE (clear source) or COPY (preserve) when program matches all train pairs.
    """
    # unify Δ
    delta_sets=[]
    for x,y in train:
        ds=enumerate_deltas_for_pair(x,y,bg)
        if not ds: return []
        delta_sets.append(set(ds))
    commons=set.intersection(*delta_sets)
    out=[]
    for Δ in commons:
        # infer k by trying small ks 1..Kmax
        for clear in (True, False):
            for k in range(1,6):  # typical ARC repetition sizes are small
                prog=MOVE_OR_COPY_OBJ_RANK(rank,Δ,k,group,clear,bg)
                if all(equal(prog(x),y) for x,y in train):
                    kind="MOVE_CHAIN" if clear else "COPY_CHAIN"
                    out.append(Rule(kind,{"rank":rank,"group":group,"delta":Δ,"k":k,"clear":clear,"bg":bg},
                                    prog,f"{'Move' if clear else 'Copy'} chain by Δ={Δ}, k={k}"))
    return out

# =============================================================================
# Beam search (receipts-first composition)
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
    rules+=induce_symmetry(train)
    rules+=induce_crop_bbox(train, bg=0)
    rules+=induce_keep_nonzero(train, bg=0)
    rules+=induce_color_perm(train)
    rules+=induce_parity_recolor_const(train)
    rules+=induce_repeat_tile(train)
    rules+=induce_repeat_object_chain(train, rank=0, group='global')
    rules+=induce_repeat_object_chain(train, rank=0, group='per_color')
    return rules

def beam_search(inst:ARCInstance, max_depth=5, beam_size=80)->Optional[List[Rule]]:
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
                # prune if any residual increases (inside must settle)
                if any(res[i]>node.res_list[i] for i in range(len(res))): continue
                tot=sum(res)
                if tot<http://node.total:
                    new.append(Node(comp,node.steps+[r],res,tot,node.depth+1))
                    if tot==0:
                        return node.steps+[r]
        if not new: break
        new.sort(key=lambda nd:(http://nd.total, nd.depth))
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

def solve_with_beam(inst:ARCInstance, max_depth=5, beam_size=80):
    steps=beam_search(inst, max_depth, beam_size)
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
# Unit tests + Mini Bench
# =============================================================================

def _unit_tests():
    # Parity recolor: even cells -> 9
    x=G([[1,1,1],[1,1,1],[1,1,1]])
    y=x.copy(); H,W=x.shape
    rr,cc=np.indices((H,W)); m=((rr+cc)&1)==0; y[m]=9
    inst=ARCInstance("parity_recolor",[(x,y)],[x],[y])
    steps,_=solve_with_beam(inst, max_depth=2, beam_size=20)
    assert steps is not None

    # Repeat tile: 2x2 motif
    motif=G([[7,0],[0,7]])
    y2=np.tile(motif,(2,3))
    inst2=ARCInstance("repeat_tile",[(G([[0]]),y2)], [G([[0,0,0,0],[0,0,0,0]])], [REPEAT_TILE(motif)(G([[0,0,0,0],[0,0,0,0]]))])
    steps2,_=solve_with_beam(inst2, max_depth=2, beam_size=20)
    assert steps2 is not None

    # Repeat object chain (copy): Δ=(0,+2), k=2
    x3=G([[2,2,0,0,0],[2,2,0,0,0]])
    y3=x3.copy(); y3[:,2:4]=2; y3[:,4:6]=2 if y3.shape[1]>=6 else y3[:,4:5]
    # Adjust to valid width = 6
    x3=G([[2,2,0,0,0,0],[2,2,0,0,0,0]])
    y3=G([[2,2,2,2,2,2],[2,2,2,2,2,2]])
    inst3=ARCInstance("repeat_chain_copy",[(x3,y3)],[x3],[y3])
    steps3,_=solve_with_beam(inst3, max_depth=3, beam_size=50)
    assert steps3 is not None

    print("All unit tests passed (parity/repetition).")

def _mini_bench():
    tasks=[
        # 1) parity recolor odd → 8
        ARCInstance("bench_parity",
            train=[(G([[1,1,1],[1,1,1]]),
                    (lambda x: (lambda z:(z.__setitem__((np.indices(z.shape).sum(0)&1)==1, 8) or z))(x.copy()))(G([[1,1,1],[1,1,1]])))],
            test_in=[G([[3,3,3],[3,3,3]])],
            test_out=[(lambda x: (lambda z:(z.__setitem__((np.indices(z.shape).sum(0)&1)==1, 8) or z))(x.copy()))(G([[3,3,3],[3,3,3]]))]),
        # 2) repeat tile
        ARCInstance("bench_tile",
            train=[(G([[0]]), np.tile(G([[5,0],[0,5]]),(2,2)))],
            test_in=[G([[0,0,0,0],[0,0,0,0]])],
            test_out=[np.tile(G([[5,0],[0,5]]),(2,2))]),
        # 3) color perm → parity recolor
        ARCInstance("bench_perm_parity",
            train=[(G([[1,1],[2,2]]), G([[7,7],[8,8]]))],
            test_in=[G([[1,2],[1,2]])],
            test_out=[G([[7,8],[7,8]])]),
        # 4) move chain (global) Δ=(+1,+2), k=1
        ARCInstance("bench_move_chain",
            train=[(G([[0,0,0,0,0],
                       [0,4,4,0,0],
                       [0,4,4,0,0],
                       [0,0,0,0,0]]),
                    G([[0,0,0,0,0],
                       [0,0,0,0,0],
                       [0,0,0,4,4],
                       [0,0,0,4,4]]))],
            test_in=[G([[0,0,0,0,0],
                        [0,6,6,0,0],
                        [0,6,6,0,0],
                        [0,0,0,0,0]])],
            test_out=[G([[0,0,0,0,0],
                         [0,0,0,0,0],
                         [0,0,0,6,6],
                         [0,0,0,6,6]])]),
    ]
    solved=0
    for inst in tasks:
        steps,recs=solve_with_beam(inst, max_depth=5, beam_size=80)
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
    print("\nFull-eval checklist:")
    print("• Add 30–60 more curated tasks; measure accuracy; iterate on failing categories.")
    print("• Run public ARC-AGI split offline; keep receipts per task (program, residual=0 on train, PCE).")
    print("• Widen beam cautiously; extend catalog with tiling-on-mask, draw/line, parity+mask combos, repetition on diagonals.")
    print("• Always refuse programs without residual=0 on train; print edit-bill deltas per composed step.")
