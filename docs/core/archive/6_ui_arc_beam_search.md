#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-UI BEAM — Receipts-First Composition Search (Universe-Functioning)
======================================================================

This is the **next complete step** the other AI asked for: a **beam search**
that composes induced operators (symmetry, masks, object-rank recolor/keep,
move/copy with Δ enumeration, color-permutation, crop) into **short programs**
that solve ARC-style tasks **with receipts**.

Universe-functioning discipline:
--------------------------------
• Inside settles → we accept a solution only if **train residual=0** on all pairs.
• Edges write → we log **edit bills** (#edits, boundary vs interior).
• Observer = observed → the **same parameters** (e.g., Δ, rank, mapping) must fit
  **all** train pairs; otherwise the candidate is pruned immediately.

What’s here:
------------
1) Invariant engine: connected components, rank-by-size, bbox/centroid.
2) DSL operators (parametric, receipts-ready):
   - Symmetries: ROT(k), FLIP(axis)
   - Crop: CROP_BBOX_NONZERO(bg)
   - Masks: KEEP_NONZERO(bg)
   - Color: COLOR_PERM (global mapping learned from train)
   - Object-rank: RECOLOR_OBJ_RANK, KEEP_OBJ_TOPK
   - MOVE/COPY: MOVE_OBJ_RANK / COPY_OBJ_RANK with **Δ enumeration** across pairs,
     plus simple collision resolution (target wins).
3) **Induction**: each operator family has an inducer that unifies parameters across
   all train pairs (observer=observed) and returns a callable `prog`.
4) **Beam search**: composes operators up to depth K; prunes any node that increases
   residual on any pair; stops at first node with residual=0 on all train pairs.
5) Receipts + PCE (Proof-Carrying English) for every solution and prediction.

Run:
----
    python arc_ui_beam.py
Dependencies: numpy only.

Extension guide (after this works):
-----------------------------------
• Add lattice/tiling, draw/line, parity, repetition, paste-between-regions.
• Increase beam width/depth judiciously; keep receipts at every partial step.
• Log edit-bill deltas per step to explain why each step was necessary.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from collections import Counter, deque
from time import perf_counter

# =============================================================================
# Types & helpers
# =============================================================================

Grid = np.ndarray
Mask = np.ndarray

def G(lst) -> Grid: return np.array(lst, dtype=int)
def assert_grid(g: Grid): assert isinstance(g, np.ndarray) and g.dtype==int and g.ndim==2

def equal(a: Grid, b: Grid) -> bool: return a.shape==b.shape and np.array_equal(a,b)
def residual(a: Grid, b: Grid) -> int: assert a.shape==b.shape; return int((a!=b).sum())

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
# Invariants: components & ranks
# =============================================================================

@dataclass

class Obj:
    color: int
    pixels: List[Tuple[int,int]]
    size: int
    bbox: Tuple[int,int,int,int]
    centroid: Tuple[float,float]

def components(g: Grid, bg: int=0) -> List[Obj]:
    assert_grid(g)
    H,W=g.shape
    vis=np.zeros_like(g,dtype=bool)
    out=[]
    for r in range(H):
        for c in range(W):
            if g[r,c]!=bg and not vis[r,c]:
                col=g[r,c]; q=deque([(r,c)]); vis[r,c]=True
                pix=[(r,c)]
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

def rank_by_size(objs: List[Obj], group: str='global') -> Dict:
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

# =============================================================================
# DSL Operators (parametric programs)
# =============================================================================

# Symmetry
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

# Crop bbox of non-zero
def bbox_nonzero(g: Grid, bg:int=0)->Tuple[int,int,int,int]:
    idx=np.argwhere(g!=bg)
    if idx.size==0: return (0,0,g.shape[0]-1,g.shape[1]-1)
    r0,c0=idx.min(axis=0); r1,c1=idx.max(axis=0)
    return (int(r0),int(c0),int(r1),int(c1))

def CROP_BBOX_NONZERO(bg:int=0)->Callable[[Grid],Grid]:
    def f(z:Grid):
        r0,c0,r1,c1=bbox_nonzero(z,bg)
        return z[r0:r1+1, c0:c1+1]
    return f

# KEEP nonzero
def KEEP_NONZERO(bg:int=0)->Callable[[Grid],Grid]:
    def f(z:Grid):
        m=(z!=bg); out=np.zeros_like(z); out[m]=z[m]; return out
    return f

# Color perm (global mapping, preserve colors not in map)
def COLOR_PERM(mapping:Dict[int,int])->Callable[[Grid],Grid]:
    def f(z:Grid):
        out=z.copy()
        for c,yc in mapping.items():
            out[z==c]=yc
        return out
    return f

# Object rank recolor / keep top-k
def MASK_OBJ_RANK(rank:int, group:str='global', bg:int=0)->Callable[[Grid],Mask]:
    def f(z:Grid)->Mask:
        objs=components(z,bg); m=np.zeros_like(z,dtype=bool)
        if not objs: return m
        ranks=rank_by_size(objs,group)
        if group=='global':
            ord_list=ranks["global"]
            if rank < len(ord_list):
                for (r,c) in objs[ord_list[rank]].pixels: m[r,c]=True
        else:
            for _,ord_list in ranks.items():
                if rank < len(ord_list):
                    for (r,c) in objs[ord_list[rank]].pixels: m[r,c]=True
        return m
    return f

def ON(mask_fn:Callable[[Grid],Mask], prog:Callable[[Grid],Grid])->Callable[[Grid],Grid]:
    def f(z:Grid):
        m=mask_fn(z); z2=prog(z.copy()); out=z.copy(); out[m]=z2[m]; return out
    return f

def RECOLOR_CONST(new_color:int)->Callable[[Grid],Grid]:
    return lambda z: np.full_like(z, new_color)

def KEEP_OBJ_TOPK(k:int, bg:int=0)->Callable[[Grid],Grid]:
    def f(z:Grid):
        objs=components(z,bg); ord_list=rank_by_size(objs,'global')["global"]
        m=np.zeros_like(z,dtype=bool)
        for rnk,idx in enumerate(ord_list):
            if rnk<k:
                for (r,c) in objs[idx].pixels: m[r,c]=True
        out=np.zeros_like(z); out[m]=z[m]; return out
    return f

# MOVE/COPY object rank by Δ, with collision priority: target wins
def MOVE_OR_COPY_OBJ_RANK(rank:int, delta:Tuple[int,int], group:str='global', clear:bool=True, bg:int=0)\
        -> Callable[[Grid],Grid]:
    dr,dc=delta
    H=W=None
    def inb(r,c): return 0<=r<H and 0<=c<W
    def f(z:Grid):
        nonlocal H,W
        H,W=z.shape
        objs=components(z,bg); out=z.copy()
        if not objs: return out
        ranks=rank_by_size(objs,group)
        idxs=[]
        if group=='global':
            ord_list=ranks["global"]; 
            if rank<len(ord_list): idxs=[ord_list[rank]]
        else:
            for _,ord_list in ranks.items():
                if rank<len(ord_list): idxs.append(ord_list[rank])
        # prepare target grid to avoid source overwriting on copy
        if clear:
            for idx in idxs:
                for (r,c) in objs[idx].pixels: out[r,c]=bg
        # paste
        for idx in idxs:
            col=objs[idx].color
            for (r,c) in objs[idx].pixels:
                nr,nc=r+dr,c+dc
                if inb(nr,nc): out[nr,nc]=col  # target wins by assignment
        return out
    return f

# =============================================================================
# Induction (receipts-first) — each returns a program or None
# =============================================================================

@dataclass

class Rule:
    name: str
    params: Dict
    prog: Callable[[Grid],Grid]
    pce: str

# Symmetry
def induce_symmetry(train)->List[Rule]:
    out=[]
    cand=[("ROT",{"k":0},ROT(0),"Rotate 0°"),
          ("ROT",{"k":1},ROT(1),"Rotate 90°"),
          ("ROT",{"k":2},ROT(2),"Rotate 180°"),
          ("ROT",{"k":3},ROT(3),"Rotate 270°"),
          ("FLIP",{"axis":"h"},FLIP('h'),"Flip horizontal"),
          ("FLIP",{"axis":"v"},FLIP('v'),"Flip vertical")]
    for name,params,prog,txt in cand:
        if all(equal(prog(x),y) for x,y in train):
            out.append(Rule(name,params,prog,txt+" (exact on train)"))
    return out

# Crop bbox non-zero
def induce_crop_bbox(train, bg=0)->List[Rule]:
    prog=CROP_BBOX_NONZERO(bg); 
    return [Rule("CROP_BBOX_NONZERO",{"bg":bg},prog,"Crop to bbox of non-zero")] if all(equal(prog(x),y) for x,y in train) else []

# Keep non-zero
def induce_keep_nonzero(train, bg=0)->List[Rule]:
    prog=KEEP_NONZERO(bg)
    return [Rule("KEEP_NONZERO",{"bg":bg},prog,"Keep non-zero")] if all(equal(prog(x),y) for x,y in train) else []

# Color perm (global)
def learn_color_perm_mapping(train)->Optional[Dict[int,int]]:
    mapping={}
    for x,y in train:
        vals=np.unique(x)
        for c in vals:
            mask=(x==c)
            if not np.any(mask): continue
            tgt_vals,counts=np.unique(y[mask],return_counts=True)
            yc=int(tgt_vals[np.argmax(counts)])
            if c in mapping and mapping[c]!=yc: return None
            mapping[c]=yc
    return mapping

def induce_color_perm(train)->List[Rule]:
    mapping=learn_color_perm_mapping(train)
    if mapping is None: return []
    prog=COLOR_PERM(mapping)
    if all(equal(prog(x),y) for x,y in train):
        return [Rule("COLOR_PERM",{"map":mapping},prog,f"Color permutation {mapping}")]
    return []

# Recolor object rank (global/per_color)
def induce_recolor_obj_rank(train, rank=0, group='global', bg=0)->List[Rule]:
    # infer target color from first pair and verify across
    target=None
    for x,y in train:
        m=MASK_OBJ_RANK(rank,group,bg)(x)
        if not np.any(m): return []
        vals=y[m]; 
        if vals.size==0: return []
        c=int(Counter(vals.tolist()).most_common(1)[0][0])
        if target is None: target=c
        elif target!=c: return []
    def recolor_prog(z):
        m=MASK_OBJ_RANK(rank,group,bg)(z); out=z.copy(); out[m]=target; return out
    if all(equal(recolor_prog(x),y) for x,y in train):
        scope="globally" if group=='global' else "per color"
        return [Rule("RECOLOR_OBJ_RANK",{"rank":rank,"group":group,"color":target,"bg":bg},
                    recolor_prog,f"Recolor {('largest' if rank==0 else f'rank {rank}')} {scope}→{target}")]
    return []

# Keep top-k
def induce_keep_topk(train, k=2, bg=0)->List[Rule]:
    prog=KEEP_OBJ_TOPK(k,bg)
    if all(equal(prog(x),y) for x,y in train):
        return [Rule("KEEP_OBJ_TOPK",{"k":k,"bg":bg},prog,f"Keep top-{k} components")]
    return []

# MOVE/COPY with Δ enumeration across ALL same-color/size targets
def enumerate_deltas_for_pair(x:Grid, y:Grid, bg=0)->List[Tuple[int,int]]:
    deltas=set()
    X=components(x,bg); Y=components(y,bg)
    if not X or not Y: return []
    H,W=x.shape
    def mask_of(obj:Obj):
        m=np.zeros((H,W),dtype=bool)
        for (r,c) in obj.pixels: m[r,c]=True
        return m
    for xi in range(len(X)):
        xobj=X[xi]
        xmask=mask_of(xobj)
        matches=[yj for yj,yo in enumerate(Y) if yo.color==xobj.color and yo.size==xobj.size]
        for yj in matches:
            dx=int(round(Y[yj].centroid[0]-xobj.centroid[0]))
            dy=int(round(Y[yj].centroid[1]-xobj.centroid[1]))
            # shape check
            rr,cc=np.where(xmask)
            nr=rr+dx; nc=cc+dy
            keep=(nr>=0)&(nr<H)&(nc>=0)&(nc<W)
            m2=np.zeros((H,W),dtype=bool); m2[nr[keep], nc[keep]]=True
            if np.array_equal(m2, mask_of(Y[yj])):
                deltas.add((dx,dy))
    return sorted(deltas)

def induce_move_copy(train, rank=0, group='global', bg=0)->List[Rule]:
    # collect candidate Δ sets across all pairs, intersect
    delta_sets=[]
    for x,y in train:
        ds=enumerate_deltas_for_pair(x,y,bg)
        if not ds: return []
        delta_sets.append(set(ds))
    common=set.intersection(*delta_sets)
    out=[]
    for Δ in common:
        move_prog=MOVE_OR_COPY_OBJ_RANK(rank,Δ,group,True,bg)
        if all(equal(move_prog(x),y) for x,y in train):
            out.append(Rule("MOVE_OBJ_RANK",{"rank":rank,"group":group,"delta":Δ,"clear":True,"bg":bg},
                            move_prog,f"Move {('largest' if rank==0 else f'rank {rank}')} {group} by Δ={Δ}"))
        copy_prog=MOVE_OR_COPY_OBJ_RANK(rank,Δ,group,False,bg)
        if all(equal(copy_prog(x),y) for x,y in train):
            out.append(Rule("COPY_OBJ_RANK",{"rank":rank,"group":group,"delta":Δ,"clear":False,"bg":bg},
                            copy_prog,f"Copy {('largest' if rank==0 else f'rank {rank}')} {group} by Δ={Δ}"))
    return out

# =============================================================================
# Beam Search (receipts-first composition)
# =============================================================================

@dataclass

class ARCInstance:
    name: str
    train: List[Tuple[Grid,Grid]]
    test_in: List[Grid]
    test_out: List[Grid]

@dataclass

class Node:
    prog: Callable[[Grid],Grid]
    steps: List[Rule]         # for PCE
    res_per_pair: List[int]   # residuals on train pairs
    total_res: int
    length: int

def compose(p:Callable[[Grid],Grid], q:Callable[[Grid],Grid])->Callable[[Grid],Grid]:
    return lambda z: q(p(z))

def apply_prog_to_pairs(pairs, prog)->List[int]:
    return [residual(prog(x),y) for x,y in pairs]

def candidate_rules(train)->List[Rule]:
    rules=[]
    rules += induce_symmetry(train)
    rules += induce_crop_bbox(train, bg=0)
    rules += induce_keep_nonzero(train, bg=0)
    rules += induce_color_perm(train)
    rules += induce_recolor_obj_rank(train, rank=0, group='global')
    rules += induce_recolor_obj_rank(train, rank=0, group='per_color')
    rules += induce_keep_topk(train, k=2)
    rules += induce_move_copy(train, rank=0, group='global')
    rules += induce_move_copy(train, rank=0, group='per_color')
    return rules

def beam_search(inst: ARCInstance, max_depth=4, beam_size=50)->Optional[List[Rule]]:
    train=inst.train
    # initial node: identity
    id_prog=lambda z:z
    init_res=apply_prog_to_pairs(train, id_prog)
    beam=[Node(id_prog, [], init_res, sum(init_res), 0)]
    best=None
    for depth in range(max_depth):
        new_nodes=[]
        # generate candidate primitive rules once (per task)
        rules=candidate_rules(train)
        for node in beam:
            base_prog=node.prog; base_res=node.res_per_pair
            base_total=http://node.total_res
            for r in rules:
                # compose: base → r
                comp_prog=compose(base_prog, r.prog)
                res=apply_prog_to_pairs(train, comp_prog)
                # prune if any residual increases
                if any(res[i] > base_res[i] for i in range(len(res))): 
                    continue
                tot=sum(res)
                # keep nodes that strictly reduce total or keep same but add step (rare)
                if tot < base_total:
                    new_nodes.append(Node(comp_prog, node.steps+[r], res, tot, node.length+1))
                    # found perfect solution
                    if tot == 0:
                        return node.steps+[r]
        if not new_nodes:
            break
        # beam prune
        new_nodes.sort(key=lambda nd: (http://nd.total_res, nd.length))
        beam=new_nodes[:beam_size]
    return None

# =============================================================================
# Receipts + PCE for a found program
# =============================================================================

@dataclass

class Receipts:
    residual: int
    edits_total: int
    edits_boundary: int
    edits_interior: int
    pce: str

def pce_for_steps(steps: List[Rule])->str:
    if not steps: return "Identity (no change)."
    return " → ".join(r.pce for r in steps)

def solve_with_beam(inst: ARCInstance, max_depth=4, beam_size=50):
    steps=beam_search(inst, max_depth=max_depth, beam_size=beam_size)
    if steps is None:
        return None, []
    # verify train receipts
    for x,y in inst.train:
        yhat=x.copy()
        for r in steps: yhat=r.prog(yhat)
        assert equal(yhat,y), "Train residual must be 0."
    # predict test + receipts
    recs=[]
    for i,x in enumerate(inst.test_in):
        ygt=inst.test_out[i]
        yhat=x.copy()
        for r in steps: yhat=r.prog(yhat)
        res=residual(yhat,ygt); eT,eB,eI=edit_counts(x,yhat)
        recs.append(Receipts(res,eT,eB,eI,f"[{http://inst.name} test#{i}] {pce_for_steps(steps)}"))
    return steps, recs

# =============================================================================
# Unit tests & Mini Bench
# =============================================================================

def _unit_tests():
    # 1) ROT90 then crop bbox
    a=G([[0,1,0],[0,0,1],[0,0,0]]); b=CROP_BBOX_NONZERO(0)(rot90(a))
    inst=ARCInstance("rot90_then_crop",[(a,b)],[G([[0,2,0],[0,0,2],[0,0,0]])],[CROP_BBOX_NONZERO(0)(rot90(G([[0,2,0],[0,0,2],[0,0,0]])))] )
    steps,_=solve_with_beam(inst, max_depth=2, beam_size=20)
    assert steps is not None and len(steps)>=1

    # 2) Color perm then move largest by (+1,+2)
    x=G([[1,1,0,0,0],[1,1,0,0,0],[0,0,0,0,0]])
    y=x.copy(); y[y==1]=3   # color 1->3
    y[0:2,0:2]=0; y[1:3,2:4]=3  # move block by (+1,+2)
    inst2=ARCInstance("perm_then_move",[(x,y)],[x],[y])
    steps2,_=solve_with_beam(inst2, max_depth=3, beam_size=50)
    assert steps2 is not None and len(steps2)>=2

    # 3) Keep nonzero then flip_h
    z=G([[0,9,0],[0,9,0],[0,0,0]])
    t=flip_h(KEEP_NONZERO(0)(z))
    inst3=ARCInstance("keep_then_flip",[(z,t)],[z],[t])
    steps3,_=solve_with_beam(inst3, max_depth=2, beam_size=20)
    assert steps3 is not None

    # 4) Recolor per-color largest then copy per-color by (0,+3)
    x4=G([[1,1,0,2,2],[1,0,0,2,0]])
    y4=x4.copy(); y4[x4==1]=7; y4[x4==2]=8
    # then copy both largest per-color by +3 cols
    yy=y4.copy()
    yy[:,3:5]=y4[:,:2]  # toy layout for demonstration
    inst4=ARCInstance("percolor_recolor_then_copy",[(x4,yy)],[x4],[yy])
    steps4,_=solve_with_beam(inst4, max_depth=4, beam_size=100)
    assert steps4 is not None

    print("All unit tests passed (beam composition).")

def _mini_bench():
    tasks=[
        # A) rot90 → crop
        ARCInstance("A_rot90_crop",
            train=[(G([[0,1,0],[0,0,1],[0,0,0]]), CROP_BBOX_NONZERO(0)(rot90(G([[0,1,0],[0,0,1],[0,0,0]]))))],
            test_in=[G([[0,2,0],[0,0,2],[0,0,0]])],
            test_out=[CROP_BBOX_NONZERO(0)(rot90(G([[0,2,0],[0,0,2],[0,0,0]]))) ]),
        # B) color perm → move
        ARCInstance("B_perm_move",
            train=[(G([[1,1,0,0,0],[1,1,0,0,0],[0,0,0,0,0]]),
                    (lambda x:(lambda z: (lambda m:(m.__setitem__(slice(0,2),0) or m)[0] )(x.copy()) )(x)) )], # placeholder replaced below
            test_in=[G([[1,1,0,0,0],[1,1,0,0,0],[0,0,0,0,0]])],
            test_out=[G([[0,0,3,3,0],[0,0,3,3,0],[0,0,0,0,0]])]),
        # C) keep nonzero → flip_h
        ARCInstance("C_keep_flip",
            train=[(G([[0,9,0],[0,9,0],[0,0,0]]), flip_h(KEEP_NONZERO(0)(G([[0,9,0],[0,9,0],[0,0,0]]))))],
            test_in=[G([[0,5,0],[0,5,0],[0,0,0]])],
            test_out=[flip_h(KEEP_NONZERO(0)(G([[0,5,0],[0,5,0],[0,0,0]]))) ]),
    ]
    # Fix train for B) programmatically (perm 1->3 then move +1,+2)
    xB=G([[1,1,0,0,0],[1,1,0,0,0],[0,0,0,0,0]])
    yB=xB.copy(); yB[yB==1]=3; yB[0:2,0:2]=0; yB[1:3,2:4]=3
    tasks[1].train=[(xB,yB)]
    total=len(tasks); solved=0
    for inst in tasks:
        steps,recs=solve_with_beam(inst, max_depth=4, beam_size=100)
        print(f"\n[{http://inst.name}]")
        if steps is None:
            print("  No solution found within beam.")
            continue
        solved+=1
        print("  Program:", " → ".join(f"{http://r.name}:{r.params}" for r in steps))
        print("  PCE:", pce_for_steps(steps))
        for rc in recs:
            print(f"  {rc.pce}  residual={rc.residual} edits={rc.edits_total} (boundary {rc.edits_boundary}, interior {rc.edits_interior})")
    print(f"\nMINI BENCH: solved {solved}/{total}")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    _unit_tests()
    _mini_bench()
    print("\nNEXT: add lattice/tiling & draw/line ops; expand rule catalog; widen beam cautiously; "
          "keep receipts-first pruning (no residual increase), and stop at residual=0 on train.")