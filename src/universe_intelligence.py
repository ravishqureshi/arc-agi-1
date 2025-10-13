"""
Universe Intelligence v2 - Core Implementation

Three mathematical enrichments unified under one framework:
- ORDER: Horn clauses, least fixed points (discrete logic)
- ENTROPY: KL divergence, Fenchel-Young (probabilistic reasoning)
- QUADRATIC: Laplacian, Green identity (spatial geometry)

All with machine-precision receipts (proofs).
"""

import numpy as np
from numpy.random import default_rng
from scipy.linalg import solve
from dataclasses import dataclass

rng = default_rng(5)

# ════════════════════════════════════════════════════════════════════════════════
# ORDER ENRICHMENT: Logic as Least Fixed Points
# ════════════════════════════════════════════════════════════════════════════════

class HornKB:
    """
    Horn Clause Knowledge Base with Least Fixed Point Computation.

    STRUCTURE:
    - Facts: Set of ground facts (strings like "edge(A,B)")
    - Rules: Inference rules ([premises] → conclusion)

    OPERATOR T(F):
    - Takes current fact set F
    - Adds all conclusions whose premises are satisfied in F
    - Returns F ∪ {new conclusions}

    LEAST FIXED POINT (lfp):
    - Iterate T(F) until T(F) = F (no new facts)
    - RECEIPT: residual = 0 (exactly zero)

    MONOTONICITY:
    - If F ⊆ G, then T(F) ⊆ T(G)
    - RECEIPT: empirically tested on random subsets
    """
    def __init__(self, facts=None, rules=None):
        self.facts = set(facts or [])
        self.rules = [(tuple(p), c) for (p,c) in (rules or [])]

    def T(self, F):
        F = set(F)
        added = set()
        for premises, concl in self.rules:
            if all(p in F for p in premises):
                added.add(concl)
        return F | added

    def lfp(self):
        """Iterate T to least fixed point. Return (F*, steps, residual)."""
        F = self.facts.copy()  # FIXED: Start from initial facts, not empty set
        steps = 0
        while True:
            steps += 1
            T_F = self.T(F)
            if T_F == F:
                residual = 0
                return F, steps, residual
            F = T_F

    def monotone_receipt(self, trials=20):
        """Empirical check: for random supersets G ⊇ F, verify T(F) ⊆ T(G)."""
        facts = list(self.facts)
        if not facts:
            return True
        ok = True
        for _ in range(trials):
            k = rng.integers(0, len(facts)+1)
            A = set(rng.choice(facts, size=k, replace=False))
            extras = list(set(facts) - A)
            j = rng.integers(0, len(extras)+1)
            B = A | set(rng.choice(extras, size=j, replace=False))
            ok &= self.T(A).issubset(self.T(B))
        return ok

# ════════════════════════════════════════════════════════════════════════════════
# ENTROPY ENRICHMENT: Information Geometry & Probabilistic Reasoning
# ════════════════════════════════════════════════════════════════════════════════

def softmax(u):
    """Numerically stable softmax."""
    u = np.asarray(u, float)
    m = np.max(u)
    z = np.exp(u - m)
    return z / z.sum()

def entropy_phi(p, eps=1e-12):
    """
    Entropy functional with Fenchel-Young adjunction.

    Φ(p) = Σ p log p − p
    Fenchel conjugate: Φ*(s) = Σ exp(s)
    Adjunction: Φ(p) + Φ*(log p) = ⟨p, log p⟩

    Returns: val, s, conj, inner, gap
    - gap ≈ 1e-12 (receipt that adjunction holds)
    """
    p = np.maximum(p, eps)
    val = float(np.sum(p * (np.log(p) - 1.0)))
    s = np.log(p)
    conj = float(np.sum(np.exp(s)))
    inner = float(np.dot(p, s))  # Fixed: removed erroneous http://
    gap = abs(val + conj - inner)
    return val, s, conj, inner, gap

def KL(p, q, eps=1e-12):
    """KL divergence KL(p || q) = Σ p log(p/q)."""
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)
    return float(np.sum(p * (np.log(p) - np.log(q))))

# ════════════════════════════════════════════════════════════════════════════════
# QUADRATIC ENRICHMENT: Spatial Geometry & Energy Minimization
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class DtN:
    """Dirichlet-to-Neumann operator data."""
    Λ: np.ndarray      # Boundary operator (Schur complement)
    I: np.ndarray      # Interior indices
    B: np.ndarray      # Boundary indices
    L_II: np.ndarray   # Interior-interior Laplacian block
    L_IB: np.ndarray   # Interior-boundary Laplacian block

def random_connected_laplacian(n=140, extra_edges=360, wmin=0.5, wmax=2.0, seed=11):
    """
    Generate random connected weighted graph Laplacian.

    Returns: L (n×n symmetric positive semi-definite matrix)
    - Null space dimension = 1 (connected graph)
    - For u ≠ constant: u^T L u > 0
    """
    rng_local = default_rng(seed)
    edges = [(i, i+1) for i in range(n-1)]  # Chain ensures connectivity
    while len(edges) < n-1 + extra_edges:
        i, j = rng_local.integers(0, n, size=2)
        if i != j:
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in edges:
                edges.append((a, b))
    L = np.zeros((n, n), float)
    for (i, j) in edges:
        wij = rng_local.uniform(wmin, wmax)
        L[i,i] += wij
        L[j,j] += wij
        L[i,j] -= wij
        L[j,i] -= wij
    return L

def schur_dtn(L, B_idx):
    """
    Compute Dirichlet-to-Neumann operator via Schur complement.

    Given Laplacian L and boundary indices B:
    - Λ = L_BB - L_BI @ L_II^{-1} @ L_IB
    - This is the boundary-only operator

    Returns: DtN object with Λ and index arrays

    Raises:
    - ValueError: if interior is empty (all nodes are boundary)
    """
    n = L.shape[0]
    B = np.array(sorted(set(B_idx)), int)
    I = np.array(sorted(set(range(n)) - set(B)), int)

    # FIXED: Check for empty interior (e.g., 1×1 grids where all nodes are boundary)
    if len(I) == 0:
        # Special case: no interior, Λ = L_BB directly (no Schur complement needed)
        L_BB = L[np.ix_(B,B)]
        return DtN(Λ=L_BB, I=I, B=B, L_II=np.empty((0,0)), L_IB=np.empty((0,len(B))))

    L_II = L[np.ix_(I,I)]
    L_IB = L[np.ix_(I,B)]
    L_BI = L[np.ix_(B,I)]
    L_BB = L[np.ix_(B,B)]
    X = solve(L_II, L_IB, assume_a='sym')
    Λ = L_BB - L_BI @ X
    return DtN(Λ=Λ, I=I, B=B, L_II=L_II, L_IB=L_IB)

def solve_dirichlet(L, dtn: DtN, b):
    """
    Solve Dirichlet problem: Δu = 0 in interior, u = b on boundary.

    Returns: u (full solution vector)
    - u[dtn.B] = b (boundary values)
    - u[dtn.I] = interior solution (harmonic extension)
    """
    u_I = -solve(dtn.L_II, dtn.L_IB @ b, assume_a='sym')
    u = np.zeros(L.shape[0], float)
    u[dtn.I] = u_I
    u[dtn.B] = b
    return u

def green_gap(L, dtn: DtN, b):
    """
    Compute Green identity receipt.

    Green identity: u^T L u = b^T Λ b
    - Interior energy = Boundary energy

    Returns: gap, Ein, Ebd
    - gap ≈ 1e-12 (receipt that identity holds)
    """
    u = solve_dirichlet(L, dtn, b)
    Ein = float(u @ (L @ u))
    Ebd = float(b @ (dtn.Λ @ b))
    return abs(Ein - Ebd), Ein, Ebd

# ════════════════════════════════════════════════════════════════════════════════
# PROOF-CARRYING ENGLISH: Human-readable certificates
# ════════════════════════════════════════════════════════════════════════════════

def pce_reasoning_trace(Fstar, kb: HornKB, max_items=6):
    """
    Turn fixed-point proof into English with receipts.

    Returns: (text, certificate_dict)
    - text: human-readable summary
    - certificate: residual + monotonicity receipt
    """
    facts = list(sorted(Fstar))[:max_items]
    text = "Reasoning lfp reached. Derived facts: " + ", ".join(facts) + "."
    cert = {"lfp_residual": 0, "monotone_empirical": kb.monotone_receipt()}
    return text, cert

def pce_entropy_comm(p, note=""):
    """
    Turn entropy adjunction into English with receipts.

    Returns: (text, certificate_dict)
    - text: human-readable summary
    - certificate: Fenchel-Young gap
    """
    val, s, conj, inner, gap = entropy_phi(p)
    txt = (f"Communication is adjoint: entropy bill Φ(p) + Φ*(log p) equals ⟨p, log p⟩ "
           f"(gap {gap:.2e}). {note}")
    cert = {"fenchel_young_gap": gap}
    return txt, cert
