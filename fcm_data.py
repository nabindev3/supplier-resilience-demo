"""Causal map (FCM) for supply-chain resilience & sustainability enablers.

Concepts and signed causal weights are illustrative but internally consistent.
They connect this project's two halves: the *enablers* on the left include
"Supplier diversification" — the very lever the allocation model in
`allocation.py` exercises — and the *targets* on the right include the
"Resilience" the disruption stress test measures.

W[i, j] = causal influence of concept i ON concept j, in [-1, 1].
"""

import numpy as np

CONCEPTS = [
    "Blockchain traceability",   # 0  enabler
    "Smart contracts",           # 1  enabler
    "Supplier diversification",  # 2  enabler / allocation lever
    "Supply-chain visibility",   # 3  intermediate
    "Disruption risk",           # 4  intermediate (undesirable)
    "Resilience",                # 5  target
    "Sustainability",            # 6  target
]

# directed, signed causal links
_EDGES = {
    ("Blockchain traceability", "Supply-chain visibility"): 0.80,
    ("Blockchain traceability", "Sustainability"): 0.55,
    ("Smart contracts", "Blockchain traceability"): 0.45,
    ("Smart contracts", "Supply-chain visibility"): 0.50,
    ("Supplier diversification", "Disruption risk"): -0.70,
    ("Supplier diversification", "Resilience"): 0.60,
    ("Supply-chain visibility", "Disruption risk"): -0.60,
    ("Supply-chain visibility", "Resilience"): 0.40,
    ("Disruption risk", "Resilience"): -0.80,
    ("Disruption risk", "Sustainability"): -0.50,
    ("Resilience", "Sustainability"): 0.40,
}


def weight_matrix() -> np.ndarray:
    idx = {c: i for i, c in enumerate(CONCEPTS)}
    W = np.zeros((len(CONCEPTS), len(CONCEPTS)))
    for (src, dst), w in _EDGES.items():
        W[idx[src], idx[dst]] = w
    return W


ENABLERS = [
    "Blockchain traceability",
    "Smart contracts",
    "Supplier diversification",
]
