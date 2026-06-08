"""
Fuzzy Cognitive Map (FCM) engine.

This is a faithful, small implementation of the FCM machinery that Dr. Yousefi
uses to model causal relationships between blockchain/operations enablers and
supply-chain performance targets — e.g.

  * Yousefi & Mohamadpour Tosarkani (2022), "An analytical approach for
    evaluating the impact of blockchain technology on sustainable supply chain
    performance," Int. J. Production Economics 246, 108429 — FCM + a hybrid
    learning algorithm + DEA.
  * "Enhancing sustainable supply chain readiness to adopt blockchain"
    (Engineering Applications of AI, 2024) — FCM (Z-number) + hybrid learning.

An FCM is a signed, weighted directed graph of *concepts*. Each concept has an
activation A_i in [0, 1]. The system evolves in discrete time by the standard
sigmoid update:

    A_i(t+1) = f( A_i(t) + Σ_{j≠i} w_{j→i} · A_j(t) ),   f(x) = 1 / (1 + e^{-λx})

Iterating to a fixed point is the "system dynamics" simulation: a change in one
concept propagates through the weighted causal links and re-settles the whole
state. `scenario()` clamps an enabler "on" and reports how every other concept
shifts versus baseline. `nhl_step()` is a Nonlinear Hebbian Learning update — the
rule family underlying the hybrid learning algorithms Yousefi tunes weights with.
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray, lam: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-lam * x))


class FCM:
    def __init__(self, concepts: list[str], weights: np.ndarray, lam: float = 1.0):
        """concepts: names; weights[i, j] = causal influence of i ON j, in [-1, 1]."""
        n = len(concepts)
        assert weights.shape == (n, n), "weight matrix must be n x n"
        self.concepts = list(concepts)
        self.W = np.asarray(weights, dtype=float)
        self.lam = lam
        self.index = {c: i for i, c in enumerate(concepts)}

    def _update(self, A: np.ndarray, clamp: dict[int, float] | None) -> np.ndarray:
        # contribution into concept i is Σ_j w_{j->i} A_j = (W^T A)_i
        nxt = sigmoid(A + self.W.T @ A, self.lam)
        if clamp:
            for i, v in clamp.items():
                nxt[i] = v
        return nxt

    def run(
        self,
        initial: np.ndarray | None = None,
        clamp: dict[str, float] | None = None,
        max_steps: int = 50,
        eps: float = 1e-5,
    ):
        """Iterate to a fixed point. Returns (final_state, trajectory).

        clamp: {concept_name: held_value} — concepts pinned each step
               (this is how an enabler is switched "on" in a scenario).
        trajectory: array of shape (steps+1, n_concepts) for plotting dynamics.
        """
        n = len(self.concepts)
        A = np.full(n, 0.5) if initial is None else np.asarray(initial, float).copy()
        clamp_idx = {self.index[k]: v for k, v in (clamp or {}).items()}
        for i, v in clamp_idx.items():
            A[i] = v

        traj = [A.copy()]
        for _ in range(max_steps):
            nxt = self._update(A, clamp_idx)
            traj.append(nxt.copy())
            if np.max(np.abs(nxt - A)) < eps:
                A = nxt
                break
            A = nxt
        return A, np.array(traj)

    def scenario(self, activate: str, value: float = 1.0, **kw):
        """Compare baseline vs. holding one enabler at `value`.

        Returns {concept: (baseline, scenario, delta)} — the dynamic, causal
        "what changes downstream when I turn this enabler on" view.
        """
        base, _ = self.run(**kw)
        scen, _ = self.run(clamp={activate: value}, **kw)
        return {
            c: (round(float(base[i]), 4), round(float(scen[i]), 4),
                round(float(scen[i] - base[i]), 4))
            for i, c in enumerate(self.concepts)
        }

    def nhl_step(self, A: np.ndarray, eta: float = 0.04, decay: float = 0.98) -> np.ndarray:
        """One Nonlinear Hebbian Learning update of the weights.

        Only existing (non-zero) causal links are adapted, preserving the
        expert-defined structure — the constraint Yousefi's hybrid learning also
        respects. Returns the updated weight matrix.
        """
        W = decay * self.W
        mask = self.W != 0
        # Δw_{j->i} = η · A_j · (A_i − w_{j->i} · A_j)
        for i in range(len(A)):
            for j in range(len(A)):
                if mask[j, i]:
                    W[j, i] += eta * A[j] * (A[i] - self.W[j, i] * A[j])
        return np.clip(W, -1.0, 1.0)

    def to_dot(self) -> str:
        """Graphviz DOT string: green = reinforcing link, red = inhibiting link."""
        lines = ["digraph FCM {", '  rankdir=LR;',
                 '  node [shape=box style=rounded fontname="Helvetica"];']
        for i, ci in enumerate(self.concepts):
            for j, cj in enumerate(self.concepts):
                w = self.W[i, j]
                if w == 0:
                    continue
                color = "forestgreen" if w > 0 else "firebrick"
                pen = 0.5 + 3.0 * abs(w)
                lines.append(
                    f'  "{ci}" -> "{cj}" '
                    f'[label="{w:+.2f}" color={color} penwidth={pen:.2f} '
                    f'fontcolor={color} fontsize=10];'
                )
        lines.append("}")
        return "\n".join(lines)
