"""Small Fuzzy Cognitive Map engine.

An FCM is a signed, weighted directed graph of concepts. Each concept has an
activation A_i in [0, 1] and the state evolves by the usual sigmoid rule:

    A_i(t+1) = f(A_i(t) + sum_j w_ji * A_j(t)),   f(x) = 1 / (1 + e^(-lam*x))

Iterate to a fixed point, clamp an enabler "on" to run a what-if scenario,
and nhl_step() does one Nonlinear Hebbian Learning update of the weights.
This is the FCM machinery used in Yousefi & Mohamadpour Tosarkani (2022,
IJPE 246) and the 2024 Eng. Appl. of AI paper, minus the full hybrid
learning algorithm.
"""

import numpy as np


def sigmoid(x: np.ndarray, lam: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-lam * x))


class FCM:
    def __init__(self, concepts: list[str], weights: np.ndarray, lam: float = 1.0):
        """weights[i, j] = causal influence of concept i ON concept j, in [-1, 1]."""
        n = len(concepts)
        assert weights.shape == (n, n), "weight matrix must be n x n"
        self.concepts = list(concepts)
        self.W = np.asarray(weights, dtype=float)
        self.lam = lam
        self.index = {c: i for i, c in enumerate(concepts)}

    def _update(self, A: np.ndarray, clamp: dict[int, float] | None) -> np.ndarray:
        # input to concept i is sum_j w_ji A_j, i.e. (W^T A)_i
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

        clamp pins concepts to a value each step, which is how a scenario
        switches an enabler on.
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
        """Baseline vs holding one enabler at `value`.

        Returns {concept: (baseline, scenario, delta)}.
        """
        base, _ = self.run(**kw)
        scen, _ = self.run(clamp={activate: value}, **kw)
        return {
            c: (round(float(base[i]), 4), round(float(scen[i]), 4),
                round(float(scen[i] - base[i]), 4))
            for i, c in enumerate(self.concepts)
        }

    def nhl_step(self, A: np.ndarray, eta: float = 0.04, decay: float = 0.98) -> np.ndarray:
        """One Nonlinear Hebbian Learning update.

        Only existing (non-zero) links are adapted, so the expert-defined
        structure is preserved. Returns the new weight matrix.
        """
        W = decay * self.W
        mask = self.W != 0
        # dw_ji = eta * A_j * (A_i - w_ji * A_j)
        for i in range(len(A)):
            for j in range(len(A)):
                if mask[j, i]:
                    W[j, i] += eta * A[j] * (A[i] - self.W[j, i] * A[j])
        return np.clip(W, -1.0, 1.0)

    def to_dot(self) -> str:
        """Graphviz DOT string, green = reinforcing, red = inhibiting."""
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
