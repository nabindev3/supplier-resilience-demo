"""
DEA module — the *recognizable DNA* of Yousefi, Jahangoshai Rezaee & Solimanpur
(2021), "Supplier selection and order allocation using two-stage hybrid supply
chain model and game-based order price," Operational Research 21(1), 553-588.

In that paper, supplier *efficiency* is computed with Data Envelopment Analysis
(DEA) and then fed into the order-allocation model so that orders flow toward
efficient suppliers rather than merely cheap ones.

Here we implement the standard input-oriented CCR (Charnes-Cooper-Rhodes) model
in its envelopment form, solved as one LP per supplier (DMU) with PuLP/CBC.

For DMU `o`:
    minimize   theta
    s.t.       sum_j lambda_j * x_ij <= theta * x_io   for each input  i
               sum_j lambda_j * y_rj >=        y_ro     for each output r
               lambda_j, theta >= 0
Efficiency score = theta*  in (0, 1];  1.0 == on the efficient frontier.
"""

import pulp


def ccr_input_efficiency(inputs: dict, outputs: dict) -> dict:
    """Return {supplier: efficiency_score} via input-oriented CCR DEA.

    inputs:  {supplier: [values to MINIMIZE]}  e.g. price, lead time, defect %
    outputs: {supplier: [values to MAXIMIZE]}  e.g. quality, on-time %, capacity
    """
    suppliers = list(inputs.keys())
    n_in = len(next(iter(inputs.values())))
    n_out = len(next(iter(outputs.values())))
    scores = {}

    for o in suppliers:
        prob = pulp.LpProblem(f"DEA_{o}", pulp.LpMinimize)
        theta = pulp.LpVariable("theta", lowBound=0)
        lam = {j: pulp.LpVariable(f"lambda_{j}", lowBound=0) for j in suppliers}

        prob += theta  # objective

        for i in range(n_in):
            prob += (
                pulp.lpSum(lam[j] * inputs[j][i] for j in suppliers)
                <= theta * inputs[o][i]
            )
        for r in range(n_out):
            prob += (
                pulp.lpSum(lam[j] * outputs[j][r] for j in suppliers)
                >= outputs[o][r]
            )

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        scores[o] = round(pulp.value(theta) or 0.0, 4)

    return scores
