"""Input-oriented CCR DEA (envelopment form), one LP per supplier.

For each DMU o:
    min  theta
    s.t. sum_j lambda_j * x_ij <= theta * x_io   for each input i
         sum_j lambda_j * y_rj >= y_ro           for each output r
         lambda_j, theta >= 0

Score = theta* in (0, 1], 1.0 means on the efficient frontier. Same role as
in Yousefi et al. (2021): the scores feed the allocation model so orders go
to efficient suppliers, not just cheap ones.
"""

import pulp


def ccr_input_efficiency(inputs: dict, outputs: dict) -> dict:
    """inputs/outputs: {supplier: [values]}. Returns {supplier: score}."""
    suppliers = list(inputs.keys())
    n_in = len(next(iter(inputs.values())))
    n_out = len(next(iter(outputs.values())))
    scores = {}

    for o in suppliers:
        prob = pulp.LpProblem(f"DEA_{o}", pulp.LpMinimize)
        theta = pulp.LpVariable("theta", lowBound=0)
        lam = {j: pulp.LpVariable(f"lambda_{j}", lowBound=0) for j in suppliers}

        prob += theta

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
