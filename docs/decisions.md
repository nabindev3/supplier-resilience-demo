# Design decisions

Notes on the non-obvious choices, mostly so I remember why I made them.

## A forecast in front of a deterministic model

The 2021 paper takes annual demand D as a given constant. That's the first
thing I wanted to change: in practice D comes from somewhere, and the
uncertainty around it matters for how much capacity you want in reserve.
Prophet over SARIMA-type models because it handles missing days natively (the
synthetic history has holes on purpose), multiplicative seasonality is a
one-flag switch, and the prediction interval comes for free — that interval
is what D_lower/D_upper in `forecast.annual_demand()` carry forward.

## DEA: cost in, quality and on-time out — capacity deliberately excluded

First version had four inputs (unit cost, holding cost, lead time, defect
rate) and capacity as the output. Two problems: with five dimensions on 10
DMUs, a third of the pool lands at exactly 1.0 and the rest crowd toward it
(too few DMUs per dimension to discriminate), and capacity as an output
double-rewards size — capacity is already a hard constraint in the MILP, it
shouldn't also buy efficiency points. The
current split asks one clean question: which supplier turns a dollar into
the most quality and reliability?

## Z2 counts selected suppliers, not allocated quantity

Z2 = sum of DEA scores over *selected* suppliers (eff_k · y_k), following the
paper's structure, rather than quantity-weighted efficiency (eff_k · q_k).
The catch: with binary terms only, the solver can set y_k = 1 everywhere and
collect efficiency points without ordering anything. The min-order linking
constraint (q_k ≥ min_order_k · y_k) closes that hole — selecting a supplier
commits real volume. Without it, every "efficiency-driven" solution is a lie.

## Range normalisation in the global criterion, not ideal-value

This one cost me an afternoon. The textbook global criterion divides each
objective's deviation by its ideal value. With that, the 10-point weight
sweep collapsed to ~3 distinct solutions: cost only moves about 5% off its
ideal across the whole Pareto set, while the efficiency sum moves about 80%,
so the Z2 term dominated at almost any weight. Dividing by the
ideal-to-nadir *range* instead maps both deviations onto [0, 1] and the sweep
spreads out properly. The nadirs come from lexicographic solves (Z2 at the
cost optimum; cheapest cost that still reaches Z2*).

## Holding cost charged on q/2

Average cycle stock: if you order q_k over the year and draw it down evenly,
you hold about half of it on average. Crude (no safety stock, no order
splitting) but linear, which keeps the model a MILP.

## Setup cost added to the supplier config

Without a fixed cost of *having* a supplier, the selection binary is
decoration — the solver would never strictly prefer fewer suppliers. The
$2-5k/year setup cost (contracting, onboarding, audits) is what Z2's push
for a broad supplier base has to fight against.

## The allocation model never goes infeasible

`allocation.py` carries an `unmet` variable with a large penalty instead of
a hard demand constraint. When a disruption removes capacity, the honest
answer is "you lose service", not "the model is infeasible". The stress test
relies on this: it reports realised service level after a supplier fails.

## Buyer budget at 95% of the no-negotiation cost

Stage 2 needs a reason for anyone to negotiate. Setting B below the
list-price purchasing cost makes the status quo unaffordable by
construction; the gap (about $215k at the default weights) is the concession
the bargaining game has to extract. The 0.95 is arbitrary and adjustable —
the point is only that it's strictly below 1.

## Synthetic data left imperfect

`demand_data.py` drops ~1% of days at random plus a ~10-day hole. Partly
realism, partly a guard: anything downstream that silently assumes a gapless
daily series should break in development, not later.
