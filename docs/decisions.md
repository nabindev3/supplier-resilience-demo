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

## Supplier margins widen toward the premium end

Stage 2 needs each supplier's production cost to compute their baseline
profit (SAP_k = margin × q*), which is what the bargaining game divides up.
I set the margins on a gradient — about 9% for the volume vendors (S01, S02,
S09) up to ~14% for the premium ones (S07, S10) — because uniform margins
make the game boring: everyone concedes proportionally. With a gradient, the
high-volume thin-margin suppliers have little room per unit but huge volume,
while the premium suppliers have room per unit but little volume at stake,
so the negotiation outcome isn't obvious in advance.

## Profit floors at 40% of baseline profit

Each supplier's walk-away point is G_k = 0.40 · SAP_k, proportional rather
than absolute so the floors scale with how much each supplier has at stake.
The 0.40 isn't free: the buyer must be able to afford every supplier sitting
at their floor price, and with the budget at 95% of list cost that breaks
just above factor ≈ 0.44 — the floor prices alone exceed B and the
bargaining set goes empty. At 0.40 there's about $15k of genuinely
negotiable surplus between "everyone at their floor" and the budget, which
is tight enough to be interesting. The two knobs (budget factor, floor
factor) trade off against each other; `frame_bargaining_problem()` reports
`bargaining_set_nonempty` so a bad combination is caught immediately.

## Nash objective in log space, solved with constrained SLSQP

The Nash product U_B · Π U_k is around 1e21 at these scales (five utilities
of 1e3-1e5 each), useless for a numeric optimiser. Maximising the sum of
logs is the same argmax (log is monotonic) and turns the product into a sum
the solver handles cleanly.

I first solved it with Nelder-Mead and a log barrier — let log(u) → −∞ at
the walls of the bargaining set and the simplex stays inside on its own.
That works but it's a penalty method standing in for constraints. The honest
formulation states them: SLSQP with the budget (Σ p_k q_k ≤ B) and each
profit floor ((p_k − prod_k)q_k ≥ G_k) as explicit inequality constraints,
prices bounded to [floor, list]. SLSQP converges in ~10-20 iterations versus
Nelder-Mead's ~190, and the constraints are now inspectable objects rather
than a side effect of the objective. One numerical detail survives the
switch: the utilities are clipped at 1e-9 inside the log so a line-search
probe just outside the set returns a finite number instead of NaN. The
optimum is strictly interior (every player keeps a positive share), so the
clip never binds at the solution — it's purely a guard.

Two facts worth remembering. First, the midpoint of the price box is *not*
a feasible start: the budget plane cuts the box very close to the floor
side (gap $215k vs $15k of surplus), so the search starts near the floors.
Second, because all utilities are linear in prices, U_B + Σ U_k is constant
(= B − cost at floors), which means the symmetric Nash solution is known in
closed form — an equal split of the surplus among the n+1 players. That's
the strongest test in the suite: the solve has to land on $3,090.30 for
every player, and it does. The symmetric game's answer doesn't depend on
q*, DEA scores or anything else, which makes it a validation tool rather
than a result — the *weighted* game is the real one. Bargaining power
follows volume share (losing 83% of your supply is a bigger threat than
losing 3%), and the same closed-form logic still checks it: with fixed
surplus the weighted solution gives each player surplus · a_i / Σa, which
the optimiser reproduces to the cent.

## Synthetic data left imperfect

`demand_data.py` drops ~1% of days at random plus a ~10-day hole. Partly
realism, partly a guard: anything downstream that silently assumes a gapless
daily series should break in development, not later.
