# Supplier selection, order allocation & disruption resilience

A working reimplementation of the supplier selection core from Yousefi,
Jahangoshai Rezaee & Solimanpur (2021), extended with the question their
deterministic model can't answer: when a supplier fails *after* orders are
committed, how much does diversification cost, and how much service does it
save?

**Live demo:** https://supplier-resilience-demo-6fuayogumnszf6bneytvbc.streamlit.app/

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## The problem

A buyer placing a year's order across several suppliers faces three coupled
decisions where "cheapest" and "safest" pull in opposite directions:

1. **Who supplies what, and how much?** pour everything into the cheapest
   vendor, or spread the order across more efficient and reliable ones?
2. **What is that spread actually worth?** diversification costs more up
   front, but how much *service* does it save when a supplier fails *after*
   orders are committed? A deterministic cost model never sees this trade off.
3. **What price to pay?** once quantities are fixed, the unit price is
   negotiated, and each supplier has a point below which it walks away.

This project answers all three on one forecast: demand is predicted (Prophet),
suppliers are scored on cost *and* quality/reliability (DEA), quantities come
from a multi-objective MILP, the plan is stress tested against a post commit
disruption, and the price is settled with a Nash bargaining game. The result
is a decision that **trades off cost, supplier quality, and disruption
resilience explicitly**, instead of optimising cost alone and discovering the
fragility too late.

## Background

> Yousefi, S., Jahangoshai Rezaee, M., & Solimanpur, M. (2021). Supplier
> selection and order allocation using two stage hybrid supply chain model and
> game-based order price. *Operational Research, 21*(1), 553-588.

The original is a two-stage model: Stage 1 fuses a buyer/vendor coordination
model with DEA so orders flow to *efficient* suppliers (not just cheap ones),
and Stage 2 sets the price through a Nash bargaining game. It's deterministic
end to end: demand is a given constant and no supplier ever fails. This repo
keeps the recognisable Stage-1 structure and fills in both gaps.

## What's here

**Forecast driven Stage 1** (`online_retail.py`, `forecast.py`, `stage1.py`).
Instead of taking demand as given, I fit Prophet on a **real** demand series —
the total daily order volume of a UK online retailer (UCI Online Retail II,
Dec 2009–Dec 2011, ~6.2M units/year) and feed the annual forecast D into a
MILP that picks suppliers and quantities. (`demand_data.py` keeps a synthetic
generator as an offline fallback.) A note on what real data costs you: I first
tried forecasting a single consistently ordered product to keep a literal
"one item" reading, but single product retail demand is far too spiky
Prophet returned a negative point estimate or a band ±5× the mean. The total
volume averages that noise out and has a clean multiplicative holiday swing
Prophet fits to within ~1% of the historical mean, so the buyer's problem
becomes sourcing the retailer's aggregate volume. Two objectives, combined
with the weighted global criterion method:

- Z1: total annual cost (purchasing + holding + setup)
- Z2: sum of DEA efficiency scores of the selected suppliers

`risk_sweep()` re-solves across 10 weight settings and plots the Pareto
frontier. One thing I learned the hard way: normalising each objective by its
ideal value makes the sweep collapse to about 3 distinct solutions, because
cost only moves ~5% off its ideal while the efficiency sum moves ~80%. You
have to normalise by the ideal-to-nadir *range* to get an even sweep.

`disruption_service()` then runs every plan on that frontier through the
stress test with S01 (the cheap workhorse every cost leaning plan relies on)
knocked out after orders are committed. That adds the third axis the 2021
model can't draw: the cost only plan keeps 17% service, the fully
diversified one 34%, and the curve between them prices the insurance
(`resilience_frontier.png`).

**Resilience extension** (`allocation.py`). A share cap and a minimum
supplier count force diversification, and `stress_test()` knocks out a
supplier after orders are committed. With the default 6 supplier data,
demand of 1,000 units and the high-volume supplier S6 failing:

| Plan | Purchasing cost | Suppliers | Service when S6 fails |
|------|----------------:|:---------:|:---------------------:|
| Cost-only | $8,150 | 2 | 30% |
| Resilient (max 40% share, min 3 suppliers) | $8,400 | 3 | 60% |

So a 3% cost premium doubles realised service under that disruption. That
trade-off is invisible to a deterministic model, which is the whole point.

**Stage 2: Nash bargaining over price** (`stage2.py`). Pulls q* from Stage 1,
prices the no negotiation baseline, sets the buyer's budget at 95% of it (so
a negotiation is forced), gives every supplier a walk-away profit floor, and
then solves the bargaining game with scipy (SLSQP, budget and floors as
explicit constraints): prices in [floor, list] maximising the Nash product
of all utilities. Because the utilities are linear in price, the symmetric
game has a closed-form answer (equal split of the surplus) that the optimiser
is tested against; the interesting version is the *weighted* game, where
bargaining power follows volume share, S01 carries 85% of the demand and
walks away with 85% of the supplier-side surplus. The whole step is wrapped
in a `GameTheoryPricingEngine` that returns a before/after dashboard, the
total savings ($2.6M, ~5.2% off a $50M list-price bill), and each supplier's
profit sacrifice.

**Fuzzy Cognitive Map** (`fcm.py`, `fcm_data.py`). A signed causal graph of
resilience and sustainability enablers (blockchain traceability, supplier
diversification, visibility, disruption risk, ...) with the standard sigmoid
state propagation, scenario clamping, and a Nonlinear Hebbian Learning step.
This connects the allocation work to the FCM methodology in Yousefi &
Mohamadpour Tosarkani (2022, 2024); the weights here are expert-defined, not
learned, and the NHL step is one rule out of their full hybrid algorithm.

## How the pieces fit

```
online_retail.py ──► demand_history.csv ──► forecast.py ──► D ± interval ──┐
   (UCI real data)   (demand_data.py = fallback)                           ├──► stage1.py ──► q*, selected ──► stage2.py
suppliers_config.py ──► dea.py ──► efficiency ─────────────────────────────┘         │
                                                                                     └──► pareto_frontier.png

data.py ──► dea.py + allocation.py ──► app.py (interactive demo, 6-supplier case)
fcm_data.py ──► fcm.py ──────────────► app.py (causal map tab)
```

The two paths share `dea.py` and the same modelling ideas but different
supplier pools: the Streamlit demo keeps the small 6 supplier case so every
number is checkable by hand, the stage-1/2 pipeline uses the 10 supplier
pool and the real forecasted demand.

The reasoning behind the less obvious modelling choices (why range
normalisation, why capacity is not a DEA output, why the budget sits at 95%)
is in [docs/decisions.md](docs/decisions.md).

## Running it

```bash
git clone https://github.com/nabindev3/supplier-resilience-demo.git
cd supplier-resilience-demo
pip install -r requirements.txt

streamlit run app.py     # interactive demo (allocation + FCM + Nash tab)
python online_retail.py  # (re)build demand_history.csv from the UCI dataset
python stage1.py         # forecast + DEA + weight sweep + pareto_frontier.png
python stage2.py         # bargaining-game setup on top of stage 1
python test_model.py     # smoke tests
```

The real `demand_history.csv` is committed, so nothing needs downloading to
run the pipeline. `python online_retail.py` regenerates it from the UCI
dataset (fetching the ~44MB workbook into `data_raw/` on first call); pass a
StockCode to inspect a single product instead of the total. The first forecast
fits Prophet and caches the result in `.annual_demand_cache.json` (keyed on a
hash of the data), so later runs of stage1.py/stage2.py are near-instant;
editing the CSV invalidates the cache automatically.

## Files

| File | What it does |
|------|--------------|
| `online_retail.py` | builds `demand_history.csv` from the real UCI Online Retail II data |
| `demand_data.py` | synthetic 5-year daily demand history (offline fallback) |
| `forecast.py` | Prophet fit, annual demand D with 90% interval |
| `suppliers_config.py` | 10-supplier candidate pool for stage 1 |
| `stage1.py` | DEA + multi-objective MILP, weight sweep, Pareto + resilience plots |
| `stage2.py` | Nash bargaining game over price (symmetric + volume-weighted, scipy) |
| `data.py` | original 6-supplier case for the interactive demo |
| `dea.py` | input-oriented CCR DEA, one LP per supplier |
| `allocation.py` | allocation MILP + post-commit stress test |
| `fcm.py`, `fcm_data.py` | Fuzzy Cognitive Map engine and the causal map |
| `app.py` | Streamlit UI |
| `test_model.py` | smoke tests |

## Limitations

- Demand is real (UCI Online Retail II), but the **supplier** data is not.
  Real procurement data unit cost, and especially each supplier's
  *production cost*, the number the Nash game bargains over is commercially
  confidential and essentially never public, so the 10-supplier pool stays a
  calibrated assumption. Its scale-dependent fields (capacity, min order,
  setup cost) were scaled to match the real ~6.2M-unit demand; the per-unit
  economics are illustrative, with margins set on a deliberate gradient (see
  [docs/decisions.md](docs/decisions.md)).
- DEA is plain CCR (constant returns to scale), no super-efficiency variant.
- Single-period, single supplier deterministic disruption. Scenario-based or
  stochastic disruptions would be the natural next step.
- The Stage-2 game uses transferable, risk-neutral utilities (linear in
  price), which is what makes the symmetric solution an exact equal split.
  Concave/risk-averse utilities would be a more realistic and genuinely
  non-linear extension.

## What's next

Roughly in order:

1. ~~Replace the synthetic series with a public demand dataset.~~ **Done** —
   the pipeline now forecasts UCI Online Retail II. The next step is sourcing
   real *supplier* attributes (the harder half), or at least calibrating the
   pool's per-unit economics against a published supplier-selection case study.
2. Try other bargaining-power definitions in the weighted Nash game (DEA
   efficiency, switching cost) and see how the negotiated prices move.
3. Multi-supplier and partial disruption scenarios for the resilience sweep,
   instead of the single S01-down case.
4. Learn the FCM weights from scenario data instead of fixing them by hand
   (the full hybrid-learning loop from the 2022 paper).

## References

Yousefi, S., Jahangoshai Rezaee, M., & Solimanpur, M. (2021). Supplier
selection and order allocation using two-stage hybrid supply chain model and
game-based order price. *Operational Research, 21*(1), 553-588.

Yousefi, S., & Mohamadpour Tosarkani, B. (2022). An analytical approach for
evaluating the impact of blockchain technology on sustainable supply chain
performance. *International Journal of Production Economics, 246*, 108429.

Yousefi, S., & Mohamadpour Tosarkani, B. (2024). Enhancing sustainable supply
chain readiness to adopt blockchain: A decision support approach for barriers
analysis. *Engineering Applications of Artificial Intelligence.*

## Data

Demand is derived from the **Online Retail II** dataset (Chen, D., 2019; UCI
Machine Learning Repository, [doi:10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D)),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The
committed `demand_history.csv` is the retailer's total daily order volume
aggregated from that dataset; `online_retail.py` regenerates it.

## License

MIT, see [LICENSE](LICENSE).
