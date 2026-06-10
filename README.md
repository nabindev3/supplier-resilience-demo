# Supplier selection, order allocation & disruption resilience

A working reimplementation of the supplier-selection core from Yousefi,
Jahangoshai Rezaee & Solimanpur (2021), extended with the question their
deterministic model can't answer: when a supplier fails *after* orders are
committed, how much does diversification cost, and how much service does it
save?

**Live demo:** https://supplier-resilience-demo-6fuayogumnszf6bneytvbc.streamlit.app/

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Background

> Yousefi, S., Jahangoshai Rezaee, M., & Solimanpur, M. (2021). Supplier
> selection and order allocation using two-stage hybrid supply chain model and
> game-based order price. *Operational Research, 21*(1), 553-588.

The original is a two-stage model: Stage 1 fuses a buyer/vendor coordination
model with DEA so orders flow to *efficient* suppliers (not just cheap ones),
and Stage 2 sets the price through a Nash bargaining game. It's deterministic
end to end: demand is a given constant and no supplier ever fails. This repo
keeps the recognisable Stage-1 structure and fills in both gaps.

## What's here

**Forecast-driven Stage 1** (`demand_data.py`, `forecast.py`, `stage1.py`).
Instead of taking demand as given, I synthesise 5 years of daily demand
(growth trend, weekly and yearly seasonality, noise), fit Prophet on it, and
feed the annual forecast D into a MILP that picks suppliers and quantities.
Two objectives, combined with the weighted global criterion method:

- Z1: total annual cost (purchasing + holding + setup)
- Z2: sum of DEA efficiency scores of the selected suppliers

`risk_sweep()` re-solves across 10 weight settings and plots the Pareto
frontier. One thing I learned the hard way: normalising each objective by its
ideal value makes the sweep collapse to about 3 distinct solutions, because
cost only moves ~5% off its ideal while the efficiency sum moves ~80%. You
have to normalise by the ideal-to-nadir *range* to get an even sweep.

**Resilience extension** (`allocation.py`). A share cap and a minimum
supplier count force diversification, and `stress_test()` knocks out a
supplier after orders are committed. With the default 6-supplier data,
demand of 1,000 units and the high-volume supplier S6 failing:

| Plan | Purchasing cost | Suppliers | Service when S6 fails |
|------|----------------:|:---------:|:---------------------:|
| Cost-only | $8,150 | 2 | 30% |
| Resilient (max 40% share, min 3 suppliers) | $8,400 | 3 | 60% |

So a 3% cost premium doubles realised service under that disruption. That
trade-off is invisible to a deterministic model, which is the whole point.

**Stage 2 framing** (`stage2.py`). The setup for the bargaining game: pulls
q* from Stage 1, computes the annual purchasing cost at list prices, and sets
the buyer's budget at 95% of that so a negotiation is actually forced. The
Nash solution itself is not implemented yet.

**Fuzzy Cognitive Map** (`fcm.py`, `fcm_data.py`). A signed causal graph of
resilience and sustainability enablers (blockchain traceability, supplier
diversification, visibility, disruption risk, ...) with the standard sigmoid
state propagation, scenario clamping, and a Nonlinear Hebbian Learning step.
This connects the allocation work to the FCM methodology in Yousefi &
Mohamadpour Tosarkani (2022, 2024); the weights here are expert-defined, not
learned, and the NHL step is one rule out of their full hybrid algorithm.

## How the pieces fit

```
demand_data.py ──► forecast.py ──► D ± interval ──┐
                                                  ├──► stage1.py ──► q*, selected ──► stage2.py
suppliers_config.py ──► dea.py ──► efficiency ────┘         │
                                                            └──► pareto_frontier.png

data.py ──► dea.py + allocation.py ──► app.py (interactive demo, 6-supplier case)
fcm_data.py ──► fcm.py ──────────────► app.py (causal map tab)
```

The two paths share `dea.py` and the same modelling ideas but different
supplier pools: the Streamlit demo keeps the small 6-supplier case so every
number is checkable by hand, the stage-1/2 pipeline uses the 10-supplier
pool and forecasted demand.

The reasoning behind the less obvious modelling choices (why range
normalisation, why capacity is not a DEA output, why the budget sits at 95%)
is in [docs/decisions.md](docs/decisions.md).

## Running it

```bash
git clone https://github.com/nabindev3/supplier-resilience-demo.git
cd supplier-resilience-demo
pip install -r requirements.txt

streamlit run app.py     # interactive demo (6-supplier model + FCM)
python stage1.py         # forecast + DEA + weight sweep + pareto_frontier.png
python stage2.py         # bargaining-game setup on top of stage 1
python test_model.py     # smoke tests
```

The Prophet fit takes ~20s; stage1.py and stage2.py both refit it from the
synthetic history (`demand_history.csv` is regenerated if missing).

## Files

| File | What it does |
|------|--------------|
| `demand_data.py` | synthetic 5-year daily demand history |
| `forecast.py` | Prophet fit, annual demand D with 90% interval |
| `suppliers_config.py` | 10-supplier candidate pool for stage 1 |
| `stage1.py` | DEA + multi-objective MILP, weight sweep, Pareto plot |
| `stage2.py` | Nash bargaining setup (baseline cost, buyer budget) |
| `data.py` | original 6-supplier case for the interactive demo |
| `dea.py` | input-oriented CCR DEA, one LP per supplier |
| `allocation.py` | allocation MILP + post-commit stress test |
| `fcm.py`, `fcm_data.py` | Fuzzy Cognitive Map engine and the causal map |
| `app.py` | Streamlit UI |
| `test_model.py` | smoke tests |

## Limitations

- All data is synthetic. The Prophet model is fit on a series I generated, so
  it demonstrates the pipeline rather than predicting anything real.
- DEA is plain CCR (constant returns to scale), no super-efficiency variant.
- Single-period, single-supplier deterministic disruption. Scenario-based or
  stochastic disruptions would be the natural next step.
- Stage 2 stops at the setup; the Nash bargaining solution is future work.

## What's next

Roughly in order:

1. Finish Stage 2: per-supplier utility functions and disagreement points,
   then solve the Nash product for the negotiated prices (non-linear, so
   probably scipy rather than PuLP).
2. Run each plan on the Pareto frontier through `stress_test()` — that adds
   realised service under disruption as a third axis to the cost/efficiency
   trade-off, which is the picture I actually want this project to end on.
3. Replace the synthetic series with a public demand dataset (M5 or similar)
   to see whether the pipeline survives contact with real data.
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

## License

MIT, see [LICENSE](LICENSE).
