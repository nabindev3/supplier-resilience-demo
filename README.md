# Supplier Selection, Order Allocation & Disruption Resilience

> A DEA-driven supplier order-allocation model, extended with **disruption-resilience
> analysis** — bridging a 2021 two-stage supply-chain model toward modern
> supply-chain risk management.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Solver](https://img.shields.io/badge/solver-PuLP%20%2F%20CBC-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

This project reproduces the **DEA + order-allocation core** of Yousefi,
Jahangoshai Rezaee & Solimanpur (2021) and extends it with a question their
deterministic model leaves open: *when a supplier fails after orders are
committed, how much does diversification cost, and how much service does it save?*

---

## Table of contents
- [Background](#background)
- [What it does](#what-it-does)
- [Results](#results)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [Reference](#reference)
- [License](#license)

## Background

> Yousefi, S., Jahangoshai Rezaee, M., & Solimanpur, M. (2021). *Supplier
> selection and order allocation using two-stage hybrid supply chain model and
> game-based order price.* **Operational Research, 21**(1), 553–588.
> [doi:10.1007/s12351-019-00456-6](https://doi.org/10.1007/s12351-019-00456-6)

The original is a two-stage hybrid model:

1. **Stage 1** — a Multi-Objective Mixed-Integer Nonlinear Program fusing a
   single-buyer/multi-vendor coordination model with **Data Envelopment Analysis
   (DEA)** to pick efficient suppliers and allocate orders at minimum cost (later
   linearised to a quadratic program).
2. **Stage 2** — a **Nash bargaining game** that sets the buyer–supplier price.

That model is **deterministic** — it has no notion of a supplier failing. This
project keeps the recognisable Stage-1 DNA (DEA efficiency → allocation) and adds
the missing **resilience** dimension.

## What it does

- **DEA efficiency scoring** — input-oriented CCR model, one LP per supplier.
- **Order allocation** — a MILP that meets demand at minimum purchasing cost, with
  an optional reward for routing orders to DEA-efficient suppliers.
- **Resilience levers** — cap any single supplier's share of demand; require a
  minimum number of active suppliers (anti single-sourcing).
- **Disruption stress test** — commit a plan, *then* knock out a supplier; units
  committed above surviving capacity are lost. Compares a **cost-only** plan
  against a **resilient** plan on *realised* service level.

## Results

Default 6-supplier scenario, demand = 1,000 units, high-volume supplier **S6**
disrupted *after* orders are committed:

| Plan | Purchasing cost | Suppliers used | Realised service when S6 fails |
|------|----------------:|:--------------:|:------------------------------:|
| Cost-only | **$8,150** | 2 | **30%** |
| Resilient (≤40% share, ≥3 suppliers) | $8,400 | 3 | **60%** |

**A 3% cost premium doubles realised service under disruption.** That cost-vs-
resilience trade-off is precisely what a static, deterministic model cannot
surface.

## Quick start

```bash
git clone https://github.com/nabindev3/supplier-resilience-demo.git
cd supplier-resilience-demo
pip install -r requirements.txt
streamlit run app.py
```

Then edit supplier data, demand, the resilience levers, and the disruption
scenario live in the browser.

## How it works

**DEA (CCR, input-oriented).** For each supplier (DMU) `o`:

```
minimize   θ
s.t.       Σ_j λ_j · x_ij ≤ θ · x_io     for each input  i   (price, lead time, defect %)
           Σ_j λ_j · y_rj ≥      y_ro     for each output r   (quality, on-time %, capacity)
           λ_j, θ ≥ 0
```
Efficiency = θ* ∈ (0, 1]; 1.0 means on the efficient frontier.

**Allocation (MILP).** Minimise `Σ price_j·q_j  +  penalty·unmet  −  w·Σ eff_j·q_j`
subject to demand balance, capacity/linking, minimum order, per-supplier share
cap, and a minimum-supplier-count constraint.

**Stress test.** Given a *committed* allocation and a disruption, fulfilled units
= `Σ min(committed_j, surviving_capacity_j)` — no re-optimisation, because orders
were placed before the failure was known.

## Project layout

| File | Role |
|------|------|
| [`dea.py`](dea.py) | Input-oriented CCR DEA efficiency (PuLP/CBC) |
| [`allocation.py`](allocation.py) | MILP order allocation + post-commit `stress_test` |
| [`data.py`](data.py) | Synthetic 6-supplier dataset + DEA input/output split |
| [`app.py`](app.py) | Streamlit UI |
| [`requirements.txt`](requirements.txt) | Dependencies |

## Limitations

- **Stage 2 (Nash bargaining price) is not reimplemented** — scope is Stage 1 plus
  the resilience extension.
- DEA is the basic CCR (constant returns to scale); no slack-based or
  super-efficiency variant.
- Single-period, single-supplier deterministic disruption. A natural next step is
  scenario-based / stochastic disruption or robust optimisation.
- Supplier data is synthetic and illustrative.

## Reference

Yousefi, S., Jahangoshai Rezaee, M., & Solimanpur, M. (2021). Supplier selection
and order allocation using two-stage hybrid supply chain model and game-based
order price. *Operational Research, 21*(1), 553–588.

## License

MIT — see [LICENSE](LICENSE).
