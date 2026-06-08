# Supplier Selection, Order Allocation & Disruption Resilience

> A DEA-driven supplier order-allocation model with **disruption-resilience
> analysis** and a **Fuzzy Cognitive Map** of resilience/sustainability enablers —
> bridging a 2021 two-stage supply-chain model toward the FCM + hybrid-learning
> methods in Dr. Yousefi's recent blockchain / sustainable-supply-chain research.

**▶ Live demo:** https://supplier-resilience-demo-6fuayogumnszf6bneytvbc.streamlit.app/ — no install needed.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://supplier-resilience-demo-6fuayogumnszf6bneytvbc.streamlit.app/)
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
- [Relation to Dr. Yousefi's research](#relation-to-dr-yousefis-research)
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
- **Fuzzy Cognitive Map (FCM)** — a signed, weighted causal graph of resilience &
  sustainability enablers (blockchain traceability, smart contracts, supplier
  diversification → visibility → disruption risk → resilience → sustainability).
  Iterating the sigmoid update rule simulates how switching one enabler "on"
  propagates through the system and re-settles every other concept, with a
  Nonlinear Hebbian Learning step for weight adaptation.

## Relation to Dr. Yousefi's research

This project is built directly on the methodological toolkit in Dr. Samuel
Yousefi's work, so each component maps to a concept from his papers:

| This repo | Concept | Source in his work |
|-----------|---------|--------------------|
| `dea.py` — CCR efficiency scoring | **Data Envelopment Analysis** to rank suppliers/enablers | Yousefi et al. (2021), *Oper. Res.* 21(1); also used in the 2022 FCM paper below |
| `allocation.py` — DEA-aware order allocation | **Supplier selection & order allocation** (Stage 1) | Yousefi, Jahangoshai Rezaee & Solimanpur (2021) |
| `allocation.py` — resilience levers + `stress_test` | **Disruption-risk / resilience** extension | the gap left open by the 2021 deterministic model; his current research agenda |
| `fcm.py` — causal graph + sigmoid state propagation | **Fuzzy Cognitive Maps** of enablers → performance targets | Yousefi & Mohamadpour Tosarkani (2022), *IJPE* 246; *Eng. Appl. of AI* (2024) |
| `fcm.py` — `run()` iterating to a fixed point | **System dynamics / state transitions over time** | the FCM simulation step in the 2022/2024 papers |
| `fcm.py` — `nhl_step()` Nonlinear Hebbian Learning | a building block of the **hybrid learning algorithm** he uses to tune FCM weights | Yousefi & Mohamadpour Tosarkani (2022); *Eng. Appl. of AI* (2024) |

**Honest scope:** the FCM weights here are expert-defined illustrative values, not
learned from data, and `nhl_step()` is a single Hebbian rule rather than his full
hybrid (Hebbian + evolutionary) algorithm. The aim is to demonstrate fluency with
the *methods* — DEA, FCM, causal state-propagation, Hebbian learning — and to show
how they connect supplier allocation to the resilience and sustainability targets
his recent work centres on.

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

**Zero install:** open the [live demo](https://supplier-resilience-demo-6fuayogumnszf6bneytvbc.streamlit.app/).

Or run it locally:

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

**FCM (Fuzzy Cognitive Map).** Concepts carry activations `A_i ∈ [0, 1]`; the
state evolves by the standard sigmoid rule until it reaches a fixed point:

```
A_i(t+1) = f( A_i(t) + Σ_{j≠i} w_{j→i} · A_j(t) ),   f(x) = 1 / (1 + e^{-λx})
```

A *scenario* clamps one enabler "on" and re-runs to convergence, so its influence
propagates through the signed weights to every downstream concept. `nhl_step()`
applies a Nonlinear Hebbian Learning update, adapting only the existing causal
links (preserving the expert-defined structure).

## Project layout

| File | Role |
|------|------|
| [`dea.py`](dea.py) | Input-oriented CCR DEA efficiency (PuLP/CBC) |
| [`allocation.py`](allocation.py) | MILP order allocation + post-commit `stress_test` |
| [`data.py`](data.py) | Synthetic 6-supplier dataset + DEA input/output split |
| [`fcm.py`](fcm.py) | Fuzzy Cognitive Map engine: state propagation, scenarios, Hebbian learning |
| [`fcm_data.py`](fcm_data.py) | Enabler/target concepts and signed causal weight matrix |
| [`app.py`](app.py) | Streamlit UI (Allocation & Resilience tab + Causal Map tab) |
| [`test_model.py`](test_model.py) | Smoke tests for the engine (allocation + FCM) |
| [`requirements.txt`](requirements.txt) | Dependencies |

## Tests

```bash
python test_model.py     # plain runner, no extra dependency
# or, if pytest is installed:
python -m pytest
```

The tests check that DEA scores stay in (0, 1], allocations meet demand, the
resilient plan diversifies, resilience pays off under a disruption, the FCM
converges, its causal signs hold, and Hebbian learning stays well-behaved.

## Limitations

- **Stage 2 (Nash bargaining price) is not reimplemented** — scope is Stage 1 plus
  the resilience extension.
- DEA is the basic CCR (constant returns to scale); no slack-based or
  super-efficiency variant.
- Single-period, single-supplier deterministic disruption. A natural next step is
  scenario-based / stochastic disruption or robust optimisation.
- Supplier data is synthetic and illustrative.
- FCM weights are expert-defined, not learned; `nhl_step()` is a single Hebbian
  rule, not a full hybrid (Hebbian + evolutionary) learning algorithm.

## References

Yousefi, S., Jahangoshai Rezaee, M., & Solimanpur, M. (2021). Supplier selection
and order allocation using two-stage hybrid supply chain model and game-based
order price. *Operational Research, 21*(1), 553–588.

Yousefi, S., & Mohamadpour Tosarkani, B. (2022). An analytical approach for
evaluating the impact of blockchain technology on sustainable supply chain
performance. *International Journal of Production Economics, 246*, 108429.

Yousefi, S., & Mohamadpour Tosarkani, B. (2024). Enhancing sustainable supply
chain readiness to adopt blockchain: A decision support approach for barriers
analysis. *Engineering Applications of Artificial Intelligence.*

## License

MIT — see [LICENSE](LICENSE).
