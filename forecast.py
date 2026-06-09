"""Stage 1 (part 3) — Prophet demand-forecasting pipeline.

This is the *predictive* half of an optimised Stage 1. The 2021 Yousefi model
allocates orders for a demand figure taken as given; here we forecast that
figure from history first, so the downstream allocation (allocation.py) is
sizing real expected demand rather than a guessed constant.

Pipeline:
    1. instantiate Prophet with multiplicative weekly + yearly seasonality
       (matching how the synthetic history in demand_data.py was built),
    2. fit it on the 5-year daily history,
    3. project a future horizon and expose the total expected demand over it,
       ready to hand to the order-allocation model.
"""

from __future__ import annotations

import logging

import pandas as pd
from prophet import Prophet

from demand_data import load_demand_history

# Prophet/cmdstanpy are chatty on stdout; quiet them for a clean demo run.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

DEFAULT_HORIZON = 90  # days to forecast (a quarter)


def build_model() -> Prophet:
    """Instantiate the Prophet pipeline.

    Seasonality is multiplicative because demand swings scale with the level of
    the (growing) trend — the same generative assumption used to synthesise the
    history, so the model is configured to recover the structure that's there.
    """
    return Prophet(
        growth="linear",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.90,  # 90% prediction intervals
    )


def fit_forecast(
    history: pd.DataFrame | None = None,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[Prophet, pd.DataFrame]:
    """Fit Prophet on `history` and forecast `horizon` days ahead.

    Returns the fitted model and Prophet's full forecast frame (yhat plus
    yhat_lower / yhat_upper and the trend/seasonality components).
    """
    if history is None:
        history = load_demand_history()

    model = build_model()
    model.fit(history)

    future = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future)
    return model, forecast


def horizon_demand(
    forecast: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
) -> dict:
    """Summarise the forecast horizon into numbers the allocator can use.

    The allocation model wants a single demand quantity; we give it the total
    expected demand over the horizon, with the prediction-interval band so the
    plan can be stress-tested against optimistic / pessimistic demand too.
    """
    tail = forecast.tail(horizon)
    return {
        "horizon_days": horizon,
        "expected_demand": round(float(tail["yhat"].sum()), 2),
        "lower_demand": round(float(tail["yhat_lower"].sum()), 2),
        "upper_demand": round(float(tail["yhat_upper"].sum()), 2),
        "mean_daily": round(float(tail["yhat"].mean()), 2),
    }


def forecast_demand(horizon: int = DEFAULT_HORIZON) -> dict:
    """End-to-end convenience: history -> Prophet -> horizon demand summary."""
    _, forecast = fit_forecast(horizon=horizon)
    return horizon_demand(forecast, horizon)


if __name__ == "__main__":
    history = load_demand_history()
    print(f"Fitting Prophet on {len(history):,} days "
          f"({history['ds'].min().date()} .. {history['ds'].max().date()}) ...")

    model, forecast = fit_forecast(history, horizon=DEFAULT_HORIZON)
    summary = horizon_demand(forecast, DEFAULT_HORIZON)

    print(f"\nForecast horizon: next {summary['horizon_days']} days")
    print(f"  expected demand : {summary['expected_demand']:>12,.0f} units")
    print(f"  90% interval    : {summary['lower_demand']:>12,.0f} .. "
          f"{summary['upper_demand']:,.0f} units")
    print(f"  mean daily      : {summary['mean_daily']:>12,.0f} units/day")
    print("\nThis expected-demand figure is what feeds order allocation "
          "(allocation.py).")
