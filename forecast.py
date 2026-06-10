"""Prophet demand forecasting.

The 2021 model takes demand as a given constant; here it comes from a
forecast instead. Fit Prophet on the 5-year synthetic history, project a
horizon, and reduce the forecast to the demand numbers the allocation model
needs (point estimate plus the prediction-interval band for uncertainty).
"""

import logging

import pandas as pd
from prophet import Prophet

from demand_data import load_demand_history

# prophet/cmdstanpy print a lot by default
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

DEFAULT_HORIZON = 90    # days, one quarter
ANNUAL_HORIZON = 365


def build_model() -> Prophet:
    # multiplicative seasonality because the synthetic history is built that
    # way (and real order volumes usually scale with the trend too)
    return Prophet(
        growth="linear",
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.90,
    )


def fit_forecast(history: pd.DataFrame | None = None, horizon: int = DEFAULT_HORIZON):
    """Fit on `history` and predict `horizon` days ahead.

    Returns (model, forecast) where forecast is Prophet's full output frame
    (yhat, yhat_lower, yhat_upper, components).
    """
    if history is None:
        history = load_demand_history()

    model = build_model()
    model.fit(history)

    future = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future)
    return model, forecast


def horizon_demand(forecast: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> dict:
    """Total expected demand over the horizon, with the interval band."""
    tail = forecast.tail(horizon)
    return {
        "horizon_days": horizon,
        "expected_demand": round(float(tail["yhat"].sum()), 2),
        "lower_demand": round(float(tail["yhat_lower"].sum()), 2),
        "upper_demand": round(float(tail["yhat_upper"].sum()), 2),
        "mean_daily": round(float(tail["yhat"].mean()), 2),
    }


def forecast_demand(horizon: int = DEFAULT_HORIZON) -> dict:
    _, forecast = fit_forecast(horizon=horizon)
    return horizon_demand(forecast, horizon)


def annual_demand() -> dict:
    """Annual demand D for the upcoming year, with its 90% interval.

    The lower/upper sums let the optimiser stress-test the plan against the
    pessimistic and optimistic ends of the forecast, not just the mean.
    """
    _, forecast = fit_forecast(horizon=ANNUAL_HORIZON)
    s = horizon_demand(forecast, ANNUAL_HORIZON)
    return {
        "D": s["expected_demand"],
        "D_lower": s["lower_demand"],
        "D_upper": s["upper_demand"],
        "mean_daily": s["mean_daily"],
        "horizon_days": ANNUAL_HORIZON,
    }


if __name__ == "__main__":
    history = load_demand_history()
    print(f"fitting on {len(history)} days "
          f"({history['ds'].min().date()} to {history['ds'].max().date()})")

    model, forecast = fit_forecast(history, horizon=DEFAULT_HORIZON)
    s = horizon_demand(forecast, DEFAULT_HORIZON)

    print(f"next {s['horizon_days']} days: {s['expected_demand']:,.0f} units expected "
          f"({s['lower_demand']:,.0f} to {s['upper_demand']:,.0f} at 90%), "
          f"about {s['mean_daily']:,.0f}/day")
