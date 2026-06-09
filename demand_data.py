"""Stage 1 (part 1) — synthetic demand history.

The 2021 Yousefi/Jahangoshai Rezaee/Solimanpur model takes demand as a single
given number. To *optimise* Stage 1 we first need something to forecast, so this
module synthesises a realistic operating history: five years of **daily** demand
built from four interpretable components,

    demand(t) = trend(t) · weekly(t) · yearly(t) · (1 + noise)

  * trend   — steady compound growth in the business (units/day rising ~8%/yr),
  * weekly  — a within-week cycle (weekday peak, weekend dip),
  * yearly  — an annual season (holiday-quarter peak, mid-year trough),
  * noise   — Gaussian day-to-day variation.

Output is a tidy DataFrame in the exact shape Prophet expects — a `ds`
(datestamp) column and a `y` (value) column — so it drops straight into the
forecasting pipeline in `forecast.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---- generative parameters (all interpretable, all in one place) ------------
HISTORY_YEARS = 5
START_DATE = "2021-01-01"
BASE_DEMAND = 1_000.0     # units/day at the start of the series
ANNUAL_GROWTH = 0.08      # +8% compound trend per year
WEEKLY_AMPLITUDE = 0.12   # ±12% weekday/weekend swing
YEARLY_AMPLITUDE = 0.20   # ±20% seasonal swing
NOISE_SD = 0.05           # 5% (relative) Gaussian daily noise
SEED = 42                 # reproducible synthetic data

CSV_PATH = "demand_history.csv"


def make_demand_history(
    years: int = HISTORY_YEARS,
    start: str = START_DATE,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return `years` of daily demand as a Prophet-ready DataFrame (ds, y).

    Components are multiplicative so the seasonal swings scale naturally with
    the growing trend, which is how real order volumes tend to behave.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=years * 365, freq="D")
    t = np.arange(len(dates))

    # Compound growth trend on the base demand level.
    trend = BASE_DEMAND * (1.0 + ANNUAL_GROWTH) ** (t / 365.0)

    # Weekly cycle: a smooth sinusoid plus an explicit weekend dip so the
    # within-week shape looks like a real order book, not a pure sine wave.
    dow = dates.dayofweek.to_numpy()                       # 0 = Monday
    weekly = 1.0 + WEEKLY_AMPLITUDE * np.sin(2 * np.pi * t / 7.0)
    weekly = weekly - WEEKLY_AMPLITUDE * (dow >= 5)        # Sat/Sun softer

    # Yearly cycle: peak late in the year (holiday demand), trough mid-year.
    doy = dates.dayofyear.to_numpy()
    yearly = 1.0 + YEARLY_AMPLITUDE * np.sin(2 * np.pi * doy / 365.0 - np.pi / 2)

    noise = rng.normal(0.0, NOISE_SD, size=len(t))
    y = trend * weekly * yearly * (1.0 + noise)
    y = np.maximum(y, 0.0).round()                         # demand is non-negative

    return pd.DataFrame({"ds": dates, "y": y})


def save_demand_history(path: str = CSV_PATH, **kwargs) -> pd.DataFrame:
    """Generate the history and persist it to CSV; return the DataFrame."""
    df = make_demand_history(**kwargs)
    df.to_csv(path, index=False)
    return df


def load_demand_history(path: str = CSV_PATH) -> pd.DataFrame:
    """Load a previously saved history, regenerating it if absent."""
    try:
        return pd.read_csv(path, parse_dates=["ds"])
    except FileNotFoundError:
        return save_demand_history(path)


if __name__ == "__main__":
    df = save_demand_history()
    print(f"Synthesised {len(df):,} days of demand -> {CSV_PATH}")
    print(f"  range : {df['ds'].min().date()} .. {df['ds'].max().date()}")
    print(f"  demand: mean {df['y'].mean():,.0f}  "
          f"min {df['y'].min():,.0f}  max {df['y'].max():,.0f}")
    print(df.head().to_string(index=False))
