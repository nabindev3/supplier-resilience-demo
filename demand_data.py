"""Synthetic daily demand history for the forecasting stage.

Five years of daily demand built as trend * weekly * yearly * noise.
Multiplicative on purpose: real order volumes swing more as the business
grows, and it also gives Prophet (in multiplicative mode) something it can
actually recover. Output columns are ds/y, the shape Prophet expects.
"""

import numpy as np
import pandas as pd

HISTORY_YEARS = 5
START_DATE = "2021-01-01"
BASE_DEMAND = 1000.0      # units/day at the start of the series
ANNUAL_GROWTH = 0.08
WEEKLY_AMPLITUDE = 0.12
YEARLY_AMPLITUDE = 0.20
NOISE_SD = 0.05
SEED = 42

CSV_PATH = "demand_history.csv"


def make_demand_history(years=HISTORY_YEARS, start=START_DATE, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=years * 365, freq="D")
    t = np.arange(len(dates))

    trend = BASE_DEMAND * (1.0 + ANNUAL_GROWTH) ** (t / 365.0)

    # sinusoid plus an explicit weekend dip; a pure sine wave looked too clean
    dow = dates.dayofweek.to_numpy()
    weekly = 1.0 + WEEKLY_AMPLITUDE * np.sin(2 * np.pi * t / 7.0)
    weekly = weekly - WEEKLY_AMPLITUDE * (dow >= 5)

    # yearly peak near year end (holiday quarter), trough mid-year
    doy = dates.dayofyear.to_numpy()
    yearly = 1.0 + YEARLY_AMPLITUDE * np.sin(2 * np.pi * doy / 365.0 - np.pi / 2)

    noise = rng.normal(0.0, NOISE_SD, size=len(t))
    y = np.maximum(trend * weekly * yearly * (1.0 + noise), 0.0).round()

    df = pd.DataFrame({"ds": dates, "y": y})

    # real demand extracts are never complete: drop ~1% of days at random,
    # plus one ~10-day hole somewhere in the middle (the kind an ERP
    # migration leaves). Prophet deals with missing dates natively, so the
    # pipeline has to cope rather than assume a gapless series.
    df = df[rng.random(len(df)) > 0.01]
    hole = int(rng.integers(200, len(df) - 200))
    df = df.drop(df.index[hole:hole + 10])
    return df.reset_index(drop=True)


def save_demand_history(path=CSV_PATH, **kwargs) -> pd.DataFrame:
    df = make_demand_history(**kwargs)
    df.to_csv(path, index=False)
    return df


def load_demand_history(path=CSV_PATH) -> pd.DataFrame:
    """Load the saved history, regenerating it if the CSV isn't there."""
    try:
        return pd.read_csv(path, parse_dates=["ds"])
    except FileNotFoundError:
        return save_demand_history(path)


if __name__ == "__main__":
    df = save_demand_history()
    print(f"wrote {len(df)} days to {CSV_PATH} "
          f"({df['ds'].min().date()} to {df['ds'].max().date()})")
    print(f"demand mean {df['y'].mean():.0f}, "
          f"min {df['y'].min():.0f}, max {df['y'].max():.0f}")
