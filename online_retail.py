"""Real demand history from the UCI Online Retail II dataset.

This replaces the synthetic series in `demand_data.py` with a real one: the
total daily order volume of a UK online retailer (Dec 2009 - Dec 2011),
across its whole product catalogue. The buyer's problem becomes sourcing that
aggregate volume (~6.2M units/year) from the supplier pool.

Why the total and not a single product: I tried forecasting one consistently
ordered item (21212, "PACK OF 72 RETRO SPOT CAKE CASES") to keep a literal
"one item" reading, but single-product retail demand is far too spiky to
forecast — Prophet returned a negative point estimate or a band ±5x the mean.
Aggregating thousands of SKUs averages that noise out (law of large numbers),
and the total has a clean multiplicative holiday swing Prophet fits well: the
point forecast lands within ~1% of the historical mean with the only
positive-throughout prediction interval. `build_demand_history(product=...)`
still exists if you want to inspect the single-product case.

Data: Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5CG6D  (CC BY 4.0). The retailer does
not trade every day (no Saturday sales), so the series has systematic gaps;
Prophet handles them natively, which is the point.

    python online_retail.py            # build demand_history.csv (total volume)
    python online_retail.py 84879      # ...for a single product code instead
"""

import sys
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = Path("data_raw")
RAW_ZIP = RAW_DIR / "online_retail_ii.zip"
RAW_XLSX = RAW_DIR / "online_retail_II.xlsx"
CLEAN_CACHE = RAW_DIR / "online_retail_clean.csv"  # slim cleaned transactions

# the single-product fallback (used only if a code is passed on the CLI)
EXAMPLE_PRODUCT = "21212"
CSV_PATH = "demand_history.csv"


def download_raw() -> Path:
    """Fetch and unzip the Excel workbook into data_raw/ if it isn't there."""
    if RAW_XLSX.exists():
        return RAW_XLSX
    RAW_DIR.mkdir(exist_ok=True)
    if not RAW_ZIP.exists():
        import urllib.request
        print(f"downloading {URL} (~45 MB)…")
        urllib.request.urlretrieve(URL, RAW_ZIP)
    with zipfile.ZipFile(RAW_ZIP) as z:
        z.extractall(RAW_DIR)
    return RAW_XLSX


def load_clean_transactions(use_cache: bool = True) -> pd.DataFrame:
    """All real-product sale lines, cancellations and returns removed.

    Parsing the 1M-row workbook takes ~50s, so the cleaned slim frame
    (StockCode, Quantity, day) is cached to CSV after the first build.
    """
    if use_cache and CLEAN_CACHE.exists():
        return pd.read_csv(CLEAN_CACHE, parse_dates=["day"],
                           dtype={"StockCode": str})

    xlsx = download_raw()
    xls = pd.ExcelFile(xlsx, engine="openpyxl")
    frames = [pd.read_excel(xls, sheet_name=s,
                            usecols=["Invoice", "StockCode", "Quantity",
                                     "InvoiceDate"])
              for s in xls.sheet_names]
    d = pd.concat(frames, ignore_index=True)
    d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"])

    # cancellations (Invoice starting 'C') and returns carry negative quantity
    d = d[~d["Invoice"].astype(str).str.startswith("C")]
    d = d[d["Quantity"] > 0].copy()
    # keep real product codes (5-6 digit), drop POSTAGE / DOT / MANUAL / etc.
    d["StockCode"] = d["StockCode"].astype(str)
    d = d[d["StockCode"].str.match(r"^\d{5}\b")]
    d["day"] = d["InvoiceDate"].dt.normalize()

    clean = d[["StockCode", "Quantity", "day"]].reset_index(drop=True)
    RAW_DIR.mkdir(exist_ok=True)
    clean.to_csv(CLEAN_CACHE, index=False)
    return clean


def build_demand_history(product: str | None = None,
                         path: str = CSV_PATH) -> pd.DataFrame:
    """Daily order volume in Prophet's ds/y shape.

    `product=None` (default) sums every product line into the retailer's total
    daily volume — the smooth, forecastable series the pipeline uses. Pass a
    StockCode to build that single product's series instead (spiky; for
    inspection only).

    Gaps (days the retailer didn't trade) are left as missing rows on purpose
    — Prophet treats them as unknown, not zero.
    """
    clean = load_clean_transactions()
    if product is not None:
        clean = clean[clean["StockCode"] == str(product)]
        if clean.empty:
            raise ValueError(f"no sales found for product {product!r}")
    daily = (clean.groupby("day")["Quantity"].sum()
             .rename("y").reset_index().rename(columns={"day": "ds"}))
    daily = daily.sort_values("ds").reset_index(drop=True)
    daily.to_csv(path, index=False)
    return daily


if __name__ == "__main__":
    product = sys.argv[1] if len(sys.argv) > 1 else None
    df = build_demand_history(product)
    span_days = (df["ds"].max() - df["ds"].min()).days
    label = f"product {product}" if product else "total volume"
    print(f"{label}: wrote {len(df)} trading days to {CSV_PATH} "
          f"({df['ds'].min().date()} to {df['ds'].max().date()})")
    print(f"demand/day  mean {df['y'].mean():.0f}  "
          f"min {df['y'].min():.0f}  max {df['y'].max():.0f}")
    print(f"~{df['y'].sum() / (span_days / 365):,.0f} units/year over the span")
