"""
DeepStack Data Updater
Fetches 3 years of historical OHLCV data for all 6 coins
Saves to local CSV files in the data/ folder
Run this anytime you want fresh data: python update_data.py
"""

import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

COINS    = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'BNB-USD', 'DOGE-USD']
DATA_DIR = 'data'
PERIOD   = '3y'
INTERVAL = '1d'

os.makedirs(DATA_DIR, exist_ok=True)


def fetch_coin(coin, retries=5):
    """Download coin data with retry logic."""
    for attempt in range(retries):
        try:
            print(f"  Downloading {coin} (attempt {attempt+1})...")
            df = yf.download(coin, period=PERIOD, interval=INTERVAL,
                             auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError("Empty dataframe returned")

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

            if len(df) < 100:
                raise ValueError(f"Not enough data: only {len(df)} rows")

            return df

        except Exception as e:
            wait = (attempt + 1) * 20
            print(f"  Failed: {e}. Waiting {wait}s before retry...")
            time.sleep(wait)

    raise RuntimeError(f"Could not download {coin} after {retries} attempts.")


def load_existing(coin):
    """Load existing CSV if available."""
    fname = coin.replace('-', '_') + '.csv'
    path  = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    return None


def merge_data(old_df, new_df):
    """Merge old and new data, keeping all history."""
    if old_df is None:
        return new_df
    combined = pd.concat([old_df, new_df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    return combined


def save_coin(coin, df):
    """Save dataframe to CSV."""
    fname = coin.replace('-', '_') + '.csv'
    path  = os.path.join(DATA_DIR, fname)
    df.to_csv(path)
    return path


def update_all():
    print("=" * 55)
    print("  DeepStack Data Updater")
    print(f"  Period: {PERIOD} | Interval: {INTERVAL}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    results = {}
    for coin in COINS:
        print(f"\n{coin}")
        try:
            # Check existing data
            old_df = load_existing(coin)
            if old_df is not None:
                print(f"  Existing: {len(old_df)} rows "
                      f"({old_df.index[0].date()} to {old_df.index[-1].date()})")

            # Fetch new data
            new_df = fetch_coin(coin)

            # Merge with existing
            final_df = merge_data(old_df, new_df)

            # Save
            path = save_coin(coin, final_df)
            print(f"  Saved:    {len(final_df)} rows "
                  f"({final_df.index[0].date()} to {final_df.index[-1].date()})")
            print(f"  File:     {path}")

            results[coin] = {'rows': len(final_df), 'status': 'OK'}
            time.sleep(3)  # polite delay between coins

        except Exception as e:
            print(f"  ERROR: {e}")
            results[coin] = {'rows': 0, 'status': f'FAILED: {e}'}

    # Summary
    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    all_ok = True
    for coin, res in results.items():
        status = "OK" if res['status'] == 'OK' else "FAILED"
        rows   = res['rows']
        print(f"  {coin:<12} {status:<8} {rows} rows")
        if res['status'] != 'OK':
            all_ok = False

    print("=" * 55)
    if all_ok:
        print("  All coins updated successfully!")
        print("  Run 'python -m streamlit run app.py' to launch.")
    else:
        print("  Some coins failed. Try again later.")
    print("=" * 55)


if __name__ == '__main__':
    update_all()
