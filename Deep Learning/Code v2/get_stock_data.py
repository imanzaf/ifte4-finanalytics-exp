import os
import yfinance as yf
import pandas as pd

def download_or_load(ticker: str, start_date: str, end_date: str, save_path: str):
    if os.path.exists(save_path):
        print(f" Loaded cached data for {ticker} from {save_path}")
        df = pd.read_csv(save_path, index_col=0, parse_dates=True)
    else:
        print(f" Downloading {ticker} from {start_date} to {end_date}")
        df = yf.download(ticker, start=start_date, end=end_date)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        df.to_csv(save_path)
        print(f"  Saved to {save_path}")
    return df

def fetch_all_data(tickers, start_date, end_date, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f" Created folder: {folder_name}")

    for ticker in tickers:
        file_path = os.path.join(folder_name, f"{ticker}_prices.csv")
        try:
            _ = download_or_load(ticker, start_date, end_date, file_path)
        except Exception as e:
            print(f" Error with {ticker}: {e}")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    benchmark = "^GSPC"

    # Toggle between crisis and stable training periods
    crisis_only = False  # Set True for crisis training

    # Periods (will readjust for covid period)
    crisis_start = "2007-10-01"
    crisis_end = "2009-03-31"
    stable_start = "2016-01-01"
    stable_end = "2018-12-31"

    # Select training period
    if crisis_only:
        train_label = "train_crisis"
        print("\n Training on CRISIS period only")
        fetch_all_data(tickers, crisis_start, crisis_end, f"stock_data/{train_label}")
    else:
        train_label = "train_stable"
        print("\n Training on STABLE period only")
        fetch_all_data(tickers, stable_start, stable_end, f"stock_data/{train_label}")

    # Test period includes both
    test_start = crisis_start
    test_end = stable_end
    print("\n Testing on BOTH periods (crisis + stable)")
    fetch_all_data(tickers, test_start, test_end, "stock_data/test")
    fetch_all_data([benchmark], test_start, test_end, "stock_data/benchmark")

    print(f"\n All data processed. Training data folder: stock_data/{train_label}")
