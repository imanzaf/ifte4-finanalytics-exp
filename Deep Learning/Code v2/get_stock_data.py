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

def merge_csvs(tickers, segments, output_folder):
    temp_folder = "stock_data/temp_test"
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    for ticker in tickers:
        dfs = []
        for (start, end) in segments:
            temp_path = os.path.join(temp_folder, f"{ticker}_{start}_{end}.csv")
            df = download_or_load(ticker, start, end, temp_path)
            dfs.append(df)
        merged_df = pd.concat(dfs).sort_index().drop_duplicates()
        merged_df.to_csv(os.path.join(output_folder, f"{ticker}_prices.csv"))
        print(f" Merged test data saved for {ticker} to {output_folder}")

    # Optional: clean up temporary files
    for file in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, file))
    os.rmdir(temp_folder)


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    benchmark = "^GSPC"

    #  Choose training period: 'crisis', 'stable', 'covid'
    train_period = "stable"  # ← change this to select different training data

    # Date ranges
    periods = {
        "crisis": ("2007-10-01", "2009-03-31"),
        "stable": ("2016-01-01", "2018-12-31"),
        "covid": ("2020-02-01", "2020-12-31"),
    }

    if train_period not in periods:
        raise ValueError(f" Invalid training period: {train_period}. Choose from {list(periods.keys())}")

    train_start, train_end = periods[train_period]
    train_label = f"train_{train_period}"
    print(f"\n Training on {train_period.upper()} period ({train_start} to {train_end})")
    fetch_all_data(tickers, train_start, train_end, f"stock_data/{train_label}")

    # Define test set: crisis + stable + covid
    print("\n Testing on CRISIS + STABLE + COVID periods")
    test_segments = [periods["crisis"], periods["stable"], periods["covid"]]
    merge_csvs(tickers, test_segments, "stock_data/test")

    # Full-range benchmark (2007–2020)
    benchmark_start = periods["crisis"][0]
    benchmark_end = periods["covid"][1]
    fetch_all_data([benchmark], benchmark_start, benchmark_end, "stock_data/benchmark")

    print(f"\n All data processed. Training data folder: stock_data/{train_label}")
