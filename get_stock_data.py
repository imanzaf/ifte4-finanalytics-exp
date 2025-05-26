import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def get_sp500_return(start_date, end_date):
    """
    Get S&P 500 return for the given period.
    """
    sp500 = yf.Ticker('^GSPC')
    hist = sp500.history(start=start_date, end=end_date)
    if hist.empty:
        raise ValueError("No S&P500 data for the given period.")
    start_price = hist['Close'].iloc[0]
    end_price = hist['Close'].iloc[-1]
    return (end_price - start_price) / start_price, start_price, end_price


def get_stock_data(stocks: list[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"], 
                        start_date: datetime = datetime(2024, 1, 1), 
                        end_date: datetime = datetime(2024, 12, 31),
                        save_to_csv: bool = True):
    """
    Retrieve market cap and historical data for a list of stocks.
    
    Parameters:
    -----------
    stocks : list of str
        List of stock symbols to retrieve (default: 6 major tech stocks)
    start_date : datetime
        Start date for historical data
    end_date : datetime
        End date for historical data
    save_to_csv : bool
        Whether to save the historical data to CSV files (default: True)
    
    Returns:
    --------
    tuple
        (stock_symbols, stock_data)
        - stock_symbols: list of stock symbols
        - stock_data: dictionary with 'market_cap' and 'data' for each symbol
    """
    print(f"Fetching data for stocks: {stocks}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Create data directory if it doesn't exist
    if save_to_csv:
        data_dir = "stock_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")
    
    market_data = {}
    for symbol in stocks:
        try:
            print(f"Processing {symbol}")
            stock = yf.Ticker(symbol)
            info = stock.info
            if 'marketCap' in info:
                hist = stock.history(start=start_date, end=end_date)
                
                # Save to CSV if requested
                if save_to_csv and not hist.empty:
                    filename = f"{data_dir}/{symbol}_prices.csv"
                    hist.to_csv(filename)
                    print(f"Saved historical data to {filename}")
                
                market_data[symbol] = {
                    'market_cap': info['marketCap'],
                    'data': hist
                }
                print(f"Successfully fetched data for {symbol}")
            else:
                print(f"No market cap data available for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            continue
    
    if not market_data:
        print("No market data was successfully retrieved")
        return [], {}
    
    # Sort by market cap and get top N (if more than 6 provided)
    sorted_stocks = sorted(market_data.items(), key=lambda x: x[1]['market_cap'], reverse=True)
    top_stocks = sorted_stocks[:len(stocks)]
    
    top_symbols = [stock[0] for stock in top_stocks]
    top_data = {symbol: market_data[symbol] for symbol in top_symbols}
    
    print(f"\nSuccessfully retrieved data for {len(top_symbols)} stocks")
    return top_symbols, top_data

# Example usage
if __name__ == "__main__":
    print("Starting stock data retrieval...")
    symbols, data = get_stock_data()
    
    if symbols:
        print("\nStocks and their market caps:")
        for symbol in symbols:
            market_cap = data[symbol]['market_cap']
            print(f"{symbol}: ${market_cap:,.2f}")
            
            # Print first few rows of historical data
            print(f"\nSample of {symbol}'s historical data:")
            print(data[symbol]['data'].head())
    else:
        print("No stocks were retrieved. Please check the error messages above.")
