import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import os

def calculate_equal_weighted_returns(position_history, stocks):
    """Calculate returns for an equal-weighted portfolio"""
    print("\n=== Stock Data Download Debug ===")
    print(f"Date range: {position_history.index[0]} to {position_history.index[-1]}")
    
    position_history.index = pd.to_datetime(position_history.index, utc=True).tz_localize(None)
    print(f"Position history index: {position_history.index}")

    # Get stock data for all stocks
    stock_data = {}
    for stock in stocks:
        try:
            data = yf.download(stock, 
                             start=position_history.index[0].date(),  # Convert to date without timezone
                             end=position_history.index[-1].date(),   # Convert to date without timezone
                             progress=False)
            if not data.empty:
                # Convert index to datetime without timezone
                data.index = pd.to_datetime(data.index)
                stock_data[stock] = data
                print(f"\n{stock} data summary:")
                print(f"Shape: {data.shape}")
                print("First few rows:")
                print(data.head())
                print("\nLast few rows:")
                print(data.tail())
                print(f"Null values in Close: {data['Close'].isnull().sum()}")
            else:
                print(f"Warning: No data downloaded for {stock}")
        except Exception as e:
            print(f"Error downloading {stock}: {str(e)}")
    
    if not stock_data:
        raise ValueError("No stock data was downloaded successfully")
    
    # Calculate daily returns for each stock
    returns = pd.DataFrame(index=position_history.index)
    for stock in stocks:
        if stock in stock_data:
            # Calculate returns and fill first day with 0
            stock_returns = stock_data[stock]['Close'].pct_change().fillna(0)
            # Reindex to match position history dates
            returns[stock] = stock_returns.reindex(position_history.index, method='ffill')
            print(f"\n{stock} returns after calculation:")
            print(returns[stock].head())
    
    # Calculate equal-weighted portfolio returns
    returns['equal_weighted'] = returns.mean(axis=1)
    print("\nEqual-weighted returns:")
    print(returns['equal_weighted'].head())
    
    # Calculate cumulative returns starting from 0
    returns['cumulative_equal_weighted'] = (1 + returns['equal_weighted']).cumprod() - 1
    print("\nCumulative equal-weighted returns:")
    print(returns['cumulative_equal_weighted'].head())
    
    # Print final debug info
    print("\nFinal returns summary:")
    print(returns.describe())
    
    return returns

def plot_portfolio_comparison():
    # Load position history
    position_history = pd.read_csv('results/position_history.csv', index_col='date', parse_dates=True)
    print("\n=== Position History Debug ===")
    print(f"Shape: {position_history.shape}")
    print("First few rows:")
    print(position_history.head())
    print("\nLast few rows:")
    print(position_history.tail())
    
    # Calculate daily returns for agent's portfolio
    position_history['portfolio_return'] = position_history['portfolio_value'].pct_change().fillna(0)
    position_history['cumulative_return'] = (1 + position_history['portfolio_return']).cumprod() - 1
    
    # Get S&P 500 data for the same period
    print("\n=== S&P 500 Data Debug ===")
    sp500 = yf.download('^GSPC', 
                       start=position_history.index[0].date(),  # Convert to date without timezone
                       end=position_history.index[-1].date(),   # Convert to date without timezone
                       progress=False)
    # Convert index to datetime without timezone
    sp500.index = pd.to_datetime(sp500.index)
    print(f"Shape: {sp500.shape}")
    print("First few rows:")
    print(sp500.head())
    print("\nLast few rows:")
    print(sp500.tail())
    
    # Calculate S&P 500 returns
    sp500['return'] = sp500['Close'].pct_change().fillna(0)
    sp500['cumulative_return'] = (1 + sp500['return']).cumprod() - 1
    
    # Calculate equal-weighted portfolio returns
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    returns = calculate_equal_weighted_returns(position_history, stocks)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: Agent vs S&P 500
    ax1.plot(position_history.index, 
             position_history['cumulative_return'] * 100,
             label='Q-Learning Agent',
             linewidth=2)
    
    ax1.plot(sp500.index,
             sp500['cumulative_return'] * 100,
             label='S&P 500',
             linewidth=2)
    
    ax1.set_title('Q-Learning Agent vs S&P 500 (2019)', fontsize=14)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Agent vs Equal-Weighted
    ax2.plot(position_history.index, 
             position_history['cumulative_return'] * 100,
             label='Q-Learning Agent',
             linewidth=2)
    
    # Ensure we're using the same index for both series
    equal_weighted_returns = returns['cumulative_equal_weighted'].reindex(position_history.index)
    ax2.plot(position_history.index,
             equal_weighted_returns * 100,
             label='Equal-Weighted Portfolio',
             linewidth=2,
             color='green')
    
    ax2.set_title('Q-Learning Agent vs Equal-Weighted Portfolio (2019)', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    ax2.tick_params(axis='x', rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/portfolio_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_portfolio_comparison() 