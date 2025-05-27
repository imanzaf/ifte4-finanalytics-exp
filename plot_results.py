import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import os

def plot_portfolio_comparison():
    # Load position history
    position_history = pd.read_csv('results/position_history.csv', index_col='date', parse_dates=True)
    
    # Calculate daily returns for agent's portfolio
    position_history['portfolio_return'] = position_history['portfolio_value'].pct_change()
    position_history['cumulative_return'] = (1 + position_history['portfolio_return']).cumprod() - 1
    
    # Get S&P 500 data for the same period
    sp500 = yf.download('^GSPC', 
                       start=position_history.index[0],
                       end=position_history.index[-1])
    
    # Calculate S&P 500 returns
    sp500['return'] = sp500['Close'].pct_change()
    sp500['cumulative_return'] = (1 + sp500['return']).cumprod() - 1
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot cumulative returns
    plt.plot(position_history.index, 
             position_history['cumulative_return'] * 100,
             label='Q-Learning Agent',
             linewidth=2)
    
    plt.plot(sp500.index,
             sp500['cumulative_return'] * 100,
             label='S&P 500',
             linewidth=2)
    
    # Customize the plot
    plt.title('Portfolio Performance Comparison (2019)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/portfolio_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_portfolio_comparison() 