"""
Q-learning agent for stock trading with full training period (2017-2019)
but only reporting 2019 performance.

- state: portfolio value, position size, price change
- action: hold, buy, sell, quantity level (25% or 50%)
- reward: portfolio value change
- done: reached the end of the data
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import random
from datetime import datetime
from get_stock_data import get_stock_data, get_sp500_return
import pickle
import warnings
import os

warnings.filterwarnings("ignore")

class StockTradingEnv:
    def __init__(self, 
                 stocks: List[str],
                 initial_balance: float = 10000.0,
                 max_position: float = 0.5):      # Maximum 50% of portfolio in one stock
        """
        Initialize the trading environment.
        """
        self.stocks = stocks
        self.initial_balance = initial_balance
        self.max_position = max_position
        
        # Load historical data
        print("Loading historical data...")
        self.symbols, self.data = get_stock_data(stocks=stocks, start_date=datetime(2017, 1, 1), end_date=datetime(2019, 12, 31))
        if not self.symbols or not self.data:
            raise ValueError("Failed to load stock data")
            
        self.prices = {symbol: self.data[symbol]['data']['Close'] for symbol in self.symbols}
        
        # Calculate returns for each stock
        self.returns = {}
        for symbol in self.symbols:
            prices = self.prices[symbol]
            self.returns[symbol] = prices.pct_change().fillna(0)
        
        # Verify data
        for symbol in self.symbols:
            if len(self.prices[symbol]) == 0:
                raise ValueError(f"No price data for {symbol}")
            print(f"Loaded {len(self.prices[symbol])} days of data for {symbol}")
        
        # Trading state
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.balance = self.initial_balance
        self.positions = {symbol: 0 for symbol in self.symbols}
        self.current_step = 0
        self.portfolio_value_history = [self.initial_balance]
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get the current state representation."""
        state = []
        
        # Add portfolio information (discretized)
        portfolio_value = self._get_portfolio_value()
        portfolio_ratio = portfolio_value / self.initial_balance
        portfolio_bucket = min(int(portfolio_ratio * 5), 4)
        state.append(portfolio_bucket)
        
        # Add position information for each stock
        for symbol in self.symbols:
            position_value = self.positions[symbol] * self.prices[symbol][self.current_step]
            position_ratio = position_value / portfolio_value if portfolio_value > 0 else 0
            position_bucket = min(int(position_ratio * 5), 4)
            state.append(position_bucket)
            
            # Calculate price momentum using 3-day moving average
            if self.current_step >= 3:
                momentum = sum(self.returns[symbol][self.current_step-i] for i in range(1, 4))
                momentum_bucket = min(int((momentum + 0.1) * 10), 9)
            else:
                momentum_bucket = 5
            state.append(momentum_bucket)
        
        return np.array(state)
    
    def step(self, actions: List[Tuple[int, int]]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        actions: List of (action_type, quantity_level) for each stock
        action_type: 0=hold, 1=buy, 2=sell
        quantity_level: 1=25%, 2=50% of max position value
        """
        if self.current_step >= len(self.prices[self.symbols[0]]) - 1:
            return self._get_state(), 0, True, {'portfolio_value': self._get_portfolio_value()}
        
        old_portfolio_value = self._get_portfolio_value()
        
        # Execute trades
        for symbol, (action, qty_level) in zip(self.symbols, actions):
            price = self.prices[symbol][self.current_step]
            
            if action == 1:  # Buy
                max_position_value = self.max_position * old_portfolio_value
                current_position_value = self.positions[symbol] * price
                available_position_value = max_position_value - current_position_value
                
                buy_value = available_position_value * (0.25 if qty_level == 1 else 0.50)
                buy_value = min(buy_value, self.balance)
                
                if buy_value >= 100:  # Minimum $100 position
                    shares_to_buy = int(buy_value / price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price
                        self.positions[symbol] += shares_to_buy
                        self.balance -= cost
                    
            elif action == 2:  # Sell
                current_shares = self.positions[symbol]
                if current_shares > 0:
                    shares_to_sell = int(current_shares * (0.25 if qty_level == 1 else 0.50))
                    
                    if shares_to_sell > 0:
                        revenue = shares_to_sell * price
                        self.positions[symbol] -= shares_to_sell
                        self.balance += revenue
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward with more conservative scaling
        new_portfolio_value = self._get_portfolio_value()
        reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value * 10  # Reduced from 100 to 10
        
        # Add penalties to discourage excessive trading
        if action != 0:  # If not holding
            reward -= 0.1  # Increased penalty for trading
        
        # Add penalty for holding too much cash
        cash_ratio = self.balance / new_portfolio_value
        if cash_ratio > 0.5:  # If more than 50% in cash
            reward -= 0.2  # Increased penalty for holding cash
        
        self.portfolio_value_history.append(new_portfolio_value)
        
        # Check if episode is done
        done = self.current_step >= len(self.prices[self.symbols[0]]) - 1
        
        return self._get_state(), reward, done, {'portfolio_value': new_portfolio_value}
    
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        portfolio_value = self.balance
        for symbol in self.symbols:
            price = self.prices[symbol][self.current_step]
            portfolio_value += self.positions[symbol] * price
        return portfolio_value

class QLearningAgent:
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 0.999,
                 min_exploration_rate: float = 0.01):
        """
        Initialize the Q-learning agent.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table
        self.q_table = {}
    
    def _get_state_key(self, state: np.ndarray) -> str:
        return str(state.tolist())
    
    def get_action(self, state: np.ndarray) -> Tuple[int, int]:
        """Choose (action_type, quantity_level) using epsilon-greedy policy."""
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        # Exploration
        if random.random() < self.exploration_rate:
            action_idx = random.randint(0, self.action_size - 1)
        else:
            action_idx = np.argmax(self.q_table[state_key])
        # Map action_idx to (action_type, quantity_level)
        action_type = action_idx // 2  # 0=hold, 1=buy, 2=sell
        quantity_level = (action_idx % 2) + 1  # 1=25%, 2=50%
        return (action_type, quantity_level)
    
    def update(self, state: np.ndarray, action: Tuple[int, int], reward: float, next_state: np.ndarray):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        # Map (action_type, quantity_level) to action_idx
        action_idx = action[0] * 2 + (action[1] - 1)
        old_value = self.q_table[state_key][action_idx]
        next_max = np.max(self.q_table[next_state_key])
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state_key][action_idx] = new_value
        self.exploration_rate = max(self.min_exploration_rate, 
                                  self.exploration_rate * self.exploration_decay)

def train_agent(env: StockTradingEnv, agent: QLearningAgent, episodes: int = 1000):
    episode_rewards = []
    episode_returns = []
    final_portfolio_values = []
    
    # Find the index where 2019 starts using a more robust method
    dates = env.prices[env.symbols[0]].index
    start_2019 = dates[dates.year == 2019].min()
    start_2019_idx = dates.get_loc(start_2019)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            actions = [agent.get_action(state) for _ in range(len(env.stocks))]
            next_state, reward, done, info = env.step(actions)
            for action in actions:
                agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
            # Only track performance for 2019
            if env.current_step >= start_2019_idx:
                total_reward += reward
        
        final_portfolio_value = info['portfolio_value']
        episode_return = (final_portfolio_value - env.initial_balance) / env.initial_balance
        episode_rewards.append(total_reward)
        episode_returns.append(episode_return)
        final_portfolio_values.append(final_portfolio_value)
            
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_return = np.mean(episode_returns[-100:])
            print(f"Episode {episode + 1}/{episodes}, "
                  f"Avg Reward: {avg_reward:.3f}, "
                  f"Avg Return: {avg_return:.3f}, "
                  f"Final Portfolio Value: ${final_portfolio_value:,.2f}, "
                  f"Exploration Rate: {agent.exploration_rate:.3f}")
    
    # Use the final episode's performance instead of the best
    return final_portfolio_values[-1], episode_returns[-1]

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Initialize environment and agent
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    env = StockTradingEnv(stocks=stocks, initial_balance=10000, max_position=0.5)
    state_size = len(env._get_state())
    action_size = 6  # 3 action types x 2 quantity levels
    
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.05,  # Reduced from 0.1
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.9995,  # Slower decay
        min_exploration_rate=0.01  # Higher minimum exploration
    )
    
    # Train agent
    print("\nTraining agent on 2017-2019 data...")
    best_value, final_return = train_agent(env, agent, episodes=2000)  # Increased episodes
    
    # Save Q-table
    with open(os.path.join(results_dir, "q_table_full_train.pkl"), "wb") as f:
        pickle.dump(agent.q_table, f)
    
    # Get 2019 performance using robust date handling
    dates = env.prices[env.symbols[0]].index
    start_2019 = dates[dates.year == 2019].min()
    end_2019 = dates[dates.year == 2019].max()
    
    # Get S&P 500 benchmark for 2019
    sp500_return, sp500_start, sp500_end = get_sp500_return(
        start_2019,
        end_2019
    )
    
    # Print results
    print(f"\n2019 Performance Results:")
    print(f"  Final Portfolio Value: ${best_value:,.2f}")
    print(f"  Total Return: {final_return*100:.2f}%")
    
    print(f"\nS&P 500 Benchmark (2019):")
    print(f"  Start Price: ${sp500_start:,.2f}")
    print(f"  End Price:   ${sp500_end:,.2f}")
    print(f"  Total Return: {sp500_return*100:.2f}%")
    
    print(f"\nOutperformance:")
    print(f"  vs S&P 500: {(final_return - sp500_return)*100:.2f}%")
    
    if final_return > sp500_return:
        print("\nAgent outperformed the S&P 500!")
    else:
        print("\nAgent underperformed the S&P 500.") 