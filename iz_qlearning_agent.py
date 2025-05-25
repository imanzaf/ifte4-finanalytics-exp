"""
simple q-learning agent for stock trading

- state: portfolio value, position size, price change
- action: hold, buy, sell
- reward: portfolio value change
- done: reached the end of the data

TODO:
- add action for quantity of shares to buy/sell
- update max position to be 100% (allow for agent to choose subset of stocks to trade)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import random
from datetime import datetime
from get_stock_data import get_stock_data

class StockTradingEnv:
    def __init__(self, 
                 stocks: List[str],
                 initial_balance: float = 100000.0,
                 max_position: float = 0.2):      # Maximum 20% of portfolio in one stock
        """
        Initialize the trading environment.
        """
        self.stocks = stocks
        self.initial_balance = initial_balance
        self.max_position = max_position
        
        # Load historical data
        print("Loading historical data...")
        self.symbols, self.data = get_stock_data(stocks=stocks, start_date=datetime(2015, 1, 1), end_date=datetime(2018, 12, 31))
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
        self.last_action = None
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get the current state representation."""
        state = []
        
        # Add portfolio information (discretized)
        portfolio_value = self._get_portfolio_value()
        portfolio_ratio = portfolio_value / self.initial_balance
        portfolio_bucket = min(int(portfolio_ratio * 5), 4)  # 5 buckets
        state.append(portfolio_bucket)
        
        # Add position information for each stock
        for symbol in self.symbols:
            # Position size (discretized)
            position_value = self.positions[symbol] * self.prices[symbol][self.current_step]
            position_ratio = position_value / portfolio_value
            position_bucket = min(int(position_ratio * 5), 4)  # 5 buckets
            state.append(position_bucket)
            
            # Price momentum (last 3 days)
            if self.current_step >= 3:
                momentum = sum(self.returns[symbol][self.current_step-i] for i in range(1, 4))
                momentum_bucket = min(int((momentum + 0.1) * 10), 9)  # 10 buckets
            else:
                momentum_bucket = 5  # neutral
            state.append(momentum_bucket)
        
        return np.array(state)
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        """
        if self.current_step >= len(self.prices[self.symbols[0]]) - 1:
            return self._get_state(), 0, True, {'portfolio_value': self._get_portfolio_value()}
        
        old_portfolio_value = self._get_portfolio_value()
        
        # Execute trades
        for symbol, action in zip(self.symbols, actions):
            price = self.prices[symbol][self.current_step]
            
            if action == 1:  # Buy
                # Buy 25% of max position
                max_shares = int((self.balance * self.max_position * 0.25) / price)
                if max_shares > 0:
                    cost = max_shares * price
                    if cost <= self.balance:
                        self.positions[symbol] += max_shares
                        self.balance -= cost
            
            elif action == 2:  # Sell
                if self.positions[symbol] > 0:
                    # Sell 25% of position
                    shares_to_sell = max(1, int(self.positions[symbol] * 0.25))
                    revenue = shares_to_sell * price
                    self.positions[symbol] -= shares_to_sell
                    self.balance += revenue
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        new_portfolio_value = self._get_portfolio_value()
        value_change = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # Add penalty for holding too much cash
        cash_ratio = self.balance / new_portfolio_value
        cash_penalty = -0.001 if cash_ratio > 0.5 else 0
        
        # Add penalty for too many transactions
        transaction_penalty = -0.0001 * sum(1 for a in actions if a != 0)
        
        # Add penalty for not diversifying
        position_ratios = [self.positions[s] * self.prices[s][self.current_step] / new_portfolio_value 
                         for s in self.symbols]
        diversification_penalty = -0.001 * max(position_ratios) if max(position_ratios) > 0.4 else 0
        
        reward = value_change + cash_penalty + transaction_penalty + diversification_penalty
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
                 exploration_decay: float = 0.995,
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
        """Convert state array to string key for Q-table."""
        return str(state.tolist())
    
    def get_action(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy."""
        state_key = self._get_state_key(state)
        
        # Exploration
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        
        # Exploitation
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return np.argmax(self.q_table[state_key])
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """Update Q-value for state-action pair."""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Initialize Q-values if not present
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update
        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_state_key])
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state_key][action] = new_value
        
        # Decay exploration rate with minimum threshold
        self.exploration_rate = max(self.min_exploration_rate, 
                                  self.exploration_rate * self.exploration_decay)

def train_agent(env: StockTradingEnv, agent: QLearningAgent, episodes: int = 1000):
    """
    Train the Q-learning agent.
    """
    best_portfolio_value = env.initial_balance
    best_episode = 0
    episode_rewards = []
    episode_returns = []  # Track percentage returns per episode
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get actions for each stock
            actions = [agent.get_action(state) for _ in range(len(env.stocks))]
            
            # Take step in environment
            next_state, reward, done, info = env.step(actions)
            
            # Update agent
            for action in actions:
                agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
        
        # Track performance
        final_portfolio_value = info['portfolio_value']
        episode_return = (final_portfolio_value - env.initial_balance) / env.initial_balance
        episode_rewards.append(total_reward)
        episode_returns.append(episode_return)
        
        if final_portfolio_value > best_portfolio_value:
            best_portfolio_value = final_portfolio_value
            best_episode = episode
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_return = np.mean(episode_returns[-100:])
            print(f"Episode {episode + 1}/{episodes}, "
                  f"Avg Reward: {avg_reward:.3f}, "
                  f"Avg Return: {avg_return:.3f}, "
                  f"Final Portfolio Value: ${final_portfolio_value:,.2f}, "
                  f"Exploration Rate: {agent.exploration_rate:.3f}")

# Example usage
if __name__ == "__main__":
    # Initialize environment
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    env = StockTradingEnv(stocks=stocks)
    
    # Initialize agent
    state_size = len(env._get_state())  # State size depends on number of stocks
    action_size = 3  # Hold, Buy, Sell
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.2,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.999,
        min_exploration_rate=0.01
    )
    
    # Train agent
    train_agent(env, agent, episodes=1000)
