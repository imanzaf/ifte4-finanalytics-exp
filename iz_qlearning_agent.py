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
                #  transaction_fee: float = 0.001,  # assume no transaction fee
                 max_position: float = 0.2):      # Maximum 20% of portfolio in one stock
        """
        Initialize the trading environment.
        
        Parameters:
        -----------
        stocks : List[str]
            List of stock symbols to trade
        initial_balance : float
            Initial cash balance
        max_position : float
            Maximum position size as a percentage of portfolio
        """
        self.stocks = stocks
        self.initial_balance = initial_balance
        self.max_position = max_position
        
        # Load historical data
        self.symbols, self.data = get_stock_data(stocks=stocks)
        self.prices = {symbol: self.data[symbol]['data']['Close'] for symbol in self.symbols}
        
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
        
        # Add portfolio information
        portfolio_value = self.balance
        for symbol in self.symbols:
            price = self.prices[symbol][self.current_step]
            portfolio_value += self.positions[symbol] * price
        
        # Normalize portfolio value
        state.append(portfolio_value / self.initial_balance)
        
        # Add position information for each stock
        for symbol in self.symbols:
            price = self.prices[symbol][self.current_step]
            position_value = self.positions[symbol] * price
            state.append(position_value / portfolio_value)  # Position as % of portfolio
            
            # Add price change information
            if self.current_step > 0:
                prev_price = self.prices[symbol][self.current_step - 1]
                price_change = (price - prev_price) / prev_price
            else:
                price_change = 0
            state.append(price_change)
        
        return np.array(state)
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Parameters:
        -----------
        actions : List[int]
            List of actions for each stock (0: hold, 1: buy, 2: sell)
            
        Returns:
        --------
        Tuple[np.ndarray, float, bool, Dict]
            (next_state, reward, done, info)
        """
        if self.current_step >= len(self.prices[self.symbols[0]]) - 1:
            return self._get_state(), 0, True, {'portfolio_value': self._get_portfolio_value()}
        
        # Execute trades
        for symbol, action in zip(self.symbols, actions):
            price = self.prices[symbol][self.current_step]
            
            if action == 1:  # Buy
                max_shares = int((self.balance * self.max_position) / price)
                if max_shares > 0:
                    cost = max_shares * price
                    if cost <= self.balance:
                        self.positions[symbol] += max_shares
                        self.balance -= cost
            
            elif action == 2:  # Sell
                if self.positions[symbol] > 0:
                    shares_to_sell = self.positions[symbol]
                    revenue = shares_to_sell * price
                    self.positions[symbol] = 0
                    self.balance += revenue
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward (portfolio value change)
        new_portfolio_value = self._get_portfolio_value()
        reward = (new_portfolio_value - self.portfolio_value_history[-1]) / self.portfolio_value_history[-1]
        self.portfolio_value_history.append(new_portfolio_value)
        
        # Check if episode is done (reached the end of the data)
        done = self.current_step >= len(self.prices[self.symbols[0]]) - 1  # -1 due to zero indexing
        
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
                 exploration_decay: float = 0.995):
        """
        Initialize the Q-learning agent.
        
        Parameters:
        -----------
        state_size : int
            Size of the state space
        action_size : int
            Size of the action space
        learning_rate : float
            Learning rate for Q-learning
        discount_factor : float
            Discount factor for future rewards
        exploration_rate : float
            Initial exploration rate
        exploration_decay : float
            Rate at which exploration rate decays
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
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
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay

def train_agent(env: StockTradingEnv, agent: QLearningAgent, episodes: int = 1000):
    """
    Train the Q-learning agent.
    
    Parameters:
    -----------
    env : StockTradingEnv
        Trading environment
    agent : QLearningAgent
        Q-learning agent
    episodes : int
        Number of episodes to train
    """
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
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}, "
                  f"Final Portfolio Value: ${info['portfolio_value']:,.2f}")

# Example usage
if __name__ == "__main__":
    # Initialize environment
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    env = StockTradingEnv(stocks=stocks)
    
    # Initialize agent
    state_size = len(env._get_state())  # State size depends on number of stocks
    action_size = 3  # Hold, Buy, Sell
    agent = QLearningAgent(state_size=state_size, action_size=action_size)
    
    # Train agent
    train_agent(env, agent, episodes=1000)
