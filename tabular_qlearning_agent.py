"""
Q-Learning Agent for Stock Trading
=================================

This implementation demonstrates a reinforcement learning approach to stock trading using Q-learning.
The agent learns to make trading decisions by interacting with a simulated stock market environment.

Key Components:
--------------
1. StockTradingEnv: Simulates the trading environment
   - State: Portfolio value, position sizes, price momentum
   - Actions: Buy, Sell, Hold with different quantity levels
   - Reward: Portfolio value change with penalties for poor practices

2. QLearningAgent: Implements the Q-learning algorithm
   - Uses epsilon-greedy exploration strategy
   - Maintains Q-table for state-action value estimates
   - Updates Q-values based on rewards and future estimates

3. Training & Testing Pipeline:
   - Train agent on historical data
   - Test performance on out-of-sample data
   - Compare against S&P 500 benchmark
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
    """
    Stock Trading Environment
    
    This class simulates a trading environment where an agent can buy and sell stocks.
    It maintains the state of the portfolio and executes trades based on agent actions.
    
    State Space:
    -----------
    - Portfolio value bucket (0-4)
    - Position size bucket for each stock (0-4)
    - Price momentum bucket for each stock (0-9)
    
    Action Space:
    ------------
    For each stock:
    - Action type: Hold (0), Buy (1), Sell (2)
    - Quantity level: 1-5 (different percentage levels)
    
    Reward Structure:
    ---------------
    - Primary: Portfolio value change
    - Penalties:
      * Holding too much cash (>50%)
      * Excessive trading
      * Insufficient diversification (<2 stocks)
    """
    
    def __init__(self, 
                 stocks: List[str],
                 initial_balance: float = 10000.0,
                 max_position: float = 0.5):
        """
        Initialize the trading environment.
        
        Args:
            stocks: List of stock symbols to trade
            initial_balance: Starting cash balance
            max_position: Maximum position size as fraction of portfolio (0.5 = 50%)
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
        # Calculate portfolio value and convert to a discrete bucket (0-4)
        # Bucket 0: Portfolio < 20% of initial balance
        # Bucket 1: 20-40% of initial balance
        # Bucket 2: 40-60% of initial balance
        # Bucket 3: 60-80% of initial balance
        # Bucket 4: 80-100%+ of initial balance
        portfolio_value = self._get_portfolio_value()
        portfolio_ratio = portfolio_value / self.initial_balance
        portfolio_bucket = min(int(portfolio_ratio * 5), 4)  # 5 buckets
        state.append(portfolio_bucket)
        
        # Add position information for each stock
        for symbol in self.symbols:
            # Position size (discretized into 5 buckets)
            # Bucket 0: Position < 20% of portfolio value
            # Bucket 1: Position 20-40% of portfolio value
            # Bucket 2: Position 40-60% of portfolio value
            # Bucket 3: Position 60-80% of portfolio value
            # Bucket 4: Position 80-100% of portfolio value
            position_value = self.positions[symbol] * self.prices[symbol][self.current_step]
            position_ratio = position_value / portfolio_value
            position_bucket = min(int(position_ratio * 5), 4)  # 5 buckets
            state.append(position_bucket)
            
            # Calculate price momentum using 3-day moving average
            if self.current_step >= 3:
                # Sum returns over last 3 days to get momentum
                momentum = sum(self.returns[symbol][self.current_step-i] for i in range(1, 4))
                # Convert momentum to discrete bucket (0-9)
                #   Bucket 0: Strong negative momentum (< -0.1)
                #   Bucket 1-4: Moderate to slight negative momentum (-0.1 to 0)
                #   Bucket 5: Neutral momentum (0)
                #   Bucket 6-9: Slight to strong positive momentum (0 to 0.1)
                # Add 0.1 offset to center buckets around 0
                # Multiply by 10 to spread values across buckets
                momentum_bucket = min(int((momentum + 0.1) * 10), 9)
            else:
                # Default to neutral bucket (5) if not enough history
                momentum_bucket = 5
            state.append(momentum_bucket)
        
        return np.array(state)
    
    def step(self, actions: List[Tuple[int, int]]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        actions: List of (action_type, quantity_level) for each stock
        action_type: 0=hold, 1=buy, 2=sell
        quantity_level: 
            For buy: 1=15%, 2=25%, 3=35%, 4=45%, 5=50% of max position value
            For sell: 1=25%, 2=35%, 3=45%, 4=55%, 5=100% of current position
        """
        if self.current_step >= len(self.prices[self.symbols[0]]) - 1:
            return self._get_state(), 0, True, {'portfolio_value': self._get_portfolio_value()}
        
        old_portfolio_value = self._get_portfolio_value()
        
        # Execute trades
        for symbol, (action, qty_level) in zip(self.symbols, actions):
            price = self.prices[symbol][self.current_step]
            
            if action == 1:  # Buy
                # Calculate maximum position value allowed
                max_position_value = self.max_position * old_portfolio_value
                current_position_value = self.positions[symbol] * price
                available_position_value = max_position_value - current_position_value
                
                # Calculate buy amount based on quantity level
                if qty_level == 1:
                    buy_value = available_position_value * 0.15
                elif qty_level == 2:
                    buy_value = available_position_value * 0.25
                elif qty_level == 3:
                    buy_value = available_position_value * 0.35
                elif qty_level == 4:
                    buy_value = available_position_value * 0.45
                else:  # qty_level == 5
                    buy_value = available_position_value * 0.50
                
                # Ensure we don't exceed available cash
                buy_value = min(buy_value, self.balance)
                
                # Calculate shares with minimum position size check
                min_position_value = 100  # Minimum $100 position
                if buy_value >= min_position_value:
                    shares_to_buy = int(buy_value / price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price
                        self.positions[symbol] += shares_to_buy
                        self.balance -= cost
                    
            elif action == 2:  # Sell
                current_shares = self.positions[symbol]
                if current_shares > 0:
                    # Calculate sell amount based on quantity level
                    if qty_level == 1:
                        shares_to_sell = int(current_shares * 0.25)
                    elif qty_level == 2:
                        shares_to_sell = int(current_shares * 0.35)
                    elif qty_level == 3:
                        shares_to_sell = int(current_shares * 0.45)
                    elif qty_level == 4:
                        shares_to_sell = int(current_shares * 0.55)
                    else:  # qty_level == 5
                        shares_to_sell = current_shares
                    
                    # Only execute if position size is meaningful
                    if shares_to_sell > 0:
                        revenue = shares_to_sell * price
                        self.positions[symbol] -= shares_to_sell
                        self.balance += revenue
            # Hold does nothing
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        new_portfolio_value = self._get_portfolio_value()
        value_change = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        
        # Penalty for holding too much cash (scaled by portfolio value)
        cash_ratio = self.balance / new_portfolio_value
        cash_penalty = -0.0001 if cash_ratio > 0.5 else 0
        
        # Penalty for too many transactions (scaled by portfolio value)
        transaction_penalty = -0.00001 * sum(1 for a, q in actions if a != 0)
        
        # Penalty for not diversifying (must have at least 2 stocks invested)
        invested_stocks = sum(1 for s in self.symbols if self.positions[s] > 0)
        diversification_penalty = -0.001 if invested_stocks < 2 else 0
        
        # Combine rewards and penalties
        reward = value_change + cash_penalty + transaction_penalty + diversification_penalty
        
        # Scale reward to be more meaningful
        reward = reward * 100  # Scale up to make rewards more significant
        
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
    """
    Q-Learning Agent for Stock Trading
    
    This class implements the Q-learning algorithm for stock trading.
    It maintains a Q-table that maps state-action pairs to their expected values.
    
    Key Features:
    ------------
    - Epsilon-greedy exploration strategy
    - Q-table for state-action value estimates
    - Learning rate and discount factor for value updates
    
    Parameters:
    ----------
    state_size: Dimension of state space
    action_size: Number of possible actions (3 action types × 5 quantity levels)
    learning_rate: Rate at which agent updates Q-values (0.1)
    discount_factor: Importance of future rewards (0.95)
    exploration_rate: Initial probability of random actions (1.0)
    exploration_decay: Rate at which exploration decreases (0.999)
    min_exploration_rate: Minimum exploration probability (0.01)
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 0.999,
                 min_exploration_rate: float = 0.01):
        """
        Initialize the Q-learning agent with specified parameters.
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
        action_type = action_idx // 5  # 0=hold, 1=buy, 2=sell
        quantity_level = (action_idx % 5) + 1  # 1=5%, 2=15%, 3=25%, 4=35%, 5=50%
        return (action_type, quantity_level)
    
    def update(self, state: np.ndarray, action: Tuple[int, int], reward: float, next_state: np.ndarray):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        # Map (action_type, quantity_level) to action_idx
        action_idx = action[0] * 5 + (action[1] - 1)
        old_value = self.q_table[state_key][action_idx]
        next_max = np.max(self.q_table[next_state_key])
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        self.q_table[state_key][action_idx] = new_value
        self.exploration_rate = max(self.min_exploration_rate, 
                                  self.exploration_rate * self.exploration_decay)

def train_agent(env: StockTradingEnv, agent: QLearningAgent, episodes: int = 1000):
    """
    Train the Q-learning agent on the trading environment.
    
    Args:
        env: StockTradingEnv instance
        agent: QLearningAgent instance
        episodes: Number of training episodes
    
    Returns:
        Tuple of (best_portfolio_value, training_return)
    """
    best_portfolio_value = env.initial_balance
    best_episode = 0
    episode_rewards = []
    episode_returns = []
    
    print("\nStarting Training...")
    print("====================")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get actions for each stock
            actions = [agent.get_action(state) for _ in range(len(env.stocks))]
            next_state, reward, done, info = env.step(actions)
            
            # Update Q-values for each action
            for action in actions:
                agent.update(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
        
        # Calculate episode metrics
        final_portfolio_value = info['portfolio_value']
        episode_return = (final_portfolio_value - env.initial_balance) / env.initial_balance
        episode_rewards.append(total_reward)
        episode_returns.append(episode_return)
        
        # Track best performance
        if final_portfolio_value > best_portfolio_value:
            best_portfolio_value = final_portfolio_value
            best_episode = episode
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_return = np.mean(episode_returns[-100:])
            print(f"\nEpisode {episode + 1}/{episodes}")
            print(f"  Average Reward: {avg_reward:.3f}")
            print(f"  Average Return: {avg_return:.3f}")
            print(f"  Final Portfolio Value: ${final_portfolio_value:,.2f}")
            print(f"  Exploration Rate: {agent.exploration_rate:.3f}")
    
    # Print training summary
    print("\nTraining Complete!")
    print("=================")
    print(f"Best Episode: {best_episode + 1}")
    print(f"Best Portfolio Value: ${best_portfolio_value:,.2f}")
    print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"Total Return: {episode_return*100:.2f}%")
    print(f"Average Return (last 100 episodes): {avg_return*100:.2f}%")
    
    return best_portfolio_value, episode_return

def test_agent(env: StockTradingEnv, agent: QLearningAgent, continue_learning: bool = True) -> Tuple[float, pd.DataFrame]:
    """
    Test the trained agent on new data.
    If continue_learning is True, the agent will continue to learn during testing.
    Returns the final portfolio value and position history.
    """
    state = env.reset()
    done = False
    
    # Initialize position history
    position_history = []
    dates = env.prices[env.symbols[0]].index
    
    while not done:
        actions = []
        for _ in range(len(env.stocks)):
            state_key = agent._get_state_key(state)
            if state_key in agent.q_table:
                action_idx = np.argmax(agent.q_table[state_key])
            else:
                action_idx = 0  # default to hold
            action_type = action_idx // 5
            quantity_level = (action_idx % 5) + 1
            actions.append((action_type, quantity_level))
        
        # Record positions before step
        current_positions = {
            'date': dates[env.current_step],
            'portfolio_value': env._get_portfolio_value(),
            'cash': env.balance
        }
        for symbol in env.symbols:
            current_positions[symbol] = env.positions[symbol]
        position_history.append(current_positions)
        
        next_state, reward, done, info = env.step(actions)
        
        # Update Q-values if continuing to learn
        if continue_learning:
            for action in actions:
                agent.update(state, action, reward, next_state)
        
        state = next_state
    
    # Convert position history to DataFrame
    position_df = pd.DataFrame(position_history)
    position_df.set_index('date', inplace=True)
    
    final_value = info['portfolio_value']
    return final_value, position_df

def calculate_equal_weighted_return(env: StockTradingEnv) -> Tuple[float, float]:
    """
    Calculate the return of an equal-weighted portfolio of all stocks.
    Returns (final_value, total_return)
    """
    initial_balance = env.initial_balance
    balance_per_stock = initial_balance / len(env.symbols)
    
    # Calculate initial shares for each stock
    initial_shares = {}
    for symbol in env.symbols:
        initial_price = env.prices[symbol][0]
        initial_shares[symbol] = int(balance_per_stock / initial_price)
    
    # Calculate final value
    final_value = 0
    for symbol in env.symbols:
        final_price = env.prices[symbol][-1]
        final_value += initial_shares[symbol] * final_price
    
    total_return = (final_value - initial_balance) / initial_balance
    return final_value, total_return

if __name__ == "__main__":
    """
    Main execution block for training and testing the Q-learning agent.
    
    This section:
    1. Sets up the training environment
    2. Trains the agent
    3. Tests the agent on new data
    4. Compares performance against S&P 500
    5. Saves results and metrics
    """
    
    # Create results directory
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Configuration
    stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    initial_balance = 10000
    max_position = 0.5
    
    # Training period
    train_start = datetime(2017, 1, 1)
    train_end = datetime(2019, 12, 31)
    
    print("\nInitializing Training Environment...")
    print("===================================")
    
    # Initialize environment
    env = StockTradingEnv(
        stocks=stocks,
        initial_balance=initial_balance,
        max_position=max_position
    )
    
    # Load training data
    env.symbols, env.data = get_stock_data(
        stocks=stocks,
        start_date=train_start,
        end_date=train_end
    )
    env.prices = {symbol: env.data[symbol]['data']['Close'] for symbol in env.symbols}
    env.returns = {symbol: env.prices[symbol].pct_change().fillna(0) for symbol in env.symbols}
    
    # Initialize agent
    state_size = len(env._get_state())
    action_size = 15  # 3 action types × 5 quantity levels
    
    agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.9995,
        min_exploration_rate=0.05
    )
    
    # Train agent
    best_training_value, training_return = train_agent(env, agent, episodes=1000)
    
    # Save trained Q-table
    with open(os.path.join(results_dir, "q_table.pkl"), "wb") as f:
        pickle.dump(agent.q_table, f)
    
    # Testing period
    test_start = datetime(2019, 1, 1)
    test_end = datetime(2019, 12, 31)
    
    print("\nInitializing Test Environment...")
    print("===============================")
    
    # Initialize test environment
    test_env = StockTradingEnv(
        stocks=stocks,
        initial_balance=initial_balance,
        max_position=max_position
    )
    
    # Load test data
    test_env.symbols, test_env.data = get_stock_data(
        stocks=stocks,
        start_date=test_start,
        end_date=test_end
    )
    test_env.prices = {symbol: test_env.data[symbol]['data']['Close'] for symbol in test_env.symbols}
    test_env.returns = {symbol: test_env.prices[symbol].pct_change().fillna(0) for symbol in test_env.symbols}
    
    # Load trained Q-table
    with open(os.path.join(results_dir, "q_table.pkl"), "rb") as f:
        agent.q_table = pickle.load(f)
    
    # Test agent
    final_value, position_history = test_agent(test_env, agent, continue_learning=False)
    agent_return = (final_value - initial_balance) / initial_balance
    
    # Save position history
    position_history.to_csv(os.path.join(results_dir, "position_history.csv"))
    
    # Get S&P 500 benchmark
    sp500_return, sp500_start, sp500_end = get_sp500_return(test_start, test_end)
    
    # Create and save results summary
    results = {
        'Metric': [
            'Initial Portfolio Value',
            'Final Portfolio Value',
            'Total Return',
            'S&P 500 Start Price',
            'S&P 500 End Price',
            'S&P 500 Return',
            'Outperformance vs S&P 500'
        ],
        'Value': [
            f'${initial_balance:,.2f}',
            f'${final_value:,.2f}',
            f'{agent_return*100:.2f}%',
            f'${sp500_start:,.2f}',
            f'${sp500_end:,.2f}',
            f'{sp500_return*100:.2f}%',
            f'{(agent_return - sp500_return)*100:.2f}%'
        ]
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, "results_summary.csv"), index=False)
    
    # Print final results
    print("\nTest Results Summary")
    print("===================")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {agent_return*100:.2f}%")
    print(f"\nS&P 500 Benchmark")
    print(f"Start Price: ${sp500_start:,.2f}")
    print(f"End Price:   ${sp500_end:,.2f}")
    print(f"Total Return: {sp500_return*100:.2f}%")
    print(f"\nOutperformance vs S&P 500: {(agent_return - sp500_return)*100:.2f}%")
    
    if agent_return > sp500_return:
        print("\nAgent outperformed the S&P 500! 🎉")
    else:
        print("\nAgent underperformed the S&P 500. 📉")
