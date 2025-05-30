import numpy as np
import pandas as pd
import os

class MultiStockEnv:
    def __init__(self, symbols,
                 data_dir="stock_data/train",
                 initial_investment=10000,
                 max_position_pct=0.5,
                 reward_type='raw',
                 normalize=True):
        self.symbols = symbols
        self.data_dir = data_dir
        self.num_stocks = len(symbols)
        self.initial_investment = initial_investment
        self.max_position_pct = max_position_pct
        self.reward_type = reward_type
        self.normalize = normalize

        self.buy_qty_map = {1: 0.15, 2: 0.25, 3: 0.35, 4: 0.45, 5: 0.50}
        self.sell_qty_map = {1: 0.25, 2: 0.35, 3: 0.45, 4: 0.55, 5: 1.0}

        self.action_dim_per_stock = 15  # 3 actions x 5 quantity levels
        self.total_action_dim = self.num_stocks * self.action_dim_per_stock

        # Load price data
        dfs = []
        for symbol in symbols:
            df = pd.read_csv(os.path.join(data_dir, f"{symbol}_prices.csv"), index_col='Date', parse_dates=True)
            dfs.append(df[['Close']].rename(columns={'Close': symbol}))
        combined = pd.concat(dfs, axis=1).dropna()
        self.dates = combined.index
        self.stock_prices = combined.values
        self.n_step = self.stock_prices.shape[0]
        self.max_price = np.max(self.stock_prices, axis=0)
        self.state_dim = self.num_stocks * 2 + 1

        self.reset()

    def reset(self):
        self.cur_step = 0
        self.stock_owned = np.zeros(self.num_stocks)
        self.cash = self.initial_investment
        self.reward_history = []
        return self._get_obs()

    def step(self, actions):
        # actions is a list of length num_stocks, each in range(15)
        prev_val = self._get_val()
        self._trade(actions)
        self.cur_step += 1
        done = self.cur_step >= self.n_step - 1
        cur_val = self._get_val()
        reward = self._calculate_reward(cur_val - prev_val)
        return self._get_obs(), reward, done, {'cur_val': cur_val}

    def _trade(self, actions):
        prices = self.stock_prices[self.cur_step]
        portfolio_value = self._get_val()

        for i, act in enumerate(actions):
            action_type = act // 5  # 0 = hold, 1 = buy, 2 = sell
            qty_level = (act % 5) + 1
            price = prices[i]

            if action_type == 1:  # Buy
                max_position_value = self.max_position_pct * portfolio_value
                current_position_value = self.stock_owned[i] * price
                available_value = max_position_value - current_position_value
                buy_value = available_value * self.buy_qty_map[qty_level]
                buy_value = min(buy_value, self.cash)

                if buy_value >= 100:
                    shares_to_buy = int(buy_value / price)
                    if shares_to_buy > 0:
                        self.stock_owned[i] += shares_to_buy
                        self.cash -= shares_to_buy * price

            elif action_type == 2:  # Sell
                current_shares = self.stock_owned[i]
                if current_shares > 0:
                    shares_to_sell = int(current_shares * self.sell_qty_map[qty_level])
                    if shares_to_sell > 0:
                        self.stock_owned[i] -= shares_to_sell
                        self.cash += shares_to_sell * price

    def _get_val(self):
        price = self.stock_prices[self.cur_step]
        return self.cash + np.sum(self.stock_owned * price)

    def _get_obs(self):
        price = self.stock_prices[self.cur_step]
        if self.normalize:
            norm_price = price / self.max_price
            norm_cash = self.cash / self.initial_investment
            return np.concatenate([self.stock_owned, norm_price, [norm_cash]])
        else:
            return np.concatenate([self.stock_owned, price, [self.cash]])

    def _calculate_reward(self, raw_reward):
        if self.reward_type == 'utility':
            return raw_reward if raw_reward > 0 else 2 * raw_reward
        elif self.reward_type == 'sharpe':
            self.reward_history.append(raw_reward)
            volatility = np.std(self.reward_history) if len(self.reward_history) > 1 else 1
            return raw_reward / (volatility + 1e-6)
        return raw_reward

    def get_portfolio_value(self):
        return self._get_val()