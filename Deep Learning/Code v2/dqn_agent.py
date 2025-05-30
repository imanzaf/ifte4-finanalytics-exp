import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNetwork(nn.Module):
    def __init__(self, input_dim, num_stocks, num_actions_per_stock):
        super(DQNetwork, self).__init__()
        self.num_stocks = num_stocks
        self.num_actions_per_stock = num_actions_per_stock
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_stocks * num_actions_per_stock)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, num_stocks, num_actions_per_stock=15, lr=1e-3,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.98, epsilon_min=0.01, batch_size=32):
        self.state_dim = state_dim
        self.num_stocks = num_stocks
        self.num_actions_per_stock = num_actions_per_stock
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNetwork(state_dim, num_stocks, num_actions_per_stock).to(self.device)
        self.target_net = DQNetwork(state_dim, num_stocks, num_actions_per_stock).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return [random.randint(0, self.num_actions_per_stock - 1) for _ in range(self.num_stocks)]
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        q_values = q_values.view(-1, self.num_stocks, self.num_actions_per_stock)
        
        # Add this print to debug:
        print("Q-values per stock:")
        for i, stock_q in enumerate(q_values[0]):
            print(f"  Stock {i}: {stock_q.cpu().numpy()}")

        return q_values[0].argmax(dim=1).tolist()

    def store(self, state, action_vector, reward, next_state, done):
        self.replay_buffer.add(state, action_vector, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        batch_q_values = self.policy_net(states).view(-1, self.num_stocks, self.num_actions_per_stock)
        next_q_values = self.target_net(next_states).view(-1, self.num_stocks, self.num_actions_per_stock)

        actions = actions.to(self.device).unsqueeze(-1)
        current_q = batch_q_values.gather(2, actions).squeeze(-1)

        max_next_q = next_q_values.max(dim=2)[0]
        target_q = rewards.unsqueeze(1) + (1 - dones.float().unsqueeze(1)) * self.gamma * max_next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
