import gym
import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from gym import spaces
import matplotlib.pyplot as plt
from ta.volatility import BollingerBands
from binance.client import Client


api_key = 'your api key here'
api_secret = 'your api secret here'

#getting the data from Binance but anything could be used to get the data
client = Client(api_key, api_secret)
df = client.get_historical_klines("FETUSDT", Client.KLINE_INTERVAL_1HOUR, "01 january 2020")

df = pd.DataFrame(df, columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
df = df.iloc[:, :6]
df = df.set_index('Time')
df.index = pd.to_datetime(df.index, unit = 'ms')
df = df.astype(float)
df = add_indicators(df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to calculate technical indicators
def add_indicators(df):
    # Calculate Relative Strength Index (RSI)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # Calculate MACD and its components
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # Calculate Bollinger Bands
    indicator_bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    # Calculate moving averages
    df['MA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    ema50 = ta.trend.EMAIndicator(df['Close'], window=50)
    df['EMA50'] = ema50.ema_indicator()
    
    # Calculate Stochastic Oscillator
    stochastic = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['stoch_k'] = stochastic.stoch()
    df['stoch_d'] = stochastic.stoch_signal()

    # Remove rows with missing values from indicator calculations
    df = df.iloc[50:]
    return df

# Trading environment class
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)  
        # Observation space: 12 technical indicators
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(12,), dtype=np.float32)
        # Initialize trading parameters
        self.balance = 10000
        self.position = 0  # 0: No position, 1: Long position
        self.net_worth = self.balance
        self.buy_price = 0

    def reset(self):
        # Reset environment to initial state
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.net_worth = self.balance
        self.buy_price = 0
        return self._next_observation()

    def _next_observation(self):
        # Create feature vector for current time step
        obs = np.array([
            self.df.iloc[self.current_step]['Close'],        # Closing price
            self.df.iloc[self.current_step]['Volume'],       # Trading volume
            self.df.iloc[self.current_step]['RSI'],          # Relative Strength Index
            self.df.iloc[self.current_step]['MACD'],         # MACD line
            self.df.iloc[self.current_step]['MACD_signal'],  # MACD signal line
            self.df.iloc[self.current_step]['MACD_diff'],    # MACD histogram
            self.df.iloc[self.current_step]['bb_bbm'],       # Bollinger Middle Band
            self.df.iloc[self.current_step]['bb_bbh'] - self.df.iloc[self.current_step]['Close'],  # Distance to upper band
            self.df.iloc[self.current_step]['bb_bbl'] - self.df.iloc[self.current_step]['Close'],  # Distance to lower band
            self.df.iloc[self.current_step]['MA20'],         # 20-day moving average
            self.df.iloc[self.current_step]['EMA50'],        # 50-day exponential moving average
            self.df.iloc[self.current_step]['stoch_k'],      # Stochastic %K
        ])
        return obs

    def step(self, action):
        done = False
        reward = 0
        current_price = self.df.iloc[self.current_step]['Close']

        # Execute trading action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.buy_price = current_price

        elif action == 2 and self.position == 1:  # Sell
            # Calculate profit (assuming position size of 100 shares)
            profit = (current_price - self.buy_price) * 100
            self.balance += profit
            reward = profit  # Immediate reward is the profit
            self.position = 0

        # Move to next time step
        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True

        # Update net worth
        self.net_worth = self.balance
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape)
        return obs, reward, done, {}

# Neural network to approximate Q-values
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        # Neural network architecture: 3 fully connected layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for each action

# DQN Agent class
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        # Hyperparameters
        self.gamma = 0.99    # Discount factor
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.lr = 0.001      # Learning rate
        self.batch_size = 64
        # Initialize networks
        self.model = DQNetwork(state_dim, action_dim).to(device)
        self.target_model = DQNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)  # Random action
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.cpu().numpy())  # Greedy action

    def replay(self):
        # Experience replay to train the network
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Calculate current Q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Calculate loss and update network
        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        # Update target network with current network weights
        self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self, filepath):
        # Save model weights
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        # Load model weights
        self.model.load_state_dict(torch.load(filepath))

# Initialize the environment and DQN agent
env = TradingEnv(df)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

print("Running: ")

# Agent training parameters
n_episodes = 1000
target_update_interval = 10  # Update target network every 10 episodes
save_interval = 100           # Save model every 100 episodes

# Training loop
for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward

    # Periodically update target network
    if episode % target_update_interval == 0:
        agent.update_target_model()

    # Save model periodically
    if episode % save_interval == 0:
        agent.save_model(f"dqn_trading_model_episode_{episode}.pth")
        print(f"Model saved at episode {episode}")

    print(f"Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}")

# Save final model after training
agent.save_model("dqn_trading_model_final.pth")
print("Final model saved.")
