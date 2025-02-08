# Deep Q-Learning Trading Bot with Binance Integration

A reinforcement learning-powered cryptocurrency trading system that leverages Deep Q-Networks (DQN) to learn optimal trading strategies using historical market data from Binance.

Key Features:
+ 🧠 Deep Reinforcement Learning - Implements DQN with experience replay and target network stabilization

+ 📈 Technical Analysis Integration - Utilizes 12 technical indicators (RSI, MACD, Bollinger Bands, etc.)

+ 💹 Binance API Integration - Fetches real historical market data (supports any Binance trading pair)

+ 🚀 GPU Acceleration - PyTorch implementation with CUDA support for faster training

+ 💰 Virtual Trading Environment - Custom Gym environment with balance tracking and position management

+ 🔄 Model Persistence - Automatic checkpoint saving and model reload capabilities

Technical Components:
1. Custom Trading Environment (TradingEnv)
   - Action space: Hold/Buy/Sell
   - Observation space: 12 normalized technical indicators
   - Profit-based reward system

2. Neural Network Architecture
      - 3-layer fully connected network (128-128 nodes)
     - Experience replay buffer (10,000 memory capacity)
     - Epsilon-greedy exploration strategy

3. Training Infrastructure
     - Periodic model checkpointing
     - Target network synchronization
     - Automatic GPU utilization


## Configure with your Binance API keys
'''
api_key = 'your_api_key_here'
api_secret = 'your_api_secret_here'
'''

## Train the model
'''
python trading_bot.py
'''
Requirements
Python 3.8+

Binance account (for API access)

PyTorch 1.12+

pandas, NumPy, TA-Lib

gym, matplotlib

Customization Options
Trading Pair: Modify FETUSDT to any Binance symbol

Timeframe: Adjust KLINE_INTERVAL_1HOUR to different intervals

Training Parameters:

Modify n_episodes for training duration

Adjust network architecture in DQNetwork

Tune hyperparameters (learning rate, gamma, epsilon decay)
