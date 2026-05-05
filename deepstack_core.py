import numpy as np
import pandas as pd
import yfinance as yf
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

# ─── 6-Coin Configuration ────────────────────────────────────────────────────
COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'BNB-USD', 'DOGE-USD']
COIN_LABELS = ['BTC', 'ETH', 'SOL', 'ADA', 'BNB', 'DOGE']

LOOKBACK_WINDOW = 20          # candles of history in each state
N_FEATURES      = 8           # OHLCV + RSI + MACD + BB
INITIAL_BALANCE = 10_000.0
TRANSACTION_FEE  = 0.001      # 0.1% per trade


def fetch_data(coins=COINS, period='1y', interval='1d'):
    """Download and align OHLCV data for all coins with retry on rate limit."""
    import time
    frames = {}
    for coin in coins:
        for attempt in range(5):
            try:
                df = yf.download(coin, period=period, interval=interval,
                                 progress=False, auto_adjust=True)
                if df.empty:
                    raise ValueError(f"Empty data for {coin}")
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                if len(df) < 50:
                    raise ValueError(f"Not enough rows: {len(df)}")
                frames[coin] = df
                print(f"  ✓ {coin}: {len(df)} rows")
                time.sleep(2)   # be polite to Yahoo
                break
            except Exception as e:
                wait = (attempt + 1) * 15
                print(f"  ✗ {coin} attempt {attempt+1}/5: {e}. Waiting {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to download {coin} after 5 attempts.")
    # Align on common dates
    common_idx = frames[coins[0]].index
    for coin in coins[1:]:
        common_idx = common_idx.intersection(frames[coin].index)
    print(f"  Common dates: {len(common_idx)} rows")
    for coin in coins:
        frames[coin] = frames[coin].loc[common_idx]
    return frames


def add_indicators(df):
    """Add RSI, MACD signal, and Bollinger Band width to a price DataFrame."""
    close = df['Close']

    # RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD signal line
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    df['MACD'] = macd - macd.ewm(span=9, adjust=False).mean()

    # Bollinger Band width (normalised)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_width'] = (2 * std20) / (sma20 + 1e-9)

    return df.dropna()


# ─── Trading Environment ──────────────────────────────────────────────────────
class MultiCoinTradingEnv(gym.Env):
    """
    A multi-asset RL trading environment for N cryptocurrencies.

    Observation : (LOOKBACK_WINDOW, N_coins * N_features) float32 array
    Action      : MultiDiscrete([3] * N_coins)  — 0=Hold, 1=Buy, 2=Sell per coin
    Reward      : Sharpe-adjusted portfolio return minus transaction costs
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, data_frames, coins=COINS):
        super().__init__()
        self.coins   = coins
        self.n_coins = len(coins)
        self.frames  = {c: add_indicators(data_frames[c].copy()) for c in coins}

        # Align lengths
        min_len = min(len(df) for df in self.frames.values())
        self.frames = {c: df.iloc[-min_len:].reset_index(drop=True)
                       for c, df in self.frames.items()}
        self.n_steps = min_len

        # Spaces
        obs_shape = (LOOKBACK_WINDOW, self.n_coins * N_FEATURES)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3] * self.n_coins)

        self.reset()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _get_prices(self, step):
        return np.array([self.frames[c]['Close'].iloc[step]
                         for c in self.coins])

    def _get_obs(self):
        obs = []
        for c in self.coins:
            df = self.frames[c]
            start = max(0, self.current_step - LOOKBACK_WINDOW)
            window = df.iloc[start: self.current_step]
            cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_width']
            feat = window[cols].values.astype(np.float32)

            # Pad with zeros if window is shorter than LOOKBACK_WINDOW
            if len(feat) < LOOKBACK_WINDOW:
                pad = np.zeros((LOOKBACK_WINDOW - len(feat), N_FEATURES), dtype=np.float32)
                feat = np.vstack([pad, feat])

            # Normalise each feature to [0,1] within the window
            feat_min = feat.min(axis=0, keepdims=True)
            feat_max = feat.max(axis=0, keepdims=True)
            feat = (feat - feat_min) / (feat_max - feat_min + 1e-9)
            obs.append(feat)
        # shape: (LOOKBACK_WINDOW, n_coins * N_FEATURES)
        return np.concatenate(obs, axis=1)

    def _portfolio_value(self, prices):
        return self.balance + np.dot(self.holdings, prices)

    # ── gym interface ─────────────────────────────────────────────────────────
    def reset(self):
        self.current_step = LOOKBACK_WINDOW
        self.balance   = INITIAL_BALANCE
        self.holdings  = np.zeros(self.n_coins)   # units held per coin
        self.prev_value = INITIAL_BALANCE
        self.portfolio_history = [INITIAL_BALANCE]
        self.trade_log = []
        return self._get_obs()

    def step(self, actions):
        prices = self._get_prices(self.current_step)
        prev_portfolio = self._portfolio_value(prices)

        for i, (action, price) in enumerate(zip(actions, prices)):
            coin = self.coins[i]

            if action == 1:   # Buy — spend 20% of current balance
                spend = self.balance * 0.20
                if spend > price:
                    units = spend / price
                    fee   = spend * TRANSACTION_FEE
                    self.balance  -= (spend + fee)
                    self.holdings[i] += units
                    self.trade_log.append(
                        (self.current_step, coin, 'BUY', price, units))

            elif action == 2:  # Sell — sell 50% of holdings
                units = self.holdings[i] * 0.50
                if units > 0:
                    proceeds = units * price
                    fee      = proceeds * TRANSACTION_FEE
                    self.balance     += (proceeds - fee)
                    self.holdings[i] -= units
                    self.trade_log.append(
                        (self.current_step, coin, 'SELL', price, units))

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        new_prices     = self._get_prices(self.current_step)
        new_portfolio  = self._portfolio_value(new_prices)
        self.portfolio_history.append(float(new_portfolio))

        # Reward: Sharpe-style — penalise volatility
        history = np.array(self.portfolio_history, dtype=np.float64)
        recent  = history[-30:] if len(history) > 30 else history
        returns = np.diff(recent)
        if len(returns) == 0:
            reward = 0.0
        else:
            sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252)
            reward = float(max(min(sharpe * 0.01, 1.0), -1.0))
        obs  = self._get_obs() if not done else np.zeros(
            self.observation_space.shape, dtype=np.float32)
        info = {
            'portfolio_value': new_portfolio,
            'balance': self.balance,
            'holdings': dict(zip(self.coins, self.holdings)),
            'step': self.current_step,
        }
        return obs, reward, done, info

    def render(self, mode='human'):
        prices = self._get_prices(self.current_step)
        pv     = self._portfolio_value(prices)
        print(f"Step {self.current_step:4d} | Portfolio: ${pv:,.2f} "
              f"| Cash: ${self.balance:,.2f}")
        for i, c in enumerate(self.coins):
            print(f"  {COIN_LABELS[i]:4s}: {self.holdings[i]:.4f} units "
                  f"@ ${prices[i]:,.4f}")
