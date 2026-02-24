"""
Strategy base classes and built-in strategies.

To create your own strategy:
1. Create a new class that inherits from Strategy
2. Implement add_indicators() to calculate your technical indicators
3. Implement generate_signals() to generate buy/sell signals

Required output columns from generate_signals():
    - signal: 1 for buy, -1 for sell, 0 for hold
    - target_qty: position size (shares for stocks, USD for crypto)
    - position: current position state (1=long, -1=short, 0=flat)

Optional output columns:
    - limit_price: if set, places a limit order instead of market

Example:
    class MyStrategy(Strategy):
        def __init__(self, lookback=20, position_size=10.0):
            self.lookback = lookback
            self.position_size = position_size

        def add_indicators(self, df):
            df['sma'] = df['Close'].rolling(self.lookback).mean()
            return df

        def generate_signals(self, df):
            df['signal'] = 0
            df.loc[df['Close'] > df['sma'], 'signal'] = 1
            df.loc[df['Close'] < df['sma'], 'signal'] = -1
            df['position'] = df['signal']
            df['target_qty'] = self.position_size
            return df
"""

import numpy as np
import pandas as pd


class Strategy:
    """
    Base Strategy interface for adding indicators and generating trading signals.

    All strategies must implement:
        - add_indicators(df): Add technical indicators to the DataFrame
        - generate_signals(df): Generate trading signals

    The DataFrame must contain these columns:
        - Datetime, Open, High, Low, Close, Volume (input)
        - signal, target_qty, position (output from generate_signals)
    """

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Add technical indicators to the DataFrame. Override this method."""
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - interface
        """Generate trading signals. Override this method."""
        raise NotImplementedError

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full strategy pipeline. Do not override."""
        df = df.copy()
        df = self.add_indicators(df)
        df = self.generate_signals(df)
        return df


class MovingAverageStrategy(Strategy):
    """
    Moving average crossover strategy with explicitly defined entry/exit rules.
    """

    def __init__(self, short_window: int = 20, long_window: int = 60, position_size: float = 10.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["MA_short"] = df["Close"].rolling(self.short_window, min_periods=1).mean()
        df["MA_long"] = df["Close"].rolling(self.long_window, min_periods=1).mean()
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["volatility"] = df["returns"].rolling(self.long_window).std().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = (df["MA_short"].shift(1) <= df["MA_long"].shift(1)) & (df["MA_short"] > df["MA_long"])
        sell = (df["MA_short"].shift(1) >= df["MA_long"].shift(1)) & (df["MA_short"] < df["MA_long"])

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = 0
        df.loc[df["MA_short"] > df["MA_long"], "position"] = 1
        df.loc[df["MA_short"] < df["MA_long"], "position"] = -1
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class TemplateStrategy(Strategy):
    """
    Starter strategy template for students. Modify the indicator and signal
    logic to build your own ideas.
    """

    def __init__(
        self,
        lookback: int = 14,
        position_size: float = 10.0,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
    ):
        if lookback < 1:
            raise ValueError("lookback must be at least 1.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.lookback = lookback
        self.position_size = position_size
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["momentum"] = df["Close"].pct_change(self.lookback).fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0

        buy = df["momentum"] > self.buy_threshold
        sell = df["momentum"] < self.sell_threshold

        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1

        df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)
        df["target_qty"] = df["position"].abs() * self.position_size
        return df


class CryptoTrendStrategy(Strategy):
    """
    Crypto trend-following strategy using fast/slow EMAs (long-only).
    """

    def __init__(self, short_window: int = 7, long_window: int = 21, position_size: float = 100.0):
        if short_window >= long_window:
            raise ValueError("short_window must be strictly less than long_window.")
        if position_size <= 0:
            raise ValueError("position_size must be positive.")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["EMA_fast"] = df["Close"].ewm(span=self.short_window, adjust=False).mean()
        df["EMA_slow"] = df["Close"].ewm(span=self.long_window, adjust=False).mean()
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        long_regime = df["EMA_fast"] > df["EMA_slow"]
        flips = long_regime.astype(int).diff().fillna(0)
        df.loc[flips > 0, "signal"] = 1
        df.loc[flips < 0, "signal"] = -1
        df["position"] = long_regime.astype(int)
        df["target_qty"] = self.position_size
        return df

class DemoStrategy(Strategy):
    """
    Simple demo strategy - buys 1 share when price up, sells 1 share when price down.
    Uses tiny position size to avoid margin/locate issues.

    Usage:
        python run_live.py --symbol AAPL --strategy demo --timeframe 1Min --sleep 5 --live
    """

    def __init__(self, position_size: float = 1.0):
        self.position_size = position_size

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["change"] = df["Close"].diff().fillna(0.0)
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        df.loc[df["change"] > 0, "signal"] = 1   # Price went up -> buy
        df.loc[df["change"] < 0, "signal"] = -1  # Price went down -> sell
        df["position"] = df["signal"]
        df["target_qty"] = self.position_size
        return df


## =============================================================================
## CREATE YOUR OWN STRATEGIES BELOW
## =============================================================================
##
## Example: RSI Strategy
##
## class RSIStrategy(Strategy):
##     """Buy when RSI is oversold, sell when overbought."""
##
##     def __init__(self, period=14, oversold=30, overbought=70, position_size=10.0):
##         self.period = period
##         self.oversold = oversold
##         self.overbought = overbought
##         self.position_size = position_size
##
##     def add_indicators(self, df):
##         delta = df['Close'].diff()
##         gain = delta.where(delta > 0, 0).rolling(self.period).mean()
##         loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
##         rs = gain / loss
##         df['RSI'] = 100 - (100 / (1 + rs))
##         return df
##
##     def generate_signals(self, df):
##         df['signal'] = 0
##         df.loc[df['RSI'] < self.oversold, 'signal'] = 1   # Buy when oversold
##         df.loc[df['RSI'] > self.overbought, 'signal'] = -1  # Sell when overbought
##         df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
##         df['target_qty'] = self.position_size
##         return df
##
## To use your strategy:
##   python run_live.py --symbol AAPL --strategy mystrategy --live
##



class MyStrategy(Strategy):
    """
    Pairs mean-reversion strategy for PM (A) vs MNST (B), matching the slide:

      - Positioning: Long A (PM) / Short B (MNST) only
      - Sizing: beta-balanced notionals (defaults ~ $61k long / $39k short for $100k gross)
      - Signal: z-score of spread = log(A) - log(B)

    Expected df columns:
      - Close_A : price of asset A (PM)
      - Close_B : price of asset B (MNST)

    Required outputs:
      - signal:   1 enter (or add) longA/shortB, -1 exit to flat, 0 hold
      - position: 1 = longA/shortB, 0 = flat
      - target_qty: shares of A to hold when in position (used by your executor)

    Extra outputs (recommended for pairs execution):
      - target_qty_b: shares of B to hold (negative when short)
      - target_notional_a, target_notional_b: dollar notionals per leg (positive numbers)
      - z, spread
    """

    def __init__(
        self,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        # Interpreting "position_size" as GROSS NOTIONAL in dollars (slide uses ~$100k gross).
        position_size: float = 100_000.0,
        # Market betas from slide (tune as needed):
        beta_a: float = 0.38,   # PM ~0.3/0.4
        beta_b: float = 0.60,   # MNST ~0.6
        # If you want to force EXACT slide notionals, set these:
        notional_a: float | None = None,  # e.g. 61_000
        notional_b: float | None = None,  # e.g. 39_000
        use_log: bool = True,
    ):
        if lookback < 5:
            raise ValueError("lookback too small; use >= 5.")
        if entry_z <= 0 or exit_z < 0:
            raise ValueError("entry_z must be >0 and exit_z must be >=0.")
        if exit_z >= entry_z:
            raise ValueError("exit_z should be < entry_z (hysteresis avoids churn).")
        if position_size <= 0:
            raise ValueError("position_size (gross notional) must be positive.")
        if beta_a <= 0 or beta_b <= 0:
            raise ValueError("beta_a and beta_b must be positive.")

        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.gross_notional = float(position_size)
        self.beta_a = float(beta_a)
        self.beta_b = float(beta_b)
        self.notional_a_fixed = None if notional_a is None else float(notional_a)
        self.notional_b_fixed = None if notional_b is None else float(notional_b)
        self.use_log = use_log

    def _compute_leg_notionals(self) -> tuple[float, float]:
        # If user pins exact slide sizing, use it.
        if self.notional_a_fixed is not None and self.notional_b_fixed is not None:
            na = self.notional_a_fixed
            nb = self.notional_b_fixed
            if na <= 0 or nb <= 0:
                raise ValueError("notional_a and notional_b must be positive if provided.")
            return na, nb

        # Otherwise beta-balance gross notional so beta_a * na ≈ beta_b * nb
        # with na + nb = gross
        na = self.gross_notional * (self.beta_b / (self.beta_a + self.beta_b))
        nb = self.gross_notional - na
        return float(na), float(nb)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "Close_A" not in df.columns or "Close_B" not in df.columns:
            raise ValueError("df must contain Close_A and Close_B.")

        px_a = df["Close_A"].astype(float)
        px_b = df["Close_B"].astype(float)

        # Spread series (log price ratio is common for pairs)
        if self.use_log:
            spread = np.where((px_a > 0) & (px_b > 0), np.log(px_a) - np.log(px_b), np.nan)
            df["spread"] = spread
        else:
            df["spread"] = np.where(px_b != 0, px_a / px_b, np.nan)

        # Rolling z-score of spread
        df["spread_mean"] = df["spread"].rolling(self.lookback, min_periods=self.lookback).mean()
        df["spread_std"] = df["spread"].rolling(self.lookback, min_periods=self.lookback).std()
        df["spread_std"] = df["spread_std"].replace(0, np.nan)

        df["z"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]
        df["z"] = df["z"].replace([np.inf, -np.inf], np.nan)

        # Precompute target sizing each bar (shares), matching slide notionals
        not_a, not_b = self._compute_leg_notionals()
        df["target_notional_a"] = not_a
        df["target_notional_b"] = not_b

        # shares = notional / price
        df["target_shares_a"] = np.where(px_a > 0, not_a / px_a, np.nan)
        df["target_shares_b"] = np.where(px_b > 0, not_b / px_b, np.nan)

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df["position"] = 0

        z = df["z"]

        pos = 0
        positions = []
        signals = []

        for i in range(len(df)):
            zi = z.iat[i]

            if np.isnan(zi):
                signals.append(0)
                positions.append(pos)
                continue

            sig = 0

            if pos == 0:
                # Slide positioning: enter ONLY when PM is "cheap" vs MNST (spread low)
                if zi <= -self.entry_z:
                    pos = 1
                    sig = 1
            else:
                # Exit when mean reversion happens
                if abs(zi) <= self.exit_z:
                    pos = 0
                    sig = -1  # exit

            signals.append(sig)
            positions.append(pos)

        df["signal"] = signals
        df["position"] = positions

        # Required output: target_qty.
        # Here: target_qty = shares of A (PM) to hold when position=1.
        df["target_qty"] = df["position"] * df["target_shares_a"]

        # Extra: shares for B (MNST). Negative means short.
        df["target_qty_b"] = -df["position"] * df["target_shares_b"]

        return df