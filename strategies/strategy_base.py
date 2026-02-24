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
    Pair mean-reversion strategy using rolling OLS hedge ratio and spread z-score.

    DataFrame requirements (must include both series):
        - Close_PM
        - Close_MNST

    Outputs:
        - signal: target regime (+1 long spread, -1 short spread, 0 flat)
        - position: same as signal, but held statefully with exits/stops
        - target_qty_PM / target_qty_MNST: target shares for each leg
        - target_qty: kept for compatibility (gross shares of PM leg)
    """

    def __init__(
        self,
        col_a: str = "Close_PM",
        col_b: str = "Close_MNST",
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        gross_notional: float = 10_000.0,   # gross dollars across both legs
        min_periods: int | None = None,
    ):
        if lookback < 5:
            raise ValueError("lookback should be at least ~5 (prefer 20-120).")
        if entry_z <= 0 or exit_z < 0 or stop_z <= entry_z:
            raise ValueError("Require: entry_z > 0, exit_z >= 0, stop_z > entry_z.")
        if gross_notional <= 0:
            raise ValueError("gross_notional must be positive.")
        self.col_a = col_a
        self.col_b = col_b
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.gross_notional = gross_notional
        self.min_periods = min_periods if min_periods is not None else lookback

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.col_a not in df.columns or self.col_b not in df.columns:
            raise ValueError(f"DataFrame must contain {self.col_a} and {self.col_b}")

        y = df[self.col_a].astype(float)  # PM
        x = df[self.col_b].astype(float)  # MNST

        # Rolling OLS of y on x using covariance formulas:
        # beta = cov(x,y) / var(x), alpha = mean(y) - beta*mean(x)
        mean_x = x.rolling(self.lookback, min_periods=self.min_periods).mean()
        mean_y = y.rolling(self.lookback, min_periods=self.min_periods).mean()
        mean_x2 = (x * x).rolling(self.lookback, min_periods=self.min_periods).mean()
        mean_xy = (x * y).rolling(self.lookback, min_periods=self.min_periods).mean()

        var_x = mean_x2 - mean_x * mean_x
        cov_xy = mean_xy - mean_x * mean_y

        beta = cov_xy / var_x.replace(0.0, np.nan)
        alpha = mean_y - beta * mean_x

        spread = y - (alpha + beta * x)

        spread_mu = spread.rolling(self.lookback, min_periods=self.min_periods).mean()
        spread_sigma = spread.rolling(self.lookback, min_periods=self.min_periods).std(ddof=0)

        z = (spread - spread_mu) / spread_sigma.replace(0.0, np.nan)

        df["hedge_beta"] = beta
        df["hedge_alpha"] = alpha
        df["spread"] = spread
        df["zscore"] = z

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        z = df["zscore"].to_numpy()
        beta = df["hedge_beta"].to_numpy()
        pa = df[self.col_a].to_numpy()  # PM price
        pb = df[self.col_b].to_numpy()  # MNST price

        n = len(df)
        pos = np.zeros(n, dtype=int)

        # Stateful position logic: enter at +/- entry_z, exit at |z|<exit_z, stop at |z|>stop_z
        current = 0
        for i in range(n):
            zi = z[i]
            if not np.isfinite(zi):
                pos[i] = current
                continue

            if current == 0:
                if zi <= -self.entry_z:
                    current = 1   # long PM, short MNST
                elif zi >= self.entry_z:
                    current = -1  # short PM, long MNST
            else:
                if abs(zi) <= self.exit_z:
                    current = 0
                elif abs(zi) >= self.stop_z:
                    current = 0

            pos[i] = current

        df["position"] = pos
        # signal as "target position" (common pattern in simple backtest engines)
        df["signal"] = df["position"]

        # Convert position into target share quantities for each leg.
        # Interpret beta as "MNST shares per 1 PM share" (from y ~ alpha + beta*x).
        qty_a = np.zeros(n, dtype=float)
        qty_b = np.zeros(n, dtype=float)

        for i in range(n):
            if pos[i] == 0:
                continue
            if not (np.isfinite(pa[i]) and np.isfinite(pb[i]) and pa[i] > 0 and pb[i] > 0):
                continue
            b = beta[i]
            if not np.isfinite(b):
                continue

            gross_per_unit = pa[i] + abs(b) * pb[i]
            if gross_per_unit <= 0:
                continue

            units = self.gross_notional / gross_per_unit  # "PM shares" scale
            qty_a[i] = pos[i] * units
            qty_b[i] = -pos[i] * b * units

        df["target_qty_PM"] = qty_a
        df["target_qty_MNST"] = qty_b

        # Compatibility fields expected by the framework (single qty column).
        df["target_qty"] = np.abs(df["target_qty_PM"])
        return df