"""
Offline backtest runner for a CSV file.

Usage:
    python run_backtest.py --csv data\\AAPL_1Min_stock_alpaca_clean.csv --strategy ma

Replace "AAPL_1Min_stock_alpaca_clean.csv" with your desired CSV file
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.backtester import Backtester, PerformanceAnalyzer, plot_equity
from core.gateway import MarketDataGateway
from core.matching_engine import MatchingEngine
from core.order_book import OrderBook
from core.order_manager import OrderLoggingGateway, OrderManager
from strategies import MovingAverageStrategy, TemplateStrategy, get_strategy_class


DATA_DIR = Path("data")


def create_sample_data(path: Path, periods: int = 200) -> None:
    df = pd.DataFrame(
        {
            "Datetime": pd.date_range(start="2024-01-01 09:30", periods=periods, freq="T"),
            "Open": np.random.uniform(100, 105, periods),
            "High": np.random.uniform(105, 110, periods),
            "Low": np.random.uniform(95, 100, periods),
            "Close": np.random.uniform(100, 110, periods),
            "Volume": np.random.randint(1_000, 5_000, periods),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline CSV backtest.")
    parser.add_argument("--csv", type=str, default="", help="Path to a CSV with OHLCV data.")
    parser.add_argument("--strategy", default="ma", help="Strategy name (ma, template, or a class name).")
    parser.add_argument("--short-window", type=int, default=20, help="Short MA window (MA strategy).")
    parser.add_argument("--long-window", type=int, default=60, help="Long MA window (MA strategy).")
    parser.add_argument("--position-size", type=float, default=10.0, help="Per-trade position size.")
    parser.add_argument("--momentum-lookback", type=int, default=14, help="Momentum lookback (template).")
    parser.add_argument("--buy-threshold", type=float, default=0.01, help="Buy threshold (template).")
    parser.add_argument("--sell-threshold", type=float, default=-0.01, help="Sell threshold (template).")
    parser.add_argument("--capital", type=float, default=50_000, help="Initial capital.")
    parser.add_argument("--plot", action="store_true", help="Plot equity curve at the end.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv) if args.csv else DATA_DIR / "sample_system_test_data.csv"
    if not csv_path.exists():
        if args.csv:
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        create_sample_data(csv_path)
        print(f"Sample data generated at {csv_path}.")

    strategy_cls = get_strategy_class(args.strategy)
    if strategy_cls is MovingAverageStrategy:
        strategy = MovingAverageStrategy(
            short_window=args.short_window,
            long_window=args.long_window,
            position_size=args.position_size,
        )
    elif strategy_cls is TemplateStrategy:
        strategy = TemplateStrategy(
            lookback=args.momentum_lookback,
            position_size=args.position_size,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
        )
    else:
        try:
            strategy = strategy_cls()
        except TypeError as exc:
            raise SystemExit(
                f"{strategy_cls.__name__} must support a no-arg constructor or use --strategy template."
            ) from exc

    gateway = MarketDataGateway(csv_path)
    order_book = OrderBook()
    order_manager = OrderManager(capital=args.capital, max_long_position=1_000, max_short_position=1_000)
    matching_engine = MatchingEngine()
    logger = OrderLoggingGateway()

    backtester = Backtester(
        data_gateway=gateway,
        strategy=strategy,
        order_manager=order_manager,
        order_book=order_book,
        matching_engine=matching_engine,
        logger=logger,
        default_position_size=int(max(1, args.position_size)),
    )

    equity_df = backtester.run()
    analyzer = PerformanceAnalyzer(equity_df["equity"].tolist(), backtester.trades)

    print("\n=== Backtest Summary ===")
    print(f"Equity data points: {len(equity_df)}")
    print(f"Trades executed: {sum(1 for t in backtester.trades if t.qty > 0)}")
    print(f"Final portfolio value: {equity_df.iloc[-1]['equity']:.2f}")
    print(f"PnL: {analyzer.pnl():.2f}")
    print(f"Sharpe: {analyzer.sharpe():.2f}")
    print(f"Max Drawdown: {analyzer.max_drawdown():.4f}")
    print(f"Win Rate: {analyzer.win_rate():.2%}")

    if args.plot:
        plot_equity(equity_df)


if __name__ == "__main__":
    main()
