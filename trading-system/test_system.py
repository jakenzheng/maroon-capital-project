"""
Lightweight integration test for the trading system components.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from core.backtester import Backtester, PerformanceAnalyzer
from core.gateway import MarketDataGateway
from core.matching_engine import MatchingEngine
from core.order_book import OrderBook
from core.order_manager import OrderManager
from strategies import MovingAverageStrategy

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
    df.to_csv(path, index=False)


def main() -> None:
    sample_csv = DATA_DIR / "sample_system_test_data.csv"
    if not sample_csv.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        create_sample_data(sample_csv)
        print("Sample data generated.")

    gateway = MarketDataGateway(sample_csv)
    strategy = MovingAverageStrategy(short_window=5, long_window=12, position_size=10)
    order_book = OrderBook()
    order_manager = OrderManager(capital=50_000, max_long_position=400, max_short_position=400)
    matching_engine = MatchingEngine()

    backtester = Backtester(
        data_gateway=gateway,
        strategy=strategy,
        order_manager=order_manager,
        order_book=order_book,
        matching_engine=matching_engine,
        logger=None,
        default_position_size=10,
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


if __name__ == "__main__":
    main()
