"""
Offline backtest runner for a CSV file.

Usage (single symbol):
    python run_backtest.py --csv data\\AAPL_1Min_stock_alpaca_clean.csv --strategy ma

Usage (pairs, MyStrategy):
    python run_backtest.py --csv data\\PM.csv --csv2 data\\MNST.csv --strategy mystrategy
    python run_backtest.py --csv data\\PM.csv,data\\MNST.csv --strategy mystrategy

Replace CSV paths with your desired files.
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
from strategies import MovingAverageStrategy, MyStrategy, TemplateStrategy, get_strategy_class


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
    parser.add_argument("--csv", type=str, default="", help="Path to CSV with OHLCV data, or 'A.csv,B.csv' for pairs.")
    parser.add_argument("--csv2", type=str, default="", help="Path to second CSV for pairs trading (leg B).")
    parser.add_argument("--strategy", default="ma", help="Strategy name (ma, template, mystrategy, or a class name).")
    parser.add_argument("--short-window", type=int, default=20, help="Short MA window (MA strategy).")
    parser.add_argument("--long-window", type=int, default=60, help="Long MA window (MA strategy).")
    parser.add_argument("--position-size", type=float, default=100000.0, help="Per-trade position size.")
    parser.add_argument("--momentum-lookback", type=int, default=14, help="Momentum lookback (template).")
    parser.add_argument("--buy-threshold", type=float, default=0.01, help="Buy threshold (template).")
    parser.add_argument("--sell-threshold", type=float, default=-0.01, help="Sell threshold (template).")
    parser.add_argument("--capital", type=float, default=100_000, help="Initial capital.")
    parser.add_argument("--plot", action="store_true", help="Plot equity curve at the end.")
    # Pairs-specific knobs
    parser.add_argument("--pairs-lookback", type=int, default=60, help="Lookback for pairs z-score (default: 60).")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Z-score entry threshold (default: 2.0).")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Z-score exit threshold (default: 0.5).")
    parser.add_argument("--fee-per-order", type=float, default=0.0, help="Flat fee per leg order in pairs backtest.")
    parser.add_argument("--save-data", action="store_true", help="Save pairs equity CSV to data/.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pairs helpers (ported from run_live.py)
# ---------------------------------------------------------------------------

def _resolve_pairs_csvs(csv_arg: str, csv2_arg: str) -> tuple[Path, Path] | None:
    """Return (path_a, path_b) if pairs mode is requested, else None."""
    if not csv_arg and not csv2_arg:
        return None

    if csv_arg and "," in csv_arg:
        parts = [p.strip() for p in csv_arg.split(",")]
        if len(parts) != 2:
            raise SystemExit("--csv with comma must be exactly 'A.csv,B.csv'")
        if csv2_arg:
            raise SystemExit("Use either --csv A.csv,B.csv OR --csv A.csv --csv2 B.csv (not both).")
        return Path(parts[0]), Path(parts[1])

    if csv_arg and csv2_arg:
        return Path(csv_arg), Path(csv2_arg)

    # Single CSV with no csv2 → not pairs mode, fall through to single-symbol path
    return None


def _read_leg_csv(path: Path, close_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"])
    if "Datetime" not in df.columns or "Close" not in df.columns:
        raise SystemExit(f"CSV {path} must include at least columns: Datetime, Close")
    return df[["Datetime", "Close"]].rename(columns={"Close": close_name}).dropna().sort_values("Datetime")


def _pairs_backtest(
    df: pd.DataFrame,
    strategy: MyStrategy,
    capital: float,
    qty_a_fixed: float,
    fee_per_order: float,
) -> pd.DataFrame:
    """
    Two-leg pairs backtest. Trades at bar close.
    Supports both explicit-leg output (target_qty / target_qty_b columns)
    and legacy output (position + beta).
    """
    out = strategy.run(df)

    cash = float(capital)
    sh_a = 0.0
    sh_b = 0.0
    last_equity = cash
    has_explicit_legs = "target_qty_b" in out.columns
    rows = []

    for i in range(len(out)):
        row = out.iloc[i]
        px_a = float(row["Close_A"])
        px_b = float(row["Close_B"])

        if has_explicit_legs:
            tgt_a = float(row.get("target_qty", 0.0)) if pd.notna(row.get("target_qty")) else 0.0
            tgt_b = float(row.get("target_qty_b", 0.0)) if pd.notna(row.get("target_qty_b")) else 0.0
            pos = 1 if (abs(tgt_a) > 1e-12 or abs(tgt_b) > 1e-12) else 0
            beta = float(row.get("beta", np.nan)) if pd.notna(row.get("beta", np.nan)) else np.nan
            qty_a = abs(tgt_a)
            qty_b = abs(tgt_b)
        else:
            pos = int(row.get("position", 0)) if pd.notna(row.get("position", 0)) else 0
            beta = row.get("beta", 1.0)
            beta = float(beta) if beta == beta else 1.0
            qty_a = float(qty_a_fixed)
            qty_b = abs(beta) * qty_a
            tgt_a = pos * qty_a
            tgt_b = -pos * qty_b

        d_a = tgt_a - sh_a
        d_b = tgt_b - sh_b
        orders = 0

        if abs(d_a) > 1e-12:
            cash -= d_a * px_a
            sh_a = tgt_a
            orders += 1
        if abs(d_b) > 1e-12:
            cash -= d_b * px_b
            sh_b = tgt_b
            orders += 1

        cash -= fee_per_order * orders
        equity = cash + sh_a * px_a + sh_b * px_b
        ret = 0.0 if i == 0 else (equity / last_equity - 1.0)
        last_equity = equity

        rows.append({
            "Datetime": row["Datetime"],
            "Close_A": px_a,
            "Close_B": px_b,
            "beta": beta,
            "z": float(row.get("z", np.nan)),
            "signal": int(row.get("signal", 0)) if pd.notna(row.get("signal", 0)) else 0,
            "position": pos,
            "qtyA": qty_a,
            "qtyB": qty_b,
            "shares_A": sh_a,
            "shares_B": sh_b,
            "cash": cash,
            "equity": equity,
            "ret": ret,
        })

    return pd.DataFrame(rows)


def _summarize_pairs(equity_df: pd.DataFrame) -> dict[str, float]:
    eq = equity_df["equity"].to_numpy(dtype=float)
    rets = equity_df["ret"].to_numpy(dtype=float)
    start = float(eq[0]) if len(eq) else 0.0
    end = float(eq[-1]) if len(eq) else 0.0

    sharpe = (
        float(np.nanmean(rets[1:]) / np.nanstd(rets[1:]))
        if rets.size > 1 and np.nanstd(rets[1:]) > 0
        else 0.0
    )

    peak, mdd = -np.inf, 0.0
    for v in eq:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, (v / peak) - 1.0)

    pos = equity_df["position"].to_numpy(dtype=int)
    trades = int(np.sum(pos[1:] != pos[:-1])) if pos.size > 1 else 0

    return {
        "start_equity": start,
        "end_equity": end,
        "net_pnl": end - start,
        "sharpe": sharpe,
        "max_drawdown": abs(mdd),
        "trades": float(trades),
    }


def _print_pairs_summary(s: dict[str, float], rows: int) -> None:
    print("")
    print("=" * 60)
    print("              PAIRS BACKTEST SUMMARY")
    print("=" * 60)
    print(f"  Rows:            {rows}")
    print(f"  Trades:          {int(s['trades'])}")
    print("-" * 60)
    print(f"  Start Equity:    ${s['start_equity']:,.2f}")
    print(f"  End Equity:      ${s['end_equity']:,.2f}")
    print(f"  Net P&L:         ${s['net_pnl']:+,.2f}")
    print("-" * 60)
    print(f"  Sharpe (bar):    {s['sharpe']:.3f}")
    print(f"  Max Drawdown:    {s['max_drawdown']:.2%}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # PAIRS MODE
    # ------------------------------------------------------------------
    pairs_csvs = _resolve_pairs_csvs(args.csv, args.csv2)
    if pairs_csvs is not None:
        if args.strategy != "mystrategy":
            raise SystemExit("Pairs backtest is only supported with --strategy mystrategy.")

        csv_a, csv_b = pairs_csvs
        for p in (csv_a, csv_b):
            if not p.exists():
                raise SystemExit(f"CSV not found: {p}")

        df_a = _read_leg_csv(csv_a, "Close_A")
        df_b = _read_leg_csv(csv_b, "Close_B")
        df = (
            df_a.merge(df_b, on="Datetime", how="inner")
            .dropna()
            .sort_values("Datetime")
            .reset_index(drop=True)
        )
        if df.empty:
            raise SystemExit("No overlapping Datetime rows between the two CSVs.")

        print(f"Pairs backtest: {csv_a.name} / {csv_b.name} | rows={len(df)}")

        strategy = MyStrategy(
            lookback=args.pairs_lookback,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            position_size=args.position_size,
        )

        equity_df = _pairs_backtest(
            df=df,
            strategy=strategy,
            capital=args.capital,
            qty_a_fixed=float(max(0.0, args.position_size)),
            fee_per_order=float(max(0.0, args.fee_per_order)),
        )

        s = _summarize_pairs(equity_df)
        _print_pairs_summary(s, len(equity_df))

        if args.save_data:
            out_path = DATA_DIR / f"backtest_pairs_{csv_a.stem}_{csv_b.stem}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            equity_df.to_csv(out_path, index=False)
            print(f"Saved equity CSV to: {out_path}")

        if args.plot:
            plot_equity(equity_df)

        return

    # ------------------------------------------------------------------
    # SINGLE-SYMBOL MODE (original behaviour)
    # ------------------------------------------------------------------
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