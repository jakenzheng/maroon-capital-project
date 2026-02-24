"""
Alpaca paper-trading runner.

Requires .env file with:
    ALPACA_API_KEY      (required)
    ALPACA_API_SECRET   (required)

Usage:
    # Single iteration
    python run_live.py --symbol AAPL --strategy ma

    # Continuous live trading
    python run_live.py --symbol AAPL --strategy ma --live

    # Dry run (no real orders)
    python run_live.py --symbol AAPL --strategy ma --dry-run

    # Crypto trading
    python run_live.py --symbol BTCUSD --asset-class crypto --strategy crypto_trend --live

Offline CSV backtest (pairs, MyStrategy):
    python run_live.py --strategy mystrategy --csv data/PM.csv --csv2 data/MNST.csv --timeframe 1D --lookback 200
    # or comma-separated:
    python run_live.py --strategy mystrategy --csv data/PM.csv,data/MNST.csv

Logs are saved to: logs/trades.csv, logs/signals.csv, logs/system.log
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.alpaca_trader import AlpacaTrader
from core.logger import get_logger, get_trade_logger
from pipeline.alpaca import clean_market_data, save_bars
from strategies import (
    MovingAverageStrategy,
    TemplateStrategy,
    CryptoTrendStrategy,
    DemoStrategy,
    MyStrategy,
    get_strategy_class,
    list_strategies,
)

logger = get_logger("run_live")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paper-trading loop with Alpaca.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available strategies: {', '.join(list_strategies())}

Examples:
  python run_live.py --symbol AAPL --strategy ma --live
  python run_live.py --symbol BTCUSD --asset-class crypto --strategy crypto_trend --live
  python run_live.py --symbol AAPL --strategy ma --dry-run --iterations 5

Offline CSV pairs backtest (MyStrategy):
  python run_live.py --strategy mystrategy --csv data/PM.csv --csv2 data/MNST.csv
  python run_live.py --strategy mystrategy --csv data/PM.csv,data/MNST.csv
        """,
    )
    parser.add_argument("--symbol", default="AAPL", help="Ticker or crypto symbol (default: AAPL)")
    parser.add_argument("--asset-class", choices=["stock", "crypto"], default="stock", help="Asset class (default: stock)")
    parser.add_argument("--timeframe", default="1Min", help="Alpaca timeframe: 1Min, 5Min, 15Min, 1H, 1D (default: 1Min)")
    parser.add_argument("--lookback", type=int, default=200, help="Bars to fetch each iteration (default: 200)")
    parser.add_argument("--strategy", default="ma", help="Strategy name (default: ma)")
    parser.add_argument("--short-window", type=int, default=20, help="Short MA window (default: 20)")
    parser.add_argument("--long-window", type=int, default=60, help="Long MA window (default: 60)")
    parser.add_argument("--position-size", type=float, default=10.0, help="Per-trade position size (default: 10.0)")
    parser.add_argument("--max-order-notional", type=float, default=None, help="Max notional per order (crypto only)")
    parser.add_argument("--momentum-lookback", type=int, default=14, help="Momentum lookback for template strategy (default: 14)")
    parser.add_argument("--buy-threshold", type=float, default=0.01, help="Buy threshold for template strategy (default: 0.01)")
    parser.add_argument("--sell-threshold", type=float, default=-0.01, help="Sell threshold for template strategy (default: -0.01)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of loops to run (default: 1)")
    parser.add_argument("--sleep", type=int, default=60, help="Seconds between loops (default: 60)")
    parser.add_argument("--live", action="store_true", help="Run continuously until Ctrl+C")
    parser.add_argument("--save-data", action="store_true", help="Save raw+clean CSVs to data/")
    parser.add_argument("--dry-run", action="store_true", help="Print decisions without placing orders")
    parser.add_argument("--feed", default=None, help="Data feed (iex or sip for stocks)")
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies and exit")

    # --- Offline CSV backtest knobs (pairs only, MyStrategy only) ---
    parser.add_argument("--csv", default=None, help="Offline backtest CSV path for leg A (or 'A.csv,B.csv').")
    parser.add_argument("--csv2", default=None, help="Offline backtest CSV path for leg B.")
    parser.add_argument("--capital", type=float, default=50_000.0, help="Initial capital for offline backtest.")
    parser.add_argument("--pairs-lookback", type=int, default=60, help="Lookback for pairs z-score (default: 60)")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Z-score threshold to enter pairs trade (default: 2.0)")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Z-score threshold to exit pairs trade (default: 0.5)")
    parser.add_argument("--fee-per-order", type=float, default=0.0, help="Flat fee per leg order in offline backtest.")

    return parser.parse_args()


def _read_leg_csv(path: Path, close_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"])
    if "Datetime" not in df.columns or "Close" not in df.columns:
        raise SystemExit(f"CSV {path} must include at least columns: Datetime, Close")
    df = df[["Datetime", "Close"]].rename(columns={"Close": close_name})
    df = df.dropna().sort_values("Datetime")
    return df


def _load_pairs_csvs(csv_arg: str | None, csv2_arg: str | None) -> tuple[Path, Path] | None:
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

    raise SystemExit("For offline pairs backtest, provide --csv A.csv --csv2 B.csv (or --csv A.csv,B.csv).")

def _pairs_backtest_mystrategy(
    df: pd.DataFrame,
    strategy: MyStrategy,
    capital: float,
    qty_a_fixed: float,   # legacy fallback only (shares of A)
    fee_per_order: float,
) -> pd.DataFrame:
    """
    Two-leg backtest:
      - Preferred: uses strategy outputs:
          target_qty   = signed shares of A (PM)
          target_qty_b = signed shares of B (MNST)
      - Fallback (legacy): uses (position, beta, qty_a_fixed) to size B as abs(beta)*qtyA.

    Trades at bar close (Close_A/Close_B). Tracks cash, holdings, equity, returns.
    """
    out = strategy.run(df)

    cash = float(capital)
    sh_a = 0.0
    sh_b = 0.0

    rows = []
    last_equity = cash

    has_explicit_legs = ("target_qty_b" in out.columns)

    for i in range(len(out)):
        row = out.iloc[i]
        px_a = float(row["Close_A"])
        px_b = float(row["Close_B"])

        # -------- target holdings --------
        if has_explicit_legs:
            # Strategy provides signed target shares for each leg
            tgt_a = row.get("target_qty", 0.0)
            tgt_b = row.get("target_qty_b", 0.0)

            tgt_a = float(tgt_a) if pd.notna(tgt_a) else 0.0
            tgt_b = float(tgt_b) if pd.notna(tgt_b) else 0.0

            # position for reporting only (1 = in trade, 0 = flat)
            pos = 1 if (abs(tgt_a) > 1e-12 or abs(tgt_b) > 1e-12) else 0
            beta = float(row.get("beta", np.nan)) if pd.notna(row.get("beta", np.nan)) else np.nan
            qty_a = abs(tgt_a)
            qty_b = abs(tgt_b)
        else:
            # Legacy sizing: position ∈ {-1,0,1}, hedge B by rolling beta
            pos = int(row.get("position", 0)) if pd.notna(row.get("position", 0)) else 0
            beta = row.get("beta", 1.0)
            beta = float(beta) if beta == beta else 1.0  # NaN -> 1.0
            qty_a = float(qty_a_fixed)
            qty_b = abs(beta) * qty_a

            tgt_a = pos * qty_a
            tgt_b = -pos * qty_b

        # -------- rebalance deltas --------
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

        rows.append(
            {
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
            }
        )

    return pd.DataFrame(rows)

def _summarize_equity(equity_df: pd.DataFrame) -> dict[str, float]:
    eq = equity_df["equity"].to_numpy(dtype=float)
    rets = equity_df["ret"].to_numpy(dtype=float)
    start = float(eq[0]) if len(eq) else 0.0
    end = float(eq[-1]) if len(eq) else 0.0
    pnl = end - start

    # Simple Sharpe using per-bar returns (no annualization here because timeframe unknown)
    if rets.size > 1 and np.nanstd(rets[1:]) > 0:
        sharpe = float(np.nanmean(rets[1:]) / np.nanstd(rets[1:]))
    else:
        sharpe = 0.0

    # Max drawdown
    peak = -np.inf
    mdd = 0.0
    for v in eq:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, (v / peak) - 1.0)
    mdd = abs(mdd)

    # Trade count = number of position changes (entry/exit/flip)
    pos = equity_df["position"].to_numpy(dtype=int)
    trades = int(np.sum(pos[1:] != pos[:-1])) if pos.size > 1 else 0

    return {
        "start_equity": start,
        "end_equity": end,
        "net_pnl": pnl,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "trades": float(trades),
    }


def main() -> None:
    args = parse_args()

    # Handle --list-strategies
    if args.list_strategies:
        print("Available strategies:")
        for name in list_strategies():
            print(f"  - {name}")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # OFFLINE CSV BACKTEST MODE (pairs only, MyStrategy only)
    # -------------------------------------------------------------------------
    csv_pair = _load_pairs_csvs(args.csv, args.csv2)
    if csv_pair is not None:
        if args.strategy != "mystrategy":
            raise SystemExit("Offline CSV backtest is supported only for --strategy mystrategy (pairs).")

        csv_a, csv_b = csv_pair
        if not csv_a.exists():
            raise SystemExit(f"CSV not found: {csv_a}")
        if not csv_b.exists():
            raise SystemExit(f"CSV not found: {csv_b}")

        df_a = _read_leg_csv(csv_a, "Close_A")
        df_b = _read_leg_csv(csv_b, "Close_B")

        df = df_a.merge(df_b, on="Datetime", how="inner").dropna().sort_values("Datetime").reset_index(drop=True)
        if df.empty:
            raise SystemExit("No overlapping Datetime rows between the two CSVs.")

        strategy = MyStrategy(
            lookback=args.pairs_lookback,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            position_size=args.position_size,  # still used inside strategy output; sizing here uses args.position_size too
        )

        logger.info(f"OFFLINE BACKTEST (pairs): {csv_a.name} / {csv_b.name} | strategy=mystrategy | rows={len(df)}")

        equity_df = _pairs_backtest_mystrategy(
            df=df,
            strategy=strategy,
            capital=args.capital,
            qty_a_fixed=float(max(0.0, args.position_size)),
            fee_per_order=float(max(0.0, args.fee_per_order)),
        )

        s = _summarize_equity(equity_df)

        # Print summary (kept similar spirit to run_live summary; minimal + relevant)
        print("")
        print("=" * 60)
        print("                 OFFLINE BACKTEST SUMMARY")
        print("=" * 60)
        print(f"  Rows:            {len(equity_df)}")
        print(f"  Trades:          {int(s['trades'])}")
        print("-" * 60)
        print(f"  Start Equity:    ${s['start_equity']:,.2f}")
        print(f"  End Equity:      ${s['end_equity']:,.2f}")
        print(f"  Net P&L:         ${s['net_pnl']:+,.2f}")
        print("-" * 60)
        print(f"  Sharpe (bar):    {s['sharpe']:.3f}")
        print(f"  Max Drawdown:    {s['max_drawdown']:.2%}")
        print("=" * 60)

        # Optional: save to data/ for inspection (very small change; piggybacks existing save_bars)
        if args.save_data:
            out_path = Path("data") / f"backtest_pairs_{csv_a.stem}_{csv_b.stem}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            equity_df.to_csv(out_path, index=False)
            logger.info(f"Saved backtest equity to: {out_path}")

        return

    # -------------------------------------------------------------------------
    # LIVE / PAPER LOOP MODE (unchanged)
    # -------------------------------------------------------------------------

    # Build strategy
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
    elif strategy_cls is CryptoTrendStrategy:
        strategy = CryptoTrendStrategy(
            short_window=args.short_window,
            long_window=args.long_window,
            position_size=args.position_size,
        )
    elif strategy_cls is DemoStrategy:
        strategy = DemoStrategy(
            position_size=args.position_size,
        )
    else:
        try:
            strategy = strategy_cls()
        except TypeError as exc:
            raise SystemExit(
                f"{strategy_cls.__name__} must support a no-arg constructor or use --strategy template."
            ) from exc

    # Log startup
    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info(f"Starting {mode} trading: {args.symbol} | strategy={args.strategy} | timeframe={args.timeframe}")

    trader = AlpacaTrader(
        symbol=args.symbol,
        asset_class=args.asset_class,
        timeframe=args.timeframe,
        lookback=args.lookback,
        strategy=strategy,
        feed=args.feed,
        dry_run=args.dry_run,
        max_order_notional=args.max_order_notional,
    )

    trade_logger = get_trade_logger()
    start_equity = trader.starting_equity
    iteration_count = 0

    def handle_iteration() -> None:
        nonlocal iteration_count
        iteration_count += 1
        logger.debug(f"Iteration {iteration_count}: fetching data for {args.symbol}")
        df = trader.run_once()
        if args.save_data and df is not None:
            raw_path = save_bars(df, args.symbol, args.timeframe, args.asset_class)
            clean_market_data(raw_path)

    def print_summary() -> None:
        summary = trade_logger.get_session_summary(start_equity)
        logger.info("")
        logger.info("=" * 60)
        logger.info("                    SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Iterations:      {iteration_count}")
        logger.info(f"  Total Trades:    {summary['total_trades']}")
        logger.info(f"  Buys / Sells:    {summary['buys']} / {summary['sells']}")
        logger.info("-" * 60)
        logger.info(f"  Wins / Losses:   {summary['wins']} / {summary['losses']}")
        logger.info(f"  Win Rate:        {summary['win_rate']:.1f}%")
        logger.info(f"  Avg Trade P&L:   ${summary['avg_trade_pnl']:+,.2f}")
        logger.info("-" * 60)
        logger.info(f"  Start Equity:    ${summary['start_equity']:,.2f}")
        logger.info(f"  End Equity:      ${summary['end_equity']:,.2f}")
        logger.info(f"  Net P&L:         ${summary['net_pnl']:+,.2f}")
        logger.info("-" * 60)
        logger.info(f"  Sharpe Ratio:    {summary['sharpe_ratio']:.2f}")
        logger.info(f"  Volatility:      {summary['volatility']:.2f}%")
        logger.info(f"  Max Drawdown:    {summary['max_drawdown']:.2f}%")
        logger.info("=" * 60)
        logger.info("Logs: logs/trades.csv, logs/system.log")

    if args.live:
        logger.info(f"Running continuously (Ctrl+C to stop). Sleep: {args.sleep}s between iterations.")
        try:
            while True:
                handle_iteration()
                time.sleep(args.sleep)
        except KeyboardInterrupt:
            logger.info("Received stop signal.")
            print_summary()
    else:
        logger.info(f"Running {args.iterations} iteration(s)...")
        for i in range(args.iterations):
            handle_iteration()
            if i < args.iterations - 1:
                time.sleep(args.sleep)
        print_summary()


if __name__ == "__main__":
    main()