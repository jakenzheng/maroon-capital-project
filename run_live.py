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

    # Pairs trading (two symbols, comma-separated)
    python run_live.py --symbol PM,MNST --strategy mystrategy --dry-run
    python run_live.py --symbol PM,MNST --strategy mystrategy --live

Logs are saved to: logs/trades.csv, logs/signals.csv, logs/system.log
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from core.alpaca_trader import AlpacaTrader
from core.logger import get_logger, get_trade_logger
from pipeline.alpaca import clean_market_data, save_bars
from strategies import MyStrategy, MovingAverageStrategy, TemplateStrategy, CryptoTrendStrategy, DemoStrategy, get_strategy_class, list_strategies

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
  python run_live.py --symbol PM,MNST --strategy mystrategy --dry-run --live
        """,
    )
    parser.add_argument("--symbol", default="AAPL", help="Ticker, crypto symbol, or comma-separated pair (e.g. PM,MNST)")
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
    parser.add_argument("--pairs-lookback", type=int, default=60, help="Lookback for pairs z-score (default: 60)")
    parser.add_argument("--entry-z", type=float, default=2.0, help="Z-score threshold to enter pairs trade (default: 2.0)")
    parser.add_argument("--exit-z", type=float, default=0.5, help="Z-score threshold to exit pairs trade (default: 0.5)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of loops to run (default: 1)")
    parser.add_argument("--sleep", type=int, default=60, help="Seconds between loops (default: 60)")
    parser.add_argument("--live", action="store_true", help="Run continuously until Ctrl+C")
    parser.add_argument("--save-data", action="store_true", help="Save raw+clean CSVs to data/")
    parser.add_argument("--dry-run", action="store_true", help="Print decisions without placing orders")
    parser.add_argument("--feed", default=None, help="Data feed (iex or sip for stocks)")
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies and exit")
    return parser.parse_args()


def is_pairs(symbol: str) -> bool:
    """Return True if symbol is a comma-separated pair like 'PM,MNST'."""
    return "," in symbol


def fetch_pairs_df(trader_a: AlpacaTrader, trader_b: AlpacaTrader) -> pd.DataFrame | None:
    """
    Fetch bars for both symbols, align on Datetime, and return a merged
    DataFrame with Close_A and Close_B columns ready for MyStrategy.
    """
    df_a = trader_a.fetch_latest_bars()
    df_b = trader_b.fetch_latest_bars()

    if df_a is None or df_b is None:
        logger.error("Failed to fetch bars for one or both symbols.")
        return None

    df_a = df_a.set_index("Datetime")[["Close"]].rename(columns={"Close": "Close_A"})
    df_b = df_b.set_index("Datetime")[["Close"]].rename(columns={"Close": "Close_B"})

    df = df_a.join(df_b, how="inner").dropna().reset_index()
    if df.empty:
        logger.error("No overlapping bars between the two symbols.")
        return None

    logger.debug(f"Merged pairs DataFrame: {len(df)} rows")
    return df


def execute_pairs_signal(signal: int, position: int, symbol_a: str, symbol_b: str,
                          trader_a: AlpacaTrader, trader_b: AlpacaTrader,
                          qty: float, dry_run: bool) -> None:
    """
    Translate a pairs signal into two orders:
      Long spread  (signal=+1): BUY  A, SELL B
      Short spread (signal=-1): SELL A, BUY  B
      Close        (signal=0, position flipping): handled by close orders
    """
    if signal == 0:
        return

    if signal == 1:
        side_a, side_b = "buy", "sell"
        logger.info(f"PAIRS LONG SPREAD: BUY {symbol_a} / SELL {symbol_b} qty={qty}")
    elif signal == -1:
        side_a, side_b = "sell", "buy"
        logger.info(f"PAIRS SHORT SPREAD: SELL {symbol_a} / BUY {symbol_b} qty={qty}")
    else:
        return

    if dry_run:
        logger.info(f"  [DRY RUN] Would {side_a} {qty} {symbol_a}")
        logger.info(f"  [DRY RUN] Would {side_b} {qty} {symbol_b}")
        return

    trader_a.place_order(side=side_a, qty=qty)
    trader_b.place_order(side=side_b, qty=qty)


def main() -> None:
    args = parse_args()

    # Handle --list-strategies
    if args.list_strategies:
        print("Available strategies:")
        for name in list_strategies():
            print(f"  - {name}")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # PAIRS MODE
    # -------------------------------------------------------------------------
    if is_pairs(args.symbol):
        symbols = [s.strip() for s in args.symbol.split(",")]
        if len(symbols) != 2:
            raise SystemExit("--symbol for pairs trading must be exactly two comma-separated tickers, e.g. PM,MNST")

        symbol_a, symbol_b = symbols
        strategy = MyStrategy(
            lookback=args.pairs_lookback,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            position_size=args.position_size,
        )

        mode = "DRY RUN" if args.dry_run else "LIVE"
        logger.info(f"Starting {mode} PAIRS trading: {symbol_a} / {symbol_b} | strategy=mystrategy | timeframe={args.timeframe}")

        # One AlpacaTrader per leg — used for data fetching and order execution
        trader_a = AlpacaTrader(
            symbol=symbol_a,
            asset_class=args.asset_class,
            timeframe=args.timeframe,
            lookback=args.lookback,
            strategy=strategy,        # not used for signal gen here, only for order plumbing
            feed=args.feed,
            dry_run=args.dry_run,
            max_order_notional=args.max_order_notional,
        )
        trader_b = AlpacaTrader(
            symbol=symbol_b,
            asset_class=args.asset_class,
            timeframe=args.timeframe,
            lookback=args.lookback,
            strategy=strategy,
            feed=args.feed,
            dry_run=args.dry_run,
            max_order_notional=args.max_order_notional,
        )

        trade_logger = get_trade_logger()
        start_equity = trader_a.starting_equity
        iteration_count = 0

        def handle_pairs_iteration() -> None:
            nonlocal iteration_count
            iteration_count += 1
            logger.debug(f"Iteration {iteration_count}: fetching bars for {symbol_a} and {symbol_b}")

            df = fetch_pairs_df(trader_a, trader_b)
            if df is None:
                return

            result = strategy.run(df)

            last = result.iloc[-1]
            signal   = int(last["signal"])
            position = int(last["position"])
            qty      = float(last["target_qty"])
            z        = round(float(last["z"]), 3) if "z" in last else float("nan")

            logger.info(f"  z={z} | signal={signal} | position={position} | qty={qty}")

            execute_pairs_signal(
                signal=signal,
                position=position,
                symbol_a=symbol_a,
                symbol_b=symbol_b,
                trader_a=trader_a,
                trader_b=trader_b,
                qty=qty,
                dry_run=args.dry_run,
            )

            if args.save_data and df is not None:
                raw_path = save_bars(df, f"{symbol_a}_{symbol_b}", args.timeframe, args.asset_class)
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
                    handle_pairs_iteration()
                    time.sleep(args.sleep)
            except KeyboardInterrupt:
                logger.info("Received stop signal.")
                print_summary()
        else:
            logger.info(f"Running {args.iterations} iteration(s)...")
            for i in range(args.iterations):
                handle_pairs_iteration()
                if i < args.iterations - 1:
                    time.sleep(args.sleep)
            print_summary()

        return  # done, skip single-symbol path below

    # -------------------------------------------------------------------------
    # SINGLE SYMBOL MODE (unchanged)
    # -------------------------------------------------------------------------
    if "," in args.symbol:
        symbol_a, symbol_b = [s.strip() for s in args.symbol.split(",")]
        strategy = MyStrategy(lookback=args.pairs_lookback, entry_z=args.entry_z,
                            exit_z=args.exit_z, position_size=args.position_size)
        trader_a = AlpacaTrader(symbol=symbol_a, asset_class=args.asset_class, timeframe=args.timeframe,
                                lookback=args.lookback, strategy=strategy, feed=args.feed,
                                dry_run=args.dry_run, max_order_notional=args.max_order_notional)
        trader_b = AlpacaTrader(symbol=symbol_b, asset_class=args.asset_class, timeframe=args.timeframe,
                                lookback=args.lookback, strategy=strategy, feed=args.feed,
                                dry_run=args.dry_run, max_order_notional=args.max_order_notional)
        trade_logger = get_trade_logger()
        start_equity = trader_a.starting_equity
        iteration_count = 0

        def handle_pairs_iteration():
            nonlocal iteration_count
            iteration_count += 1
            df_a = trader_a.get_bars().set_index("Datetime")[["Close"]].rename(columns={"Close": "Close_A"})
            df_b = trader_b.get_bars().set_index("Datetime")[["Close"]].rename(columns={"Close": "Close_B"})
            df = df_a.join(df_b, how="inner").dropna().reset_index()
            result = strategy.run(df)
            last = result.iloc[-1]
            signal, qty = int(last["signal"]), float(last["target_qty"])
            logger.info(f"z={last['z']:.3f} | signal={signal} | position={int(last['position'])} | qty={qty}")
            if signal == 1:
                side_a, side_b = "buy", "sell"
            elif signal == -1:
                side_a, side_b = "sell", "buy"
            else:
                return
            if args.dry_run:
                logger.info(f"[DRY RUN] {side_a} {qty} {symbol_a}, {side_b} {qty} {symbol_b}")
            else:
                trader_a.place_order(side=side_a, qty=qty)
                trader_b.place_order(side=side_b, qty=qty)

        # re-use the existing loop/summary pattern below by swapping handle_iteration
        args._handle_iteration = handle_pairs_iteration
        args._trade_logger = trade_logger
        args._start_equity = start_equity
        args._iteration_count_ref = [iteration_count]
    
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