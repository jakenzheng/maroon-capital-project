"""
Alpaca data helpers for downloading and cleaning market data.
"""

from __future__ import annotations

import os
from datetime import timezone
from pathlib import Path
from typing import Optional

import alpaca_trade_api as tradeapi
import pandas as pd
from dotenv import load_dotenv


DATA_DIR = Path("data")
DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"
DEFAULT_DATA_FEED = "iex"
DEFAULT_FALLBACK_DAYS = 10


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_env() -> None:
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    loaded = False
    for path in env_paths:
        if path.exists():
            load_dotenv(path, override=False)
            loaded = True
    if not loaded:
        load_dotenv(override=False)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Set it in your .env or environment.")
    return value


def _parse_timeframe(timeframe: str):
    try:
        return tradeapi.TimeFrame(timeframe)
    except Exception:
        return timeframe


def _to_rfc3339(ts: pd.Timestamp) -> str:
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_rest() -> tradeapi.REST:
    _load_env()
    api_key = _require_env("ALPACA_API_KEY")
    api_secret = _require_env("ALPACA_API_SECRET")
    base_url = os.environ.get("ALPACA_API_URL", DEFAULT_BASE_URL).rstrip("/")
    if not base_url:
        base_url = DEFAULT_BASE_URL
    return tradeapi.REST(api_key, api_secret, base_url, api_version="v2")


def _normalize_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])

    if isinstance(df.index, pd.MultiIndex):
        level0 = df.index.get_level_values(0)
        symbol_key = symbol.upper()
        if symbol in level0:
            df = df.xs(symbol, level=0)
        elif symbol_key in level0:
            df = df.xs(symbol_key, level=0)
        else:
            lower_map = {str(val).lower(): val for val in level0}
            match = lower_map.get(symbol.lower())
            if match is not None:
                df = df.xs(match, level=0)

    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

    rename_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower in {"timestamp", "time", "t", "index", "datetime", "date"}:
            rename_map[col] = "Datetime"
        elif col_lower in {"open", "o"}:
            rename_map[col] = "Open"
        elif col_lower in {"high", "h"}:
            rename_map[col] = "High"
        elif col_lower in {"low", "l"}:
            rename_map[col] = "Low"
        elif col_lower in {"close", "c"}:
            rename_map[col] = "Close"
        elif col_lower in {"volume", "v"}:
            rename_map[col] = "Volume"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    if "Datetime" not in df.columns:
        raise ValueError("Alpaca bars are missing timestamp data.")

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df.dropna(subset=["Datetime"], inplace=True)
    required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Alpaca bars are missing columns: {', '.join(missing)}")

    return df[required]


def fetch_stock_bars(
    symbol: str,
    timeframe: str = "1Min",
    limit: int = 1000,
    feed: Optional[str] = None,
    fallback_days: int = DEFAULT_FALLBACK_DAYS,
    api: Optional[tradeapi.REST] = None,
) -> pd.DataFrame:
    api = api or get_rest()
    feed = feed or os.environ.get("ALPACA_DATA_FEED", DEFAULT_DATA_FEED)
    tf = _parse_timeframe(timeframe)
    bars = api.get_bars(symbol, tf, limit=limit, feed=feed).df
    df = _normalize_bars(bars, symbol)
    if df.empty and fallback_days > 0:
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(days=fallback_days)
        bars = api.get_bars(
            symbol,
            tf,
            start=_to_rfc3339(start),
            end=_to_rfc3339(end),
            limit=limit,
            feed=feed,
        ).df
        df = _normalize_bars(bars, symbol)
    if df.empty:
        raise ValueError(
            f"No stock bars returned for {symbol}. Market may be closed. "
            "Try --timeframe 1Day or run during market hours."
        )
    return df


def fetch_crypto_bars(
    symbol: str,
    timeframe: str = "1Min",
    limit: int = 1000,
    fallback_days: int = DEFAULT_FALLBACK_DAYS,
    api: Optional[tradeapi.REST] = None,
) -> pd.DataFrame:
    api = api or get_rest()
    tf = _parse_timeframe(timeframe)
    if hasattr(api, "get_crypto_bars"):
        bars = api.get_crypto_bars(symbol, tf, limit=limit).df
    else:
        raise RuntimeError("alpaca_trade_api does not support get_crypto_bars in this version.")
    df = _normalize_bars(bars, symbol)
    if df.empty and fallback_days > 0:
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(days=fallback_days)
        bars = api.get_crypto_bars(
            symbol,
            tf,
            start=_to_rfc3339(start),
            end=_to_rfc3339(end),
            limit=limit,
        ).df
        df = _normalize_bars(bars, symbol)
    if df.empty:
        raise ValueError(f"No crypto bars returned for {symbol}.")
    return df


def save_bars(df: pd.DataFrame, symbol: str, timeframe: str, asset_class: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = asset_class.lower()
    safe_symbol = symbol.upper().replace("/", "")
    raw_path = DATA_DIR / f"{safe_symbol}_{timeframe}_{suffix}_alpaca_raw.csv"
    df.to_csv(raw_path, index=False)
    return raw_path


def clean_market_data(
    csv_path: Path,
    dest_dir: Optional[Path] = None,
    add_features: bool = True,
) -> Path:
    """
    Clean Alpaca candle data and optionally add derived features.
    """
    df = pd.read_csv(csv_path)
    if "Datetime" not in df.columns:
        raise ValueError("Input CSV must contain a Datetime column.")

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df.dropna(subset=["Datetime"], inplace=True)
    df.drop_duplicates(subset=["Datetime"], inplace=True)
    df.sort_values("Datetime", inplace=True)

    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
    df.set_index("Datetime", inplace=True)

    if add_features:
        df["returns"] = df["Close"].pct_change().fillna(0.0)
        df["rolling_volatility"] = df["returns"].rolling(60).std().fillna(0.0)
        df["rolling_volume"] = df["Volume"].rolling(60).mean().fillna(method="bfill")
        df["momentum"] = df["Close"].diff().fillna(0.0)

    dest_dir = _ensure_dir(dest_dir or DATA_DIR)
    stem = Path(csv_path).stem.replace("_raw", "")
    out_path = dest_dir / f"{stem}_clean.csv"
    df.to_csv(out_path)
    return out_path
