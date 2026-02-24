from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Union
import time

import pandas as pd

CsvArg = Union[str, Path, Sequence[Union[str, Path]]]


class MarketDataGateway:
    """
    Streams historical market data to consumers.

    Single CSV: yields rows with Datetime, Open, High, Low, Close, Volume.
    Two CSVs: merges on Datetime and yields rows with *_A and *_B columns
              (Close_A, Close_B, etc.). Also sets .symbol = "A/B".
    """

    def __init__(self, csv_path: CsvArg, symbol: Optional[str] = None):
        paths = self._normalize_paths(csv_path)
        if len(paths) not in (1, 2):
            raise ValueError("MarketDataGateway supports 1 CSV or 2 CSVs (pairs).")

        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"CSV file not found: {p}")

        self.csv_paths = paths

        if len(paths) == 1:
            self.symbol = symbol or self._infer_symbol(paths[0])
            self.symbols = (self.symbol,)
            self.data = self._read_one(paths[0])
        else:
            sym_a = symbol or self._infer_symbol(paths[0])
            sym_b = self._infer_symbol(paths[1])
            self.symbols = (sym_a, sym_b)
            self.symbol = f"{sym_a}/{sym_b}"
            self.data = self._read_two(paths[0], paths[1])

        self.length = len(self.data)
        self.pointer = 0

    def _normalize_paths(self, csv_path: CsvArg) -> list[Path]:
        if isinstance(csv_path, (str, Path)):
            s = str(csv_path)
            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip()]
                return [Path(p) for p in parts]
            return [Path(csv_path)]
        return [Path(p) for p in csv_path]

    def _infer_symbol(self, path: Path) -> str:
        stem = path.stem
        token = stem.split("_")[0] if stem else "ASSET"
        return token.upper()

    def _read_one(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["Datetime"])
        if "Datetime" not in df.columns:
            raise ValueError(f"CSV must contain a Datetime column: {path}")
        df.sort_values("Datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _read_two(self, path_a: Path, path_b: Path) -> pd.DataFrame:
        df_a = self._read_one(path_a).rename(columns={c: f"{c}_A" for c in self._read_one(path_a).columns if c != "Datetime"})
        df_b = self._read_one(path_b).rename(columns={c: f"{c}_B" for c in self._read_one(path_b).columns if c != "Datetime"})

        df = pd.merge(df_a, df_b, on="Datetime", how="outer").sort_values("Datetime")
        df = df.ffill()
        df.reset_index(drop=True, inplace=True)

        if "Close_A" not in df.columns or "Close_B" not in df.columns:
            raise ValueError("Pairs mode requires both CSVs to have a Close column.")
        return df

    # Iterator protocol
    def __iter__(self) -> Iterator[Dict]:
        self.reset()
        return self

    def __next__(self) -> Dict:
        if self.pointer >= self.length:
            raise StopIteration
        row = self.data.iloc[self.pointer].to_dict()
        row["Datetime"] = pd.Timestamp(row["Datetime"])
        self.pointer += 1
        return row

    # Helpers
    def reset(self) -> None:
        self.pointer = 0

    def has_next(self) -> bool:
        return self.pointer < self.length

    def get_next(self) -> Optional[Dict]:
        try:
            return next(self)
        except StopIteration:
            return None

    def peek(self) -> Optional[Dict]:
        if not self.has_next():
            return None
        row = self.data.iloc[self.pointer].to_dict()
        row["Datetime"] = pd.Timestamp(row["Datetime"])
        return row

    # Generator
    def stream(self, delay: Optional[float] = None, reset: bool = False):
        if reset:
            self.reset()
        while self.has_next():
            row = next(self)
            yield row
            if delay:
                time.sleep(delay)


Gateway = MarketDataGateway