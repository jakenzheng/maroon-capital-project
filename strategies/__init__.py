"""Strategy implementations and lookup helpers."""

from __future__ import annotations

import inspect
from typing import Dict, Type

from . import strategy_base
from .strategy_base import (
    CryptoTrendStrategy,
    DemoStrategy,
    MovingAverageStrategy,
    Strategy,
    TemplateStrategy,
    MyStrategy
)


def _build_registry() -> Dict[str, Type[Strategy]]:
    registry: Dict[str, Type[Strategy]] = {}
    for name, obj in inspect.getmembers(strategy_base, inspect.isclass):
        if obj is Strategy:
            continue
        if issubclass(obj, Strategy) and obj.__module__ == strategy_base.__name__:
            registry[name.lower()] = obj

    registry.setdefault("ma", MovingAverageStrategy)
    registry.setdefault("moving_average", MovingAverageStrategy)
    registry.setdefault("template", TemplateStrategy)
    registry.setdefault("student", TemplateStrategy)
    registry.setdefault("crypto", CryptoTrendStrategy)
    registry.setdefault("crypto_trend", CryptoTrendStrategy)
    registry.setdefault("crypto_trend_ema", CryptoTrendStrategy)
    registry.setdefault("demo", DemoStrategy)
    registry.setdefault("fast", DemoStrategy)
    registry.setdefault("my", MyStrategy)
    return registry


_REGISTRY = _build_registry()


def get_strategy_class(name: str) -> Type[Strategy]:
    key = name.strip().lower()
    if not key:
        raise ValueError("Strategy name cannot be empty.")
    if key not in _REGISTRY:
        options = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{name}'. Available: {options}")
    return _REGISTRY[key]


def list_strategies() -> list[str]:
    return sorted(_REGISTRY.keys())


__all__ = [
    "Strategy",
    "TemplateStrategy",
    "MovingAverageStrategy",
    "CryptoTrendStrategy",
    "DemoStrategy",
    "MyStrategy",
    "get_strategy_class",
    "list_strategies",
]
