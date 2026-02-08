import json
import time
from pathlib import Path


class OrderManager:
    """
    Validates new orders and tracks capital/position state for risk checks.
    """

    def __init__(
        self,
        capital: float = 100_000.0,
        max_long_position: int = 500,
        max_short_position: int = 500,
        max_orders_per_min: int = 30,
    ):
        self.initial_capital = float(capital)
        self.cash = self.initial_capital
        self.max_long_position = max_long_position
        self.max_short_position = max_short_position
        self.max_orders_per_min = max_orders_per_min

        self.order_timestamps: list[float] = []
        self.long_position = 0
        self.short_position = 0

    # ------------------------------------------------------------------ utils

    @property
    def net_position(self) -> int:
        return self.long_position - self.short_position

    def portfolio_value(self, price: float) -> float:
        return self.cash + self.long_position * price - self.short_position * price

    # ----------------------------------------------------------------- checks

    def _check_capital(self, order) -> bool:
        if order.side == "buy":
            return order.price * order.qty <= self.cash
        return True

    def _project_positions(self, order):
        long_after = self.long_position
        short_after = self.short_position
        qty_remaining = order.qty

        if order.side == "buy":
            if short_after > 0:
                cover = min(qty_remaining, short_after)
                short_after -= cover
                qty_remaining -= cover
            long_after += qty_remaining
        else:
            if long_after > 0:
                cover = min(qty_remaining, long_after)
                long_after -= cover
                qty_remaining -= cover
            short_after += qty_remaining

        return long_after, short_after

    def _check_position_limit(self, order) -> bool:
        long_after, short_after = self._project_positions(order)
        return (long_after <= self.max_long_position) and (
            short_after <= self.max_short_position
        )

    def _check_order_rate(self) -> bool:
        now = time.time()
        self.order_timestamps = [t for t in self.order_timestamps if now - t < 60]
        return len(self.order_timestamps) < self.max_orders_per_min

    # ----------------------------------------------------------------- public

    def validate(self, order):
        if not self._check_capital(order):
            return False, "Not enough capital"
        if not self._check_position_limit(order):
            return False, "Position limit exceeded"
        if not self._check_order_rate():
            return False, "Order rate limit exceeded"

        self.order_timestamps.append(time.time())
        return True, "Order approved"

    def record_execution(self, order, filled_qty: int, price: float) -> None:
        """
        Update capital and open positions after an execution report.
        """
        if filled_qty <= 0:
            return

        qty_remaining = filled_qty
        if order.side == "buy":
            if self.short_position > 0:
                cover = min(qty_remaining, self.short_position)
                self.short_position -= cover
                qty_remaining -= cover
                self.cash -= price * cover
            if qty_remaining > 0:
                self.long_position += qty_remaining
                self.cash -= price * qty_remaining
        else:
            if self.long_position > 0:
                close = min(qty_remaining, self.long_position)
                self.long_position -= close
                qty_remaining -= close
                self.cash += price * close
            if qty_remaining > 0:
                self.short_position += qty_remaining
                self.cash += price * qty_remaining


class OrderLoggingGateway:
    """
    Logs all order events: new, modified, canceled, filled.
    """

    def __init__(self, file_path="data/order_log.json"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type, data):
        event = {"event": event_type, "timestamp": time.time(), "data": data}
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
