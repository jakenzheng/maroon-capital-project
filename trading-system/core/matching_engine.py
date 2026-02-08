import random

from core.order_book import Order


class MatchingEngine:
    """
    Simulates exchange execution outcomes such as fills, partial fills, and
    cancellations for the provided trade intent.
    """

    def simulate_execution(self, order: Order, intended_qty: int, trade_price: float):
        r = random.random()

        if r < 0.70:
            filled_qty = intended_qty
            status = "filled"
        elif r < 0.90:
            filled_qty = max(1, int(intended_qty * random.uniform(0.1, 0.9)))
            status = "partial"
        else:
            filled_qty = 0
            status = "cancelled"

        return {
            "order_id": order.order_id,
            "status": status,
            "filled_qty": filled_qty,
            "avg_price": trade_price,
        }
