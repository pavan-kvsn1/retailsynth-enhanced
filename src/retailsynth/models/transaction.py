"""
Transaction data models.
"""
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class TransactionItem:
    """A single line item in a transaction"""
    line_number: int
    product_id: int
    quantity: int
    unit_price: float
    line_total: float
    transaction_id: Optional[int] = None

@dataclass
class Transaction:
    """A complete shopping transaction"""
    transaction_id: int
    customer_id: int
    store_id: int
    transaction_date: datetime
    transaction_time: str
    week_number: int
    total_items_count: int
    total_revenue: float
    total_margin: float
    total_discount: float
    promotional_items_count: int
    satisfaction_score: float
    created_at: datetime