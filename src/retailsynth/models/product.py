"""
Product data model.
Currently: Simple dataclass for product attributes
Future: Will be replaced by real Dunnhumby products
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Product:
    """
    Represents a single product in the retail system.
    """
    product_id: int
    department: str
    category: str
    subcategory: str
    product_name: str
    brand: str
    base_price: float
    assortment_role: str  # 'lpg_line', 'front_basket', 'mid_basket', 'back_basket'
    launch_week: Optional[int] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation"""
        return {
            'product_id': self.product_id,
            'department': self.department,
            'category': self.category,
            'subcategory': self.subcategory,
            'product_name': self.product_name,
            'brand': self.brand,
            'base_price': self.base_price,
            'assortment_role': self.assortment_role,
            'launch_week': self.launch_week,
            'created_at': self.created_at
        }