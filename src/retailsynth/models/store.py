"""
Store data model.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Store:
    """
    Represents a single retail store location.
    """
    store_id: int
    store_name: str
    store_type: str  # 'Supermarket', 'Hypermarket', 'Convenience', 'Supercenter'
    region: str
    square_feet: int
    established_year: int
    has_pharmacy: bool
    has_deli: bool
    has_bakery: bool
    parking_spaces: int
    created_at: datetime
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation"""
        return {
            'store_id': self.store_id,
            'store_name': self.store_name,
            'store_type': self.store_type,
            'region': self.region,
            'square_feet': self.square_feet,
            'established_year': self.established_year,
            'has_pharmacy': self.has_pharmacy,
            'has_deli': self.has_deli,
            'has_bakery': self.has_bakery,
            'parking_spaces': self.parking_spaces,
            'created_at': self.created_at
        }