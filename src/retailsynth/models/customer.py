"""
Customer data model.
Currently: Simple dataclass for customer attributes
Future: Will be enhanced with state tracking for purchase history
"""
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

@dataclass
class Customer:
    """
    Represents a single customer in the retail system.
    """
    customer_id: int
    age: int
    household_size: int
    marital_status: str
    children_count: int
    income_bracket: str
    shopping_personality: str  # 'price_anchor', 'convenience', 'planned', 'impulse'
    price_sensitivity: str  # 'high', 'medium', 'low'
    store_loyalty_level: float
    days_since_last_visit: int
    mobile_usage_propensity: float
    sustainability_preference: float
    utility_params: Dict[str, float]  # beta_price, beta_brand, beta_promotion, beta_assortment_role
    brand_preferences: Dict[str, float]
    created_at: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation"""
        return {
            'customer_id': self.customer_id,
            'age': self.age,
            'household_size': self.household_size,
            'marital_status': self.marital_status,
            'children_count': self.children_count,
            'income_bracket': self.income_bracket,
            'shopping_personality': self.shopping_personality,
            'price_sensitivity': self.price_sensitivity,
            'store_loyalty_level': self.store_loyalty_level,
            'days_since_last_visit': self.days_since_last_visit,
            'mobile_usage_propensity': self.mobile_usage_propensity,
            'sustainability_preference': self.sustainability_preference,
            'utility_params': self.utility_params,
            'brand_preferences': self.brand_preferences,
            'created_at': self.created_at
        }