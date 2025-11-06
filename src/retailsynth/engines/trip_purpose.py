"""
Trip Purpose Framework (Sprint 1.4)

Defines shopping mission types and their characteristics:
- Trip purpose taxonomy (stock-up, fill-in, meal-prep, etc.)
- Basket size distributions per trip type
- Category priorities and requirements
- Trip purpose selection probabilities

Based on retail industry research and Dunnhumby shopping patterns.

Author: RetailSynth Team
Date: November 2024
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


class TripPurpose(Enum):
    """
    Shopping trip purpose taxonomy
    
    Based on retail research:
    - Stock-up: Weekly/bi-weekly major grocery run
    - Fill-in: Quick top-up between major trips
    - Meal-prep: Recipe-focused shopping
    - Convenience: Grab-and-go, immediate needs
    - Special: Party, holiday, entertaining
    """
    STOCK_UP = "stock_up"
    FILL_IN = "fill_in"
    MEAL_PREP = "meal_prep"
    CONVENIENCE = "convenience"
    SPECIAL_OCCASION = "special_occasion"


@dataclass
class TripCharacteristics:
    """
    Characteristics of a shopping trip type
    
    Attributes:
        basket_size_mean: Average number of items
        basket_size_std: Standard deviation of basket size
        min_items: Minimum items in basket
        max_items: Maximum items in basket
        required_categories: Must-have categories
        optional_categories: Nice-to-have categories
        min_categories: Minimum number of categories
        max_same_product: Maximum quantity of same product
        category_diversity: How diverse categories should be (0-1)
    """
    basket_size_mean: float
    basket_size_std: float
    min_items: int
    max_items: int
    required_categories: List[str]
    optional_categories: List[str]
    min_categories: int
    max_same_product: int
    category_diversity: float  # 0.0 = focused, 1.0 = diverse


# Industry-standard trip characteristics
TRIP_CHARACTERISTICS = {
    TripPurpose.STOCK_UP: TripCharacteristics(
        basket_size_mean=28.0,
        basket_size_std=8.0,
        min_items=15,
        max_items=50,
        required_categories=['DAIRY', 'PRODUCE', 'MEAT', 'GROCERY'],
        optional_categories=['FROZEN', 'BAKERY', 'DELI', 'BEVERAGE', 'SNACKS'],
        min_categories=5,
        max_same_product=3,
        category_diversity=0.8
    ),
    
    TripPurpose.FILL_IN: TripCharacteristics(
        basket_size_mean=5.5,
        basket_size_std=2.0,
        min_items=2,
        max_items=12,
        required_categories=['DAIRY', 'PRODUCE'],
        optional_categories=['BAKERY', 'BEVERAGE', 'SNACKS'],
        min_categories=2,
        max_same_product=2,
        category_diversity=0.4
    ),
    
    TripPurpose.MEAL_PREP: TripCharacteristics(
        basket_size_mean=12.0,
        basket_size_std=4.0,
        min_items=6,
        max_items=20,
        required_categories=['MEAT', 'PRODUCE', 'GROCERY'],
        optional_categories=['DAIRY', 'FROZEN', 'DELI'],
        min_categories=3,
        max_same_product=2,
        category_diversity=0.6
    ),
    
    TripPurpose.CONVENIENCE: TripCharacteristics(
        basket_size_mean=3.0,
        basket_size_std=1.5,
        min_items=1,
        max_items=6,
        required_categories=[],  # No required categories
        optional_categories=['BEVERAGE', 'SNACKS', 'DAIRY', 'BAKERY'],
        min_categories=1,
        max_same_product=2,
        category_diversity=0.2
    ),
    
    TripPurpose.SPECIAL_OCCASION: TripCharacteristics(
        basket_size_mean=22.0,
        basket_size_std=7.0,
        min_items=10,
        max_items=40,
        required_categories=['MEAT', 'PRODUCE', 'BEVERAGE'],
        optional_categories=['FROZEN', 'BAKERY', 'DELI', 'SNACKS', 'DAIRY'],
        min_categories=4,
        max_same_product=4,
        category_diversity=0.7
    )
}


# Trip purpose probabilities by customer shopping personality
TRIP_PURPOSE_PROBABILITIES = {
    'price_anchor': {
        TripPurpose.STOCK_UP: 0.45,
        TripPurpose.FILL_IN: 0.30,
        TripPurpose.MEAL_PREP: 0.15,
        TripPurpose.CONVENIENCE: 0.05,
        TripPurpose.SPECIAL_OCCASION: 0.05
    },
    'convenience': {
        TripPurpose.STOCK_UP: 0.15,
        TripPurpose.FILL_IN: 0.35,
        TripPurpose.MEAL_PREP: 0.10,
        TripPurpose.CONVENIENCE: 0.35,
        TripPurpose.SPECIAL_OCCASION: 0.05
    },
    'planned': {
        TripPurpose.STOCK_UP: 0.40,
        TripPurpose.FILL_IN: 0.25,
        TripPurpose.MEAL_PREP: 0.25,
        TripPurpose.CONVENIENCE: 0.05,
        TripPurpose.SPECIAL_OCCASION: 0.05
    },
    'impulse': {
        TripPurpose.STOCK_UP: 0.25,
        TripPurpose.FILL_IN: 0.30,
        TripPurpose.MEAL_PREP: 0.15,
        TripPurpose.CONVENIENCE: 0.20,
        TripPurpose.SPECIAL_OCCASION: 0.10
    }
}


class TripPurposeSelector:
    """
    Selects trip purpose for a shopping occasion
    
    Considers:
    - Customer shopping personality
    - Time since last trip
    - Day of week
    - Season/holidays
    """
    
    def __init__(self):
        self.trip_characteristics = TRIP_CHARACTERISTICS
        self.trip_probabilities = TRIP_PURPOSE_PROBABILITIES
    
    def select_trip_purpose(
        self,
        shopping_personality: str,
        weeks_since_last_trip: int = 1,
        day_of_week: Optional[int] = None,
        week_of_year: Optional[int] = None
    ) -> TripPurpose:
        """
        Select trip purpose based on customer and context
        
        Args:
            shopping_personality: Customer type (price_anchor, convenience, planned, impulse)
            weeks_since_last_trip: Weeks since customer's last shopping trip
            day_of_week: Day of week (0=Monday, 6=Sunday)
            week_of_year: Week of year (for holiday detection)
        
        Returns:
            Selected TripPurpose
        """
        # Get base probabilities
        base_probs = self.trip_probabilities.get(
            shopping_personality,
            self.trip_probabilities['planned']
        )
        
        # Adjust probabilities based on context
        adjusted_probs = self._adjust_probabilities(
            base_probs,
            weeks_since_last_trip,
            day_of_week,
            week_of_year
        )
        
        # Sample trip purpose
        trip_purposes = list(adjusted_probs.keys())
        probabilities = list(adjusted_probs.values())
        
        return np.random.choice(trip_purposes, p=probabilities)
    
    def _adjust_probabilities(
        self,
        base_probs: Dict[TripPurpose, float],
        weeks_since_last_trip: int,
        day_of_week: Optional[int],
        week_of_year: Optional[int]
    ) -> Dict[TripPurpose, float]:
        """
        Adjust trip probabilities based on context
        
        Rules:
        - Long time since last trip → more likely stock-up
        - Weekend → more likely stock-up
        - Holiday weeks → more likely special occasion
        - Weekday → more likely fill-in/convenience
        """
        adjusted = base_probs.copy()
        
        # Time since last trip effect
        if weeks_since_last_trip >= 2:
            # Haven't shopped in 2+ weeks → need stock-up
            adjusted[TripPurpose.STOCK_UP] *= 1.5
            adjusted[TripPurpose.FILL_IN] *= 0.5
        elif weeks_since_last_trip == 0:
            # Shopped very recently → likely fill-in
            adjusted[TripPurpose.STOCK_UP] *= 0.3
            adjusted[TripPurpose.FILL_IN] *= 1.8
        
        # Day of week effect
        if day_of_week is not None:
            if day_of_week in [5, 6]:  # Saturday, Sunday
                adjusted[TripPurpose.STOCK_UP] *= 1.3
                adjusted[TripPurpose.MEAL_PREP] *= 1.2
            else:  # Weekday
                adjusted[TripPurpose.CONVENIENCE] *= 1.4
                adjusted[TripPurpose.FILL_IN] *= 1.2
        
        # Holiday effect
        if week_of_year is not None:
            holiday_weeks = [47, 48, 51, 52]  # Thanksgiving, Christmas
            if week_of_year in holiday_weeks:
                adjusted[TripPurpose.SPECIAL_OCCASION] *= 3.0
                adjusted[TripPurpose.STOCK_UP] *= 1.5
        
        # Normalize probabilities
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}
    
    def get_basket_size(self, trip_purpose: TripPurpose) -> int:
        """
        Sample basket size for a trip purpose
        
        Args:
            trip_purpose: Trip purpose type
        
        Returns:
            Number of items in basket
        """
        chars = self.trip_characteristics[trip_purpose]
        
        # Sample from normal distribution
        size = np.random.normal(chars.basket_size_mean, chars.basket_size_std)
        
        # Clip to min/max
        size = int(np.clip(size, chars.min_items, chars.max_items))
        
        return size
    
    def get_required_categories(self, trip_purpose: TripPurpose) -> List[str]:
        """Get required categories for trip purpose"""
        return self.trip_characteristics[trip_purpose].required_categories
    
    def get_optional_categories(self, trip_purpose: TripPurpose) -> List[str]:
        """Get optional categories for trip purpose"""
        return self.trip_characteristics[trip_purpose].optional_categories
    
    def get_max_same_product(self, trip_purpose: TripPurpose) -> int:
        """Get maximum quantity of same product"""
        return self.trip_characteristics[trip_purpose].max_same_product
    
    def get_min_categories(self, trip_purpose: TripPurpose) -> int:
        """Get minimum number of categories"""
        return self.trip_characteristics[trip_purpose].min_categories


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_trip_purpose_distribution(shopping_personality: str) -> Dict[TripPurpose, float]:
    """
    Get trip purpose probability distribution for a customer type
    
    Args:
        shopping_personality: Customer type
    
    Returns:
        Dictionary mapping TripPurpose to probability
    """
    return TRIP_PURPOSE_PROBABILITIES.get(
        shopping_personality,
        TRIP_PURPOSE_PROBABILITIES['planned']
    )


def estimate_trip_frequency(shopping_personality: str) -> float:
    """
    Estimate average trips per week for customer type
    
    Args:
        shopping_personality: Customer type
    
    Returns:
        Average trips per week
    """
    # Based on retail research
    frequencies = {
        'price_anchor': 0.8,    # ~3 trips per month
        'convenience': 1.5,     # ~6 trips per month
        'planned': 1.0,         # ~4 trips per month
        'impulse': 1.2          # ~5 trips per month
    }
    
    return frequencies.get(shopping_personality, 1.0)
