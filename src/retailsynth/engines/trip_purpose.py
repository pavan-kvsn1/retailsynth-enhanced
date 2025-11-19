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


    # DEPRECATED: These are now generated from config in TripPurposeSelector.__init__()
    # Kept for reference but not used
    _LEGACY_TRIP_CHARACTERISTICS = {
        # These values were too high - now configurable via EnhancedRetailConfig
        # STOCK_UP: basket_size_mean=28.0 (was way too high!)
        # SPECIAL_OCCASION: basket_size_mean=22.0 (was too high!)
        # See config.trip_*_basket_mean parameters
    }

    _LEGACY_TRIP_PURPOSE_PROBABILITIES = {
        # These probabilities were too skewed toward large trips
        # price_anchor: STOCK_UP=0.45 (was too high!)
        # See config.trip_prob_* parameters
    }


def build_trip_characteristics_from_config(config):
    """Build TRIP_CHARACTERISTICS dict from config parameters"""
    return {
        TripPurpose.STOCK_UP: TripCharacteristics(
            basket_size_mean=config.trip_stock_up_basket_mean,
            basket_size_std=config.trip_stock_up_basket_std,
            min_items=max(1, int(config.trip_stock_up_basket_mean - config.trip_stock_up_basket_std * 2)),
            max_items=int(config.trip_stock_up_basket_mean + config.trip_stock_up_basket_std * 3),
            required_categories=['DAIRY', 'PRODUCE', 'MEAT', 'GROCERY'],
            optional_categories=['FROZEN', 'BAKERY', 'DELI', 'BEVERAGE', 'SNACKS'],
            min_categories=5,
            max_same_product=3,
            category_diversity=0.8
        ),
        
        TripPurpose.FILL_IN: TripCharacteristics(
            basket_size_mean=config.trip_fill_in_basket_mean,
            basket_size_std=config.trip_fill_in_basket_std,
            min_items=max(1, int(config.trip_fill_in_basket_mean - config.trip_fill_in_basket_std * 1.5)),
            max_items=int(config.trip_fill_in_basket_mean + config.trip_fill_in_basket_std * 2.5),
            required_categories=['DAIRY', 'PRODUCE'],
            optional_categories=['BAKERY', 'BEVERAGE', 'SNACKS'],
            min_categories=2,
            max_same_product=2,
            category_diversity=0.4
        ),
        
        TripPurpose.MEAL_PREP: TripCharacteristics(
            basket_size_mean=config.trip_meal_prep_basket_mean,
            basket_size_std=config.trip_meal_prep_basket_std,
            min_items=max(1, int(config.trip_meal_prep_basket_mean - config.trip_meal_prep_basket_std * 2)),
            max_items=int(config.trip_meal_prep_basket_mean + config.trip_meal_prep_basket_std * 2.5),
            required_categories=['MEAT', 'PRODUCE', 'GROCERY'],
            optional_categories=['DAIRY', 'FROZEN', 'DELI'],
            min_categories=3,
            max_same_product=2,
            category_diversity=0.6
        ),
        
        TripPurpose.CONVENIENCE: TripCharacteristics(
            basket_size_mean=config.trip_convenience_basket_mean,
            basket_size_std=config.trip_convenience_basket_std,
            min_items=1,
            max_items=int(config.trip_convenience_basket_mean + config.trip_convenience_basket_std * 2),
            required_categories=[],  # No required categories
            optional_categories=['BEVERAGE', 'SNACKS', 'DAIRY', 'BAKERY'],
            min_categories=1,
            max_same_product=2,
            category_diversity=0.2
        ),
        
        TripPurpose.SPECIAL_OCCASION: TripCharacteristics(
            basket_size_mean=config.trip_special_basket_mean,
            basket_size_std=config.trip_special_basket_std,
            min_items=max(1, int(config.trip_special_basket_mean - config.trip_special_basket_std * 2)),
            max_items=int(config.trip_special_basket_mean + config.trip_special_basket_std * 3),
            required_categories=['MEAT', 'PRODUCE', 'BEVERAGE'],
            optional_categories=['FROZEN', 'BAKERY', 'DELI', 'SNACKS', 'DAIRY'],
            min_categories=4,
            max_same_product=4,
            category_diversity=0.7
        )
    }


def build_trip_probabilities_from_config(config):
    """Build TRIP_PURPOSE_PROBABILITIES dict from config parameters"""
    # Calculate remaining probabilities to ensure they sum to 1.0
    # If configured values sum > 1.0, normalize them to prevent negative remainders
    
    # Price anchor - normalize if needed
    pa_sum = (config.trip_prob_price_anchor_stock_up + 
              config.trip_prob_price_anchor_fill_in + 
              config.trip_prob_price_anchor_convenience)
    if pa_sum > 1.0:
        # Normalize to 0.9 to leave 10% for remaining
        scale = 0.9 / pa_sum
        pa_stock = config.trip_prob_price_anchor_stock_up * scale
        pa_fill = config.trip_prob_price_anchor_fill_in * scale
        pa_conv = config.trip_prob_price_anchor_convenience * scale
        price_anchor_remaining = 0.1
    else:
        pa_stock = config.trip_prob_price_anchor_stock_up
        pa_fill = config.trip_prob_price_anchor_fill_in
        pa_conv = config.trip_prob_price_anchor_convenience
        price_anchor_remaining = 1.0 - pa_sum
    
    price_anchor_meal = max(0.0, price_anchor_remaining * 0.6)  # 60% meal prep
    price_anchor_special = max(0.0, price_anchor_remaining * 0.4)  # 40% special
    
    # Convenience - normalize if needed
    conv_sum = (config.trip_prob_convenience_convenience + 
                config.trip_prob_convenience_fill_in + 
                config.trip_prob_convenience_stock_up)
    if conv_sum > 1.0:
        scale = 0.9 / conv_sum
        conv_conv = config.trip_prob_convenience_convenience * scale
        conv_fill = config.trip_prob_convenience_fill_in * scale
        conv_stock = config.trip_prob_convenience_stock_up * scale
        convenience_remaining = 0.1
    else:
        conv_conv = config.trip_prob_convenience_convenience
        conv_fill = config.trip_prob_convenience_fill_in
        conv_stock = config.trip_prob_convenience_stock_up
        convenience_remaining = 1.0 - conv_sum
    
    convenience_meal = max(0.0, convenience_remaining * 0.5)
    convenience_special = max(0.0, convenience_remaining * 0.5)
    
    # Planned - normalize if needed
    plan_sum = (config.trip_prob_planned_stock_up + 
                config.trip_prob_planned_meal_prep + 
                config.trip_prob_planned_fill_in)
    if plan_sum > 1.0:
        scale = 0.9 / plan_sum
        plan_stock = config.trip_prob_planned_stock_up * scale
        plan_meal = config.trip_prob_planned_meal_prep * scale
        plan_fill = config.trip_prob_planned_fill_in * scale
        planned_remaining = 0.1
    else:
        plan_stock = config.trip_prob_planned_stock_up
        plan_meal = config.trip_prob_planned_meal_prep
        plan_fill = config.trip_prob_planned_fill_in
        planned_remaining = 1.0 - plan_sum
    
    planned_convenience = max(0.0, planned_remaining * 0.4)
    planned_special = max(0.0, planned_remaining * 0.6)
    
    # Impulse - normalize if needed
    imp_sum = (config.trip_prob_impulse_convenience + 
               config.trip_prob_impulse_fill_in + 
               config.trip_prob_impulse_special)
    if imp_sum > 1.0:
        scale = 0.9 / imp_sum
        imp_conv = config.trip_prob_impulse_convenience * scale
        imp_fill = config.trip_prob_impulse_fill_in * scale
        imp_special = config.trip_prob_impulse_special * scale
        impulse_remaining = 0.1
    else:
        imp_conv = config.trip_prob_impulse_convenience
        imp_fill = config.trip_prob_impulse_fill_in
        imp_special = config.trip_prob_impulse_special
        impulse_remaining = 1.0 - imp_sum
    
    impulse_stock = max(0.0, impulse_remaining * 0.4)
    impulse_meal = max(0.0, impulse_remaining * 0.6)
    
    return {
        'price_anchor': {
            TripPurpose.STOCK_UP: pa_stock,
            TripPurpose.FILL_IN: pa_fill,
            TripPurpose.CONVENIENCE: pa_conv,
            TripPurpose.MEAL_PREP: price_anchor_meal,
            TripPurpose.SPECIAL_OCCASION: price_anchor_special
        },
        'convenience': {
            TripPurpose.CONVENIENCE: conv_conv,
            TripPurpose.FILL_IN: conv_fill,
            TripPurpose.STOCK_UP: conv_stock,
            TripPurpose.MEAL_PREP: convenience_meal,
            TripPurpose.SPECIAL_OCCASION: convenience_special
        },
        'planned': {
            TripPurpose.STOCK_UP: plan_stock,
            TripPurpose.MEAL_PREP: plan_meal,
            TripPurpose.FILL_IN: plan_fill,
            TripPurpose.CONVENIENCE: planned_convenience,
            TripPurpose.SPECIAL_OCCASION: planned_special
        },
        'impulse': {
            TripPurpose.CONVENIENCE: imp_conv,
            TripPurpose.FILL_IN: imp_fill,
            TripPurpose.SPECIAL_OCCASION: imp_special,
            TripPurpose.STOCK_UP: impulse_stock,
            TripPurpose.MEAL_PREP: impulse_meal
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
    
    def __init__(self, config=None):
        """Initialize with config-based trip characteristics and probabilities
        
        Args:
            config: EnhancedRetailConfig instance. If None, uses legacy hardcoded values.
        """
        if config is not None:
            # Build from config parameters (NEW - replaces hardcoded values)
            self.trip_characteristics = build_trip_characteristics_from_config(config)
            self.trip_probabilities = build_trip_probabilities_from_config(config)
        else:
            # Fallback: Use legacy hardcoded values (should rarely happen)
            print("⚠️  WARNING: TripPurposeSelector initialized without config, using legacy hardcoded values")
            # Create minimal defaults
            from retailsynth.config import EnhancedRetailConfig
            default_config = EnhancedRetailConfig()
            self.trip_characteristics = build_trip_characteristics_from_config(default_config)
            self.trip_probabilities = build_trip_probabilities_from_config(default_config)
    
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
        
        # Use Gamma distribution (better for count-like continuous data)
        # Gamma is naturally right-skewed and doesn't require truncation
        shape = (chars.basket_size_mean / chars.basket_size_std) ** 2
        scale = chars.basket_size_std ** 2 / chars.basket_size_mean
        size = int(np.random.gamma(shape, scale))
        
        # Clip to min/max (soft boundaries, not artificial truncation)
        size = np.clip(size, chars.min_items, chars.max_items)
        
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
