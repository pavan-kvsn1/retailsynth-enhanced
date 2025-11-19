"""
Marketing Signal Calculator (Phase 2.3)

Calculates marketing signal strength from promotional activity to influence store visit probability.

Key Components:
1. Signal strength from discount depth
2. Signal boost from displays (end caps, features)
3. Signal boost from advertising (in-ad, mailer)
4. Store-level promotional intensity
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MarketingSignalCalculator:
    """
    Calculates marketing signal strength from promotional context
    
    Signal influences customer store visit probability.
    Stronger signals = more likely to visit store that week.
    
    Components:
    - Discount depth (deeper = stronger signal)
    - Display prominence (end caps > features > shelf tags)
    - Advertising reach (in-ad + mailer)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize marketing signal calculator
        
        Args:
            config: Optional configuration dict with weights
        """
        self.config = config or {}
        
        # Signal component weights
        self.weights = {
            'discount_depth': self.config.get('discount_weight', 0.4),
            'display_prominence': self.config.get('display_weight', 0.3),
            'advertising_reach': self.config.get('advertising_weight', 0.3)
        }
        
        # Display type signal multipliers
        self.display_multipliers = {
            'feature_display': 1.0,   # Strongest signal
            'end_cap': 0.7,           # Strong signal
            'shelf_tag': 0.3,         # Moderate signal
            'none': 0.0               # No display signal
        }
        
        # Advertising reach multipliers
        self.ad_multipliers = {
            'in_ad_and_mailer': 1.0,  # Both channels
            'in_ad_only': 0.7,        # Single channel (higher reach)
            'mailer_only': 0.5,       # Single channel (targeted)
            'none': 0.0               # No advertising
        }
        
        logger.info("MarketingSignalCalculator initialized")
    
    def calculate_signal_strength(self, promo_context) -> float:
        """
        Calculate overall marketing signal strength for a store-week
        
        Args:
            promo_context: StorePromoContext with promotional data
        
        Returns:
            float: Signal strength [0.0, 1.0]
                  0.0 = no promotional activity
                  1.0 = maximum promotional intensity
        """
        if not promo_context.promoted_products:
            return 0.0
        
        # Component 1: Discount depth signal
        discount_signal = self._calculate_discount_signal(promo_context)
        
        # Component 2: Display prominence signal
        display_signal = self._calculate_display_signal(promo_context)
        
        # Component 3: Advertising reach signal
        advertising_signal = self._calculate_advertising_signal(promo_context)
        
        # Weighted combination
        total_signal = (
            self.weights['discount_depth'] * discount_signal +
            self.weights['display_prominence'] * display_signal +
            self.weights['advertising_reach'] * advertising_signal
        )
        
        # Normalize to [0, 1]
        total_signal = np.clip(total_signal, 0.0, 1.0)
        
        logger.debug(f"Store {promo_context.store_id}, Week {promo_context.week_number}: "
                    f"Signal strength = {total_signal:.3f} "
                    f"(discount={discount_signal:.3f}, display={display_signal:.3f}, "
                    f"ads={advertising_signal:.3f})")
        
        return total_signal
    
    def _calculate_discount_signal(self, promo_context) -> float:
        """
        Calculate signal from discount depths
        
        Logic: Deeper discounts = stronger signal
        """
        if not promo_context.promo_depths:
            return 0.0
        
        # Average discount depth across all promos
        avg_discount = promo_context.avg_discount_depth
        
        # Normalize: 0% = 0.0, 50%+ = 1.0
        # Use sigmoid-like curve for realistic scaling
        signal = min(avg_discount / 0.5, 1.0)
        
        # Boost for deep discounts
        if promo_context.n_deep_discounts > 0:
            deep_boost = min(promo_context.n_deep_discounts / 10.0, 0.2)
            signal = min(signal + deep_boost, 1.0)
        
        return signal
    
    def _calculate_display_signal(self, promo_context) -> float:
        """
        Calculate signal from display prominence
        
        Logic: Prominent displays = stronger signal
        """
        if not promo_context.display_types:
            return 0.0
        
        # Count displays by type
        display_counts = {
            'feature_display': len(promo_context.feature_display_products),
            'end_cap': len(promo_context.end_cap_products),
            'shelf_tag': sum(1 for d in promo_context.display_types.values() if d == 'shelf_tag'),
            'none': sum(1 for d in promo_context.display_types.values() if d == 'none')
        }
        
        # Weighted average of display types
        total_displays = sum(display_counts.values())
        if total_displays == 0:
            return 0.0
        
        weighted_signal = sum(
            count * self.display_multipliers[display_type]
            for display_type, count in display_counts.items()
        ) / total_displays
        
        return weighted_signal
    
    def _calculate_advertising_signal(self, promo_context) -> float:
        """
        Calculate signal from advertising reach
        
        Logic: More advertising channels = stronger signal
        """
        n_in_ad = len(promo_context.in_ad_products)
        n_mailer = len(promo_context.mailer_products)
        
        if n_in_ad == 0 and n_mailer == 0:
            return 0.0
        
        # Categorize products by advertising coverage
        in_ad_set = set(promo_context.in_ad_products)
        mailer_set = set(promo_context.mailer_products)
        
        n_both = len(in_ad_set & mailer_set)  # In both channels
        n_in_ad_only = len(in_ad_set - mailer_set)
        n_mailer_only = len(mailer_set - in_ad_set)
        
        total_advertised = n_both + n_in_ad_only + n_mailer_only
        if total_advertised == 0:
            return 0.0
        
        # Weighted average by channel combination
        weighted_signal = (
            n_both * self.ad_multipliers['in_ad_and_mailer'] +
            n_in_ad_only * self.ad_multipliers['in_ad_only'] +
            n_mailer_only * self.ad_multipliers['mailer_only']
        ) / total_advertised
        
        # Scale by coverage (% of promos that are advertised)
        coverage = total_advertised / max(len(promo_context.promoted_products), 1)
        signal = weighted_signal * coverage
        
        return signal
    
    def calculate_visit_probability_boost(self, signal_strength: float, 
                                         base_probability: float = 0.3) -> float:
        """
        Calculate boost to store visit probability from marketing signal
        
        Args:
            signal_strength: Marketing signal strength [0.0, 1.0]
            base_probability: Base visit probability without promotions
        
        Returns:
            float: Boosted visit probability
        
        Logic:
            - Weak signal (0.0-0.3): +5-15% boost
            - Medium signal (0.3-0.6): +15-30% boost
            - Strong signal (0.6-1.0): +30-50% boost
        """
        # Convert signal to boost multiplier (1.05 to 1.50)
        boost_multiplier = 1.0 + (signal_strength * 0.5)
        
        # Apply boost to base probability
        boosted_prob = base_probability * boost_multiplier
        
        # Cap at 0.95 (realistic maximum)
        boosted_prob = min(boosted_prob, 0.95)
        
        return boosted_prob
    
    def get_signal_breakdown(self, promo_context) -> Dict:
        """
        Get detailed breakdown of signal components for analysis
        
        Returns:
            Dict with component signals and total signal
        """
        discount_signal = self._calculate_discount_signal(promo_context)
        display_signal = self._calculate_display_signal(promo_context)
        advertising_signal = self._calculate_advertising_signal(promo_context)
        total_signal = self.calculate_signal_strength(promo_context)
        
        return {
            'store_id': promo_context.store_id,
            'week_number': promo_context.week_number,
            'discount_signal': discount_signal,
            'display_signal': display_signal,
            'advertising_signal': advertising_signal,
            'total_signal': total_signal,
            'n_promotions': len(promo_context.promoted_products),
            'avg_discount': promo_context.avg_discount_depth,
            'n_end_caps': len(promo_context.end_cap_products),
            'n_features': len(promo_context.feature_display_products),
            'n_in_ad': len(promo_context.in_ad_products),
            'n_mailer': len(promo_context.mailer_products)
        }
