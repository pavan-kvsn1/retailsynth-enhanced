"""
Promotional Response Calculator (Phase 2.5)

Calculates customer-specific promotional response using:
1. Individual promo responsiveness (from Phase 2.4 heterogeneity)
2. Marketing signal strength (from Phase 2.3)
3. Arc elasticity for discount sensitivity
4. Display and advertising receptivity

Key Innovation: Every customer responds differently to the same promotion!
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromoResponse:
    """
    Individual customer's response to a promotion
    """
    customer_id: int
    product_id: int
    base_utility: float
    promo_boost: float
    final_utility: float
    elasticity: float
    response_probability: float
    
    # Component breakdowns
    discount_boost: float
    display_boost: float
    advertising_boost: float
    signal_multiplier: float


class PromoResponseCalculator:
    """
    Calculates customer-specific promotional response
    
    Integrates:
    - Phase 2.4: Individual heterogeneity parameters
    - Phase 2.3: Marketing signal strength
    - Arc elasticity for price sensitivity
    - Display and advertising receptivity
    
    Key Concept: Same promotion → different response per customer!
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize promotional response calculator
        
        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        
        # Base elasticity parameters (will be modulated by individual params)
        self.base_price_elasticity = self.config.get('base_price_elasticity', -2.0)
        self.base_promo_elasticity = self.config.get('base_promo_elasticity', -3.5)
        
        # Response curve parameters
        self.discount_threshold = self.config.get('discount_threshold', 0.05)  # 5% minimum
        self.saturation_discount = self.config.get('saturation_discount', 0.50)  # 50% saturation
        
        # Boost multipliers for displays
        self.display_boosts = {
            'feature_display': 0.25,  # +25% utility boost
            'end_cap': 0.15,          # +15% utility boost
            'shelf_tag': 0.05,        # +5% utility boost
            'none': 0.0
        }
        
        # Boost multipliers for advertising
        self.advertising_boosts = {
            'in_ad_and_mailer': 0.20,  # +20% utility boost
            'in_ad_only': 0.12,        # +12% utility boost
            'mailer_only': 0.08,       # +8% utility boost
            'none': 0.0
        }
        
        logger.info("PromoResponseCalculator initialized")
    
    def calculate_promo_response(
        self,
        customer_params: Dict,
        base_utility: float,
        discount_depth: float,
        marketing_signal: float,
        display_type: str = 'none',
        advertising_type: str = 'none',
        product_id: Optional[int] = None
    ) -> PromoResponse:
        """
        Calculate individual customer's response to a promotion
        
        Args:
            customer_params: Customer heterogeneity parameters (from Phase 2.4)
            base_utility: Base utility without promotion
            discount_depth: Discount depth [0.0, 1.0] (e.g., 0.2 = 20% off)
            marketing_signal: Marketing signal strength [0.0, 1.0] (from Phase 2.3)
            display_type: Display type (feature_display, end_cap, shelf_tag, none)
            advertising_type: Advertising type (in_ad_and_mailer, in_ad_only, mailer_only, none)
            product_id: Optional product ID for logging
        
        Returns:
            PromoResponse object with detailed breakdown
        """
        customer_id = customer_params.get('customer_id', 0)
        
        # Extract individual parameters (from Phase 2.4)
        promo_responsiveness = customer_params.get('promo_responsiveness_param', 1.0)
        display_sensitivity = customer_params.get('display_sensitivity_param', 0.7)
        advertising_receptivity = customer_params.get('advertising_receptivity_param', 0.8)
        price_sensitivity = customer_params.get('price_sensitivity_param', 1.2)
        
        # Component 1: Discount boost (arc elasticity)
        discount_boost = self._calculate_discount_boost(
            discount_depth,
            promo_responsiveness,
            price_sensitivity
        )
        
        # Component 2: Display boost
        display_boost = self._calculate_display_boost(
            display_type,
            display_sensitivity
        )
        
        # Component 3: Advertising boost
        advertising_boost = self._calculate_advertising_boost(
            advertising_type,
            advertising_receptivity
        )
        
        # Component 4: Marketing signal multiplier (from Phase 2.3)
        signal_multiplier = self._calculate_signal_multiplier(
            marketing_signal,
            promo_responsiveness
        )
        
        # Combine components (multiplicative for synergy)
        base_promo_boost = discount_boost + display_boost + advertising_boost
        
        # Apply marketing signal multiplier
        total_promo_boost = base_promo_boost * signal_multiplier
        
        # Calculate final utility
        final_utility = base_utility + total_promo_boost
        
        # Calculate arc elasticity
        elasticity = self._calculate_arc_elasticity(
            discount_depth,
            promo_responsiveness,
            price_sensitivity
        )
        
        # Calculate response probability (utility → probability)
        response_probability = self._utility_to_probability(
            base_utility,
            final_utility
        )
        
        logger.debug(
            f"Customer {customer_id}, Product {product_id}: "
            f"Discount={discount_depth:.1%} → Boost={total_promo_boost:.3f}, "
            f"Elasticity={elasticity:.2f}, P(response)={response_probability:.3f}"
        )
        
        return PromoResponse(
            customer_id=customer_id,
            product_id=product_id or 0,
            base_utility=base_utility,
            promo_boost=total_promo_boost,
            final_utility=final_utility,
            elasticity=elasticity,
            response_probability=response_probability,
            discount_boost=discount_boost,
            display_boost=display_boost,
            advertising_boost=advertising_boost,
            signal_multiplier=signal_multiplier
        )
    
    def _calculate_discount_boost(
        self,
        discount_depth: float,
        promo_responsiveness: float,
        price_sensitivity: float
    ) -> float:
        """
        Calculate utility boost from discount using arc elasticity
        
        Individual customers have different discount sensitivity curves!
        
        Args:
            discount_depth: Discount [0.0, 1.0]
            promo_responsiveness: Customer promo responsiveness [0.5, 2.0]
            price_sensitivity: Customer price sensitivity [0.5, 2.5]
        
        Returns:
            Utility boost from discount
        """
        if discount_depth <= 0:
            return 0.0
        
        # Combine promo responsiveness and price sensitivity
        # High promo responsiveness = strong reaction to discounts
        # High price sensitivity = value discounts highly
        individual_sensitivity = (promo_responsiveness + price_sensitivity) / 2
        
        # Non-linear discount response (diminishing returns)
        # Small discounts have threshold effect, deep discounts saturate
        normalized_discount = discount_depth / self.saturation_discount
        
        # Use sigmoid-like curve for realistic response
        # S(x) = x / (1 + x) scaled by individual sensitivity
        discount_effect = normalized_discount / (1 + normalized_discount)
        
        # Apply individual sensitivity
        boost = discount_effect * individual_sensitivity * 0.5  # Scale to reasonable range
        
        return boost
    
    def _calculate_display_boost(
        self,
        display_type: str,
        display_sensitivity: float
    ) -> float:
        """
        Calculate utility boost from in-store displays
        
        Args:
            display_type: Display type
            display_sensitivity: Customer display sensitivity [0.3, 1.2]
        
        Returns:
            Utility boost from display
        """
        base_boost = self.display_boosts.get(display_type, 0.0)
        
        # Modulate by individual sensitivity
        individual_boost = base_boost * display_sensitivity
        
        return individual_boost
    
    def _calculate_advertising_boost(
        self,
        advertising_type: str,
        advertising_receptivity: float
    ) -> float:
        """
        Calculate utility boost from advertising
        
        Args:
            advertising_type: Advertising type
            advertising_receptivity: Customer ad receptivity [0.3, 1.5]
        
        Returns:
            Utility boost from advertising
        """
        base_boost = self.advertising_boosts.get(advertising_type, 0.0)
        
        # Modulate by individual receptivity
        individual_boost = base_boost * advertising_receptivity
        
        return individual_boost
    
    def _calculate_signal_multiplier(
        self,
        marketing_signal: float,
        promo_responsiveness: float
    ) -> float:
        """
        Calculate multiplier from marketing signal strength (Phase 2.3)
        
        Strong marketing signals amplify promotional response
        
        Args:
            marketing_signal: Signal strength [0.0, 1.0]
            promo_responsiveness: Customer promo responsiveness [0.5, 2.0]
        
        Returns:
            Signal multiplier [1.0, 2.0]
        """
        # Base multiplier: 1.0 = no signal, 2.0 = maximum signal
        base_multiplier = 1.0 + marketing_signal
        
        # Modulate by promo responsiveness
        # Highly responsive customers amplify strong signals more
        individual_multiplier = 1.0 + (marketing_signal * promo_responsiveness * 0.5)
        
        # Average the two approaches
        multiplier = (base_multiplier + individual_multiplier) / 2
        
        return np.clip(multiplier, 1.0, 2.5)
    
    def _calculate_arc_elasticity(
        self,
        discount_depth: float,
        promo_responsiveness: float,
        price_sensitivity: float
    ) -> float:
        """
        Calculate arc elasticity for the discount
        
        Arc elasticity = % change in quantity / % change in price
        
        For promotions, this measures how much demand increases
        per percentage point of discount
        
        Args:
            discount_depth: Discount [0.0, 1.0]
            promo_responsiveness: Customer promo responsiveness [0.5, 2.0]
            price_sensitivity: Customer price sensitivity [0.5, 2.5]
        
        Returns:
            Elasticity (negative value, more negative = more elastic)
        """
        if discount_depth <= 0:
            return 0.0
        
        # Individual elasticity based on parameters
        # More price-sensitive customers have higher (more negative) elasticity
        individual_elasticity = self.base_promo_elasticity * (
            (price_sensitivity + promo_responsiveness) / 2
        )
        
        # Arc elasticity depends on discount depth (non-linear)
        # Small discounts: steep response
        # Large discounts: diminishing returns
        if discount_depth < 0.15:
            # Steep response to small discounts
            elasticity = individual_elasticity * 1.2
        elif discount_depth < 0.30:
            # Moderate response
            elasticity = individual_elasticity * 1.0
        else:
            # Diminishing returns for deep discounts
            elasticity = individual_elasticity * 0.7
        
        return elasticity
    
    def _utility_to_probability(
        self,
        base_utility: float,
        promo_utility: float
    ) -> float:
        """
        Convert utility change to response probability
        
        Uses logistic function to convert utility difference to probability
        
        Args:
            base_utility: Utility without promotion
            promo_utility: Utility with promotion
        
        Returns:
            Probability of responding to promotion [0.0, 1.0]
        """
        utility_gain = promo_utility - base_utility
        
        # Logistic function: P = 1 / (1 + exp(-utility_gain))
        # Scaled to be more sensitive to moderate utility changes
        scaled_gain = utility_gain * 3.0  # Sensitivity factor
        
        probability = 1.0 / (1.0 + np.exp(-scaled_gain))
        
        return np.clip(probability, 0.0, 1.0)
    
    def calculate_population_response(
        self,
        customers_df: pd.DataFrame,
        base_utilities: np.ndarray,
        discount_depth: float,
        marketing_signal: float,
        display_types: Optional[np.ndarray] = None,
        advertising_types: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Calculate promotional response for entire population
        
        Vectorized for performance
        
        Args:
            customers_df: DataFrame with customer heterogeneity parameters
            base_utilities: Array of base utilities (one per customer)
            discount_depth: Discount depth [0.0, 1.0]
            marketing_signal: Marketing signal strength [0.0, 1.0]
            display_types: Optional array of display types per customer
            advertising_types: Optional array of advertising types per customer
        
        Returns:
            DataFrame with promotional response for each customer
        """
        n_customers = len(customers_df)
        
        if display_types is None:
            display_types = np.array(['none'] * n_customers)
        if advertising_types is None:
            advertising_types = np.array(['none'] * n_customers)
        
        responses = []
        
        for i in range(n_customers):
            customer_params = {
                'customer_id': customers_df.loc[i, 'customer_id'],
                'promo_responsiveness_param': customers_df.loc[i, 'promo_responsiveness'],
                'display_sensitivity_param': customers_df.loc[i, 'display_sensitivity'],
                'advertising_receptivity_param': customers_df.loc[i, 'advertising_receptivity'],
                'price_sensitivity_param': customers_df.loc[i, 'price_sensitivity']
            }
            
            response = self.calculate_promo_response(
                customer_params=customer_params,
                base_utility=base_utilities[i],
                discount_depth=discount_depth,
                marketing_signal=marketing_signal,
                display_type=display_types[i],
                advertising_type=advertising_types[i]
            )
            
            responses.append({
                'customer_id': response.customer_id,
                'base_utility': response.base_utility,
                'promo_boost': response.promo_boost,
                'final_utility': response.final_utility,
                'elasticity': response.elasticity,
                'response_probability': response.response_probability,
                'discount_boost': response.discount_boost,
                'display_boost': response.display_boost,
                'advertising_boost': response.advertising_boost,
                'signal_multiplier': response.signal_multiplier
            })
        
        return pd.DataFrame(responses)
    
    def get_elasticity_summary(
        self,
        customers_df: pd.DataFrame,
        discount_range: Tuple[float, float] = (0.05, 0.50),
        n_points: int = 10
    ) -> pd.DataFrame:
        """
        Calculate elasticity curves across discount range
        
        Shows how different customers respond to varying discounts
        
        Args:
            customers_df: DataFrame with customer parameters
            discount_range: (min, max) discount depths
            n_points: Number of discount levels to test
        
        Returns:
            DataFrame with elasticity at each discount level
        """
        discount_levels = np.linspace(discount_range[0], discount_range[1], n_points)
        
        elasticity_data = []
        
        for discount in discount_levels:
            for i in range(len(customers_df)):
                promo_resp = customers_df.loc[i, 'promo_responsiveness']
                price_sens = customers_df.loc[i, 'price_sensitivity']
                
                elasticity = self._calculate_arc_elasticity(
                    discount,
                    promo_resp,
                    price_sens
                )
                
                elasticity_data.append({
                    'customer_id': customers_df.loc[i, 'customer_id'],
                    'discount_depth': discount,
                    'elasticity': elasticity,
                    'promo_responsiveness': promo_resp,
                    'price_sensitivity': price_sens
                })
        
        return pd.DataFrame(elasticity_data)
