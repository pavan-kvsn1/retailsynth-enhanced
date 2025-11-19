"""
Arc Price Elasticity Engine (Sprint 2.3)

This module implements intertemporal price effects (arc elasticity).
Customers form expectations about future prices and adjust current purchases accordingly:
- Stock up when prices are low (anticipate future high prices)
- Defer when prices are high (anticipate future promotions)

Arc elasticity = Expected utility from buying now vs. later

Modeled as:
    U(buy_now) = immediate_utility + β * E[future_utility | current_price]

Where β is discount factor and E[future_utility] depends on price expectations
learned from HMM.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class ArcPriceElasticityEngine:
    """
    Model intertemporal price effects (arc elasticity)
    
    Customers form expectations about future prices based on:
    1. HMM state (if in discount state, expect return to regular)
    2. Purchase history (inventory depletion rate)
    3. Product category (stockpiling propensity)
    
    Attributes:
        price_hmm: HMM model for price state transitions
        inventory_decay_rate: Weekly inventory depletion rate
        future_discount_factor: Discount factor for future utility
    """
    
    def __init__(self, 
                 price_hmm,
                 inventory_decay_rate: float = 0.25,
                 future_discount_factor: float = 0.95):
        """
        Initialize arc elasticity engine
        
        Args:
            price_hmm: PriceStateHMM instance with learned parameters
            inventory_decay_rate: Weekly inventory depletion (default: 25%)
            future_discount_factor: Weekly discount factor (default: 0.95)
        """
        self.price_hmm = price_hmm
        self.inventory_decay_rate = inventory_decay_rate
        self.future_discount_factor = future_discount_factor
        
        logger.info(f"Initialized ArcPriceElasticityEngine")
        logger.info(f"  Inventory decay rate: {inventory_decay_rate:.1%}/week")
        logger.info(f"  Future discount factor: {future_discount_factor:.3f}")
    
    def save_parameters(self, output_path: Path):
        """
        Save arc elasticity parameters to file
        
        Note: The HMM model is saved separately. This only saves configuration.
        
        Args:
            output_path: Path to save parameters (will save as .pkl)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration parameters (HMM is saved separately)
        params = {
            'inventory_decay_rate': self.inventory_decay_rate,
            'future_discount_factor': self.future_discount_factor,
            'version': '1.0'
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(params, f)
        
        logger.info(f"✅ Arc elasticity parameters saved to {output_path}")
    
    @classmethod
    def load_parameters(cls, params_path: Path, price_hmm):
        """
        Load arc elasticity parameters from file
        
        Args:
            params_path: Path to saved parameters
            price_hmm: PriceStateHMM instance (must be loaded separately)
        
        Returns:
            ArcPriceElasticityEngine instance
        """
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
        
        instance = cls(
            price_hmm=price_hmm,
            inventory_decay_rate=params['inventory_decay_rate'],
            future_discount_factor=params['future_discount_factor']
        )
        
        logger.info(f"✅ Arc elasticity parameters loaded from {params_path}")
        return instance
    
    def calculate_arc_effect(self,
                            product_id: int,
                            current_price: float,
                            current_state: int,
                            customer_inventory: float,
                            weeks_since_purchase: int,
                            product_category: Optional[str] = None) -> float:
        """
        Calculate utility adjustment from arc elasticity
        
        Args:
            product_id: Product being evaluated
            current_price: Current price
            current_state: Current HMM price state (0=regular, 1=feature, 2=deep)
            customer_inventory: Estimated inventory level (0-1 scale)
            weeks_since_purchase: Weeks since last purchase
            product_category: Product category for stockpiling propensity
        
        Returns:
            Utility adjustment (positive = buy now, negative = defer)
        """
        
        # Component 1: Inventory urgency
        # As inventory depletes, urgency to purchase increases
        inventory_urgency = self._calculate_inventory_urgency(
            customer_inventory,
            weeks_since_purchase
        )
        
        # Component 2: Price expectation
        # Predict future price based on HMM state transitions
        expected_future_price = self._predict_future_price(
            product_id,
            current_price,
            current_state,
            horizon=4  # 4 weeks ahead
        )
        
        # Price advantage: current vs. expected future
        price_advantage = (expected_future_price - current_price) / (current_price + 0.01)
        
        # Component 3: Stockpiling incentive
        # If current price much lower than expected, incentive to buy extra
        stockpile_bonus = self._calculate_stockpile_bonus(
            price_advantage,
            current_state,
            product_category
        )
        
        # Component 4: Deferral penalty
        # If high price and sufficient inventory, defer purchase
        deferral_penalty = self._calculate_deferral_penalty(
            price_advantage,
            customer_inventory,
            current_state
        )
        
        # Total arc effect (weighted combination)
        arc_adjustment = (
            inventory_urgency * 2.0 +      # Inventory most important
            price_advantage * 1.5 +         # Price expectations
            stockpile_bonus +               # Stockpiling behavior
            deferral_penalty                # Deferral behavior
        )
        
        return arc_adjustment
    
    def _calculate_inventory_urgency(self,
                                    customer_inventory: float,
                                    weeks_since_purchase: int) -> float:
        """
        Calculate urgency to purchase based on inventory level
        
        Args:
            customer_inventory: Current inventory (0-1)
            weeks_since_purchase: Weeks since last purchase
        
        Returns:
            Urgency score (negative = high urgency)
        """
        # Low inventory = high urgency (negative value increases utility)
        base_urgency = -customer_inventory
        
        # If haven't purchased in a while, increase urgency
        time_urgency = min(weeks_since_purchase / 10.0, 0.5)
        
        return base_urgency + time_urgency
    
    def _predict_future_price(self,
                             product_id: int,
                             current_price: float,
                             current_state: int,
                             horizon: int = 4) -> float:
        """
        Predict expected price in 'horizon' weeks using HMM
        
        Uses transition matrix to compute expected future state,
        then emission distribution for that state.
        
        Args:
            product_id: Product ID
            current_price: Current price
            current_state: Current HMM state
            horizon: Weeks ahead to predict
        
        Returns:
            Expected future price
        """
        if product_id not in self.price_hmm.transition_matrices:
            # Default expectation: prices revert to mean (slightly higher)
            return current_price * 1.10
        
        transition_matrix = self.price_hmm.transition_matrices[product_id]
        emission_dists = self.price_hmm.emission_distributions[product_id]
        
        # Determine actual number of states from transition matrix
        n_actual_states = transition_matrix.shape[0]
        
        # Compute state distribution after 'horizon' steps
        # P(state_t+horizon | state_t) = transition_matrix ^ horizon
        state_probs = np.zeros(n_actual_states)
        state_probs[current_state] = 1.0
        
        for _ in range(horizon):
            state_probs = state_probs @ transition_matrix
        
        # Expected price = weighted average across states
        expected_price = 0.0
        for state in range(n_actual_states):
            if state in emission_dists:
                state_price = emission_dists[state]['price_mean']
                expected_price += state_probs[state] * state_price
        
        return expected_price
    
    def _calculate_stockpile_bonus(self,
                                  price_advantage: float,
                                  current_state: int,
                                  product_category: Optional[str] = None) -> float:
        """
        Calculate bonus for stockpiling behavior
        
        Args:
            price_advantage: (future_price - current_price) / current_price
            current_state: Current HMM state
            product_category: Product category
        
        Returns:
            Stockpile bonus (positive if good deal)
        """
        # Only stockpile if significant price advantage
        if price_advantage < 0.15:  # Future price not much higher
            return 0.0
        
        # Base stockpile incentive
        stockpile_bonus = 0.5 * price_advantage
        
        # Deep discount state increases stockpiling
        if current_state == 2:  # Deep discount
            stockpile_bonus *= 1.5
        
        # Category-specific stockpiling propensity
        # (Some categories like canned goods stockpile more)
        if product_category:
            stockpile_multiplier = self._get_stockpile_propensity(product_category)
            stockpile_bonus *= stockpile_multiplier
        
        return stockpile_bonus
    
    def _calculate_deferral_penalty(self,
                                   price_advantage: float,
                                   customer_inventory: float,
                                   current_state: int) -> float:
        """
        Calculate penalty for deferring purchase
        
        Args:
            price_advantage: (future_price - current_price) / current_price
            customer_inventory: Current inventory level
            current_state: Current HMM state
        
        Returns:
            Deferral penalty (negative if should defer)
        """
        # Only defer if current price high and inventory sufficient
        if price_advantage >= -0.10 or customer_inventory < 0.3:
            return 0.0
        
        # Current price high, expect lower future price
        # Sufficient inventory to wait
        deferral_penalty = -0.3
        
        # If in regular state (not on sale), more likely to defer
        if current_state == 0:
            deferral_penalty *= 1.2
        
        return deferral_penalty
    
    def _get_stockpile_propensity(self, product_category: str) -> float:
        """
        Get stockpiling propensity for product category
        
        Args:
            product_category: Product category/commodity
        
        Returns:
            Multiplier for stockpiling (1.0 = average)
        """
        # Categories with high stockpiling propensity
        high_stockpile = [
            'SOFT DRINKS', 'CANNED GOODS', 'PAPER PRODUCTS',
            'CEREAL', 'SNACKS', 'PASTA', 'RICE'
        ]
        
        # Categories with low stockpiling propensity
        low_stockpile = [
            'PRODUCE', 'DAIRY', 'MEAT', 'BAKERY', 'DELI'
        ]
        
        category_upper = product_category.upper()
        
        if any(cat in category_upper for cat in high_stockpile):
            return 1.5
        elif any(cat in category_upper for cat in low_stockpile):
            return 0.5
        else:
            return 1.0
    
    def update_customer_inventory(self,
                                  customer_id: int,
                                  product_id: int,
                                  quantity_purchased: int,
                                  current_inventory: float,
                                  weeks_elapsed: int = 1) -> float:
        """
        Update customer's estimated inventory after purchase
        
        Simple model:
        inventory_new = min(1.0, current_inventory * (1-decay)^weeks + quantity_normalized)
        
        Args:
            customer_id: Customer ID (for future personalization)
            product_id: Product ID
            quantity_purchased: Quantity purchased
            current_inventory: Current inventory level (0-1)
            weeks_elapsed: Weeks since last update
        
        Returns:
            Updated inventory level (0-1)
        """
        # Normalize quantity (assume 1 unit = 1 week consumption)
        quantity_normalized = quantity_purchased / 1.0
        
        # Decay current inventory
        decay_factor = (1 - self.inventory_decay_rate) ** weeks_elapsed
        inventory_after_decay = current_inventory * decay_factor
        
        # Add purchase
        new_inventory = min(1.0, inventory_after_decay + quantity_normalized)
        
        return new_inventory
    
    def get_expected_purchase_timing(self,
                                    product_id: int,
                                    current_price: float,
                                    current_state: int,
                                    customer_inventory: float) -> int:
        """
        Estimate when customer is likely to purchase next
        
        Args:
            product_id: Product ID
            current_price: Current price
            current_state: Current HMM state
            customer_inventory: Current inventory level
        
        Returns:
            Expected weeks until next purchase
        """
        # Base timing from inventory depletion
        weeks_until_depletion = int(customer_inventory / self.inventory_decay_rate)
        
        # Adjust based on price expectations
        expected_future_price = self._predict_future_price(
            product_id, current_price, current_state
        )
        
        price_advantage = (expected_future_price - current_price) / (current_price + 0.01)
        
        # If good deal now, purchase sooner
        if price_advantage > 0.15:
            weeks_until_depletion = max(0, weeks_until_depletion - 2)
        # If bad deal now, defer
        elif price_advantage < -0.10:
            weeks_until_depletion += 2
        
        return max(0, weeks_until_depletion)
    
    def get_stockpile_quantity(self,
                              product_id: int,
                              current_price: float,
                              current_state: int,
                              base_quantity: int = 1) -> int:
        """
        Determine how many units customer should buy (stockpiling)
        
        Args:
            product_id: Product ID
            current_price: Current price
            current_state: Current HMM state
            base_quantity: Base quantity (default: 1)
        
        Returns:
            Recommended purchase quantity
        """
        # Predict future price
        expected_future_price = self._predict_future_price(
            product_id, current_price, current_state
        )

        # Calculate price advantage
        price_advantage = (expected_future_price - current_price) / (current_price + 0.0000001)
        
        print("Expected future price:", expected_future_price)
        print("Price advantage:", price_advantage)
        # Stockpile if significant discount
        if price_advantage < -0.20:  # Future price 20%+ higher
            if current_state == 2:  # Deep discount
                return base_quantity * 3  # Buy 3x
            else:
                return base_quantity * 2  # Buy 2x
        elif price_advantage < -0.10:
            return base_quantity * 2  # Buy 2x
        elif price_advantage < -0.05:
            return base_quantity * 1.5  # Buy 1.5x
        else:
            return base_quantity  # Buy normal amount