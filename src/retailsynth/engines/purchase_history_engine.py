"""
Purchase History Engine (Sprint 1.3)

Calculates history-dependent utility adjustments based on:
- Brand loyalty from past purchases
- Product-specific habits
- Category inventory needs
- Variety-seeking behavior
- Inter-purchase timing

Integrates with CustomerState to provide realistic state-dependent shopping.

Author: RetailSynth Team
Date: November 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .customer_state import CustomerState, get_depletion_rates_by_assortment


class PurchaseHistoryEngine:
    """
    Engine for calculating history-dependent utility adjustments
    
    Modifies base utilities based on:
    1. Brand loyalty (positive experiences increase utility)
    2. Habit strength (repeated purchases create habits)
    3. Inventory needs (low stock triggers repurchase)
    4. Variety-seeking (occasional exploration of new products)
    5. Recency effects (recently purchased items more likely)
    """
    
    def __init__(
        self,
        products: pd.DataFrame,
        loyalty_weight: float = 0.3,
        habit_weight: float = 0.4,
        inventory_weight: float = 0.5,
        variety_weight: float = 0.2,
        price_memory_weight: float = 0.1,
        inventory_depletion_rate: float = 0.1,
        replenishment_threshold: float = 0.3
    ):
        """
        Initialize purchase history engine
        
        Args:
            products: Product catalog with PRODUCT_ID, BRAND, COMMODITY, assortment_role
            loyalty_weight: Weight for loyalty bonus (default 0.3)
            habit_weight: Weight for habit strength (default 0.4)
            inventory_weight: Weight for inventory need (default 0.5)
            variety_weight: Weight for variety-seeking (default 0.2)
            price_memory_weight: Weight for reference price effect (default 0.1)
            inventory_depletion_rate: Daily inventory depletion rate (default 0.1)
            replenishment_threshold: Inventory level that triggers restocking (default 0.3)
        """
        self.products = products
        self.loyalty_weight = loyalty_weight
        self.habit_weight = habit_weight
        self.inventory_weight = inventory_weight
        self.variety_weight = variety_weight
        self.price_memory_weight = price_memory_weight
        self.inventory_depletion_rate = inventory_depletion_rate
        self.replenishment_threshold = replenishment_threshold
        
        # Create product lookup dictionaries
        self._build_product_mappings()
        
        # Get depletion rates by assortment role (using config parameter)
        self.depletion_rates = get_depletion_rates_by_assortment(inventory_depletion_rate)
    
    def _build_product_mappings(self):
        """Build efficient lookup dictionaries for product attributes"""
        self.product_to_brand = dict(zip(
            self.products['product_id'],
            self.products['brand']
        ))
        
        self.product_to_category = dict(zip(
            self.products['product_id'],
            self.products['commodity_desc']
        ))
        
        self.product_to_assortment = dict(zip(
            self.products['product_id'],
            self.products.get('assortment_role', 'mid_basket')
        ))
    
    def calculate_history_utility(
        self,
        customer_state: CustomerState,
        product_ids: np.ndarray,
        base_utilities: np.ndarray,
        current_week: int,
        current_prices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate history-dependent utility adjustments
        
        Args:
            customer_state: Customer's purchase history state
            product_ids: Array of product IDs
            base_utilities: Base utility values (before history adjustment)
            current_week: Current week number
            current_prices: Optional array of current prices for price memory effect
        
        Returns:
            Adjusted utilities incorporating history effects
        """
        n_products = len(product_ids)
        adjusted_utilities = base_utilities.copy()
        
        for i, product_id in enumerate(product_ids):
            # Get product attributes
            brand = self.product_to_brand.get(product_id, 'Unknown')
            category = self.product_to_category.get(product_id, 'Unknown')
            assortment_role = self.product_to_assortment.get(product_id, 'mid_basket')
            
            # 1. Loyalty bonus (brand + habit + recency)
            loyalty_bonus = customer_state.get_loyalty_bonus(product_id, brand)
            adjusted_utilities[i] += self.loyalty_weight * loyalty_bonus
            
            # 2. Inventory need (category-level urgency)
            inventory_need = customer_state.get_inventory_need(category, assortment_role, self.replenishment_threshold)
            adjusted_utilities[i] += self.inventory_weight * inventory_need
            
            # 3. Price memory effect (if prices provided)
            if current_prices is not None:
                price_memory = customer_state.get_price_memory_effect(product_id, current_prices[i])
                # Use configurable weight for price memory
                adjusted_utilities[i] += self.price_memory_weight * price_memory
            
            # 4. Variety-seeking penalty (reduce utility of frequently purchased)
            if product_id in customer_state.products_tried:
                # Reduce utility slightly for already-tried products
                purchase_count = customer_state.purchase_count.get(product_id, 0)
                if purchase_count > 5:  # Satiation after 5+ purchases
                    satiation_penalty = min(purchase_count * 0.05, 0.5)
                    adjusted_utilities[i] -= self.variety_weight * satiation_penalty
            else:
                # Boost utility for new products if in exploration mode
                if customer_state.should_try_new_product():
                    novelty_bonus = 0.5
                    adjusted_utilities[i] += self.variety_weight * novelty_bonus
        
        return adjusted_utilities
    
    def update_customer_after_purchase(
        self,
        customer_state: CustomerState,
        product_id: int,
        week: int,
        price: float,
        base_price: float,
        quantity: int = 1
    ):
        """
        Update customer state after a purchase
        
        Args:
            customer_state: Customer's state to update
            product_id: Purchased product ID
            week: Current week
            price: Actual price paid
            base_price: Regular price
            quantity: Quantity purchased
        """
        # Get product attributes
        brand = self.product_to_brand.get(product_id, 'Unknown')
        category = self.product_to_category.get(product_id, 'Unknown')
        
        # Calculate satisfaction (higher for discounts)
        discount_pct = (base_price - price) / base_price if base_price > 0 else 0.0
        satisfaction = 0.5 + min(discount_pct * 2.0, 0.4)  # 0.5-0.9 range
        
        # Update state
        customer_state.update_after_purchase(
            product_id=product_id,
            brand=brand,
            category=category,
            week=week,
            satisfaction=satisfaction,
            quantity=quantity,
            price=price
        )
    
    def get_repeat_purchase_probability(
        self,
        customer_state: CustomerState,
        product_id: int
    ) -> float:
        """
        Calculate probability of repeat purchase
        
        Args:
            customer_state: Customer state
            product_id: Product to evaluate
        
        Returns:
            Probability in range [0, 1]
        """
        # Base probability
        base_prob = 0.1
        
        # Increase with purchase count
        purchase_count = customer_state.purchase_count.get(product_id, 0)
        if purchase_count == 0:
            return 0.0
        
        # Loyalty component
        brand = self.product_to_brand.get(product_id, 'Unknown')
        brand_loyalty = customer_state.brand_experience.get(brand, 0.0)
        loyalty_boost = min(brand_loyalty / 20.0, 0.3)
        
        # Habit component
        habit_strength = customer_state.habit_strength.get(product_id, 0.0)
        habit_boost = habit_strength * 0.4
        
        # Recency component
        weeks_since = customer_state.current_week - customer_state.last_purchase_week.get(product_id, 999)
        if weeks_since < 8:
            recency_boost = 0.2 * (8 - weeks_since) / 8
        else:
            recency_boost = 0.0
        
        repeat_prob = base_prob + loyalty_boost + habit_boost + recency_boost
        return min(repeat_prob, 0.95)  # Cap at 95%
    
    def get_brand_switching_probability(
        self,
        customer_state: CustomerState,
        current_brand: str,
        alternative_brand: str
    ) -> float:
        """
        Calculate probability of switching brands
        
        Args:
            customer_state: Customer state
            current_brand: Currently preferred brand
            alternative_brand: Alternative brand to consider
        
        Returns:
            Switching probability in range [0, 1]
        """
        # Base switching probability
        base_switch_prob = 0.15
        
        # Loyalty to current brand reduces switching
        current_loyalty = customer_state.brand_experience.get(current_brand, 0.0)
        loyalty_barrier = min(current_loyalty / 15.0, 0.4)
        
        # Experience with alternative brand increases switching
        alt_experience = customer_state.brand_experience.get(alternative_brand, 0.0)
        alt_attraction = min(alt_experience / 20.0, 0.3)
        
        switch_prob = base_switch_prob - loyalty_barrier + alt_attraction
        return np.clip(switch_prob, 0.0, 0.8)
    
    def get_category_purchase_timing(
        self,
        customer_state: CustomerState,
        category: str,
        assortment_role: str
    ) -> int:
        """
        Estimate weeks until next category purchase
        
        Args:
            customer_state: Customer state
            category: Product category
            assortment_role: Assortment role (lpg_line, front_basket, etc.)
        
        Returns:
            Estimated weeks until next purchase
        """
        # Get depletion rate
        depletion_rate = self.depletion_rates.get(assortment_role, 0.15)
        
        # Get current inventory
        current_inventory = customer_state.category_inventory.get(category, 0.5)
        
        # Calculate weeks until inventory depletes to 20% (repurchase threshold)
        if current_inventory <= 0.2:
            return 0  # Need to buy now
        
        weeks_until_repurchase = (current_inventory - 0.2) / depletion_rate
        return int(np.ceil(weeks_until_repurchase))
    
    def analyze_customer_loyalty(
        self,
        customer_state: CustomerState
    ) -> Dict[str, float]:
        """
        Analyze customer loyalty patterns
        
        Args:
            customer_state: Customer state
        
        Returns:
            Dictionary with loyalty metrics
        """
        # Brand concentration (Herfindahl index)
        total_purchases = sum(customer_state.brand_purchase_count.values())
        if total_purchases == 0:
            brand_concentration = 0.0
        else:
            brand_shares = [
                count / total_purchases 
                for count in customer_state.brand_purchase_count.values()
            ]
            brand_concentration = sum(share ** 2 for share in brand_shares)
        
        # Average brand loyalty
        avg_loyalty = np.mean(list(customer_state.brand_experience.values())) if customer_state.brand_experience else 0.0
        
        # Habit strength
        avg_habit = np.mean(list(customer_state.habit_strength.values())) if customer_state.habit_strength else 0.0
        
        # Variety-seeking score
        variety_score = customer_state.variety_seeking_score
        
        return {
            'brand_concentration': brand_concentration,
            'avg_brand_loyalty': avg_loyalty,
            'avg_habit_strength': avg_habit,
            'variety_seeking_score': variety_score,
            'num_brands_tried': len(customer_state.brand_purchase_count),
            'num_products_tried': len(customer_state.products_tried)
        }


class InterPurchaseTimingModel:
    """
    Model inter-purchase timing based on category and customer behavior
    
    Predicts when customers will repurchase products based on:
    - Category consumption rates
    - Historical purchase patterns
    - Inventory depletion
    """
    
    def __init__(self, products: pd.DataFrame):
        """
        Initialize timing model
        
        Args:
            products: Product catalog
        """
        self.products = products
        self.depletion_rates = get_depletion_rates_by_assortment()
        
        # Expected inter-purchase intervals by assortment role
        self.expected_intervals = {
            'lpg_line': 3.5,      # ~3-4 weeks (milk, bread)
            'front_basket': 4.0,  # ~4 weeks (planned staples)
            'mid_basket': 6.5,    # ~6-7 weeks (regular items)
            'back_basket': 20.0   # ~20 weeks (occasional)
        }
    
    def predict_next_purchase_week(
        self,
        customer_state: CustomerState,
        product_id: int,
        current_week: int
    ) -> int:
        """
        Predict when customer will next purchase product
        
        Args:
            customer_state: Customer state
            product_id: Product to predict
            current_week: Current week
        
        Returns:
            Predicted week of next purchase
        """
        # Get last purchase week
        last_purchase = customer_state.last_purchase_week.get(product_id, 0)
        
        # Get product assortment role
        product_row = self.products[self.products['PRODUCT_ID'] == product_id]
        if len(product_row) == 0:
            return current_week + 4  # Default 4 weeks
        
        assortment_role = product_row.iloc[0].get('assortment_role', 'mid_basket')
        
        # Get expected interval
        expected_interval = self.expected_intervals.get(assortment_role, 6.5)
        
        # Adjust for habit strength (strong habits → shorter intervals)
        habit_strength = customer_state.habit_strength.get(product_id, 0.0)
        interval_adjustment = 1.0 - (habit_strength * 0.3)  # Up to 30% shorter
        
        adjusted_interval = expected_interval * interval_adjustment
        
        # Add some randomness
        actual_interval = np.random.normal(adjusted_interval, adjusted_interval * 0.2)
        actual_interval = max(1, int(actual_interval))
        
        return last_purchase + actual_interval
    
    def get_purchase_probability_by_week(
        self,
        customer_state: CustomerState,
        product_id: int,
        current_week: int
    ) -> float:
        """
        Get probability of purchase in current week
        
        Args:
            customer_state: Customer state
            product_id: Product to evaluate
            current_week: Current week
        
        Returns:
            Purchase probability [0, 1]
        """
        predicted_week = self.predict_next_purchase_week(
            customer_state, product_id, current_week
        )
        
        # Probability peaks at predicted week
        weeks_diff = abs(current_week - predicted_week)
        
        if weeks_diff == 0:
            return 0.8  # High probability at predicted week
        elif weeks_diff == 1:
            return 0.4  # Moderate probability ±1 week
        elif weeks_diff == 2:
            return 0.2  # Low probability ±2 weeks
        else:
            return 0.05  # Very low probability otherwise


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_repeat_purchase_rate(transactions: pd.DataFrame) -> float:
    """
    Calculate repeat purchase rate from transaction data
    
    Args:
        transactions: DataFrame with customer_id, product_id
    
    Returns:
        Repeat purchase rate (0-1)
    """
    customer_product_counts = transactions.groupby(['customer_id', 'product_id']).size()
    repeat_purchases = (customer_product_counts > 1).sum()
    total_customer_products = len(customer_product_counts)
    
    return repeat_purchases / total_customer_products if total_customer_products > 0 else 0.0


def calculate_brand_loyalty_metrics(
    transactions: pd.DataFrame,
    products: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate brand loyalty metrics by customer
    
    Args:
        transactions: Transaction data
        products: Product catalog with BRAND column
    
    Returns:
        DataFrame with loyalty metrics per customer
    """
    # Merge to get brands
    trans_with_brands = transactions.merge(
        products[['PRODUCT_ID', 'BRAND']],
        on='PRODUCT_ID',
        how='left'
    )
    
    # Calculate brand concentration per customer
    loyalty_metrics = []
    
    for customer_id in trans_with_brands['customer_id'].unique():
        customer_trans = trans_with_brands[trans_with_brands['customer_id'] == customer_id]
        
        # Brand purchase counts
        brand_counts = customer_trans['BRAND'].value_counts()
        total_purchases = len(customer_trans)
        
        # Herfindahl index (brand concentration)
        brand_shares = brand_counts / total_purchases
        herfindahl = (brand_shares ** 2).sum()
        
        # Top brand share
        top_brand_share = brand_shares.iloc[0] if len(brand_shares) > 0 else 0.0
        
        loyalty_metrics.append({
            'customer_id': customer_id,
            'brand_concentration': herfindahl,
            'top_brand_share': top_brand_share,
            'num_brands': len(brand_counts)
        })
    
    return pd.DataFrame(loyalty_metrics)
