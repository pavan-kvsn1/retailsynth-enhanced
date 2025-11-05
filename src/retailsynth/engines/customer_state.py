"""
Customer State Tracking System (Sprint 1.3)

Tracks individual customer purchase history, brand loyalty, inventory levels,
and habit formation to enable realistic state-dependent shopping behavior.

Key Features:
- Purchase history tracking (last purchase, frequency)
- Brand loyalty accumulation
- Category-level inventory depletion
- Habit strength formation
- Variety-seeking behavior
- Efficient state updates

Author: RetailSynth Team
Date: November 2024
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
from pathlib import Path


@dataclass
class CustomerState:
    """
    Tracks complete purchase history and behavioral state for a single customer
    
    State Components:
    1. Purchase History: What, when, how often
    2. Brand Loyalty: Accumulated satisfaction per brand
    3. Inventory: Category-level stock estimates
    4. Habits: Product-specific habit strength
    5. Variety-Seeking: Exploration behavior
    """
    
    customer_id: int
    current_week: int = 0
    
    # ========== Purchase History ==========
    # Track when each product was last purchased
    last_purchase_week: Dict[int, int] = field(default_factory=dict)  # product_id -> week
    
    # Count total purchases per product
    purchase_count: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Track first purchase week (for tenure calculation)
    first_purchase_week: Dict[int, int] = field(default_factory=dict)
    
    # ========== Brand Loyalty ==========
    # Accumulated satisfaction per brand (increases with positive experiences)
    brand_experience: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    # Number of purchases per brand
    brand_purchase_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Last purchase week per brand
    brand_last_purchase: Dict[str, int] = field(default_factory=dict)
    
    # ========== Category Inventory ==========
    # Current inventory level per category (0.0 = empty, 1.0 = full)
    category_inventory: Dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 0.5))
    
    # Last purchase week per category
    last_category_purchase: Dict[str, int] = field(default_factory=dict)
    
    # ========== Habit Formation ==========
    # Habit strength per product (0.0 = no habit, 1.0 = strong habit)
    habit_strength: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    
    # Consecutive purchase streak (for habit reinforcement)
    purchase_streak: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # ========== Variety-Seeking ==========
    # Weeks since tried a new product
    weeks_since_new_product: int = 0
    
    # Set of all products ever purchased
    products_tried: Set[int] = field(default_factory=set)
    
    # Variety-seeking tendency (0.0 = loyal, 1.0 = explorer)
    variety_seeking_score: float = 0.1  # Default 10% exploration
    
    # ========== Satisfaction Tracking ==========
    # Average satisfaction per product (for loyalty calculation)
    product_satisfaction: Dict[int, float] = field(default_factory=lambda: defaultdict(lambda: 0.5))
    
    # Recent satisfaction history (last 5 purchases)
    recent_satisfaction: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def update_after_purchase(
        self,
        product_id: int,
        brand: str,
        category: str,
        week: int,
        satisfaction: float = 0.7,
        quantity: int = 1
    ):
        """
        Update customer state after a purchase
        
        Args:
            product_id: ID of purchased product
            brand: Brand name
            category: Product category
            week: Current week number
            satisfaction: Satisfaction with purchase (0-1)
            quantity: Quantity purchased
        """
        self.current_week = week
        
        # 1. Update purchase history
        self.last_purchase_week[product_id] = week
        self.purchase_count[product_id] += 1
        
        if product_id not in self.first_purchase_week:
            self.first_purchase_week[product_id] = week
            self.products_tried.add(product_id)
            self.weeks_since_new_product = 0
        
        # 2. Update brand loyalty
        self.brand_purchase_count[brand] += 1
        self.brand_last_purchase[brand] = week
        
        # Accumulate brand experience (satisfaction weighted by recency)
        self.brand_experience[brand] += satisfaction * 1.0
        
        # 3. Update category inventory (replenish based on quantity)
        replenish_amount = min(quantity * 0.25, 1.0)  # Each unit adds 25% inventory
        self.category_inventory[category] = min(
            self.category_inventory[category] + replenish_amount,
            1.0
        )
        self.last_category_purchase[category] = week
        
        # 4. Update habit strength
        self._update_habit_strength(product_id)
        
        # 5. Update satisfaction tracking
        self.product_satisfaction[product_id] = (
            0.7 * self.product_satisfaction[product_id] + 0.3 * satisfaction
        )
        
        # Keep last 5 satisfaction scores
        self.recent_satisfaction[product_id].append(satisfaction)
        if len(self.recent_satisfaction[product_id]) > 5:
            self.recent_satisfaction[product_id].pop(0)
    
    def _update_habit_strength(self, product_id: int):
        """
        Update habit strength based on purchase frequency
        
        Habit formation:
        - 1-2 purchases: No habit (0.0)
        - 3-5 purchases: Weak habit (0.2-0.4)
        - 6-10 purchases: Moderate habit (0.5-0.7)
        - 11+ purchases: Strong habit (0.8-1.0)
        """
        count = self.purchase_count[product_id]
        
        if count <= 2:
            self.habit_strength[product_id] = 0.0
        elif count <= 5:
            self.habit_strength[product_id] = 0.2 + (count - 2) * 0.1
        elif count <= 10:
            self.habit_strength[product_id] = 0.5 + (count - 5) * 0.05
        else:
            self.habit_strength[product_id] = min(0.8 + (count - 10) * 0.02, 1.0)
        
        # Update streak
        self.purchase_streak[product_id] += 1
    
    def decay_state(self, weeks_elapsed: int = 1):
        """
        Apply time-based decay to state components
        
        Args:
            weeks_elapsed: Number of weeks since last update
        """
        decay_rate = 0.05 * weeks_elapsed  # 5% decay per week
        
        # 1. Decay brand loyalty (loyalty fades without purchases)
        for brand in self.brand_experience:
            self.brand_experience[brand] *= (1.0 - decay_rate)
        
        # 2. Decay habit strength (habits weaken without reinforcement)
        for product_id in self.habit_strength:
            weeks_since = self.current_week - self.last_purchase_week.get(product_id, 0)
            if weeks_since > 4:  # Start decaying after 4 weeks
                decay = 0.1 * (weeks_since - 4)
                self.habit_strength[product_id] *= (1.0 - min(decay, 0.5))
        
        # 3. Increment variety-seeking counter
        self.weeks_since_new_product += weeks_elapsed
    
    def deplete_inventory(self, depletion_rates: Dict[str, float]):
        """
        Deplete category inventory based on consumption rates
        
        Args:
            depletion_rates: Dict mapping category -> weekly depletion rate
        """
        for category, rate in depletion_rates.items():
            if category in self.category_inventory:
                self.category_inventory[category] = max(
                    0.0,
                    self.category_inventory[category] - rate
                )
    
    def get_loyalty_bonus(self, product_id: int, brand: str) -> float:
        """
        Calculate loyalty bonus for a product
        
        Returns:
            Utility bonus in range [0, 3.0]
        """
        # 1. Brand loyalty component
        brand_loyalty = self.brand_experience.get(brand, 0.0)
        loyalty_bonus = min(brand_loyalty / 10.0, 2.0)  # Cap at +2.0
        
        # 2. Habit strength component
        habit = self.habit_strength.get(product_id, 0.0)
        habit_bonus = habit * 1.0  # Up to +1.0
        
        # 3. Recency bonus (bought recently → higher utility)
        weeks_since = self.current_week - self.last_purchase_week.get(product_id, 999)
        if weeks_since < 4:
            recency_bonus = 0.3 * (4 - weeks_since) / 4  # Up to +0.3
        else:
            recency_bonus = 0.0
        
        return loyalty_bonus + habit_bonus + recency_bonus
    
    def get_inventory_need(self, category: str, assortment_role: str) -> float:
        """
        Calculate inventory-based need for a category
        
        Args:
            category: Product category
            assortment_role: One of lpg_line, front_basket, mid_basket, back_basket
        
        Returns:
            Utility bonus in range [0, 5.0]
        """
        current_inventory = self.category_inventory.get(category, 0.5)
        
        # Urgency increases as inventory depletes
        if current_inventory < 0.2:
            urgency = 5.0  # Critical need
        elif current_inventory < 0.4:
            urgency = 3.0  # High need
        elif current_inventory < 0.6:
            urgency = 1.5  # Moderate need
        else:
            urgency = 0.0  # No immediate need
        
        # High-frequency items (lpg_line, front_basket) have higher urgency
        if assortment_role in ['lpg_line', 'front_basket']:
            urgency *= 1.2
        
        return urgency
    
    def should_try_new_product(self) -> bool:
        """
        Determine if customer should explore a new product
        
        Returns:
            True if customer is in exploration mode
        """
        # Variety-seeking increases with time since last new product
        exploration_probability = min(
            self.variety_seeking_score + (self.weeks_since_new_product * 0.02),
            0.4  # Cap at 40%
        )
        
        return np.random.random() < exploration_probability
    
    def get_state_summary(self) -> Dict:
        """Get summary statistics of customer state"""
        return {
            'customer_id': self.customer_id,
            'current_week': self.current_week,
            'total_products_purchased': len(self.products_tried),
            'total_purchases': sum(self.purchase_count.values()),
            'num_brands_tried': len(self.brand_purchase_count),
            'avg_brand_loyalty': np.mean(list(self.brand_experience.values())) if self.brand_experience else 0.0,
            'avg_habit_strength': np.mean(list(self.habit_strength.values())) if self.habit_strength else 0.0,
            'avg_category_inventory': np.mean(list(self.category_inventory.values())),
            'weeks_since_new_product': self.weeks_since_new_product
        }


class CustomerStateManager:
    """
    Manages states for all customers in the simulation
    
    Provides efficient batch operations and state persistence
    """
    
    def __init__(self, n_customers: int):
        """
        Initialize state manager
        
        Args:
            n_customers: Number of customers to track
        """
        self.n_customers = n_customers
        self.states: Dict[int, CustomerState] = {}
        
        # Initialize all customer states
        for customer_id in range(n_customers):
            self.states[customer_id] = CustomerState(customer_id=customer_id)
    
    def get_state(self, customer_id: int) -> CustomerState:
        """Get state for a specific customer"""
        return self.states[customer_id]
    
    def update_all_states(self, current_week: int):
        """
        Update all customer states for the current week
        
        Args:
            current_week: Current simulation week
        """
        for state in self.states.values():
            state.current_week = current_week
    
    def decay_all_states(self, weeks_elapsed: int = 1):
        """Apply decay to all customer states"""
        for state in self.states.values():
            state.decay_state(weeks_elapsed)
    
    def deplete_all_inventory(self, depletion_rates: Dict[str, float]):
        """Deplete inventory for all customers"""
        for state in self.states.values():
            state.deplete_inventory(depletion_rates)
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for all customers
        
        Returns:
            DataFrame with one row per customer
        """
        summaries = [state.get_state_summary() for state in self.states.values()]
        return pd.DataFrame(summaries)
    
    def save_states(self, filepath: str):
        """
        Save all customer states to disk
        
        Args:
            filepath: Path to save pickle file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.states, f)
        print(f"✅ Saved {len(self.states)} customer states to {filepath}")
    
    def load_states(self, filepath: str):
        """
        Load customer states from disk
        
        Args:
            filepath: Path to pickle file
        """
        with open(filepath, 'rb') as f:
            self.states = pickle.load(f)
        print(f"✅ Loaded {len(self.states)} customer states from {filepath}")
    
    def checkpoint(self, checkpoint_dir: str, week: int):
        """
        Create checkpoint of current states
        
        Args:
            checkpoint_dir: Directory for checkpoints
            week: Current week number
        """
        checkpoint_path = Path(checkpoint_dir) / f"customer_states_week_{week}.pkl"
        self.save_states(str(checkpoint_path))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def initialize_customer_states(n_customers: int) -> CustomerStateManager:
    """
    Initialize customer state manager for simulation
    
    Args:
        n_customers: Number of customers
    
    Returns:
        CustomerStateManager instance
    """
    return CustomerStateManager(n_customers)


def get_depletion_rates_by_assortment() -> Dict[str, float]:
    """
    Get weekly inventory depletion rates by assortment role
    
    Based on industry-standard retail assortment roles:
    - lpg_line: High frequency staples (milk, bread) - 30% per week
    - front_basket: Planned purchases - 25% per week
    - mid_basket: Regular purchases - 15% per week
    - back_basket: Occasional purchases - 5% per week
    
    Returns:
        Dict mapping assortment_role -> weekly depletion rate
    """
    return {
        'lpg_line': 0.30,      # Buy every 3-4 weeks
        'front_basket': 0.25,  # Buy every 4 weeks
        'mid_basket': 0.15,    # Buy every 6-7 weeks
        'back_basket': 0.05    # Buy every 20 weeks
    }
