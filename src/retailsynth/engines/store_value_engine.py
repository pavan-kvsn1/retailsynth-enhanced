"""
Store Value (SV) Engine - Implements Bain et al.'s Inclusive Value Calculation

This engine calculates the "store inclusive value" (SV) which represents the expected
utility a customer gets from visiting a store, based on the available products and
their utilities.

Key Components:
1. Store Value (SV): log(sum(exp(CV_ct)) for c in categories)
2. Visit Utility: X_store = γ₀ + γ₁*SV_{t-1} + β*Marketing
3. Recursive Probability: P(Visit_t) = θ*P(Visit_{t-1}) + (1-θ)*Logit(X_store)

This creates a self-reinforcing loop:
- Better products → Higher SV → More visits → Experience SV → More likely to return

Reference: Bain et al. (2023) "A Dynamic Model of Grocery Shopping"
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple
import numpy as np


class StoreValueEngine:
    """
    Calculates store inclusive value and recursive visit probabilities
    
    This implements the Bain paper's approach where:
    1. Customer utilities → Store value (SV)
    2. SV + Marketing → Visit utility
    3. Visit utility + Memory → Visit probability (recursive)
    """
    
    def __init__(self, config):
        """
        Initialize store value engine
        
        Args:
            config: EnhancedRetailConfig with store value parameters
        """
        self.config = config
        
        # Store value → visit utility weights (Bain equation 4)
        self.gamma_0 = getattr(config, 'store_base_utility', 0.5)           # Base store utility
        self.gamma_1 = getattr(config, 'store_value_weight', 0.6)           # SV → visit utility weight
        self.beta_marketing = getattr(config, 'marketing_visit_weight', 0.4) # Marketing → visit weight
        
        # Recursive probability parameter (Bain equation 5)
        self.theta = getattr(config, 'visit_memory_weight', 0.3)            # Memory weight (0-1)
        
        print(f"   StoreValueEngine initialized:")
        print(f"      γ₀ (base utility): {self.gamma_0:.2f}")
        print(f"      γ₁ (SV weight): {self.gamma_1:.2f}")
        print(f"      β (marketing weight): {self.beta_marketing:.2f}")
        print(f"      θ (memory): {self.theta:.2f}")
    
    @partial(jit, static_argnums=(0, 3))
    def compute_store_value_gpu(self,
                               product_utilities: jnp.ndarray,
                               product_categories: jnp.ndarray,
                               n_categories: int) -> jnp.ndarray:
        """
        Calculate store inclusive value (SV) for each customer
        
        SV = log(sum(exp(CV_c)) for c in categories)
        CV_c = log(sum(exp(utility_p)) for p in category c)
        
        This is a nested log-sum-exp calculation that captures the
        "option value" of having many good choices in each category.
        
        Args:
            product_utilities: [n_customers, n_products] - utility of each product
            product_categories: [n_products] - category index for each product (0 to n_categories-1)
            n_categories: Number of unique categories
        
        Returns:
            store_values: [n_customers] - SV for each customer
        """
        n_customers = product_utilities.shape[0]
        n_products = product_utilities.shape[1]
        
        # Step 1: Calculate category values (CV) using log-sum-exp per category
        # CV_c = log(sum(exp(u_p)) for p in category c)
        category_values = jnp.zeros((n_customers, n_categories))
        
        for cat_idx in range(n_categories):
            # Create mask for products in this category
            cat_mask = product_categories == cat_idx
            
            # Set non-category products to -inf (exp(-inf) = 0, doesn't contribute)
            cat_utilities = jnp.where(cat_mask, product_utilities, -jnp.inf)
            
            # Numerical stability: log-sum-exp trick
            # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
            max_util = jnp.max(cat_utilities, axis=1, keepdims=True)
            max_util = jnp.where(jnp.isinf(max_util), 0.0, max_util)  # Handle empty categories
            
            # CV_c = max + log(sum(exp(utilities - max)))
            log_sum_exp = max_util.squeeze() + jnp.log(
                jnp.sum(jnp.exp(cat_utilities - max_util), axis=1) + 1e-10  # Small epsilon for stability
            )
            
            category_values = category_values.at[:, cat_idx].set(log_sum_exp)
        
        # Step 2: Store value = log-sum-exp across categories
        # SV = log(sum(exp(CV_c)) for c in categories)
        max_cv = jnp.max(category_values, axis=1, keepdims=True)
        max_cv = jnp.where(jnp.isinf(max_cv), 0.0, max_cv)
        
        store_values = max_cv.squeeze() + jnp.log(
            jnp.sum(jnp.exp(category_values - max_cv), axis=1) + 1e-10
        )
        
        return store_values
    
    @partial(jit, static_argnums=(0,))
    def compute_visit_utilities_gpu(self,
                                   store_values: jnp.ndarray,
                                   prev_store_values: jnp.ndarray,
                                   marketing_signals: jnp.ndarray,
                                   visited_last_period: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate visit utility from store value and marketing signals
        
        Bain equation 4:
        X_store_ut = γ₀ + γ₁*SV_{u,t-1} + β*Marketing_ut
        
        Key insight: Previous period's SV (if visited) affects this period's visit decision.
        This creates temporal dependence and self-reinforcement.
        
        Args:
            store_values: [n_customers] - Current period SV (for context)
            prev_store_values: [n_customers] - Previous period SV (if visited)
            marketing_signals: [n_customers] - Marketing signal strength (sum of discounts)
            visited_last_period: [n_customers] - Binary flag if visited last period
        
        Returns:
            visit_utilities: [n_customers] - Utility of visiting this period
        """
        # Only use previous SV if customer actually visited last period
        # If didn't visit, they don't have experience → sv_component = 0
        sv_component = jnp.where(visited_last_period > 0, prev_store_values, 0.0)
        
        # Bain's visit utility formula
        visit_utilities = (
            self.gamma_0 +                           # Base utility (everyone has this)
            self.gamma_1 * sv_component +            # Experience from last visit
            self.beta_marketing * marketing_signals  # Marketing pull (promotions, ads)
        )
        
        return visit_utilities
    
    @partial(jit, static_argnums=(0,))
    def compute_visit_probabilities_recursive_gpu(self,
                                                 visit_utilities: jnp.ndarray,
                                                 prev_visit_probs: jnp.ndarray,
                                                 customer_theta: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate recursive visit probabilities with memory
        
        Bain equation 5:
        P(Visit_ut) = θ_u * P(Visit_{u,t-1}) + (1-θ_u) * Logit(X_store_ut)
        
        Args:
            visit_utilities: [n_customers] - Visit utility X_store_ut
            prev_visit_probs: [n_customers] - Previous period visit probability
            customer_theta: [n_customers] - Memory weight per customer (0-1)
        
        Returns:
            visit_probs: [n_customers] - Probability of visiting this period
        """
        # Aggressive dampening to prevent saturation
        # Scale down utilities and add noise to prevent convergence
        dampened_utilities = visit_utilities * 0.3  # More aggressive scaling
        
        # Add small random noise to prevent perfect convergence
        noise = jax.random.normal(jax.random.PRNGKey(42), shape=dampened_utilities.shape) * 0.05
        dampened_utilities = dampened_utilities + noise
        
        # Logit transform: P = sigmoid(X) = exp(X) / (1 + exp(X))
        current_prob = jax.nn.sigmoid(dampened_utilities)
        
        # Memory decay: reduce previous probabilities to prevent accumulation
        # This prevents the "ratchet effect" where probabilities only go up
        memory_decay = 0.95  # Slight decay each period
        decayed_prev_probs = prev_visit_probs * memory_decay
        
        # Recursive mixture: weighted average of decayed past probability and current utility
        visit_probs = customer_theta * decayed_prev_probs + (1.0 - customer_theta) * current_prob
        
        # Additional safety: cap probabilities at 0.85 to prevent saturation
        return jnp.clip(visit_probs, 0.0, 0.85)
    
    def compute_marketing_signals(self,
                                 promo_depths: dict,
                                 customer_product_relevance: np.ndarray = None) -> np.ndarray:
        """
        Calculate marketing signal strength for each customer
        
        Marketing_ut = sum of discount depths for products relevant to customer
        
        Bain paper: "Marketing signal is the sum of all discounts available that week"
        This captures the promotional intensity that draws customers to the store.
        
        Args:
            promo_depths: {product_id: discount_depth} - Active promotions
            customer_product_relevance: [n_customers, n_products] - Optional relevance weights
        
        Returns:
            marketing_signals: [n_customers] - Signal strength per customer
        """
        if len(promo_depths) == 0:
            # No promotions this week
            return np.zeros(1) if customer_product_relevance is None else np.zeros(len(customer_product_relevance))
        
        # Simple version: sum all discount depths (same signal for all customers)
        total_discount = sum(promo_depths.values())
        
        if customer_product_relevance is None:
            return np.array([total_discount])
        else:
            # Future enhancement: weight by customer-product relevance
            # For now, everyone sees the same marketing signal
            n_customers = len(customer_product_relevance)
            return np.full(n_customers, total_discount)


class VisitStateTracker:
    """
    Tracks visit state across time periods for recursive probability calculation
    
    This maintains the "memory" that makes the Bain model recursive.
    """
    
    def __init__(self, n_customers: int, base_visit_probability: float = 0.5):
        """
        Initialize visit state tracker
        
        Args:
            n_customers: Number of customers to track
            base_visit_probability: Initial visit probability (t=0)
        """
        self.n_customers = n_customers
        
        # State arrays (initialized for t=0)
        self.prev_visit_probs = np.full(n_customers, base_visit_probability)
        self.prev_store_values = np.zeros(n_customers)
        self.visited_last_period = np.zeros(n_customers, dtype=bool)
        
        print(f"   VisitStateTracker initialized for {n_customers} customers")
    
    def update(self, 
               visited_customers: np.ndarray,
               store_values: np.ndarray,
               visit_probs: np.ndarray):
        """
        Update state after a period
        
        Args:
            visited_customers: [n_customers] - Binary array of who visited
            store_values: [n_customers] - Store values experienced this period
            visit_probs: [n_customers] - Visit probabilities for next period
        """
        self.visited_last_period = visited_customers.astype(bool)
        self.prev_store_values = store_values
        self.prev_visit_probs = visit_probs
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get current state for computing next period's visit probabilities
        
        Returns:
            (prev_visit_probs, prev_store_values, visited_last_period)
        """
        return (
            self.prev_visit_probs,
            self.prev_store_values,
            self.visited_last_period
        )
    
    def reset(self, base_visit_probability: float = 0.5):
        """Reset state (e.g., for new simulation)"""
        self.prev_visit_probs = np.full(self.n_customers, base_visit_probability)
        self.prev_store_values = np.zeros(self.n_customers)
        self.visited_last_period = np.zeros(self.n_customers, dtype=bool)
