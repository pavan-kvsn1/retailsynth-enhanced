import numpy as np
import jax.numpy as jnp
from jax import jit
from jax import random
from typing import Dict
from functools import partial
import jax

# ============================================================================
# GPU-ACCELERATED UTILITY ENGINE (v3.3, v3.4, v3.5)
# ============================================================================

class GPUUtilityEngine:
    """
    GPU-accelerated utility engine with fixed JIT compilation (v3.5).
    Combines speed of v3.3 with stability fixes from v3.5.
    """
    
    def __init__(self, calibration_params: Dict):
        self.calib = calibration_params
    
    @partial(jit, static_argnums=(0,))
    def compute_all_utilities_gpu(self,
                                  prices: jnp.ndarray,
                                  beta_price: jnp.ndarray,
                                  brand_prefs: jnp.ndarray,
                                  beta_brand: jnp.ndarray,
                                  promo_flags: jnp.ndarray,
                                  beta_promo: jnp.ndarray,
                                  role_prefs: jnp.ndarray,
                                  beta_role: jnp.ndarray,
                                  random_key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Compute utilities for ALL customers × ALL products in parallel.
        JIT-compiled for maximum performance.
        """
        # Price utility: β_price[i] × ln(price[j])
        log_prices = jnp.log(prices + 1e-6)
        price_utility = jnp.outer(beta_price, log_prices)
        
        # Brand utility: β_brand[i] × brand_pref[i,j]
        brand_utility = beta_brand[:, None] * brand_prefs
        
        # Promotion utility: β_promo[i] × promo_flag[j]
        promo_utility = jnp.outer(beta_promo, promo_flags.astype(jnp.float32))
        
        # Role utility: β_role[i] × role_pref[i,j]
        role_utility = beta_role[:, None] * role_prefs
        
        # Total utility
        base_utility = -1.0
        total_utility = base_utility + price_utility + brand_utility + promo_utility + role_utility
        
        # Add Gumbel noise for multinomial logit
        noise = jax.random.gumbel(random_key, shape=total_utility.shape) * 0.6
        
        return total_utility + noise
    
    @partial(jit, static_argnums=(0,))
    def compute_store_visit_probabilities_gpu(self,
                                             days_since_visit: jnp.ndarray,
                                             loyalty_levels: jnp.ndarray,
                                             random_key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Compute store visit probabilities for all customers in parallel.
        JIT-compiled for performance.
        """
        base_prob = 0.5
        loyalty_boost = loyalty_levels * 0.2
        days_factor = jnp.minimum(0.3, days_since_visit / 30.0)
        visit_probs = base_prob + loyalty_boost + days_factor
        return jnp.clip(visit_probs, 0.0, 1.0)
    
    def sample_product_choices_numpy(self,
                                    utilities: np.ndarray,
                                    n_choices_per_customer: np.ndarray) -> np.ndarray:
        """
        Sample product choices using numpy (v3.5 fix).
        NOT JIT-compiled due to dynamic shapes.
        """
        n_customers, n_products = utilities.shape
        max_choices = int(np.max(n_choices_per_customer))
        
        choices = np.zeros((n_customers, max_choices), dtype=np.int32)
        
        # Compute softmax probabilities
        exp_utilities = np.exp(utilities - np.max(utilities, axis=1, keepdims=True))
        probs = exp_utilities / np.sum(exp_utilities, axis=1, keepdims=True)
        
        # Sample for each customer
        for i in range(n_customers):
            n_choices = int(n_choices_per_customer[i])
            if n_choices > 0:
                available_probs = probs[i].copy()
                for j in range(min(n_choices, max_choices)):
                    if np.sum(available_probs) > 0:
                        available_probs = available_probs / np.sum(available_probs)
                        choice = np.random.choice(n_products, p=available_probs)
                        choices[i, j] = choice
                        available_probs[choice] = 0
        
        return choices

