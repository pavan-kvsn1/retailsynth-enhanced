import numpy as np
import jax.numpy as jnp
from jax import jit
from jax import random
from typing import Dict, Optional, Tuple
from functools import partial
import jax

# Phase 2: Import StoreValueEngine for Bain recursive visit probability
from .store_value_engine import StoreValueEngine, VisitStateTracker

# ============================================================================
# GPU-ACCELERATED UTILITY ENGINE (v3.3, v3.4, v3.5, v4.1 Phase 2.6)
# ============================================================================

class GPUUtilityEngine:
    """
    GPU-accelerated utility engine with Phase 2.6 non-linear utilities.
    
    Supports two modes:
    - Linear utilities (legacy): β * price, β * quality
    - Non-linear utilities (Phase 2.6): log-price, quadratic quality, 
      reference prices, psychological thresholds
    
    Phase 2: Now includes Bain's recursive visit probability with store value (SV)
    """
    
    def __init__(self, calibration_params: Dict, 
                 enable_nonlinear: bool = True,
                 use_log_price: bool = True,
                 use_quadratic_quality: bool = True,
                 log_price_scale: float = 10.0,
                 quality_diminishing_factor: float = 0.8,
                 config = None,
                 n_customers: int = None):
        """
        Initialize utility engine
        
        Args:
            calibration_params: Calibration parameters
            enable_nonlinear: Use non-linear utilities (Phase 2.6)
            use_log_price: Use log-price instead of linear price
            use_quadratic_quality: Use quadratic quality utility
            log_price_scale: Scale factor for log transformation
            quality_diminishing_factor: Coefficient for Q² term
            config: Config object (needed for Phase 2 SV engine)
            n_customers: Number of customers (needed for state tracking)
        """
        self.calib = calibration_params
        self.enable_nonlinear = enable_nonlinear
        self.use_log_price = use_log_price
        self.use_quadratic_quality = use_quadratic_quality
        self.log_price_scale = log_price_scale
        self.quality_diminishing_factor = quality_diminishing_factor
        
        # Phase 2: Initialize StoreValueEngine if config provided
        self.config = config
        self.store_value_engine = None
        self.visit_state_tracker = None
        
        if config is not None:
            self.store_value_engine = StoreValueEngine(config)
            if n_customers is not None:
                base_visit_prob = getattr(config, 'base_visit_probability', 0.5)
                self.visit_state_tracker = VisitStateTracker(n_customers, base_visit_prob)
                print(f"   Phase 2: Recursive visit probability ENABLED")

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
        
        Phase 2.6: Uses non-linear utilities if enabled.
        """
        # PHASE 2.6: Log-price utility (replaces linear)
        if self.enable_nonlinear and self.use_log_price:
            # U = -β * scale * log(price)
            log_prices = jnp.log(prices + 1e-6)
            price_utility = -jnp.outer(beta_price, log_prices) * self.log_price_scale
        else:
            # Legacy linear: U = β * ln(price) [original implementation]
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
    
    def compute_all_utilities_gpu_with_quality(self,
                                               prices: jnp.ndarray,
                                               quality: jnp.ndarray,
                                               beta_price: jnp.ndarray,
                                               beta_quality: jnp.ndarray,
                                               brand_prefs: jnp.ndarray,
                                               beta_brand: jnp.ndarray,
                                               promo_flags: jnp.ndarray,
                                               beta_promo: jnp.ndarray,
                                               role_prefs: jnp.ndarray,
                                               beta_role: jnp.ndarray,
                                               random_key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Compute utilities with quality term (Phase 2.6).
        
        This variant includes product quality, which may be quadratic if enabled.
        """
        # PHASE 2.6: Log-price utility
        if self.enable_nonlinear and self.use_log_price:
            log_prices = jnp.log(prices + 1e-6)
            price_utility = -jnp.outer(beta_price, log_prices) * self.log_price_scale
        else:
            log_prices = jnp.log(prices + 1e-6)
            price_utility = jnp.outer(beta_price, log_prices)
        
        # PHASE 2.6: Quadratic quality utility
        if self.enable_nonlinear and self.use_quadratic_quality:
            # U = α*Q - γ*Q²
            linear_term = beta_quality[:, None] * quality[None, :]
            quadratic_term = -self.quality_diminishing_factor * beta_quality[:, None] * (quality[None, :] ** 2)
            quality_utility = linear_term + quadratic_term
        else:
            # Legacy linear: U = α * Q
            quality_utility = beta_quality[:, None] * quality[None, :]
        
        # Other utilities (same as before)
        brand_utility = beta_brand[:, None] * brand_prefs
        promo_utility = jnp.outer(beta_promo, promo_flags.astype(jnp.float32))
        role_utility = beta_role[:, None] * role_prefs
        
        # Total utility
        base_utility = -1.0
        total_utility = (base_utility + price_utility + quality_utility + 
                        brand_utility + promo_utility + role_utility)
        
        # Add Gumbel noise
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
        
        LEGACY METHOD: Simple additive formula (pre-Phase 2)
        Kept for backward compatibility. New code should use compute_visit_probabilities_with_sv().
        """
        base_prob = 0.5
        loyalty_boost = loyalty_levels * 0.2
        days_factor = jnp.minimum(0.3, days_since_visit / 30.0)
        visit_probs = base_prob + loyalty_boost + days_factor
        return jnp.clip(visit_probs, 0.0, 1.0)
    
    def compute_visit_probabilities_with_sv(self,
                                           product_utilities: jnp.ndarray,
                                           product_categories: jnp.ndarray,
                                           promo_context: dict,
                                           n_categories: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute visit probabilities using Bain's recursive mechanism with store value (Phase 2)
        
        This implements the full Bain model:
        1. Calculate store inclusive value (SV) from product utilities
        2. Get previous state (prev_visit_probs, prev_store_values, visited_last_period)
        3. Compute visit utilities from SV + marketing signals
        4. Calculate recursive probabilities with memory
        
        Args:
            product_utilities: [n_customers, n_products] - Current period utilities
            product_categories: [n_products] - Category index for each product
            promo_context: Dict with 'promo_depths' {product_id: discount}
            n_categories: Number of product categories
        
        Returns:
            visit_probs: [n_customers] - Probability of visiting this period
            store_values: [n_customers] - Store values (for next period's state)
        """
        if self.store_value_engine is None or self.visit_state_tracker is None:
            raise ValueError("Phase 2 not enabled! Initialize with config and n_customers.")
        
        n_customers = product_utilities.shape[0]
        
        # Step 1: Calculate current period store values (SV)
        store_values = self.store_value_engine.compute_store_value_gpu(
            product_utilities,
            product_categories,
            n_categories
        )
        
        # Step 2: Get previous state from tracker
        prev_visit_probs, prev_store_values, visited_last_period = self.visit_state_tracker.get_state()
        
        # Convert numpy arrays to JAX arrays
        prev_visit_probs_jax = jnp.array(prev_visit_probs)
        prev_store_values_jax = jnp.array(prev_store_values)
        visited_last_period_jax = jnp.array(visited_last_period, dtype=jnp.float32)
        
        # Step 3: Get marketing signal from promo_context
        # CRITICAL FIX: Use the marketing_signal calculated by promotional engine
        # instead of recalculating from promo_depths
        marketing_signal_strength = promo_context.get('marketing_signal', 0.0)
        
        # Convert to array for all customers
        # For now, same signal for all customers (store-level signal)
        # Future: could be customer-specific based on preferences
        marketing_signals_np = np.full(n_customers, marketing_signal_strength)
        marketing_signals_jax = jnp.array(marketing_signals_np)
        
        # Optional: Apply additional boost based on signal strength
        # Strong promotions → higher visit probability
        if marketing_signal_strength > 0:
            # Log the boost for debugging
            if hasattr(self, '_log_marketing_boost'):
                print(f"  Marketing signal: {marketing_signal_strength:.3f} → boosting visit probability")
        
        # Step 4: Compute visit utilities (γ₀ + γ₁*SV_{t-1} + β*Marketing)
        visit_utilities = self.store_value_engine.compute_visit_utilities_gpu(
            store_values,
            prev_store_values_jax,
            marketing_signals_jax,
            visited_last_period_jax
        )
        
        # LOGGING: Track utility components
        if marketing_signal_strength > 0.3:  # Log when signal is strong
            base_utility = self.store_value_engine.gamma_0
            sv_contribution = float((self.store_value_engine.gamma_1 * store_values).mean())
            marketing_contribution = float((self.store_value_engine.beta_marketing * marketing_signals_jax).mean())
            total_utility = float(visit_utilities.mean())
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Visit Utility Breakdown:")
            logger.info(f"  Base (γ₀): {base_utility:.3f}")
            logger.info(f"  SV contrib (γ₁*SV): {sv_contribution:.3f}")
            logger.info(f"  Marketing contrib (β*M): {marketing_contribution:.3f}")
            logger.info(f"  Total utility: {total_utility:.3f}")
            logger.info(f"  β_marketing: {self.store_value_engine.beta_marketing:.3f}")
        
        # Step 5: Calculate recursive probabilities (θ*P_{t-1} + (1-θ)*Logit(X))
        theta = jnp.full(n_customers, self.store_value_engine.theta)
        visit_probs = self.store_value_engine.compute_visit_probabilities_recursive_gpu(
            visit_utilities,
            prev_visit_probs_jax,
            theta
        )
        
        return visit_probs, store_values
    
    def compute_visit_rates_with_sv(self,
                                   product_utilities: jnp.ndarray,
                                   product_categories: jnp.ndarray,
                                   promo_context: dict,
                                   n_categories: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute visit RATES (for Poisson model) using Store Value mechanism
        
        This enables multiple visits per week instead of binary visit/no-visit
        
        Returns:
            visit_rates: Lambda parameters for Poisson distribution [n_customers]
            store_values: Store inclusive values [n_customers]
        """
        if self.store_value_engine is None or self.visit_state_tracker is None:
            raise ValueError("Phase 2 not enabled! Initialize with config and n_customers.")
        
        n_customers = product_utilities.shape[0]
        
        # Step 1: Calculate current period store values (SV)
        store_values = self.store_value_engine.compute_store_value_gpu(
            product_utilities,
            product_categories,
            n_categories
        )
        
        # Step 2: Get previous state from tracker (using rates instead of probs)
        prev_visit_probs, prev_store_values, visited_last_period = self.visit_state_tracker.get_state()
        
        # Convert previous probabilities to rates for continuity
        # If upgrading from probability model, convert: rate ≈ -log(1-prob)
        prev_visit_rates = jnp.where(
            prev_visit_probs < 0.99,
            -jnp.log(1.0 - prev_visit_probs + 1e-6),  # Convert prob to rate
            2.0  # Default rate for high probability customers
        )
        
        # Convert numpy arrays to JAX arrays
        prev_visit_rates_jax = jnp.array(prev_visit_rates)
        prev_store_values_jax = jnp.array(prev_store_values)
        visited_last_period_jax = jnp.array(visited_last_period, dtype=jnp.float32)
        
        # Step 3: Get marketing signal from promo_context
        marketing_signal_strength = promo_context.get('marketing_signal', 0.0)
        
        # Convert to array for all customers
        marketing_signals_np = np.full(n_customers, marketing_signal_strength)
        marketing_signals_jax = jnp.array(marketing_signals_np)
        
        # Step 4: Compute visit utilities (γ₀ + γ₁*SV_{t-1} + β*Marketing)
        visit_utilities = self.store_value_engine.compute_visit_utilities_gpu(
            store_values,
            prev_store_values_jax,
            marketing_signals_jax,
            visited_last_period_jax
        )
        
        # LOGGING: Track utility components for rates model
        if marketing_signal_strength > 0.3:
            base_utility = self.store_value_engine.gamma_0
            sv_contribution = float((self.store_value_engine.gamma_1 * store_values).mean())
            marketing_contribution = float((self.store_value_engine.beta_marketing * marketing_signals_jax).mean())
            total_utility = float(visit_utilities.mean())
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Visit Rate Model - Utility Breakdown:")
            logger.info(f"  Base (γ₀): {base_utility:.3f}")
            logger.info(f"  SV contrib (γ₁*SV): {sv_contribution:.3f}")
            logger.info(f"  Marketing contrib (β*M): {marketing_contribution:.3f}")
            logger.info(f"  Total utility: {total_utility:.3f}")
        
        # Step 5: Calculate recursive rates (θ*Rate_{t-1} + (1-θ)*f(X))
        theta = jnp.full(n_customers, self.store_value_engine.theta)
        visit_rates = self.store_value_engine.compute_visit_rates_recursive_gpu(
            visit_utilities,
            prev_visit_rates_jax,
            theta
        )
        
        return visit_rates, store_values
    
    def update_visit_state(self, visiting_customers: np.ndarray, store_values: jnp.ndarray, visit_probs: jnp.ndarray):
        """
        Update visit state after a period (Phase 2)
        
        Call this after each period to update the recursive state.
        
        Args:
            visiting_customers: [n_customers] - Binary array of who visited
            visited_customers: [n_customers] - Binary array of who visited
            store_values: [n_customers] - Store values from this period
            visit_probs: [n_customers] - Visit probabilities for reference
        """
        if self.visit_state_tracker is not None:
            # Convert JAX arrays to numpy if needed
            if isinstance(store_values, jnp.ndarray):
                store_values = np.array(store_values)
            if isinstance(visit_probs, jnp.ndarray):
                visit_probs = np.array(visit_probs)
            
            self.visit_state_tracker.update(visiting_customers, store_values, visit_probs)

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
