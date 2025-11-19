import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from retailsynth.calibration import CalibrationEngine
from retailsynth.engines import GPUUtilityEngine, StoreLoyaltyEngine, VectorizedPreComputationEngine
from datetime import datetime
from retailsynth.config import EnhancedRetailConfig
from typing import Tuple, List, Dict, Optional
from datetime import date

# Sprint 1.3: Import purchase history components
from retailsynth.engines.customer_state import CustomerStateManager
from retailsynth.engines.purchase_history_engine import PurchaseHistoryEngine

# Sprint 1.4: Import basket composition components
from retailsynth.engines.basket_composer import BasketComposer

# ============================================================================
# TRANSACTION GENERATOR (v3.3, v3.5, v3.6 + Sprint 1.3 + Sprint 1.4)
# ============================================================================

class ComprehensiveTransactionGenerator:
    """
    Transaction generator combining all optimizations:
    - GPU acceleration (v3.3)
    - Fixed JIT issues (v3.5)
    - Store loyalty (v3.6)
    - Purchase history & state dependence (Sprint 1.3)
    - Basket composition logic (Sprint 1.4)
    - Promotional response (Sprint 2.5)
    - Non-linear utilities (Sprint 2.6)
    - SV-based visit probability (Phase 2)
    """
    
    def __init__(self, precomp: VectorizedPreComputationEngine, 
                 utility_engine: GPUUtilityEngine,
                 store_loyalty: StoreLoyaltyEngine,
                 config: EnhancedRetailConfig,
                 state_manager: Optional[CustomerStateManager] = None,
                 history_engine: Optional[PurchaseHistoryEngine] = None,
                 basket_composer: Optional[BasketComposer] = None,
                 promo_response_calc = None,  # Sprint 2.5
                 nonlinear_engine = None,  # Sprint 2.6
                 products = None):  # Phase 2: For SV-based visit probability
        self.precomp = precomp
        self.utility_engine = utility_engine
        self.store_loyalty = store_loyalty
        self.config = config
        self.rng_key = jax.random.PRNGKey(config.random_seed)
        
        # Sprint 1.3: Purchase history components
        self.state_manager = state_manager
        self.history_engine = history_engine
        self.enable_history = (state_manager is not None and history_engine is not None)
        
        # Sprint 1.4: Basket composition
        self.basket_composer = basket_composer
        self.enable_basket_composition = (basket_composer is not None)
        
        # Sprint 2.5: Promotional response
        self.promo_response_calc = promo_response_calc
        
        # Sprint 2.6: Non-linear utilities
        self.nonlinear_engine = nonlinear_engine
        
        # Phase 2: Products for SV calculation
        self.products = products
    
    def generate_week_transactions_vectorized(self, 
                                             week_number: int,
                                             current_prices: np.ndarray,
                                             promo_flags: np.ndarray,
                                             week_date: date,
                                             store_promo_contexts: dict = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate all transactions for a week using vectorized GPU operations.
        Includes store loyalty logic (v3.6), purchase history (Sprint 1.3), basket composition (Sprint 1.4),
        and promotional contexts (Sprint 2).
        """
        # Sprint 1.3: Update all customer states for current week
        if self.enable_history:
            self.state_manager.update_all_states(week_number)
        
        # Sprint 2: Store promotional contexts for this week
        self.store_promo_contexts = store_promo_contexts or {}
        
        # Convert to JAX arrays
        current_prices_jax = jnp.array(current_prices, dtype=jnp.float32)
        promo_flags_jax = jnp.array(promo_flags, dtype=jnp.float32)
        
        # Step 1: Compute utilities for ALL customers × ALL products (GPU)
        # Note: We need utilities BEFORE visit decision for Phase 2 Store Value calculation
        self.rng_key, subkey = jax.random.split(self.rng_key)
        all_utilities = self.utility_engine.compute_all_utilities_gpu(
            current_prices_jax,
            self.precomp.beta_price_jax,
            self.precomp.brand_pref_matrix_jax,
            self.precomp.beta_brand_jax,
            promo_flags_jax,
            self.precomp.beta_promo_jax,
            self.precomp.role_pref_matrix_jax,
            self.precomp.beta_role_jax,
            subkey
        )
        
        # Step 2: Calculate visit probabilities using Phase 2 method (Store Value + Marketing Signal)
        if self.products is not None and hasattr(self.utility_engine, 'compute_visit_probabilities_with_sv'):
            # Phase 2: Use Bain's recursive model with Store Value
            # Create numeric category mapping from commodity_desc (string) -> category_id (int)
            unique_categories = self.products['commodity_desc'].unique()
            category_to_id = {cat: idx for idx, cat in enumerate(unique_categories)}
            product_categories = jnp.array([category_to_id[cat] for cat in self.products['commodity_desc'].values])
            n_categories = len(unique_categories)
            
            visit_probs, store_values = self.utility_engine.compute_visit_probabilities_with_sv(
                all_utilities,
                product_categories,
                self.store_promo_contexts,
                n_categories
            )
        else:
            # Fallback: Use legacy method (should rarely happen)
            print("⚠️  WARNING: Phase 2 not available, using legacy visit probability (hardcoded)")
            self.rng_key, subkey = jax.random.split(self.rng_key)
            visit_probs = self.utility_engine.compute_store_visit_probabilities_gpu(
                self.precomp.days_since_visit_jax,
                self.precomp.loyalty_levels_jax,
                subkey
            )
        
        # Step 3: Sample visits based on calculated probabilities
        self.rng_key, subkey = jax.random.split(self.rng_key)
        visit_random = jax.random.uniform(subkey, shape=(self.precomp.n_customers,))
        visiting_customers = (visit_random < visit_probs).astype(jnp.int32)
        visiting_indices = jnp.where(visiting_customers)[0]
        
        if len(visiting_indices) == 0:
            return [], []
        
        # Step 4: Extract utilities for visiting customers only (for product selection)
        all_utilities_for_visitors = all_utilities[visiting_indices]
        
        # Convert to numpy for product sampling
        all_utilities_np = np.array(all_utilities_for_visitors)
        
        # Sprint 1.3: Apply history-dependent utility adjustments
        if self.enable_history:
            all_utilities_np = self._apply_history_adjustments(
                all_utilities_np,
                visiting_indices,
                week_number,
                current_prices
            )
        
        # Step 3 & 4: Generate baskets (Sprint 1.4: Use basket composer if enabled)
        if self.enable_basket_composition:
            # NEW: Trip-purpose driven basket composition (with Priority 2B promo boost)
            baskets = self._generate_baskets_with_composer(
                visiting_indices,
                all_utilities_np,
                week_number,
                week_date,
                promo_flags=np.array(promo_flags_jax) if promo_flags_jax is not None else None
            )
        else:
            # LEGACY: Independent product sampling
            baskets = self._generate_baskets_legacy(
                visiting_indices,
                all_utilities_np
            )
        
        # Step 5: Create transaction records
        transactions = []
        transaction_items = []
        transaction_id = week_number * 1000000 + 1
        
        for i, customer_idx in enumerate(visiting_indices):
            customer_id = self.precomp.customer_ids[customer_idx]
            
            # Select store using loyalty engine (v3.6)
            store_id = self.store_loyalty.select_store_for_customer(customer_id, week_number)
            
            # Get basket for this customer (list of (product_id, quantity) tuples)
            basket = baskets[i]
            
            if len(basket) == 0:
                continue
            
            # Build line items
            line_items = []
            totals = {'revenue': 0, 'margin': 0, 'discount': 0, 'items': 0, 'promo_items': 0}
            
            for line_num, (product_id, quantity) in enumerate(basket):
                # Find product index
                product_idx = np.where(self.precomp.product_ids == product_id)[0]
                if len(product_idx) == 0:
                    continue
                product_idx = product_idx[0]
                
                base_price = self.precomp.base_prices[product_idx]
                final_price = current_prices[product_idx]
                is_promoted = promo_flags[product_idx]
                
                # Calculate amounts
                line_total = final_price * quantity
                discount_amount = (base_price - final_price) if is_promoted else 0
                margin_amount = (final_price - base_price * 0.6) * quantity  # Assume 40% cost
                
                line_items.append({
                    'line_number': line_num + 1,
                    'product_id': int(product_id),
                    'quantity': int(quantity),
                    'unit_price': round(float(final_price), 2),
                    'line_total': round(float(line_total), 2)
                })
                
                totals['revenue'] += line_total
                totals['margin'] += margin_amount
                totals['discount'] += discount_amount * quantity
                totals['items'] += quantity
                if is_promoted:
                    totals['promo_items'] += quantity
                
                # Sprint 1.3: Update customer state after purchase
                if self.enable_history:
                    customer_state = self.state_manager.get_state(customer_id)
                    self.history_engine.update_customer_after_purchase(
                        customer_state=customer_state,
                        product_id=int(product_id),
                        week=week_number,
                        price=float(final_price),
                        base_price=float(base_price),
                        quantity=int(quantity)
                    )
            
            if not line_items:
                continue
            
            # Satisfaction score (influenced by promotions and store service)
            satisfaction = 0.7 + np.random.uniform(0, 0.2)
            if totals['discount'] > 0:
                satisfaction += 0.1
            satisfaction = min(0.99, satisfaction)
            
            # Update store loyalty based on satisfaction (v3.6)
            self.store_loyalty.update_store_preference(
                customer_id, 
                store_id, 
                satisfaction,
                week_number
            )
            
            # Create transaction record
            transaction = {
                'transaction_id': transaction_id,
                'customer_id': int(customer_id),
                'store_id': int(store_id),
                'week_number': week_number,
                'transaction_date': week_date,
                'transaction_time': self._generate_shopping_time(),
                'basket_size': len(line_items),
                'total_items': int(totals['items']),
                'total_revenue': round(float(totals['revenue']), 2),
                'total_margin': round(float(totals['margin']), 2),
                'total_discount': round(float(totals['discount']), 2),
                'promo_items': int(totals['promo_items']),
                'satisfaction_score': round(satisfaction, 3)
            }
            
            transactions.append(transaction)
            
            # Add line items
            for item in line_items:
                item['transaction_id'] = transaction_id
                transaction_items.append(item)
            
            transaction_id += 1
        
        return transactions, transaction_items
    
    def _apply_history_adjustments(
        self,
        base_utilities: np.ndarray,
        visiting_indices: np.ndarray,
        week_number: int,
        current_prices: np.ndarray
    ) -> np.ndarray:
        """
        Apply history-dependent utility adjustments (Sprint 1.3)
        
        Args:
            base_utilities: Base utility matrix (n_customers × n_products)
            visiting_indices: Indices of visiting customers
            week_number: Current week
            current_prices: Current prices for all products
        
        Returns:
            Adjusted utilities incorporating purchase history
        """
        adjusted_utilities = base_utilities.copy()
        
        for i, customer_idx in enumerate(visiting_indices):
            customer_id = self.precomp.customer_ids[customer_idx]
            customer_state = self.state_manager.get_state(customer_id)
            
            # Calculate history-dependent adjustments (with price memory)
            adjusted_utilities[i] = self.history_engine.calculate_history_utility(
                customer_state=customer_state,
                product_ids=self.precomp.product_ids,
                base_utilities=base_utilities[i],
                current_week=week_number,
                current_prices=current_prices
            )
        
        return adjusted_utilities
    
    def _generate_baskets_with_composer(
        self,
        visiting_indices: np.ndarray,
        all_utilities: np.ndarray,
        week_number: int,
        week_date: date,
        promo_flags: Optional[np.ndarray] = None
    ) -> List[List[Tuple[int, int]]]:
        """
        Generate baskets using trip-purpose driven basket composition (Sprint 1.4)
        
        Args:
            visiting_indices: Indices of visiting customers
            all_utilities: Utility matrix (n_customers × n_products)
            week_number: Current week
            week_date: Current week date
            promo_flags: Optional promotional flags (Priority 2B)
        
        Returns:
            List of baskets (each basket is a list of (product_id, quantity))
        """
        baskets = []
        
        for i, customer_idx in enumerate(visiting_indices):
            customer_id = self.precomp.customer_ids[customer_idx]
            shopping_personality = self.precomp.shopping_personalities[customer_idx]
            
            # Get customer state if available
            customer_state = None
            if self.enable_history:
                customer_state = self.state_manager.get_state(customer_id)
            
            # Get day of week from date
            day_of_week = week_date.weekday() if week_date else None
            
            # Generate basket using composer (with promo flags for Priority 2B)
            basket = self.basket_composer.generate_basket(
                customer_id=int(customer_id),
                shopping_personality=shopping_personality,
                utilities=all_utilities[i],
                product_ids=self.precomp.product_ids,
                customer_state=customer_state,
                week_number=week_number,
                day_of_week=day_of_week,
                promo_flags=promo_flags  # Priority 2B: Enable promotional quantity boost
            )
            
            baskets.append(basket)
        
        return baskets
    
    def _generate_baskets_legacy(
        self,
        visiting_indices: np.ndarray,
        all_utilities: np.ndarray
    ) -> List[List[Tuple[int, int]]]:
        """
        Generate baskets using independent product sampling (LEGACY)
        
        Args:
            visiting_indices: Indices of visiting customers
            all_utilities: Utility matrix (n_customers × n_products)
        
        Returns:
            List of baskets (each basket is a list of (product_id, quantity))
        """
        baskets = []
        n_visiting = len(visiting_indices)
        
        # Determine basket size per customer (personality-based)
        n_products_per_customer = np.zeros(n_visiting, dtype=int)
        for i, idx in enumerate(visiting_indices):
            personality = self.precomp.shopping_personalities[idx]
            # Use config-based basket size (simplified from trip-based to personality-based)
            if personality == 'impulse':
                n_products_per_customer[i] = max(1, int(np.random.poisson(self.config.basket_size_lambda * 0.7)))
            elif personality == 'planned':
                n_products_per_customer[i] = max(1, int(np.random.poisson(self.config.basket_size_lambda * 1.2)))
            elif personality == 'convenience':
                n_products_per_customer[i] = max(1, int(np.random.poisson(self.config.basket_size_lambda * 0.5)))
            else:  # price_anchor
                n_products_per_customer[i] = max(1, int(np.random.poisson(self.config.basket_size_lambda * 0.8)))
        
        # Sample product choices
        product_choices = self.utility_engine.sample_product_choices_numpy(
            all_utilities,
            n_products_per_customer
        )
        
        # Convert to basket format
        for i in range(len(visiting_indices)):
            chosen_products = product_choices[i, :n_products_per_customer[i]]
            
            basket = []
            for product_idx in chosen_products:
                product_id = self.precomp.product_ids[product_idx]
                # Use config-based quantity distribution
                quantity = max(1, int(np.random.normal(self.config.quantity_mean, self.config.quantity_std)))
                quantity = min(quantity, self.config.quantity_max)  # Cap at max
                basket.append((int(product_id), int(quantity)))

            baskets.append(basket)
        
        return baskets
    
    def _generate_shopping_time(self) -> str:
        """Generate realistic shopping time"""
        hour = np.random.choice(
            range(8, 20), 
            p=[0.05, 0.08, 0.12, 0.15, 0.12, 0.08, 0.05, 0.05, 0.05, 0.08, 0.12, 0.05]
        )
        minute = np.random.randint(0, 60)
        return f"{hour:02d}:{minute:02d}:00"