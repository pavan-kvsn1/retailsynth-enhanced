import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from retailsynth.calibration import CalibrationEngine
from retailsynth.engines import GPUUtilityEngine, StoreLoyaltyEngine, VectorizedPreComputationEngine
from datetime import datetime
from retailsynth.config import EnhancedRetailConfig
from typing import Tuple, List, Dict
from datetime import date

# ============================================================================
# TRANSACTION GENERATOR (v3.3, v3.5, v3.6 combined)
# ============================================================================

class ComprehensiveTransactionGenerator:
    """
    Transaction generator combining all optimizations:
    - GPU acceleration (v3.3)
    - Fixed JIT issues (v3.5)
    - Store loyalty (v3.6)
    """
    
    def __init__(self, precomp: VectorizedPreComputationEngine, 
                 utility_engine: GPUUtilityEngine,
                 store_loyalty: StoreLoyaltyEngine,
                 config: EnhancedRetailConfig):
        self.precomp = precomp
        self.utility_engine = utility_engine
        self.store_loyalty = store_loyalty
        self.config = config
        self.rng_key = jax.random.PRNGKey(config.random_seed)
    
    def generate_week_transactions_vectorized(self, 
                                             week_number: int,
                                             current_prices: np.ndarray,
                                             promo_flags: np.ndarray,
                                             week_date: date) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate all transactions for a week using vectorized GPU operations.
        Includes store loyalty logic (v3.6).
        """
        # Convert to JAX arrays
        current_prices_jax = jnp.array(current_prices, dtype=jnp.float32)
        promo_flags_jax = jnp.array(promo_flags, dtype=jnp.float32)
        
        # Step 1: Store visit decisions (vectorized on GPU)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        visit_probs = self.utility_engine.compute_store_visit_probabilities_gpu(
            self.precomp.days_since_visit_jax,
            self.precomp.loyalty_levels_jax,
            subkey
        )
        
        # Sample visits
        self.rng_key, subkey = jax.random.split(self.rng_key)
        visit_random = jax.random.uniform(subkey, shape=(self.precomp.n_customers,))
        visiting_customers = (visit_random < visit_probs).astype(jnp.int32)
        visiting_indices = jnp.where(visiting_customers)[0]
        
        if len(visiting_indices) == 0:
            return [], []
        
        # Step 2: Compute utilities for ALL visiting customers × ALL products (GPU)
        self.rng_key, subkey = jax.random.split(self.rng_key)
        all_utilities = self.utility_engine.compute_all_utilities_gpu(
            current_prices_jax,
            self.precomp.beta_price_jax[visiting_indices],
            self.precomp.brand_pref_matrix_jax[visiting_indices],
            self.precomp.beta_brand_jax[visiting_indices],
            promo_flags_jax,
            self.precomp.beta_promo_jax[visiting_indices],
            self.precomp.role_pref_matrix_jax[visiting_indices],
            self.precomp.beta_role_jax[visiting_indices],
            subkey
        )
        
        # Convert to numpy for product sampling
        all_utilities_np = np.array(all_utilities)
        
        # Step 3: Determine number of products per customer
        n_visiting = len(visiting_indices)
        n_products_per_customer = np.ones(n_visiting, dtype=np.int32)
        
        for i, idx in enumerate(visiting_indices):
            personality = self.precomp.shopping_personalities[idx]
            if personality == 'impulse':
                n_products_per_customer[i] = np.random.choice([2, 3, 4, 5], p=[0.3, 0.3, 0.25, 0.15])
            elif personality == 'planned':
                n_products_per_customer[i] = np.random.choice([3, 4, 5, 6], p=[0.3, 0.4, 0.2, 0.1])
            elif personality == 'convenience':
                n_products_per_customer[i] = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            else:  # price_anchor
                n_products_per_customer[i] = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
        
        # Step 4: Sample product choices (numpy - v3.5 fix)
        product_choices = self.utility_engine.sample_product_choices_numpy(
            all_utilities_np,
            n_products_per_customer
        )
        
        # Step 5: Create transaction records
        transactions = []
        transaction_items = []
        transaction_id = week_number * 1000000 + 1
        
        for i, customer_idx in enumerate(visiting_indices):
            customer_id = self.precomp.customer_ids[customer_idx]
            
            # Select store using loyalty engine (v3.6)
            store_id = self.store_loyalty.select_store_for_customer(customer_id, week_number)
            
            # Get product choices for this customer
            chosen_products = product_choices[i, :n_products_per_customer[i]]
            
            if len(chosen_products) == 0:
                continue
            
            # Build line items
            line_items = []
            totals = {'revenue': 0, 'margin': 0, 'discount': 0, 'items': 0, 'promo_items': 0}
            
            for line_num, product_idx in enumerate(chosen_products):
                product_id = self.precomp.product_ids[product_idx]
                base_price = self.precomp.base_prices[product_idx]
                final_price = current_prices[product_idx]
                is_promoted = promo_flags[product_idx]
                
                # Quantity (typically 1, sometimes 2-3)
                quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
                
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
            
            if not line_items:
                continue
            
            # Satisfaction score (influenced by promotions and store service)
            satisfaction = 0.7 + np.random.uniform(0, 0.2)
            if totals['discount'] > 0:
                satisfaction += 0.1
            satisfaction = min(0.99, satisfaction)
            
            # Update store loyalty based on satisfaction (v3.6)
            self.store_loyalty.update_store_preference(
                customer_id, store_id, satisfaction, week_number
            )
            
            # Transaction record
            transaction = {
                'transaction_id': transaction_id,
                'customer_id': int(customer_id),
                'store_id': int(store_id),
                'transaction_date': week_date,
                'transaction_time': self._generate_shopping_time(),
                'week_number': int(week_number),
                'total_items_count': int(totals['items']),
                'total_revenue': round(totals['revenue'], 2),
                'total_margin': round(totals['margin'], 2),
                'total_discount': round(totals['discount'], 2),
                'promotional_items_count': int(totals['promo_items']),
                'satisfaction_score': round(satisfaction, 3),
                'created_at': datetime.now()
            }
            
            transactions.append(transaction)
            
            # Transaction items
            for item in line_items:
                item['transaction_id'] = transaction_id
                transaction_items.append(item)
            
            transaction_id += 1
        
        return transactions, transaction_items
    
    def _generate_shopping_time(self) -> str:
        """Generate realistic shopping time"""
        hour = np.random.choice(
            range(8, 20), 
            p=[0.05, 0.08, 0.12, 0.15, 0.12, 0.08, 0.05, 0.05, 0.05, 0.08, 0.12, 0.05]
        )
        minute = np.random.randint(0, 60)
        return f"{hour:02d}:{minute:02d}:00"

        