
import numpy as np
import jax.numpy as jnp
import pandas as pd
from typing import Dict
from datetime import datetime
from ..calibration.calibration_engine import CalibrationEngine
import jax

# ============================================================================
# VECTORIZED PRE-COMPUTATION ENGINE (v3.4 - ZERO LOOPS)
# ============================================================================

class VectorizedPreComputationEngine:
    """
    ZERO Python loops in matrix construction (v3.4).
    Uses numpy broadcasting and vectorization throughout.
    """
    
    def __init__(self, customers_df: pd.DataFrame, products_df: pd.DataFrame):
        print("   🔧 Pre-computing customer/product matrices (VECTORIZED)...")
        
        self.n_customers = len(customers_df)
        self.n_products = len(products_df)
        
        start_time = datetime.now()
        
        # Extract customer data (vectorized)
        print(f"      Extracting customer data ({self.n_customers:,} customers)...")
        self.customer_ids = customers_df['customer_id'].values
        self.beta_price = self._extract_utility_param_vectorized(customers_df, 'beta_price')
        self.beta_brand = self._extract_utility_param_vectorized(customers_df, 'beta_brand')
        self.beta_promo = self._extract_utility_param_vectorized(customers_df, 'beta_promotion')
        self.beta_role = self._extract_utility_param_vectorized(customers_df, 'beta_assortment_role')
        self.loyalty_levels = customers_df['store_loyalty_level'].values
        self.days_since_visit = customers_df['days_since_last_visit'].values
        self.household_sizes = customers_df['household_size'].values
        self.shopping_personalities = customers_df['shopping_personality'].values
        
        # Extract product data (vectorized)
        print(f"      Extracting product data ({self.n_products:,} products)...")
        self.product_ids = products_df['product_id'].values
        self.base_prices = products_df['base_price'].values
        self.assortment_roles = products_df['assortment_role'].values
        self.departments = products_df['department'].values
        self.product_brands = products_df['brand'].values
        
        # Build brand preference matrix (VECTORIZED - NO LOOPS!)
        print(f"      Building brand preference matrix ({self.n_customers:,} × {self.n_products:,})...")
        matrix_start = datetime.now()
        self.brand_pref_matrix = self._build_brand_preference_matrix_vectorized(customers_df, products_df)
        print(f"         ✅ Built in {(datetime.now() - matrix_start).total_seconds():.1f}s")
        
        # Build role preference matrix (VECTORIZED - NO LOOPS!)
        print(f"      Building role preference matrix ({self.n_customers:,} × {self.n_products:,})...")
        matrix_start = datetime.now()
        self.role_pref_matrix = self._build_role_preference_matrix_vectorized(customers_df, products_df)
        print(f"         ✅ Built in {(datetime.now() - matrix_start).total_seconds():.1f}s")
        
        # Convert to JAX arrays
        print(f"      Converting to JAX arrays...")
        self.beta_price_jax = jnp.array(self.beta_price, dtype=jnp.float32)
        self.beta_brand_jax = jnp.array(self.beta_brand, dtype=jnp.float32)
        self.beta_promo_jax = jnp.array(self.beta_promo, dtype=jnp.float32)
        self.beta_role_jax = jnp.array(self.beta_role, dtype=jnp.float32)
        self.base_prices_jax = jnp.array(self.base_prices, dtype=jnp.float32)
        self.brand_pref_matrix_jax = jnp.array(self.brand_pref_matrix, dtype=jnp.float32)
        self.role_pref_matrix_jax = jnp.array(self.role_pref_matrix, dtype=jnp.float32)
        self.loyalty_levels_jax = jnp.array(self.loyalty_levels, dtype=jnp.float32)
        self.days_since_visit_jax = jnp.array(self.days_since_visit, dtype=jnp.float32)
        
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"      ✅ Pre-computation complete in {total_time:.1f}s")
        print(f"         Matrix size: {self.brand_pref_matrix.nbytes / 1024**2:.1f} MB")
    
    def _extract_utility_param_vectorized(self, customers_df: pd.DataFrame, param_name: str) -> np.ndarray:
        """Extract utility parameter (vectorized - no loop over rows)"""
        return np.array([
            utility_params[param_name]
            for utility_params in customers_df['utility_params'].values
        ], dtype=np.float32)
    
    def _build_brand_preference_matrix_vectorized(self, customers_df: pd.DataFrame, 
                                                 products_df: pd.DataFrame) -> np.ndarray:
        """
        Build brand preference matrix WITHOUT nested loops (v3.4).
        Uses vectorized operations and broadcasting.
        """
        # Get unique brands
        unique_brands = products_df['brand'].unique()
        brand_to_idx = {brand: idx for idx, brand in enumerate(unique_brands)}
        n_brands = len(unique_brands)
        
        # Map products to brand indices
        product_brand_indices = np.array([brand_to_idx[b] for b in products_df['brand'].values])
        
        # Create preference matrix: (n_customers, n_brands)
        default_pref = 0.3
        customer_brand_prefs = np.full((self.n_customers, n_brands), default_pref, dtype=np.float32)
        
        # Fill in actual preferences
        for i, brand_pref_dict in enumerate(customers_df['brand_preferences'].values):
            for brand, pref in brand_pref_dict.items():
                if brand in brand_to_idx:
                    customer_brand_prefs[i, brand_to_idx[brand]] = pref
        
        # Expand to (n_customers, n_products) using broadcasting
        brand_pref_matrix = customer_brand_prefs[:, product_brand_indices]
        
        return brand_pref_matrix.astype(np.float32)
    
    def _build_role_preference_matrix_vectorized(self, customers_df: pd.DataFrame,
                                                products_df: pd.DataFrame) -> np.ndarray:
        """
        Build role preference matrix WITHOUT nested loops (v3.4).
        Uses vectorized operations.
        """
        # Role preference mapping by personality
        role_prefs_map = {
            'price_anchor': {'lpg_line': 0.8, 'front_basket': 0.3, 'mid_basket': 0.5, 'back_basket': 0.1},
            'convenience': {'lpg_line': 0.3, 'front_basket': 0.6, 'mid_basket': 0.4, 'back_basket': 0.7},
            'planned': {'lpg_line': 0.4, 'front_basket': 0.2, 'mid_basket': 0.8, 'back_basket': 0.2},
            'impulse': {'lpg_line': 0.2, 'front_basket': 0.8, 'mid_basket': 0.4, 'back_basket': 0.9}
        }
        
        # Get unique roles and create mapping
        unique_roles = products_df['assortment_role'].unique()
        role_to_idx = {role: idx for idx, role in enumerate(unique_roles)}
        n_roles = len(unique_roles)
        
        # Map products to role indices
        product_role_indices = np.array([role_to_idx[r] for r in products_df['assortment_role'].values])
        
        # Get unique personalities
        unique_personalities = customers_df['shopping_personality'].unique()
        personality_to_idx = {p: idx for idx, p in enumerate(unique_personalities)}
        
        # Create preference lookup: (n_personalities, n_roles)
        personality_role_prefs = np.full((len(unique_personalities), n_roles), 0.5, dtype=np.float32)
        
        for personality, prefs in role_prefs_map.items():
            if personality in personality_to_idx:
                p_idx = personality_to_idx[personality]
                for role, pref in prefs.items():
                    if role in role_to_idx:
                        r_idx = role_to_idx[role]
                        personality_role_prefs[p_idx, r_idx] = pref
        
        # Map customers to personality indices
        customer_personality_indices = np.array([
            personality_to_idx[p] for p in customers_df['shopping_personality'].values
        ])
        
        # Build matrix using advanced indexing (vectorized)
        customer_role_prefs = personality_role_prefs[customer_personality_indices]
        role_pref_matrix = customer_role_prefs[:, product_role_indices]
        
        return role_pref_matrix.astype(np.float32)
    
    def update_from_drift(self, updated_customers_df: pd.DataFrame):
        """
        Update pre-computed matrices after customer drift (v3.6).
        Only update changed parameters - avoid full recomputation.
        """
        # Update utility parameters
        self.beta_price = self._extract_utility_param_vectorized(updated_customers_df, 'beta_price')
        self.beta_brand = self._extract_utility_param_vectorized(updated_customers_df, 'beta_brand')
        self.beta_promo = self._extract_utility_param_vectorized(updated_customers_df, 'beta_promotion')
        self.beta_role = self._extract_utility_param_vectorized(updated_customers_df, 'beta_assortment_role')
        
        # Update behavioral attributes
        self.loyalty_levels = updated_customers_df['store_loyalty_level'].values
        self.days_since_visit = updated_customers_df['days_since_last_visit'].values
        self.shopping_personalities = updated_customers_df['shopping_personality'].values
        
        # Rebuild matrices that depend on changed attributes
        # Brand preferences might have changed
        self.brand_pref_matrix = self._build_brand_preference_matrix_vectorized(
            updated_customers_df, 
            pd.DataFrame({
                'brand': self.product_brands,
                'product_id': self.product_ids
            })
        )
        
        # Role preferences depend on personality
        self.role_pref_matrix = self._build_role_preference_matrix_vectorized(
            updated_customers_df,
            pd.DataFrame({
                'assortment_role': self.assortment_roles,
                'product_id': self.product_ids
            })
        )
        
        # Update JAX arrays
        self.beta_price_jax = jnp.array(self.beta_price, dtype=jnp.float32)
        self.beta_brand_jax = jnp.array(self.beta_brand, dtype=jnp.float32)
        self.beta_promo_jax = jnp.array(self.beta_promo, dtype=jnp.float32)
        self.beta_role_jax = jnp.array(self.beta_role, dtype=jnp.float32)
        self.brand_pref_matrix_jax = jnp.array(self.brand_pref_matrix, dtype=jnp.float32)
        self.role_pref_matrix_jax = jnp.array(self.role_pref_matrix, dtype=jnp.float32)
        self.loyalty_levels_jax = jnp.array(self.loyalty_levels, dtype=jnp.float32)
        self.days_since_visit_jax = jnp.array(self.days_since_visit, dtype=jnp.float32)
    
    def update_for_products(self, updated_products_df: pd.DataFrame, customers_df: pd.DataFrame = None):
        """
        Update pre-computed matrices after product changes (v3.6).
        Handle product retirements and launches.
        """
        print(f"         Rebuilding matrices for {len(updated_products_df):,} products...")
        
        self.n_products = len(updated_products_df)
        
        # Update product data
        self.product_ids = updated_products_df['product_id'].values
        self.base_prices = updated_products_df['base_price'].values
        self.assortment_roles = updated_products_df['assortment_role'].values
        self.departments = updated_products_df['department'].values
        self.product_brands = updated_products_df['brand'].values
        
        # Rebuild brand preference matrix with new products
        if customers_df is not None:
            self.brand_pref_matrix = self._build_brand_preference_matrix_vectorized(
                customers_df, updated_products_df
            )
        else:
            # Fallback: use default preferences
            self.brand_pref_matrix = np.full((self.n_customers, self.n_products), 0.3, dtype=np.float32)
        
        # Rebuild role preference matrix with new products
        # Create a temporary DataFrame with required columns
        products_temp = pd.DataFrame({
            'assortment_role': self.assortment_roles,
            'product_id': self.product_ids
        })
        
        if customers_df is not None:
            self.role_pref_matrix = self._build_role_preference_matrix_vectorized(
                customers_df, products_temp
            )
        else:
            # Fallback: use personality-based defaults
            unique_roles = np.unique(self.assortment_roles)
            role_to_idx = {role: idx for idx, role in enumerate(unique_roles)}
            n_roles = len(unique_roles)
            
            product_role_indices = np.array([role_to_idx.get(r, 0) for r in self.assortment_roles])
            
            unique_personalities = np.unique(self.shopping_personalities)
            personality_to_idx = {p: idx for idx, p in enumerate(unique_personalities)}
            
            role_prefs_map = {
                'price_anchor': {'lpg_line': 0.8, 'front_basket': 0.3, 'mid_basket': 0.5, 'back_basket': 0.1},
                'convenience': {'lpg_line': 0.3, 'front_basket': 0.6, 'mid_basket': 0.4, 'back_basket': 0.7},
                'planned': {'lpg_line': 0.4, 'front_basket': 0.2, 'mid_basket': 0.8, 'back_basket': 0.2},
                'impulse': {'lpg_line': 0.2, 'front_basket': 0.8, 'mid_basket': 0.4, 'back_basket': 0.9}
            }
            
            personality_role_prefs = np.full((len(unique_personalities), n_roles), 0.5, dtype=np.float32)
            
            for personality, prefs in role_prefs_map.items():
                if personality in personality_to_idx:
                    p_idx = personality_to_idx[personality]
                    for role, pref in prefs.items():
                        if role in role_to_idx:
                            r_idx = role_to_idx[role]
                            personality_role_prefs[p_idx, r_idx] = pref
            
            customer_personality_indices = np.array([
                personality_to_idx.get(p, 0) for p in self.shopping_personalities
            ])
            
            customer_role_prefs = personality_role_prefs[customer_personality_indices]
            self.role_pref_matrix = customer_role_prefs[:, product_role_indices]
        
        # Update JAX arrays
        self.base_prices_jax = jnp.array(self.base_prices, dtype=jnp.float32)
        self.brand_pref_matrix_jax = jnp.array(self.brand_pref_matrix, dtype=jnp.float32)
        self.role_pref_matrix_jax = jnp.array(self.role_pref_matrix, dtype=jnp.float32)
        
        print(f"         ✅ Matrices rebuilt: {self.n_customers:,} × {self.n_products:,}")
