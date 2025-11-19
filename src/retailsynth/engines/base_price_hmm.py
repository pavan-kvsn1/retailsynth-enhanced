"""
Base Price Hidden Markov Model (Sprint 2.1)

Models strategic base price dynamics (slow-moving, monthly transitions).
Separate from promotional discounts - this captures only the underlying
base price that products are sold at during non-promotional periods.

States represent price tiers within each product's price distribution:
- State 0: Low Price (P0-P25)
- State 1: Mid-Low Price (P25-P50)
- State 2: Mid-High Price (P50-P75)
- State 3: High Price (P75-P100)

Key Characteristics:
- Slow transitions (monthly/quarterly price changes)
- High diagonal probabilities (0.85-0.95) - prices are sticky
- Learned from NON-promotional weeks only
- Product-specific transition matrices and emission distributions

Theoretical Foundation:
    Base prices follow strategic decisions (cost changes, competitive positioning)
    Separate from tactical promotional discounts (temporary, weekly)
    
References:
    - PRICING_PROMO_SEPARATION_DESIGN.md
    - Sprint 2 Goal 1: Clean separation of pricing and promotions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from tqdm import tqdm
import pickle
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class BasePriceHMM:
    """
    Hidden Markov Model for strategic base price dynamics
    
    Learns product-specific base price states and transitions from
    historical non-promotional transaction data.
    
    Attributes:
        n_states (int): Number of price states (default: 4)
        products_df (pd.DataFrame): Product catalog
        transition_matrices (dict): Product-specific transition matrices
        emission_distributions (dict): Price distributions per state per product
        initial_state_probs (dict): Initial state probabilities per product
        state_price_ranges (dict): Price ranges defining each state per product
    """
    
    def __init__(self, products_df: pd.DataFrame, n_states: int = 4):
        """
        Initialize Base Price HMM
        
        Args:
            products_df: Product catalog with PRODUCT_ID column
            n_states: Number of price states (default: 4)
        """
        self.products_df = products_df
        self.n_states = n_states
        
        # State definitions (relative to product's price distribution)
        self.state_names = {
            0: 'low_price',
            1: 'mid_low_price',
            2: 'mid_high_price',
            3: 'high_price'
        }
        
        # Learned parameters (populated by learn_from_data)
        self.transition_matrices = {}  # product_id -> 4x4 matrix
        self.emission_distributions = {}  # product_id -> {state: {mean, std}}
        self.initial_state_probs = {}  # product_id -> {state: prob}
        self.state_price_ranges = {}  # product_id -> {state: (min, max)}
        
        # Current states (for simulation)
        self.current_states = {}  # product_id -> current_state
        
        logger.info(f"Initialized BasePriceHMM with {len(products_df)} products, {n_states} states")
    
    def learn_from_data(self, 
                       transactions_df: pd.DataFrame, 
                       causal_df: Optional[pd.DataFrame] = None,
                       min_observations: int = 10):
        """
        Learn base price HMM parameters from non-promotional transaction data
        
        Args:
            transactions_df: Transaction data with columns:
                - PRODUCT_ID, WEEK_NO, SALES_VALUE, QUANTITY, RETAIL_DISC
            causal_df: Optional promotional data with columns:
                - PRODUCT_ID, STORE_ID, WEEK_NO, DISPLAY, MAILER
            min_observations: Minimum weeks of data required per product
        
        Process:
            1. Filter to non-promotional weeks only
            2. Aggregate to product-week level
            3. Calculate base prices
            4. Define state boundaries (quartiles)
            5. Classify states per week
            6. Estimate transition matrices
            7. Estimate emission distributions
        """
        logger.info("Learning Base Price HMM from non-promotional transaction data...")
        
        # Step 1: Filter to non-promotional weeks
        logger.info("Step 1: Filtering to non-promotional weeks...")
        base_price_data = self._filter_non_promo_weeks(transactions_df, causal_df)
        
        if len(base_price_data) == 0:
            logger.error("No non-promotional data found! Cannot learn base prices.")
            return
        
        logger.info(f"  Found {len(base_price_data):,} non-promotional product-weeks")
        
        # Step 2: Calculate base prices per product-week
        logger.info("Step 2: Calculating base prices...")
        product_week_prices = self._calculate_base_prices(base_price_data)
        
        # Step 3: Learn parameters for each product
        logger.info("Step 3: Learning product-specific parameters...")
        products_learned = 0
        products_skipped = 0
        
        for product_id in tqdm(self.products_df['PRODUCT_ID'].unique(), desc="Learning products"):
            product_data = product_week_prices[
                product_week_prices['PRODUCT_ID'] == product_id
            ]
            
            if len(product_data) < min_observations:
                products_skipped += 1
                continue
            
            # Learn for this product
            success = self._learn_product_parameters(product_id, product_data)
            if success:
                products_learned += 1
        
        logger.info(f"✅ Learned parameters for {products_learned:,} products")
        logger.info(f"   Skipped {products_skipped:,} products (insufficient data)")
        
        # Initialize current states
        self._initialize_states()
    
    def _filter_non_promo_weeks(self, 
                                transactions_df: pd.DataFrame,
                                causal_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Filter to non-promotional weeks only
        
        Uses multiple methods to identify base price weeks:
        1. No retail discount (RETAIL_DISC == 0)
        2. No coupon discount (COUPON_DISC == 0 if available)
        3. No display/mailer activity (if causal data available)
        """
        # Method 1: Discount-based filtering
        base_data = transactions_df[
            transactions_df['RETAIL_DISC'] == 0
        ].copy()
        
        # Also filter coupon discounts if available
        if 'COUPON_DISC' in transactions_df.columns:
            base_data = base_data[base_data['COUPON_DISC'] == 0]
        
        # Method 2: Causal data filtering (if available)
        if causal_df is not None:
            # Merge with causal data
            base_data = base_data.merge(
                causal_df[['PRODUCT_ID', 'STORE_ID', 'WEEK_NO', 'DISPLAY', 'MAILER']],
                on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],
                how='left'
            )
            
            # Filter out weeks with display or mailer activity
            base_data = base_data[
                (base_data['DISPLAY'].isna() | (base_data['DISPLAY'] == '')) &
                (base_data['MAILER'].isna() | (base_data['MAILER'] == ''))
            ]
        
        return base_data
    
    def _calculate_base_prices(self, base_price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate average base price per product-week"""
        product_week_prices = base_price_data.groupby(['PRODUCT_ID', 'WEEK_NO']).agg({
            'SALES_VALUE': 'sum',
            'QUANTITY': 'sum'
        }).reset_index()
        
        # Calculate average price
        product_week_prices['base_price'] = (
            product_week_prices['SALES_VALUE'] / 
            product_week_prices['QUANTITY'].replace(0, np.nan)
        )
        
        # Remove invalid prices
        product_week_prices = product_week_prices.dropna(subset=['base_price'])
        product_week_prices = product_week_prices[product_week_prices['base_price'] > 0]
        
        return product_week_prices
    
    def _learn_product_parameters(self, 
                                  product_id: int, 
                                  product_data: pd.DataFrame) -> bool:
        """
        Learn HMM parameters for a single product
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Step 1: Define state boundaries using quartiles
            prices = product_data['base_price'].values
            quartiles = np.percentile(prices, [25, 50, 75])
            
            self.state_price_ranges[product_id] = {
                0: (prices.min(), quartiles[0]),
                1: (quartiles[0], quartiles[1]),
                2: (quartiles[1], quartiles[2]),
                3: (quartiles[2], prices.max())
            }
            
            # Step 2: Classify each week into a state
            product_data = product_data.copy()
            product_data['state'] = product_data['base_price'].apply(
                lambda p: self._classify_price_to_state(p, product_id)
            )
            
            # Sort by week for transition counting
            product_data = product_data.sort_values('WEEK_NO')
            
            # Step 3: Estimate transition matrix
            transition_counts = np.zeros((self.n_states, self.n_states))
            states = product_data['state'].values
            
            for i in range(len(states) - 1):
                from_state = states[i]
                to_state = states[i + 1]
                transition_counts[from_state, to_state] += 1
            
            # Normalize to probabilities (with smoothing)
            transition_matrix = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states):
                row_sum = transition_counts[i].sum()
                if row_sum > 0:
                    transition_matrix[i] = transition_counts[i] / row_sum
                else:
                    # Default: stay in same state (sticky prices)
                    transition_matrix[i, i] = 0.90
                    # Small probability to adjacent states
                    if i > 0:
                        transition_matrix[i, i-1] = 0.05
                    if i < self.n_states - 1:
                        transition_matrix[i, i+1] = 0.05
            
            self.transition_matrices[product_id] = transition_matrix
            
            # Step 4: Estimate emission distributions (mean, std per state)
            emissions = {}
            for state in range(self.n_states):
                state_prices = product_data[product_data['state'] == state]['base_price']
                if len(state_prices) > 0:
                    emissions[state] = {
                        'mean': state_prices.mean(),
                        'std': state_prices.std() if len(state_prices) > 1 else 0.01
                    }
                else:
                    # Use state range midpoint as fallback
                    range_min, range_max = self.state_price_ranges[product_id][state]
                    emissions[state] = {
                        'mean': (range_min + range_max) / 2,
                        'std': (range_max - range_min) / 4
                    }
            
            self.emission_distributions[product_id] = emissions
            
            # Step 5: Estimate initial state probabilities
            state_counts = product_data['state'].value_counts()
            initial_probs = {}
            for state in range(self.n_states):
                initial_probs[state] = state_counts.get(state, 0) / len(product_data)
            
            self.initial_state_probs[product_id] = initial_probs
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to learn parameters for product {product_id}: {e}")
            return False
    
    def _classify_price_to_state(self, price: float, product_id: int) -> int:
        """Classify a price into a state based on learned ranges"""
        if product_id not in self.state_price_ranges:
            return 1  # Default to mid-low state
        
        ranges = self.state_price_ranges[product_id]
        for state in range(self.n_states):
            min_price, max_price = ranges[state]
            if state == self.n_states - 1:  # Last state includes upper bound
                if min_price <= price <= max_price:
                    return state
            else:
                if min_price <= price < max_price:
                    return state
        
        # Fallback: return closest state
        return 1
    
    def _initialize_states(self):
        """Initialize current states for all products"""
        for product_id in self.initial_state_probs.keys():
            probs = self.initial_state_probs[product_id]
            states = list(probs.keys())
            probabilities = list(probs.values())
            
            # Sample initial state
            self.current_states[product_id] = np.random.choice(states, p=probabilities)
    
    def sample_base_prices(self, 
                          product_ids: Optional[List[int]] = None,
                          week: Optional[int] = None) -> Dict[int, float]:
        """
        Sample base prices for products
        
        Args:
            product_ids: List of product IDs (if None, use all learned products)
            week: Current week number (for logging/debugging)
        
        Returns:
            Dict mapping product_id -> base_price
        """
        if product_ids is None:
            product_ids = list(self.current_states.keys())
        
        base_prices = {}
        
        for product_id in product_ids:
            if product_id not in self.current_states:
                # Product not learned - use fallback
                base_prices[product_id] = self._get_fallback_price(product_id)
                continue
            
            # Get current state
            current_state = self.current_states[product_id]
            
            # Sample price from emission distribution
            emission = self.emission_distributions[product_id][current_state]
            price = np.random.normal(emission['mean'], emission['std'])
            
            # Ensure price is positive and within reasonable bounds
            price = max(0.01, price)
            
            base_prices[product_id] = price
        
        return base_prices
    
    def transition_states(self, product_ids: Optional[List[int]] = None):
        """
        Transition to next states for products (call this weekly/monthly)
        
        Args:
            product_ids: List of product IDs (if None, transition all)
        """
        if product_ids is None:
            product_ids = list(self.current_states.keys())
        
        for product_id in product_ids:
            if product_id not in self.current_states:
                continue
            
            current_state = self.current_states[product_id]
            transition_matrix = self.transition_matrices[product_id]
            
            # Sample next state
            next_state = np.random.choice(
                self.n_states,
                p=transition_matrix[current_state]
            )
            
            self.current_states[product_id] = next_state
    
    def _get_fallback_price(self, product_id: int) -> float:
        """Get fallback price for products without learned parameters"""
        # Try to get from product catalog
        product_row = self.products_df[
            self.products_df['PRODUCT_ID'] == product_id
        ]
        
        if len(product_row) > 0 and 'avg_price' in product_row.columns:
            return float(product_row['avg_price'].iloc[0])
        
        # Ultimate fallback
        return 3.99
    
    def get_transition_matrix(self, product_id: int) -> Optional[np.ndarray]:
        """Get transition matrix for a product"""
        return self.transition_matrices.get(product_id)
    
    def get_state_info(self, product_id: int) -> Dict:
        """Get current state information for a product"""
        if product_id not in self.current_states:
            return {}
        
        current_state = self.current_states[product_id]
        emission = self.emission_distributions[product_id][current_state]
        price_range = self.state_price_ranges[product_id][current_state]
        
        return {
            'current_state': current_state,
            'state_name': self.state_names[current_state],
            'mean_price': emission['mean'],
            'std_price': emission['std'],
            'price_range': price_range
        }
    
    def save_parameters(self, filepath: Path):
        """Save learned parameters to file"""
        params = {
            'n_states': self.n_states,
            'state_names': self.state_names,
            'transition_matrices': self.transition_matrices,
            'emission_distributions': self.emission_distributions,
            'initial_state_probs': self.initial_state_probs,
            'state_price_ranges': self.state_price_ranges,
            'current_states': self.current_states
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        
        logger.info(f"✅ Saved Base Price HMM parameters to {filepath}")
    
    def load_parameters(self, filepath: Path):
        """Load learned parameters from file"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        self.n_states = params['n_states']
        self.state_names = params['state_names']
        self.transition_matrices = params['transition_matrices']
        self.emission_distributions = params['emission_distributions']
        self.initial_state_probs = params['initial_state_probs']
        self.state_price_ranges = params['state_price_ranges']
        self.current_states = params['current_states']
        
        logger.info(f"✅ Loaded Base Price HMM parameters from {filepath}")
        logger.info(f"   Products with learned parameters: {len(self.transition_matrices):,}")
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of learned parameters"""
        if len(self.transition_matrices) == 0:
            return {}
        
        # Average transition matrix
        avg_transition = np.mean(
            list(self.transition_matrices.values()),
            axis=0
        )
        
        # State prevalence
        state_counts = defaultdict(int)
        for probs in self.initial_state_probs.values():
            for state, prob in probs.items():
                state_counts[state] += prob
        
        total = sum(state_counts.values())
        state_prevalence = {
            state: count / total 
            for state, count in state_counts.items()
        }
        
        # Diagonal strength (stickiness)
        diagonal_strength = np.mean(np.diag(avg_transition))
        
        return {
            'n_products_learned': len(self.transition_matrices),
            'avg_transition_matrix': avg_transition,
            'state_prevalence': state_prevalence,
            'diagonal_strength': diagonal_strength,
            'state_names': self.state_names
        }
