"""
Hidden Markov Model for Product Price States (Sprint 2.1)

This module implements HMM-based price dynamics learning from Dunnhumby data.
Price states capture unobservable market conditions:
- State 0: Regular pricing (baseline)
- State 1: Feature pricing (in-ad, moderate discount)
- State 2: Deep discount (TPR - Temporary Price Reduction)

Theoretical Foundation:
    State transitions follow Markov property:
    P(State_t | State_{t-1}, State_{t-2}, ...) = P(State_t | State_{t-1})

References:
    - RetailSynth paper methodology
    - Dunnhumby Complete Journey dataset structure
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import pickle
import logging

logger = logging.getLogger(__name__)


class PriceStateHMM:
    """
    Hidden Markov Model for product price states
    
    Learns product-specific transition matrices and emission distributions
    from historical transaction and promotional data.
    
    Attributes:
        n_states (int): Number of hidden states (default: 3)
        states (dict): State definitions with discount ranges
        transition_matrices (dict): Product-specific transition matrices
        emission_distributions (dict): Price distributions per state
        initial_state_probs (dict): Initial state probabilities
    """
    
    def __init__(self, products_df: pd.DataFrame, n_states: int = 4):
        """
        Initialize HMM with product catalog
        
        Args:
            products_df: Product catalog with PRODUCT_ID
            n_states: Number of hidden states (default: 3)
        """
        self.products = products_df
        self.n_states = n_states
        
        # State definitions (empirically calibrated from retail data)
        self.states = {
            0: {'name': 'regular', 'discount_range': (0.0, -0.05)},
            1: {'name': 'feature', 'discount_range': (-0.10, -0.25)},
            2: {'name': 'deep_discount', 'discount_range': (-0.25, -0.50)},
            3: {'name': 'clearance', 'discount_range': (-0.50, -1.00)},
        }
        
        # Learned parameters (populated by learn_from_data)
        self.transition_matrices = {}
        self.emission_distributions = {}
        self.initial_state_probs = {}
        
        logger.info(f"Initialized PriceStateHMM with {len(products_df)} products, {n_states} states")
    
    def learn_from_data(self, 
                       transactions_df: pd.DataFrame, 
                       causal_df: Optional[pd.DataFrame] = None):
        """
        Learn HMM parameters from Dunnhumby transaction data
        
        Args:
            transactions_df: Transaction data with columns:
                - PRODUCT_ID, WEEK_NO, SALES_VALUE, QUANTITY, RETAIL_DISC
            causal_df: Optional promotional data with columns:
                - PRODUCT_ID, STORE_ID, WEEK_NO, DISPLAY, MAILER
        
        Process:
            1. Aggregate to product-week level
            2. Calculate discount percentages
            3. Infer hidden states from discounts + promotions
            4. Estimate transition matrices
            5. Estimate emission distributions
        """
        logger.info("Learning HMM parameters from transaction data...")
        
        # Step 1: Aggregate to product-week level
        product_week_prices = transactions_df.groupby(['PRODUCT_ID', 'WEEK_NO']).agg({'SALES_VALUE': 'sum',
                                                                                      'QUANTITY': 'sum',
                                                                                      'RETAIL_DISC': 'sum'}).reset_index()
        
        # Calculate average price and discount percentage
        product_week_prices['avg_price'] = product_week_prices['SALES_VALUE'] / product_week_prices['QUANTITY']
        product_week_prices['discount_pct'] = product_week_prices['RETAIL_DISC'] / product_week_prices['SALES_VALUE']
        
        # Step 2: Merge with promotional data if available
        if causal_df is not None:
            # Aggregate promotional flags to product-week
            promo_agg = causal_df.groupby(['PRODUCT_ID', 'WEEK_NO']).agg({
                'DISPLAY': lambda x: (x == 'A').any(),  # Any display activity
                'MAILER': lambda x: (x == 'A').any()     # Any mailer activity
            }).reset_index()
            
            product_week_prices = product_week_prices.merge(promo_agg, on=['PRODUCT_ID', 'WEEK_NO'], how='left')
            product_week_prices['DISPLAY'] = product_week_prices['DISPLAY'].fillna(False)
            product_week_prices['MAILER'] = product_week_prices['MAILER'].fillna(False)
        else:
            product_week_prices['DISPLAY'] = False
            product_week_prices['MAILER'] = False
        
        # Step 3: Infer hidden states
        product_week_prices['state'] = product_week_prices.apply(self._classify_state, axis=1)
        
        # Step 4: Learn parameters for each product
        logger.info(f"Learning parameters for {len(self.products)} products...")
        
        learned_count = 0
        for product_id in tqdm(self.products['PRODUCT_ID'].unique(), desc="Learning HMM"):
            product_data = product_week_prices[product_week_prices['PRODUCT_ID'] == product_id].sort_values('WEEK_NO')
            
            if len(product_data) < 20:  # Need sufficient history
                continue
            
            # Learn transition matrix
            transition_matrix = self._estimate_transition_matrix(product_data['state'].values)
            self.transition_matrices[product_id] = transition_matrix
            
            # Learn emission distributions (price given state)
            emission_dists = self._estimate_emission_distributions(product_data[['state', 'avg_price', 'discount_pct']])
            self.emission_distributions[product_id] = emission_dists
            
            # Initial state probabilities
            initial_probs = product_data['state'].value_counts(normalize=True).to_dict()
            # Ensure all states have some probability
            for state in [0, 1, 2, 3]:
                if state not in initial_probs:
                    initial_probs[state] = 0.01
            # Normalize
            total = sum(initial_probs.values())
            initial_probs = {k: v/total for k, v in initial_probs.items()}
            self.initial_state_probs[product_id] = initial_probs
            
            learned_count += 1
        
        logger.info(f"✅ Learned HMM parameters for {learned_count:,} products")
        
        # Log summary statistics
        self._log_summary_statistics()
    
    def _classify_state(self, row: pd.Series) -> int:
        """
        Classify price state from discount and promotional flags
        
        Args:
            row: DataFrame row with discount_pct, DISPLAY, MAILER
        
        Returns:
            State index (0, 1, or 2)
        """
        discount = row['discount_pct']
        has_display = row['DISPLAY']
        has_mailer = row['MAILER']
        
        # Deep discount: high discount OR both display and mailer
        if discount >= -0.10 or (has_display and has_mailer):
            return 0
        # Deep discount: high discount OR both display and mailer
        if discount >= -0.25 or (has_display and has_mailer):
            return 1
        # Feature: moderate discount OR any promotion
        elif discount >= -0.50 or has_display or has_mailer:
            return 2
        # Regular: low/no discount, no promotion
        else:
            return 3
    
    def _estimate_transition_matrix(self, state_sequence: np.ndarray) -> np.ndarray:
        """
        Estimate state transition matrix from observed sequence
        
        Args:
            state_sequence: Array of observed states over time
        
        Returns:
            3x3 matrix where element [i,j] = P(state_j | state_i)
        """
        transitions = np.zeros((self.n_states, self.n_states))
        
        # Count transitions
        for t in range(len(state_sequence) - 1):
            current_state = int(state_sequence[t])
            next_state = int(state_sequence[t + 1])
            transitions[current_state, next_state] += 1
        
        # Normalize rows with Laplace smoothing
        transitions += 1  # Add-one smoothing
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_matrix = transitions / row_sums
        
        return transition_matrix
    
    def _estimate_emission_distributions(self, data: pd.DataFrame) -> Dict:
        """
        Estimate price distribution for each state
        
        Args:
            data: DataFrame with columns: state, avg_price, discount_pct
        
        Returns:
            Dictionary mapping state -> {price_mean, price_std, discount_mean, discount_std}
            Only includes states that were actually observed (no fake values)
        """
        emissions = {}
        
        for state in range(self.n_states):
            state_data = data[data['state'] == state]
            
            # Only add emission parameters if we have sufficient observations
            if len(state_data) >= 2:  # Need at least 2 observations for meaningful std
                # Calculate std with NaN handling
                price_std = state_data['avg_price'].std()
                discount_std = state_data['discount_pct'].std()
                
                # Handle NaN or zero std (all identical values)
                if pd.isna(price_std) or price_std == 0:
                    price_std = 0.01
                if pd.isna(discount_std) or discount_std == 0:
                    discount_std = 0.01
                
                emissions[state] = {
                    'price_mean': state_data['avg_price'].mean(),
                    'price_std': price_std,
                    'discount_mean': state_data['discount_pct'].mean(),
                    'discount_std': discount_std
                }
            # If < 2 observations, don't add this state - insufficient data
        
        return emissions
    
    def generate_price_sequence(self, 
                                product_id: int, 
                                n_weeks: int,
                                base_price: float,
                                random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate price sequence for a product using learned HMM
        
        Args:
            product_id: Product ID
            n_weeks: Number of weeks to generate
            base_price: Base price (regular price)
            random_seed: Random seed for reproducibility
        
        Returns:
            Tuple of (prices, states):
                - prices: Array of prices for each week
                - states: Array of hidden states for each week
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Get learned parameters (or use similar product)
        if product_id not in self.transition_matrices:
            product_id = self._get_similar_product(product_id)
        
        transition_matrix = self.transition_matrices[product_id]
        emission_dists = self.emission_distributions[product_id]
        initial_probs = self.initial_state_probs.get(
            product_id, 
            {0: 0.7, 1: 0.2, 2: 0.1}
        )
        
        # Generate state sequence
        states = np.zeros(n_weeks, dtype=int)
        
        # Determine actual number of states from transition matrix
        n_actual_states = transition_matrix.shape[0]
        state_range = list(range(n_actual_states))

        # Initial state
        state_probs = [initial_probs.get(s, 0.33) for s in state_range]
        state_probs = np.array(state_probs) / sum(state_probs)  # Normalize
        states[0] = np.random.choice(state_range, p=state_probs)
        
        # State transitions
        for t in range(1, n_weeks):
            prev_state = states[t-1]
            transition_probs = transition_matrix[prev_state]
            states[t] = np.random.choice(state_range, p=transition_probs)
        
        # Generate prices from states
        prices = np.zeros(n_weeks)
        
        for t in range(n_weeks):
            state = states[t]
            emission = emission_dists.get(state, None)
            
            if emission is None:
                # State not observed in training data - use default discount
                if state == 0:  # Regular
                    discount = 0.0
                elif state == 1:  # Feature
                    discount = 0.10
                elif state == 2:  # Deep discount
                    discount = 0.25
                else:  # Clearance
                    discount = 0.40
            else:
                # Apply discount based on learned emission
                discount = max(0, min(0.6, np.random.normal(
                    emission['discount_mean'],
                    emission['discount_std']
                )))
            
            prices[t] = base_price * (1 - discount)
            
            # Add small noise for realism
            prices[t] *= np.random.normal(1.0, 0.02)
            prices[t] = max(0.50, prices[t])  # Floor price
        
        return prices, states
    
    def _get_similar_product(self, product_id: int) -> int:
        """
        Find product with similar characteristics for fallback
        
        Args:
            product_id: Target product ID
        
        Returns:
            Similar product ID with learned parameters
        """
        if product_id not in self.products['product_id'].values:
            # Return any product with parameters
            if len(self.transition_matrices) > 0:
                return list(self.transition_matrices.keys())[0]
            else:
                raise ValueError("No HMM parameters learned yet")
        
        product = self.products[self.products['product_id'] == product_id].iloc[0]
        
        # Find products in same commodity
        similar = self.products[
            self.products['commodity_desc'] == product.get('commodity_desc', '')
        ]
        
        # Return one with learned parameters
        for similar_id in similar['product_id']:
            if similar_id in self.transition_matrices:
                return similar_id
        
        # Fallback to any product with parameters
        if len(self.transition_matrices) > 0:
            return list(self.transition_matrices.keys())[0]
        else:
            raise ValueError("No HMM parameters learned yet")
    
    def _log_summary_statistics(self):
        """Log summary statistics of learned parameters"""
        if len(self.transition_matrices) == 0:
            logger.warning("No HMM parameters learned")
            return
        
        # Average transition probabilities
        avg_transition = np.mean(
            [tm for tm in self.transition_matrices.values()],
            axis=0
        )
        
        logger.info("\nAverage Transition Matrix:")
        logger.info(f"  Regular -> Regular: {avg_transition[0,0]:.3f}")
        logger.info(f"  Regular -> Feature: {avg_transition[0,1]:.3f}")
        logger.info(f"  Regular -> Deep: {avg_transition[0,2]:.3f}")
        logger.info(f"  Regular -> Clearance: {avg_transition[0,3]:.3f}")
        logger.info(f"  Feature -> Regular: {avg_transition[1,0]:.3f}")
        logger.info(f"  Feature -> Feature: {avg_transition[1,1]:.3f}")
        logger.info(f"  Feature -> Deep: {avg_transition[1,2]:.3f}")
        logger.info(f"  Feature -> Clearance: {avg_transition[1,3]:.3f}")
        logger.info(f"  Deep -> Regular: {avg_transition[2,0]:.3f}")
        logger.info(f"  Deep -> Feature: {avg_transition[2,1]:.3f}")
        logger.info(f"  Deep -> Deep: {avg_transition[2,2]:.3f}")
        logger.info(f"  Deep -> Clearance: {avg_transition[2,3]:.3f}")
        logger.info(f"  Clearance -> Regular: {avg_transition[3,0]:.3f}")
        logger.info(f"  Clearance -> Feature: {avg_transition[3,1]:.3f}")
        logger.info(f"  Clearance -> Deep: {avg_transition[3,2]:.3f}")
        logger.info(f"  Clearance -> Clearance: {avg_transition[3,3]:.3f}")
        
        
        # State prevalence
        state_prevalence = {0: 0, 1: 0, 2: 0, 3: 0}
        for probs in self.initial_state_probs.values():
            for state, prob in probs.items():
                state_prevalence[state] += prob
        
        total = sum(state_prevalence.values())
        logger.info("\nAverage State Prevalence:")
        logger.info(f"  Regular: {state_prevalence[0]/total:.1%}")
        logger.info(f"  Feature: {state_prevalence[1]/total:.1%}")
        logger.info(f"  Deep Discount: {state_prevalence[2]/total:.1%}")
        logger.info(f"  Clearance: {state_prevalence[3]/total:.1%}")
    
    def save_parameters(self, output_path: str):
        """
        Save learned HMM parameters to disk
        
        Args:
            output_path: Path to save pickle file
        """
        hmm_data = {
            'transition_matrices': self.transition_matrices,
            'emission_distributions': self.emission_distributions,
            'initial_state_probs': self.initial_state_probs,
            'states': self.states,
            'n_states': self.n_states
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(hmm_data, f)
        
        logger.info(f"✅ Saved HMM parameters to {output_path}")
    
    def load_parameters(self, input_path: str):
        """
        Load learned HMM parameters from disk
        
        Args:
            input_path: Path to pickle file
        """
        with open(input_path, 'rb') as f:
            hmm_data = pickle.load(f)
        
        self.transition_matrices = hmm_data['transition_matrices']
        self.emission_distributions = hmm_data['emission_distributions']
        self.initial_state_probs = hmm_data['initial_state_probs']
        self.states = hmm_data['states']
        self.n_states = hmm_data['n_states']
        
        logger.info(f"✅ Loaded HMM parameters from {input_path}")
        logger.info(f"   Products with parameters: {len(self.transition_matrices):,}")