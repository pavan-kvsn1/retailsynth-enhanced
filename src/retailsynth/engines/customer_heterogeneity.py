"""
Customer Heterogeneity Engine (Phase 2.4)

Replaces discrete customer archetypes with continuous parameter distributions.
Every customer is unique with individual behavioral parameters.

Key Parameters:
1. Price Sensitivity - How much price changes affect utility
2. Quality Preference - Preference for higher quality products
3. Promotional Responsiveness - Response to promotional signals
4. Variety Seeking - Tendency to try new products
5. Brand Loyalty - Stickiness to preferred brands
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class CustomerParameters:
    """
    Individual customer behavioral parameters
    
    All parameters are continuous values sampled from distributions
    """
    customer_id: int
    
    # Core utility parameters
    price_sensitivity: float      # [0.5, 2.5] - Higher = more price sensitive
    quality_preference: float      # [0.3, 1.5] - Higher = prefers quality over price
    
    # Promotional response
    promo_responsiveness: float    # [0.5, 2.0] - Response to discounts/promotions
    display_sensitivity: float     # [0.3, 1.2] - Response to in-store displays
    advertising_receptivity: float # [0.3, 1.5] - Response to ads (in-ad, mailer)
    
    # Shopping behavior
    variety_seeking: float         # [0.3, 1.2] - Tendency to try new products
    brand_loyalty: float           # [0.2, 1.5] - Stickiness to brands
    store_loyalty: float           # [0.3, 1.3] - Stickiness to stores
    
    # Shopping patterns
    basket_size_preference: float  # [0.5, 2.0] - Preferred basket size (relative)
    impulsivity: float            # [0.2, 1.5] - Unplanned purchase tendency
    
    # Derived segment (for analysis only)
    segment_label: str = "individual"  # No longer used for behavior


class CustomerHeterogeneityEngine:
    """
    Generates individual customer parameters from continuous distributions
    
    Instead of discrete types (Budget, Premium, etc.), every customer gets
    unique parameters sampled from realistic distributions.
    
    Distribution Types:
    - Beta distributions (bounded [0,1], then scaled)
    - Log-normal distributions (for right-skewed parameters)
    - Truncated normal distributions (for symmetric bounded parameters)
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize heterogeneity engine
        
        Args:
            random_seed: Optional seed for reproducibility
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Define parameter distributions
        self._init_distributions()
        
        logger.info("CustomerHeterogeneityEngine initialized")
    
    def _init_distributions(self):
        """
        Initialize parameter distributions
        
        Uses mixture of distribution types for realism:
        - Beta: Bounded with flexible shapes
        - Log-normal: Right-skewed (e.g., price sensitivity)
        - Truncated normal: Symmetric bounded
        """
        
        # Price Sensitivity: Log-normal (right-skewed, some very price sensitive)
        # Mean ~1.2, range [0.5, 2.5]
        self.price_sensitivity_dist = {
            'type': 'lognormal',
            'params': {'mean': 0.15, 'sigma': 0.4},
            'bounds': (0.5, 2.5)
        }
        
        # Quality Preference: Beta distribution (flexible shape)
        # Mean ~0.9, range [0.3, 1.5]
        self.quality_preference_dist = {
            'type': 'beta',
            'params': {'alpha': 5, 'beta': 3},
            'bounds': (0.3, 1.5)
        }
        
        # Promo Responsiveness: Beta (most moderate, some extreme)
        # Mean ~1.2, range [0.5, 2.0]
        self.promo_responsiveness_dist = {
            'type': 'beta',
            'params': {'alpha': 3, 'beta': 2},
            'bounds': (0.5, 2.0)
        }
        
        # Display Sensitivity: Beta (moderate)
        # Mean ~0.7, range [0.3, 1.2]
        self.display_sensitivity_dist = {
            'type': 'beta',
            'params': {'alpha': 3, 'beta': 3},
            'bounds': (0.3, 1.2)
        }
        
        # Advertising Receptivity: Beta (varied)
        # Mean ~0.8, range [0.3, 1.5]
        self.advertising_receptivity_dist = {
            'type': 'beta',
            'params': {'alpha': 2.5, 'beta': 3},
            'bounds': (0.3, 1.5)
        }
        
        # Variety Seeking: Beta (left-skewed, most habitual)
        # Mean ~0.6, range [0.3, 1.2]
        self.variety_seeking_dist = {
            'type': 'beta',
            'params': {'alpha': 2, 'beta': 4},
            'bounds': (0.3, 1.2)
        }
        
        # Brand Loyalty: Beta (U-shaped, either loyal or switchers)
        # Mean ~0.8, range [0.2, 1.5]
        self.brand_loyalty_dist = {
            'type': 'beta',
            'params': {'alpha': 3, 'beta': 2},
            'bounds': (0.2, 1.5)
        }
        
        # Store Loyalty: Beta (moderate)
        # Mean ~0.8, range [0.3, 1.3]
        self.store_loyalty_dist = {
            'type': 'beta',
            'params': {'alpha': 4, 'beta': 3},
            'bounds': (0.3, 1.3)
        }
        
        # Basket Size Preference: Log-normal (right-skewed)
        # Mean ~1.0, range [0.5, 2.0]
        self.basket_size_dist = {
            'type': 'lognormal',
            'params': {'mean': 0.0, 'sigma': 0.3},
            'bounds': (0.5, 2.0)
        }
        
        # Impulsivity: Beta (right-skewed, most controlled)
        # Mean ~0.6, range [0.2, 1.5]
        self.impulsivity_dist = {
            'type': 'beta',
            'params': {'alpha': 2, 'beta': 3.5},
            'bounds': (0.2, 1.5)
        }
    
    def _sample_parameter(self, dist_config: Dict) -> float:
        """
        Sample a parameter from its distribution
        
        Args:
            dist_config: Distribution configuration dict
        
        Returns:
            float: Sampled parameter value
        """
        dist_type = dist_config['type']
        params = dist_config['params']
        bounds = dist_config['bounds']
        
        if dist_type == 'beta':
            # Sample from Beta(alpha, beta), then scale to bounds
            raw_value = np.random.beta(params['alpha'], params['beta'])
            value = bounds[0] + raw_value * (bounds[1] - bounds[0])
        
        elif dist_type == 'lognormal':
            # Sample from log-normal, then clip to bounds
            raw_value = np.random.lognormal(params['mean'], params['sigma'])
            value = np.clip(raw_value, bounds[0], bounds[1])
        
        elif dist_type == 'truncnorm':
            # Sample from truncated normal
            a = (bounds[0] - params['mean']) / params['std']
            b = (bounds[1] - params['mean']) / params['std']
            value = stats.truncnorm.rvs(a, b, loc=params['mean'], scale=params['std'])
        
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
        
        return float(value)
    
    def generate_customer_parameters(self, customer_id: int) -> CustomerParameters:
        """
        Generate individual parameters for a customer
        
        Args:
            customer_id: Customer identifier
        
        Returns:
            CustomerParameters: Individual behavioral parameters
        """
        params = CustomerParameters(
            customer_id=customer_id,
            price_sensitivity=self._sample_parameter(self.price_sensitivity_dist),
            quality_preference=self._sample_parameter(self.quality_preference_dist),
            promo_responsiveness=self._sample_parameter(self.promo_responsiveness_dist),
            display_sensitivity=self._sample_parameter(self.display_sensitivity_dist),
            advertising_receptivity=self._sample_parameter(self.advertising_receptivity_dist),
            variety_seeking=self._sample_parameter(self.variety_seeking_dist),
            brand_loyalty=self._sample_parameter(self.brand_loyalty_dist),
            store_loyalty=self._sample_parameter(self.store_loyalty_dist),
            basket_size_preference=self._sample_parameter(self.basket_size_dist),
            impulsivity=self._sample_parameter(self.impulsivity_dist),
            segment_label=self._assign_segment_label(customer_id)  # For analysis only
        )
        
        return params
    
    def generate_population_parameters(self, n_customers: int) -> pd.DataFrame:
        """
        Generate parameters for entire customer population
        
        Args:
            n_customers: Number of customers
        
        Returns:
            DataFrame: Customer parameters for all customers
        """
        logger.info(f"Generating heterogeneous parameters for {n_customers:,} customers...")
        
        customers = []
        for customer_id in range(1, n_customers + 1):
            params = self.generate_customer_parameters(customer_id)
            customers.append({
                'customer_id': params.customer_id,
                'price_sensitivity': params.price_sensitivity,
                'quality_preference': params.quality_preference,
                'promo_responsiveness': params.promo_responsiveness,
                'display_sensitivity': params.display_sensitivity,
                'advertising_receptivity': params.advertising_receptivity,
                'variety_seeking': params.variety_seeking,
                'brand_loyalty': params.brand_loyalty,
                'store_loyalty': params.store_loyalty,
                'basket_size_preference': params.basket_size_preference,
                'impulsivity': params.impulsivity,
                'segment_label': params.segment_label
            })
        
        df = pd.DataFrame(customers)
        
        # Log distribution statistics
        self._log_distribution_stats(df)
        
        return df
    
    def _assign_segment_label(self, customer_id: int) -> str:
        """
        Assign segment label based on parameters (for analysis/visualization only)
        
        This is NOT used for behavior - it's just for understanding the population
        """
        # Use hash to consistently assign segment across calls
        np.random.seed(customer_id)
        segment_probs = [0.25, 0.30, 0.25, 0.20]  # Budget, Balanced, Premium, Cherry-picker
        segment = np.random.choice(
            ['budget_oriented', 'balanced', 'premium_seeker', 'deal_hunter'],
            p=segment_probs
        )
        
        # Reset seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed + customer_id)
        
        return segment
    
    def _log_distribution_stats(self, df: pd.DataFrame):
        """Log summary statistics of parameter distributions"""
        logger.info("Parameter Distribution Statistics:")
        
        params_to_log = [
            'price_sensitivity', 'quality_preference', 'promo_responsiveness',
            'display_sensitivity', 'advertising_receptivity', 'variety_seeking',
            'brand_loyalty', 'store_loyalty'
        ]
        
        for param in params_to_log:
            mean = df[param].mean()
            std = df[param].std()
            min_val = df[param].min()
            max_val = df[param].max()
            logger.info(f"  {param}: μ={mean:.3f}, σ={std:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
    
    def get_distribution_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for all parameters
        
        Args:
            df: DataFrame with customer parameters
        
        Returns:
            Dict: Summary statistics
        """
        summary = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'customer_id':
                summary[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q25': float(df[col].quantile(0.25)),
                    'q50': float(df[col].quantile(0.50)),
                    'q75': float(df[col].quantile(0.75))
                }
        
        return summary
