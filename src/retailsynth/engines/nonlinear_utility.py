"""
Non-Linear Utility Engine (Phase 2.6)

This module implements psychologically realistic non-linear utility functions:
1. Log-price utility (diminishing marginal disutility)
2. Reference price tracking with loss aversion
3. Psychological price thresholds
4. Quadratic quality utility

Key Innovations:
- Loss aversion: 2.5x stronger response to price increases
- Psychological thresholds: $0.99 vs $1.00 effects  
- EWMA reference prices: adaptive expectations
- Diminishing returns to quality

Author: RetailSynth Team
Sprint: 2, Phase: 2.6
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NonLinearUtilityConfig:
    """Configuration for non-linear utility calculations"""
    
    # Log-price utility
    use_log_price: bool = True
    log_price_scale: float = 10.0  # Scale factor for log transformation
    
    # Reference prices & loss aversion
    use_reference_prices: bool = True
    ewma_alpha: float = 0.3  # Smoothing parameter for EWMA (0.3 = give 30% weight to new price)
    loss_aversion_lambda: float = 2.5  # Loss aversion coefficient (Kahneman & Tversky: ~2.25)
    
    # Psychological thresholds
    use_psychological_thresholds: bool = True
    threshold_bonus: float = 0.15  # Bonus utility for prices ending in .99, .95, .49
    
    # Quadratic quality utility
    use_quadratic_quality: bool = True
    quality_diminishing_factor: float = 0.8  # Coefficient for quadratic term (negative)
    
    # General
    enable_all: bool = True  # Master switch


@dataclass
class NonLinearAdjustment:
    """Result of non-linear utility adjustment"""
    base_utility: float
    reference_price_effect: float
    threshold_bonus: float
    final_utility: float


class NonLinearUtilityEngine:
    """
    Computes non-linear utility adjustments for products
    
    Applies behavioral economics principles to make utilities more realistic:
    - Log-price: U = -β * log(price) instead of -β * price
    - Reference prices: Penalties for prices above expected
    - Psychological thresholds: Bonus for charm pricing
    - Quadratic quality: Diminishing marginal utility
    """
    
    def __init__(self, config: Optional[NonLinearUtilityConfig] = None):
        """
        Initialize non-linear utility engine
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or NonLinearUtilityConfig()
        
        # Reference price tracking (product_id → reference_price)
        self.reference_prices: Dict[int, float] = {}
        
        # Track initialization
        self.initialized = False
        
        logger.info("NonLinearUtilityEngine initialized")
        logger.info(f"  • Log-price utility: {'✅' if self.config.use_log_price else '❌'}")
        logger.info(f"  • Reference prices: {'✅' if self.config.use_reference_prices else '❌'}")
        logger.info(f"  • Psychological thresholds: {'✅' if self.config.use_psychological_thresholds else '❌'}")
        logger.info(f"  • Quadratic quality: {'✅' if self.config.use_quadratic_quality else '❌'}")
    
    def initialize_reference_prices(self, products_df: pd.DataFrame, 
                                    price_column: str = 'base_price'):
        """
        Initialize reference prices from base prices
        
        Args:
            products_df: Products DataFrame with price information
            price_column: Column name for prices
        """
        for _, product in products_df.iterrows():
            base_price = float(product[price_column])
            self.reference_prices[product['product_id']] = base_price
        
        self.initialized = True
        logger.info(f"Initialized reference prices for {len(self.reference_prices):,} products")
    
    def compute_log_price_utility(self, prices: np.ndarray, 
                                  price_sensitivity: np.ndarray) -> np.ndarray:
        """
        Compute log-price utility: U = -β * log(price)
        
        This captures diminishing marginal disutility - the difference between
        $1 and $2 matters more than between $10 and $11.
        
        Args:
            prices: Array of current prices
            price_sensitivity: Array of price sensitivity parameters (β)
        
        Returns:
            Log-price utility contribution
        """
        if not self.config.use_log_price:
            # Linear utility: U = -β * price
            return -price_sensitivity * prices
        
        # Log utility: U = -β * scale * log(price)
        # Add small epsilon to avoid log(0)
        log_prices = np.log(prices + 1e-6)
        utility = -price_sensitivity * self.config.log_price_scale * log_prices
        
        return utility
    
    def compute_reference_price_effect(self, 
                                       product_ids: np.ndarray,
                                       current_prices: np.ndarray,
                                       price_sensitivity: np.ndarray) -> np.ndarray:
        """
        Compute reference price effect with loss aversion
        
        Loss aversion: Losses loom larger than gains
        - Price increase (current > reference): Strong negative effect (λ = 2.5)
        - Price decrease (current < reference): Weaker positive effect (λ = 1.0)
        
        Args:
            product_ids: Array of product IDs
            current_prices: Array of current prices
            price_sensitivity: Array of price sensitivity parameters
        
        Returns:
            Reference price utility adjustment
        """
        if not self.config.use_reference_prices or not self.initialized:
            return np.zeros_like(current_prices)
        
        reference_effect = np.zeros_like(current_prices, dtype=float)
        
        for i, product_id in enumerate(product_ids):
            ref_price = self.reference_prices.get(int(product_id))
            
            if ref_price is None:
                continue
            
            price_diff = current_prices[i] - ref_price
            
            if price_diff > 0:
                # Price increase: loss aversion applies (λ = 2.5)
                penalty = -self.config.loss_aversion_lambda * price_sensitivity[i] * price_diff
                reference_effect[i] = penalty
            elif price_diff < 0:
                # Price decrease: gain (λ = 1.0)
                bonus = -price_sensitivity[i] * price_diff  # Negative diff means positive bonus
                reference_effect[i] = bonus
            # If price_diff == 0, no effect (stays at 0)
        
        return reference_effect
    
    def compute_psychological_threshold_bonus(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute psychological threshold bonus for charm pricing
        
        Prices ending in .99, .95, .49 are perceived as significantly cheaper
        than round numbers (e.g., $0.99 vs $1.00).
        
        Args:
            prices: Array of current prices
        
        Returns:
            Threshold bonus array
        """
        if not self.config.use_psychological_thresholds:
            return np.zeros_like(prices)
        
        bonus = np.zeros_like(prices)
        
        # Extract cents portion (e.g., 0.99 from $5.99)
        cents = (prices * 100) % 100
        
        # Check for charm pricing patterns
        charm_prices = (
            (np.abs(cents - 99) < 0.5) |  # .99
            (np.abs(cents - 95) < 0.5) |  # .95
            (np.abs(cents - 49) < 0.5)    # .49
        )
        
        # Apply bonus where charm pricing is detected
        bonus[charm_prices] = self.config.threshold_bonus
        
        return bonus
    
    def compute_quadratic_quality_utility(self,
                                         quality: np.ndarray,
                                         quality_preference: np.ndarray) -> np.ndarray:
        """
        Compute quadratic quality utility: U = α*Q - γ*Q²
        
        This captures diminishing marginal utility of quality.
        The first quality increase matters more than subsequent ones.
        
        Args:
            quality: Array of product quality scores [0, 1]
            quality_preference: Array of quality preference parameters (α)
        
        Returns:
            Quadratic quality utility
        """
        if not self.config.use_quadratic_quality:
            # Linear quality utility: U = α * Q
            return quality_preference * quality
        
        # Quadratic utility: U = α*Q - γ*Q²
        linear_term = quality_preference * quality
        quadratic_term = -self.config.quality_diminishing_factor * quality_preference * (quality ** 2)
        
        return linear_term + quadratic_term
    
    def update_reference_prices(self, product_ids: np.ndarray, 
                               observed_prices: np.ndarray):
        """
        Update reference prices using EWMA
        
        EWMA: R_new = α * P_observed + (1 - α) * R_old
        where α = smoothing parameter (default: 0.3)
        
        Args:
            product_ids: Array of product IDs
            observed_prices: Array of observed prices this period
        """
        if not self.config.use_reference_prices:
            return
        
        alpha = self.config.ewma_alpha
        
        for i, product_id in enumerate(product_ids):
            product_id = int(product_id)
            observed_price = float(observed_prices[i])
            
            if product_id in self.reference_prices:
                # Update using EWMA
                old_ref = self.reference_prices[product_id]
                new_ref = alpha * observed_price + (1 - alpha) * old_ref
                self.reference_prices[product_id] = new_ref
            else:
                # Initialize if not seen before
                self.reference_prices[product_id] = observed_price
    
    def compute_all_nonlinear_effects(self,
                                     product_ids: np.ndarray,
                                     prices: np.ndarray,
                                     quality: np.ndarray,
                                     price_sensitivity: np.ndarray,
                                     quality_preference: np.ndarray,
                                     base_utility: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute all non-linear utility effects at once
        
        Args:
            product_ids: Array of product IDs
            prices: Array of current prices
            quality: Array of product quality scores
            price_sensitivity: Array of price sensitivity parameters
            quality_preference: Array of quality preference parameters
            base_utility: Optional base utility (if None, will be computed from scratch)
        
        Returns:
            Dict with keys:
                - 'log_price_utility': Log-price utility component
                - 'reference_price_effect': Reference price adjustment
                - 'threshold_bonus': Psychological threshold bonus
                - 'quality_utility': Quadratic quality utility
                - 'total_nonlinear_utility': Sum of all effects
        """
        results = {}
        
        # 1. Log-price utility (replaces linear price utility)
        results['log_price_utility'] = self.compute_log_price_utility(
            prices, price_sensitivity
        )
        
        # 2. Reference price effect (additive)
        results['reference_price_effect'] = self.compute_reference_price_effect(
            product_ids, prices, price_sensitivity
        )
        
        # 3. Psychological threshold bonus (additive)
        results['threshold_bonus'] = self.compute_psychological_threshold_bonus(prices)
        
        # 4. Quadratic quality utility (replaces linear quality utility)
        results['quality_utility'] = self.compute_quadratic_quality_utility(
            quality, quality_preference
        )
        
        # 5. Total non-linear utility
        # Note: log_price_utility and quality_utility REPLACE their linear counterparts
        # reference_price_effect and threshold_bonus are ADDITIVE
        if base_utility is not None:
            # If base utility provided, add the additive effects
            results['total_nonlinear_utility'] = (
                base_utility +
                results['reference_price_effect'] +
                results['threshold_bonus']
            )
        else:
            # If no base utility, sum all components
            results['total_nonlinear_utility'] = (
                results['log_price_utility'] +
                results['reference_price_effect'] +
                results['threshold_bonus'] +
                results['quality_utility']
            )
        
        return results
    
    def get_reference_price(self, product_id: int) -> Optional[float]:
        """Get current reference price for a product"""
        return self.reference_prices.get(product_id)
    
    def reset_reference_prices(self):
        """Reset all reference prices (for testing or re-initialization)"""
        self.reference_prices.clear()
        self.initialized = False
        logger.info("Reference prices reset")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about non-linear utility effects
        
        Returns:
            Dict with statistics
        """
        stats = {
            'n_products_tracked': len(self.reference_prices),
            'config': {
                'use_log_price': self.config.use_log_price,
                'use_reference_prices': self.config.use_reference_prices,
                'loss_aversion_lambda': self.config.loss_aversion_lambda,
                'use_psychological_thresholds': self.config.use_psychological_thresholds,
                'use_quadratic_quality': self.config.use_quadratic_quality,
            }
        }
        
        if self.reference_prices:
            ref_prices_array = np.array(list(self.reference_prices.values()))
            stats['reference_price_stats'] = {
                'mean': float(np.mean(ref_prices_array)),
                'std': float(np.std(ref_prices_array)),
                'min': float(np.min(ref_prices_array)),
                'max': float(np.max(ref_prices_array)),
            }
        
        return stats
    
    def calculate_nonlinear_adjustment(self,
                                      customer_params: Dict,
                                      base_utility: float,
                                      current_price: float,
                                      product_id: int) -> NonLinearAdjustment:
        """
        Calculate non-linear adjustments for a single customer-product pair
        
        This is used in transaction generation to add reference price effects
        and psychological threshold bonuses on top of base utility.
        
        Args:
            customer_params: Customer heterogeneity parameters (Phase 2.4)
            base_utility: Base utility already computed (includes log-price if enabled)
            current_price: Current price of product
            product_id: Product ID
        
        Returns:
            NonLinearAdjustment with breakdown and final utility
        """
        # Get customer price sensitivity
        price_sensitivity = customer_params.get('price_sensitivity_param', 1.0)
        
        # 1. Reference price effect (additive)
        ref_effect = 0.0
        if self.config.use_reference_prices and self.initialized:
            ref_price = self.reference_prices.get(product_id)
            
            if ref_price is not None:
                price_diff = current_price - ref_price
                
                if price_diff > 0:
                    # Price increase: loss aversion (λ = 2.5)
                    ref_effect = -self.config.loss_aversion_lambda * price_sensitivity * price_diff
                elif price_diff < 0:
                    # Price decrease: gain (λ = 1.0)
                    ref_effect = -price_sensitivity * price_diff
        
        # 2. Psychological threshold bonus (additive)
        threshold_bonus = 0.0
        if self.config.use_psychological_thresholds:
            cents = (current_price * 100) % 100
            
            # Check for charm pricing
            is_charm = (
                abs(cents - 99) < 0.5 or
                abs(cents - 95) < 0.5 or
                abs(cents - 49) < 0.5
            )
            
            if is_charm:
                threshold_bonus = self.config.threshold_bonus
        
        # Final utility = base + additive effects
        final_utility = base_utility + ref_effect + threshold_bonus
        
        return NonLinearAdjustment(
            base_utility=base_utility,
            reference_price_effect=ref_effect,
            threshold_bonus=threshold_bonus,
            final_utility=final_utility
        )


# Convenience function for quick testing
def demonstrate_nonlinear_effects():
    """Demonstrate non-linear utility effects with example data"""
    print("="*70)
    print("NON-LINEAR UTILITY DEMONSTRATION")
    print("="*70)
    
    # Create engine
    engine = NonLinearUtilityEngine()
    
    # Example products
    product_ids = np.array([1, 2, 3, 4, 5])
    base_prices = np.array([5.00, 9.99, 10.00, 15.95, 20.00])
    current_prices = np.array([5.50, 8.99, 10.00, 14.95, 18.00])  # Some increases, some decreases
    quality = np.array([0.7, 0.8, 0.6, 0.9, 0.85])
    price_sensitivity = np.array([1.2, 1.5, 1.0, 1.3, 1.1])
    quality_preference = np.array([0.9, 1.0, 0.8, 1.1, 0.95])
    
    # Initialize reference prices with base prices
    products_df = pd.DataFrame({
        'product_id': product_ids,
        'base_price': base_prices
    })
    engine.initialize_reference_prices(products_df)
    
    # Compute effects
    results = engine.compute_all_nonlinear_effects(
        product_ids, current_prices, quality,
        price_sensitivity, quality_preference
    )
    
    # Display results
    print("\n📊 Product Comparisons:")
    print(f"{'ID':<5} {'Base':<8} {'Current':<8} {'Ref Effect':<12} {'Threshold':<10} {'Total Δ':<10}")
    print("-" * 70)
    
    for i in range(len(product_ids)):
        print(f"{product_ids[i]:<5} "
              f"${base_prices[i]:<7.2f} "
              f"${current_prices[i]:<7.2f} "
              f"{results['reference_price_effect'][i]:>11.3f} "
              f"{results['threshold_bonus'][i]:>9.2f} "
              f"{results['total_nonlinear_utility'][i]:>9.2f}")
    
    print("\n💡 Key Insights:")
    print("  • Product 2 ($9.99): Gets threshold bonus for charm pricing")
    print("  • Product 1: Price increased (+$0.50) → strong negative effect (loss aversion)")
    print("  • Product 5: Price decreased (-$2.00) → positive effect")
    print("  • Products with .99/.95 pricing get psychological bonuses")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    # Run demonstration
    demonstrate_nonlinear_effects()
