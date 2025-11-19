import numpy as np
from typing import Tuple, List, Optional, Dict
import pandas as pd

# ============================================================================
# PRICING EVOLUTION ENGINE (v4.0 - Sprint 2.1)
# Base price evolution only - promotions handled separately
# ============================================================================

class PricingEvolutionEngine:
    """
    Models realistic base price dynamics over time including:
    - Cost inflation
    - Competitive pressure
    - Market dynamics
    
    Note: Promotional pricing is now handled by PromotionalEngine
    """
    
    def __init__(self, n_products: int, config: Optional[Dict] = None):
        self.n_products = n_products
        
        # Load from config or use defaults
        if config:
            self.inflation_rate = config.get('inflation_rate', 0.0005)
            self.competitive_pressure = config.get('competitive_pressure', 0.001)
            self.price_volatility = config.get('price_volatility', 0.02)
        else:
            self.inflation_rate = 0.0005  # ~2.6% annual
            self.competitive_pressure = 0.001
            self.price_volatility = 0.02  # ±2% random fluctuations
        
        self.min_price = 0.50  # Minimum product price
    
    def evolve_prices(self, base_prices: np.ndarray, week_number: int, 
                     product_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evolve base prices based on market dynamics.
        
        Components:
        1. Inflation - Long-term price increase (~2.6% annual)
        2. Competitive pressure - Gradual price reduction
        3. Random fluctuations - Short-term price volatility
        
        Args:
            base_prices: Array of base prices for products
            week_number: Current week number
            product_ids: Optional array of product IDs (for future use)
        
        Returns:
            np.ndarray: Current base prices (no promotional discounts)
        """
        # Get current number of products (may have changed due to lifecycle)
        n_current_products = len(base_prices)
        
        # Start with base prices
        current_prices = base_prices.copy()
        
        # 1. Apply inflation (cost increases over time)
        inflation_factor = 1.0 + (self.inflation_rate * week_number)
        current_prices *= inflation_factor
        
        # 2. Competitive pressure (slight price reduction over time)
        # Square root to slow down the effect
        competitive_factor = 1.0 - (self.competitive_pressure * np.sqrt(week_number))
        competitive_factor = np.maximum(competitive_factor, 0.90)  # Cap at 10% reduction
        current_prices *= competitive_factor
        
        # 3. Add small random fluctuations (±2%)
        noise = np.random.normal(1.0, self.price_volatility, size=n_current_products)
        current_prices *= noise
        
        # 4. Ensure minimum price
        current_prices = np.maximum(current_prices, self.min_price)
        
        return current_prices
    
    def get_base_price_at_week(self, initial_price: float, week_number: int) -> float:
        """
        Calculate base price for a single product at a given week
        
        Useful for reference price calculations and validation
        """
        price = initial_price
        
        # Apply inflation
        price *= (1.0 + self.inflation_rate * week_number)
        
        # Apply competitive pressure
        competitive_factor = 1.0 - (self.competitive_pressure * np.sqrt(week_number))
        competitive_factor = max(competitive_factor, 0.90)
        price *= competitive_factor
        
        # Ensure minimum
        price = max(price, self.min_price)
        
        return price
    
    def visualize_price_evolution(self, pricing_history: List[Dict], 
                                 product_ids: List[int]) -> pd.DataFrame:
        """
        Create visualization dataset for base price evolution
        
        Note: This now shows base prices only (no promotional effects)
        """
        price_data = []
        
        for week_prices in pricing_history:
            week_number = week_prices['week']
            prices = week_prices['prices']
            
            for product_id in product_ids:
                if product_id in prices:
                    price_data.append({
                        'week_number': week_number,
                        'product_id': product_id,
                        'base_price': prices[product_id]
                    })
        
        return pd.DataFrame(price_data)
    
    def get_price_dynamics_summary(self, week_number: int) -> Dict:
        """
        Get summary of price dynamics at a given week
        
        Useful for debugging and validation
        """
        return {
            'week_number': week_number,
            'inflation_factor': 1.0 + (self.inflation_rate * week_number),
            'competitive_factor': max(1.0 - (self.competitive_pressure * np.sqrt(week_number)), 0.90),
            'expected_price_change': ((1.0 + self.inflation_rate * week_number) * 
                                     max(1.0 - self.competitive_pressure * np.sqrt(week_number), 0.90))
        }
