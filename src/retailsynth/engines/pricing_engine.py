import numpy as np
from typing import Tuple, List, Optional, Dict
import pandas as pd

# ============================================================================
# PRICING EVOLUTION ENGINE (v3.2)
# ============================================================================

class PricingEvolutionEngine:
    """
    Models realistic price dynamics over time including:
    - Competitive pressure
    - Promotional cycles
    - Cost inflation
    - Dynamic pricing strategies
    """
    
    def __init__(self, n_products: int):
        self.n_products = n_products
        self.inflation_rate = 0.0005  # ~2.6% annual
        self.competitive_pressure = 0.001
    
    def evolve_prices(self, base_prices: np.ndarray, week_number: int, 
                     product_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve prices based on market dynamics.
        Returns: (current_prices, promotion_flags)
        """
        # Get current number of products (may have changed due to lifecycle)
        n_current_products = len(base_prices)
        
        # Start with base prices
        current_prices = base_prices.copy()
        
        # Apply inflation
        inflation_factor = 1.0 + (self.inflation_rate * week_number)
        current_prices *= inflation_factor
        
        # Competitive pressure (slight price reduction over time)
        competitive_factor = 1.0 - (self.competitive_pressure * np.sqrt(week_number))
        current_prices *= competitive_factor
        
        # Promotional pricing (15-20% of products on promotion each week)
        n_promotions = int(n_current_products * np.random.uniform(0.15, 0.20))
        promo_indices = np.random.choice(n_current_products, size=n_promotions, replace=False)
        promotion_flags = np.zeros(n_current_products, dtype=bool)
        promotion_flags[promo_indices] = True
        
        # Apply promotion discounts (10-30% off)
        promo_discounts = np.random.uniform(0.10, 0.30, size=n_promotions)
        current_prices[promo_indices] *= (1.0 - promo_discounts)
        
        # Add small random fluctuations (±2%)
        noise = np.random.normal(1.0, 0.02, size=n_current_products)
        current_prices *= noise
        
        # Ensure minimum price
        current_prices = np.maximum(current_prices, 0.50)
        
        return current_prices, promotion_flags
    
    def visualize_price_evolution(self, pricing_history: List[Dict], 
                                 product_ids: List[int]) -> pd.DataFrame:
        """Create visualization dataset for price evolution"""
        price_data = []
        
        for week_prices in pricing_history:
            week_number = week_prices['week']
            prices = week_prices['prices']
            promos = week_prices['promotions']
            
            for product_id in product_ids:
                if product_id in prices:
                    price_data.append({
                        'week_number': week_number,
                        'product_id': product_id,
                        'price': prices[product_id],
                        'on_promotion': promos.get(product_id, False)
                    })
        
        return pd.DataFrame(price_data)
