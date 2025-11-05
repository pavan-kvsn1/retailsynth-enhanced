import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from retailsynth.config import EnhancedRetailConfig

# ============================================================================
# MARKET CONTEXT GENERATOR (v3.2)
# ============================================================================

class MarketContextGenerator:
    """
    Generates market context with realistic attributes and pricing.
    """
    
    @staticmethod
    def generate_market_context(config: EnhancedRetailConfig) -> pd.DataFrame:
        """Generate market-level context and economic indicators"""
        market_data = []
        
        for week in range(config.simulation_weeks):
            week_date = config.start_date + timedelta(weeks=week)
            
            # Economic indicators with realistic trends
            base_consumer_confidence = 65
            seasonal_boost = 10 * np.sin(2 * np.pi * week / 52)
            confidence = base_consumer_confidence + seasonal_boost + np.random.normal(0, 3)
            
            # Unemployment rate (decreasing slightly over time)
            unemployment = 4.5 - (week * 0.01) + np.random.normal(0, 0.2)
            
            # Inflation (gradual increase)
            inflation = 2.5 + (week * 0.01) + np.random.normal(0, 0.1)
            
            market_data.append({
                'week_number': week + 1,
                'week_start_date': week_date,
                'consumer_confidence_index': round(confidence, 2),
                'unemployment_rate': round(max(3.0, unemployment), 2),
                'inflation_rate_annual': round(max(1.0, inflation), 2),
                'retail_sales_growth': round(np.random.normal(3.5, 1.5), 2),
                'created_at': datetime.now()
            })
        
        return pd.DataFrame(market_data)