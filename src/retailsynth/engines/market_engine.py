
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


# ============================================================================
# MARKET DYNAMICS ENGINE (v3.2)
# ============================================================================

class MarketDynamicsEngine:
    """
    Models competitive market dynamics and store-level performance.
    """
    
    def __init__(self, n_stores: int):
        self.n_stores = n_stores
        self.market_share = np.ones(n_stores) / n_stores  # Initialize equal
    
    def update_market_shares(self, week_number: int, 
                           store_performance: Dict[int, float]) -> np.ndarray:
        """
        Update market shares based on store performance.
        Better performing stores gain market share.
        """
        if not store_performance:
            return self.market_share
        
        # Calculate performance scores
        store_ids = sorted(store_performance.keys())
        scores = np.array([store_performance[sid] for sid in store_ids])
        
        # Normalize scores to get new shares
        total_score = np.sum(scores)
        if total_score > 0:
            new_shares = scores / total_score
            
            # Smooth transition (80% old, 20% new)
            self.market_share = 0.8 * self.market_share + 0.2 * new_shares
        
        return self.market_share
    
    def visualize_market_share_evolution(self, 
                                        market_share_history: List[Dict]) -> pd.DataFrame:
        """Create visualization dataset for market share evolution"""
        share_data = []
        
        for week_data in market_share_history:
            week_number = week_data['week']
            shares = week_data['shares']
            
            for store_id, share in shares.items():
                share_data.append({
                    'week_number': week_number,
                    'store_id': store_id,
                    'market_share': round(share, 4)
                })
        
        return pd.DataFrame(share_data)
