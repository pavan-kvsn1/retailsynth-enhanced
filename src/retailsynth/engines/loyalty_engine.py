
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

# ============================================================================
# STORE LOYALTY ENGINE (v3.6)
# ============================================================================

class StoreLoyaltyEngine:
    """
    Manages realistic store preferences and habitual shopping patterns (v3.6).
    Each customer has 2-4 preferred stores with evolving preferences.
    """
    
    def __init__(self, customers_df: pd.DataFrame, stores_df: pd.DataFrame):
        print("   🔧 Initializing store loyalty engine...")
        self.customers = customers_df
        self.stores = stores_df
        self.n_customers = len(customers_df)
        self.n_stores = len(stores_df)
        
        # Initialize store preferences for each customer
        self.store_preferences = self._initialize_store_preferences()
    
    def _initialize_store_preferences(self) -> Dict[int, Dict]:
        """
        Initialize store preferences for all customers.
        Each customer has 2-4 preferred stores.
        """
        preferences = {}
        
        for _, customer in self.customers.iterrows():
            customer_id = customer['customer_id']
            
            # Number of preferred stores
            n_preferred = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2])
            preferred_stores = np.random.choice(
                self.stores['store_id'].values,
                size=n_preferred,
                replace=False
            )
            
            # Assign preference weights (sum to ~0.8, leaving 0.2 for exploration)
            weights = np.random.dirichlet(np.ones(n_preferred))
            weights = weights * 0.8  # 80% to preferred stores
            
            store_prefs = {}
            for store_id, weight in zip(preferred_stores, weights):
                store_prefs[int(store_id)] = float(weight)
            
            preferences[customer_id] = {
                'preferred_stores': store_prefs,
                'satisfaction_history': defaultdict(list)
            }
        
        return preferences
    
    def select_store_for_customer(self, customer_id: int, week_number: int) -> int:
        """
        Select store for customer's shopping trip.
        80% preferred stores, 15% exploration, 5% impulse/convenience
        """
        if customer_id not in self.store_preferences:
            # Fallback to random store
            return np.random.choice(self.stores['store_id'].values)
        
        prefs = self.store_preferences[customer_id]
        preferred_stores = prefs['preferred_stores']
        
        # Decision: preferred (80%), explore (15%), impulse (5%)
        choice_type = np.random.choice(['preferred', 'explore', 'impulse'], 
                                      p=[0.80, 0.15, 0.05])
        
        if choice_type == 'preferred' and preferred_stores:
            # Sample from preferred stores based on weights
            stores = list(preferred_stores.keys())
            weights = list(preferred_stores.values())
            # Normalize weights
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                return np.random.choice(stores, p=weights)
        
        # Exploration or impulse: random store
        return np.random.choice(self.stores['store_id'].values)
    
    def update_store_preference(self, customer_id: int, store_id: int, 
                               satisfaction_score: float, week_number: int):
        """
        Update store preference based on satisfaction.
        High satisfaction strengthens preference, low satisfaction weakens it.
        """
        if customer_id not in self.store_preferences:
            return
        
        prefs = self.store_preferences[customer_id]
        preferred_stores = prefs['preferred_stores']
        satisfaction_history = prefs['satisfaction_history']
        
        # Record satisfaction
        satisfaction_history[store_id].append(satisfaction_score)
        
        # Update preference based on satisfaction
        if store_id in preferred_stores:
            current_pref = preferred_stores[store_id]
            
            if satisfaction_score > 0.8:
                # High satisfaction: strengthen preference
                new_pref = min(0.5, current_pref * 1.05)
            elif satisfaction_score < 0.6:
                # Low satisfaction: weaken preference
                new_pref = max(0.05, current_pref * 0.95)
            else:
                # Neutral: small random walk
                new_pref = current_pref * (1 + np.random.uniform(-0.02, 0.02))
            
            preferred_stores[store_id] = new_pref
        else:
            # New store: add to preferences if satisfaction is high
            if satisfaction_score > 0.75 and len(preferred_stores) < 4:
                preferred_stores[store_id] = 0.1
        
        # Normalize preferences to sum to ~0.8
        total = sum(preferred_stores.values())
        if total > 0:
            for store_id in preferred_stores:
                preferred_stores[store_id] = (preferred_stores[store_id] / total) * 0.8

