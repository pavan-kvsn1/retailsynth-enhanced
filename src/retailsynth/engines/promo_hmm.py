"""
Promotional HMM (Sprint 2.1) - Tactical promotional state dynamics

States: No Promo (0), Light (1), Moderate (2), Heavy (3)
Learns from promotional weeks only (RETAIL_DISC > 0, DISPLAY/MAILER present)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from pathlib import Path
from tqdm import tqdm
import pickle
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PromoMechanics:
    """Promotional mechanics for a product"""
    discount_depth: float
    has_display: bool
    has_ad: bool
    promo_state: int


class PromoHMM:
    """HMM for tactical promotional dynamics"""
    
    def __init__(self, products_df: pd.DataFrame, n_states: int = 4):
        self.products_df = products_df
        self.n_states = n_states
        
        self.state_names = {0: 'no_promo', 1: 'light_promo', 2: 'moderate_promo', 3: 'heavy_promo'}
        self.state_discount_ranges = {0: (0.00, 0.05), 1: (0.05, 0.15), 2: (0.15, 0.30), 3: (0.30, 0.50)}
        
        self.transition_matrices = {}
        self.emission_distributions = {}
        self.initial_state_probs = {}
        self.promo_tendencies = {}
        self.current_states = {}
        
        logger.info(f"Initialized PromoHMM with {len(products_df)} products")
    
    def learn_from_data(self, transactions_df: pd.DataFrame, causal_df: Optional[pd.DataFrame] = None, min_observations: int = 5):
        """Learn promotional HMM from promotional weeks"""
        logger.info("Learning Promotional HMM...")
        
        # Filter to promo weeks
        promo_data = transactions_df[transactions_df['RETAIL_DISC'] > 0].copy()
        
        if len(promo_data) == 0:
            logger.warning("No promotional data found!")
            return
        
        # Calculate metrics
        product_week_promos = promo_data.groupby(['PRODUCT_ID', 'WEEK_NO']).agg({
            'SALES_VALUE': 'sum', 'QUANTITY': 'sum', 'RETAIL_DISC': 'sum'
        }).reset_index()
        
        product_week_promos['discount_depth'] = (
            product_week_promos['RETAIL_DISC'] / product_week_promos['SALES_VALUE'].replace(0, np.nan)
        ).clip(0, 0.75)
        
        # Add causal flags
        if causal_df is not None:
            causal_agg = causal_df.groupby(['PRODUCT_ID', 'WEEK_NO']).agg({
                'DISPLAY': lambda x: (x.notna() & (x != '')).any(),
                'MAILER': lambda x: (x.notna() & (x != '')).any()
            }).reset_index()
            product_week_promos = product_week_promos.merge(causal_agg, on=['PRODUCT_ID', 'WEEK_NO'], how='left')
        
        product_week_promos['DISPLAY'] = product_week_promos.get('DISPLAY', False).fillna(False)
        product_week_promos['MAILER'] = product_week_promos.get('MAILER', False).fillna(False)
        product_week_promos = product_week_promos.dropna(subset=['discount_depth'])
        
        # Learn per product
        for product_id in tqdm(self.products_df['PRODUCT_ID'].unique(), desc="Learning"):
            product_data = product_week_promos[product_week_promos['PRODUCT_ID'] == product_id]
            if len(product_data) >= min_observations:
                self._learn_product(product_id, product_data)
        
        self._initialize_states()
        logger.info(f"✅ Learned {len(self.transition_matrices)} products")
    
    def _learn_product(self, product_id: int, data: pd.DataFrame):
        """Learn parameters for one product"""
        data = data.copy()
        data['state'] = data.apply(lambda r: self._classify_state(r['discount_depth'], r['DISPLAY'], r['MAILER']), axis=1)
        data = data.sort_values('WEEK_NO')
        
        # Transition matrix
        tm = np.zeros((self.n_states, self.n_states))
        states = data['state'].values
        for i in range(len(states) - 1):
            tm[states[i], states[i + 1]] += 1
        
        for i in range(self.n_states):
            if tm[i].sum() > 0:
                tm[i] = tm[i] / tm[i].sum()
            else:
                tm[i] = [0.7, 0.2, 0.08, 0.02]
        
        self.transition_matrices[product_id] = tm
        
        # Emissions
        emissions = {}
        for state in range(self.n_states):
            state_data = data[data['state'] == state]
            if len(state_data) > 0:
                emissions[state] = {
                    'discount_mean': state_data['discount_depth'].mean(),
                    'discount_std': state_data['discount_depth'].std() if len(state_data) > 1 else 0.02,
                    'display_prob': state_data['DISPLAY'].mean(),
                    'ad_prob': state_data['MAILER'].mean()
                }
            else:
                r = self.state_discount_ranges[state]
                emissions[state] = {'discount_mean': (r[0] + r[1]) / 2, 'discount_std': (r[1] - r[0]) / 4,
                                   'display_prob': 0.5 if state >= 2 else 0.1, 'ad_prob': 0.5 if state >= 2 else 0.05}
        
        self.emission_distributions[product_id] = emissions
        
        # Initial probs
        counts = data['state'].value_counts()
        self.initial_state_probs[product_id] = {s: counts.get(s, 0) / len(data) for s in range(self.n_states)}
        self.promo_tendencies[product_id] = (data['state'] > 0).mean()
    
    def _classify_state(self, discount: float, display: bool, ad: bool) -> int:
        """Classify promo state"""
        if discount < 0.05 and not display and not ad:
            return 0
        elif discount < 0.15:
            return 1
        elif discount < 0.30 or (display and not ad):
            return 2
        else:
            return 3
    
    def _initialize_states(self):
        """Initialize current states"""
        for pid in self.initial_state_probs.keys():
            probs = list(self.initial_state_probs[pid].values())
            self.current_states[pid] = np.random.choice(self.n_states, p=probs)
    
    def sample_promo_mechanics(self, product_ids: Optional[List[int]] = None) -> Dict[int, PromoMechanics]:
        """Sample promotional mechanics"""
        if product_ids is None:
            product_ids = list(self.current_states.keys())
        
        mechanics = {}
        for pid in product_ids:
            if pid not in self.current_states:
                mechanics[pid] = PromoMechanics(0.0, False, False, 0)
                continue
            
            state = self.current_states[pid]
            em = self.emission_distributions[pid][state]
            
            discount = max(0, min(0.75, np.random.normal(em['discount_mean'], em['discount_std'])))
            display = np.random.random() < em['display_prob']
            ad = np.random.random() < em['ad_prob']
            
            mechanics[pid] = PromoMechanics(discount, display, ad, state)
        
        return mechanics
    
    def transition_states(self, product_ids: Optional[List[int]] = None):
        """Transition to next states"""
        if product_ids is None:
            product_ids = list(self.current_states.keys())
        
        for pid in product_ids:
            if pid in self.current_states:
                state = self.current_states[pid]
                tm = self.transition_matrices[pid]
                self.current_states[pid] = np.random.choice(self.n_states, p=tm[state])
    
    def save_parameters(self, filepath: Path):
        """Save parameters"""
        params = {
            'n_states': self.n_states, 'state_names': self.state_names,
            'transition_matrices': self.transition_matrices,
            'emission_distributions': self.emission_distributions,
            'initial_state_probs': self.initial_state_probs,
            'promo_tendencies': self.promo_tendencies,
            'current_states': self.current_states
        }
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        logger.info(f"✅ Saved PromoHMM to {filepath}")
    
    def load_parameters(self, filepath: Path):
        """Load parameters"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.n_states = params['n_states']
        self.state_names = params['state_names']
        self.transition_matrices = params['transition_matrices']
        self.emission_distributions = params['emission_distributions']
        self.initial_state_probs = params['initial_state_probs']
        self.promo_tendencies = params['promo_tendencies']
        self.current_states = params['current_states']
        logger.info(f"✅ Loaded PromoHMM from {filepath}")
