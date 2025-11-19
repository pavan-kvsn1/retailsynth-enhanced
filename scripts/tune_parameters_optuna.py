"""
Automated Parameter Tuning with Optuna

This script uses Bayesian optimization to find optimal RetailSynth parameters
that best match real Dunnhumby data distributions.

**Sprint 2 Complete**: Now includes ALL Sprint 2 phases (2.1-2.7):
  - Phase 2.1-2.2: Promotional frequency and mechanics
  - Phase 2.3: Marketing signal weights
  - Phase 2.4: Customer heterogeneity distributions
  - Phase 2.5: Promotional response (already in original Tier 2)
  - Phase 2.6: Non-linear utilities (loss aversion, reference prices)
  - Phase 2.7: Seasonality learning (confidence threshold)

**Parameter Tiers**:
- Tier 1 (15 params): Directly calibratable - direct observable effect
- Tier 2 (20 params): Indirectly calibratable - aggregate effect (8 original + 12 Sprint 2)
- Tier 3 (29 params): Not calibratable - fixed at literature defaults

See docs/PARAMETER_TIER_CLASSIFICATION.md for full rationale.

Usage:
    # Tune Tier 1 only (recommended, fast)
    python scripts/tune_parameters_optuna.py \
        --real-data data/raw/dunnhumby/transaction_data.csv \
        --tier 1 \
        --n-trials 50 \
        --output outputs/tuning_tier1
    
    # Tune Tier 1 + 2 (slower, includes behavioral economics & seasonality)
    python scripts/tune_parameters_optuna.py \
        --real-data data/raw/dunnhumby/transaction_data.csv \
        --tier 1,2 \
        --n-trials 100 \
        --output outputs/tuning_tier1_2

Author: RetailSynth Team
Date: November 2024
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import optuna
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retailsynth.config import EnhancedRetailConfig
from retailsynth.generators.main_generator import EnhancedRetailSynthV4_1


class ParameterTuner:
    """
    Tunes RetailSynth parameters using Optuna to match real data distributions
    """
    
    def __init__(self, real_data_path: str, objective: str = 'combined', tiers: list = [1], synthetic_data_path: str = None):
        """
        Initialize tuner
        
        Args:
            real_data_path: Path to real transaction data CSV
            objective: Optimization objective ('basket_size', 'revenue', 'visit_frequency', 'combined')
            tiers: Which parameter tiers to tune (1, 2, or [1,2])
            synthetic_data_path: Optional path to pre-generated synthetic data (for faster tuning)
        """
        self.real_data_path = real_data_path
        self.objective = objective
        self.tiers = tiers if isinstance(tiers, list) else [tiers]
        self.synthetic_data_path = synthetic_data_path
        
        # Load and compute target distributions
        print(f"\n📊 Loading real data from: {real_data_path}")
        self.real_df = pd.read_csv(real_data_path)
        self._compute_target_distributions()
        
        print(f"✅ Target distributions computed")
        print(f"   • Basket size: mean={self.target_basket_size_mean:.2f}, std={self.target_basket_size_std:.2f}")
        print(f"   • Revenue: mean=${self.target_revenue_mean:.2f}, std=${self.target_revenue_std:.2f}")
        print(f"   • Visit frequency: mean={self.target_visit_freq_mean:.2f}, std={self.target_visit_freq_std:.2f}")
        print(f"\n🎯 Tuning Tier(s): {', '.join(map(str, self.tiers))}")
        
        tier_counts = {1: 15, 2: 27, 3: 29}  # Tier 2: 8 original + 12 Sprint 2 + 7 distribution fixes
        total_params = sum(tier_counts[t] for t in self.tiers)
        print(f"   • Parameters to tune: {total_params}")
         
    def _compute_target_distributions(self):
        """Compute target distributions from real data"""
        # Basket size distribution
        basket_sizes = self.real_df.groupby('transaction_id').size()
        self.target_basket_size_dist = basket_sizes.values
        self.target_basket_size_mean = basket_sizes.mean()
        self.target_basket_size_std = basket_sizes.std()
        
        # Revenue distribution
        revenues = self.real_df.groupby('transaction_id')['line_total'].sum()
        self.target_revenue_dist = revenues.values
        self.target_revenue_mean = revenues.mean()
        self.target_revenue_std = revenues.std()
        
        # Visit frequency distribution (NORMALIZED by time period)
        # Instead of total visits, compute visits per week
        # This makes it comparable across different time periods
        self.real_df['transaction_date'] = pd.to_datetime(self.real_df['transaction_date'])

        # CRITICAL FIX: Count distinct WEEKS visited, not transaction IDs (same as synthetic data)
        real_customer_visits = self.real_df.groupby('customer_id')['transaction_date'].agg([
            ('weeks_visited', lambda x: x.dt.isocalendar().week.nunique()),  # Distinct weeks
            ('first_date', 'min'),
            ('last_date', 'max')
        ]).reset_index()

        # Calculate weeks span
        real_customer_visits['weeks_span'] = (
            (real_customer_visits['last_date'] - real_customer_visits['first_date']).dt.days / 7 + 1
        )

        # Visit frequency = weeks visited / weeks span
        visit_freq_per_week = real_customer_visits['weeks_visited'] / real_customer_visits['weeks_span']
        self.target_visit_freq_dist = visit_freq_per_week.values
        self.target_visit_freq_mean = visit_freq_per_week.mean()
        self.target_visit_freq_std = visit_freq_per_week.std()
        
        # Quantity distribution
        self.target_quantity_dist = self.real_df['quantity'].values
        self.target_quantity_max_limit = self.real_df['quantity'].mean() + 3 * self.real_df['quantity'].std()
        self.target_quantity_min_limit = self.real_df['quantity'].mean() - 3 * self.real_df['quantity'].std()
        self.target_qunatity_dist = self.target_quantity_dist[(self.target_quantity_dist > self.target_quantity_min_limit) & (self.target_quantity_dist < self.target_quantity_max_limit)]
        self.target_quantity_mean = self.target_qunatity_dist.mean()
        self.target_quantity_std = self.target_qunatity_dist.std()
    
        # Target marketing signal strength (promotional penetration from real data)
        real_promo_penetration = len(self.real_df[self.real_df['retail_discount'] < 0]) / len(self.real_df)
        self.target_marketing_signal = real_promo_penetration
    
    def _suggest_parameters(self, trial: optuna.Trial) -> EnhancedRetailConfig:
        """
        Suggest parameter values for this trial
        
        Only tunes Tier 1 & 2 parameters. Tier 3 fixed at defaults.
        
        IMPORTANT: Uses SMALL SCALE for fast tuning (~2-3 min/trial)
        - 1,000 customers (vs 10,000+ in production)
        - 1,000 products (vs 15,000+ in production)
        - 20 weeks (vs 52+ in production)
        
        Rationale: Parameter effects on distributions are scale-invariant.
        A parameter that improves basket size distribution at 1k customers
        will also improve it at 10k customers.
        
        Returns:
            Config object with suggested parameters
        """
        config = EnhancedRetailConfig()
        
        # ===================================================================
        # TUNING SCALE: Small for fast iteration
        # ===================================================================
        config.n_customers = 500       # ~2-3 min generation time
        config.n_products = 3000        # Enough for distribution patterns
        config.n_stores = 10             # Reduced from 10
        config.simulation_weeks = 10    # Enough for behavioral patterns
        config.enable_multiple_visits_per_week = False

        print(f"   🔧 Tuning scale: {config.n_customers} customers, {config.n_products} products, {config.simulation_weeks} weeks")
        
        # ===================================================================
        # TIER 1: DIRECTLY CALIBRATABLE (15 parameters)
        # ===================================================================
        
        if 1 in self.tiers:
            print(f"   🔧 Tuning Tier 1 parameters...")
            
            # 1. Visit Behavior (1 param)
            config.base_visit_probability = trial.suggest_float('base_visit_prob', 0.15, 0.5)
            
            # 2. Basket Size (1 param)
            config.basket_size_lambda = trial.suggest_float('basket_size_lambda', 1.0, 15.0)
            
            # 3. Quantity Distribution (3 params)
            config.quantity_mean = trial.suggest_float('quantity_mean', 1.2, 5)
            config.quantity_std = trial.suggest_float('quantity_std', 0.5, 2.5)
            config.quantity_max = trial.suggest_int('quantity_max', 5, 15)
            
            # 4. Temporal Dynamics (2 params)
            config.inventory_depletion_rate = trial.suggest_float('inventory_depletion_rate', 0.05, 0.30)
            config.replenishment_threshold = trial.suggest_float('replenishment_threshold', 0.2, 0.5)
            
            # 5. Basket Composition (3 params)
            config.complement_probability = trial.suggest_float('complement_probability', 0.2, 0.7)
            config.substitute_avoidance = trial.suggest_float('substitute_avoidance', 0.4, 0.8)
            config.category_diversity_weight = trial.suggest_float('category_diversity_weight', 0.1, 0.6)
            
            # 6. Purchase History Weights (5 params)
            config.loyalty_weight = trial.suggest_float('loyalty_weight', 0.2, 0.5)  # Was 0.1-0.6, raised for stronger feedback
            config.habit_weight = trial.suggest_float('habit_weight', 0.2, 0.5)      # Was 0.2-0.7, raised for temporal dynamics
            config.inventory_weight = trial.suggest_float('inventory_weight', 0.2, 0.5)
            config.variety_weight = trial.suggest_float('variety_weight', 0.2, 0.5)
            config.price_memory_weight = trial.suggest_float('price_memory_weight', 0.05, 0.5)
            
            # 7. Trip Purpose Basket Sizes (5 params) - NEW: Replaces hardcoded TRIP_CHARACTERISTICS
            config.trip_stock_up_basket_mean = trial.suggest_float('trip_stock_up_basket', 8.0, 15.0)
            config.trip_fill_in_basket_mean = trial.suggest_float('trip_fill_in_basket', 3.0, 8.0)
            config.trip_convenience_basket_mean = trial.suggest_float('trip_convenience_basket', 2.0, 5.0)
            config.trip_meal_prep_basket_mean = trial.suggest_float('trip_meal_prep_basket', 6.0, 12.0)
            config.trip_special_basket_mean = trial.suggest_float('trip_special_basket', 10.0, 18.0)
            
            # 8. Trip Purpose Probabilities (9 params) - NEW: Replaces hardcoded TRIP_PURPOSE_PROBABILITIES
            # Price anchor customers (favor stock-up but not too much)
            config.trip_prob_price_anchor_stock_up = trial.suggest_float('trip_prob_pa_stock_up', 0.15, 0.35)
            config.trip_prob_price_anchor_fill_in = trial.suggest_float('trip_prob_pa_fill_in', 0.30, 0.50)
            config.trip_prob_price_anchor_convenience = trial.suggest_float('trip_prob_pa_convenience', 0.10, 0.25)
            
            # Convenience customers (favor small trips)
            config.trip_prob_convenience_convenience = trial.suggest_float('trip_prob_conv_convenience', 0.25, 0.45)
            config.trip_prob_convenience_fill_in = trial.suggest_float('trip_prob_conv_fill_in', 0.25, 0.45)
            config.trip_prob_convenience_stock_up = trial.suggest_float('trip_prob_conv_stock_up', 0.10, 0.25)
            
            # Planned customers (balanced but organized)
            config.trip_prob_planned_stock_up = trial.suggest_float('trip_prob_plan_stock_up', 0.20, 0.40)
            config.trip_prob_planned_meal_prep = trial.suggest_float('trip_prob_plan_meal_prep', 0.25, 0.45)
            config.trip_prob_planned_fill_in = trial.suggest_float('trip_prob_plan_fill_in', 0.15, 0.35)
        
        # ===================================================================
        # TIER 2: INDIRECTLY CALIBRATABLE (20 parameters)
        # Original 8: Promo response (3), Store loyalty (4), Customer drift (1)
        # Sprint 2 adds 12: Promo frequency (2), Marketing signal (3), 
        #                   Heterogeneity (4), Non-linear utilities (2), Seasonality (1)
        # Distribution fixes add 7: Days since last visit (2), Drift mixture (3), Inventory (2)

        # ===================================================================
        
        if 2 in self.tiers:
            print(f"   🔧 Tuning Tier 2 parameters...")
            
            # 1. Promotion Response (3 params) - Original Tier 2
            config.promotion_sensitivity_mean = trial.suggest_float('promo_sensitivity_mean', 0.3, 0.7)
            config.promotion_sensitivity_std = trial.suggest_float('promo_sensitivity_std', 0.1, 0.3)
            config.promotion_quantity_boost = trial.suggest_float('promo_quantity_boost', 1.2, 2.5)
            
            # 2. Store Loyalty (4 params) - Original Tier 2
            config.store_loyalty_weight = trial.suggest_float('store_loyalty_weight', 0.25, 0.6)  # Was 0.4-0.8, raised for stronger feedback
            config.store_switching_probability = trial.suggest_float('store_switching_prob', 0.05, 0.30)
            config.distance_weight = trial.suggest_float('distance_weight', 0.2, 0.6)
            config.satisfaction_weight = trial.suggest_float('satisfaction_weight', 0.3, 0.7)
            
            # 3. Customer Drift (1 param) - Original Tier 2
            config.drift_rate = trial.suggest_float('drift_rate', 0.01, 0.25)
            
            # 4. Phase 2.1 & 2.2: Promotional Frequency (2 params) - NEW Sprint 2
            config.promo_frequency_min = trial.suggest_float('promo_frequency_min', 0.03, 0.08)
            config.promo_frequency_max = trial.suggest_float('promo_frequency_max', 0.08, 0.15)
            
            # 5. Phase 2.3: Marketing Signal Weights (5 params) - NEW Sprint 2
            config.marketing_discount_weight = trial.suggest_float('marketing_discount_weight', 0.05, 0.25)
            config.marketing_display_weight = trial.suggest_float('marketing_display_weight', 0.05, 0.25)
            config.marketing_advertising_weight = trial.suggest_float('marketing_advertising_weight', 0.05, 0.25)
            # HEAVILY REDUCED: Prevent marketing feedback loops causing probability saturation
            config.marketing_visit_weight = trial.suggest_float('marketing_visit_weight', 0.05, 0.25)  # Much lower
            # MINIMAL MEMORY: Prevent probability accumulation
            config.visit_memory_weight = trial.suggest_float('visit_memory_weight', 0.01, 0.25)  # Much lower
            
            # 6. Phase 2.4: Heterogeneity Distribution (4 params) - NEW Sprint 2
            config.hetero_promo_alpha = trial.suggest_float('hetero_promo_alpha', 2.0, 5.0)
            config.hetero_promo_beta = trial.suggest_float('hetero_promo_beta', 1.5, 4.0)
            config.hetero_display_alpha = trial.suggest_float('hetero_display_alpha', 2.0, 5.0)
            config.hetero_display_beta = trial.suggest_float('hetero_display_beta', 2.0, 5.0)
            
            # 7. Phase 2.6: Non-linear Utilities (2 params) - NEW Sprint 2
            config.loss_aversion_lambda = trial.suggest_float('loss_aversion_lambda', 1.5, 3.5)
            config.ewma_alpha = trial.suggest_float('ewma_alpha', 0.1, 0.5)
            
            # 8. Phase 2.7: Seasonality Learning (1 param) - NEW Sprint 2
            config.seasonality_min_confidence = trial.suggest_float('seasonality_min_confidence', 0.2, 0.5)

            # 9. Phase 2.8: Days Since Last Visit (2 params) - NEW Sprint 2
            config.days_since_last_visit_shape = trial.suggest_float('days_since_last_visit_shape', 1.5, 4.0)
            config.days_since_last_visit_scale = trial.suggest_float('days_since_last_visit_scale', 2.0, 5.0)
            
            # 10. Customer Drift Mixture Model (4 params) - NEW Distribution Fix
            # Mixture: 90% small drift + 10% life events (large shifts)
            config.drift_probability = trial.suggest_float('drift_probability', 0.05, 0.20)
            config.drift_life_event_probability = trial.suggest_float('drift_life_event_probability', 0.05, 0.20)
            config.drift_life_event_multiplier = trial.suggest_float('drift_life_event_multiplier', 3.0, 8.0)
            
            # 11. Inventory Dynamics (2 params) - NEW Visit Frequency Fix
            # Faster depletion + higher threshold = more frequent visits
            config.inventory_depletion_rate = trial.suggest_float('inventory_depletion_rate', 0.08, 0.20)
            config.replenishment_threshold = trial.suggest_float('replenishment_threshold', 0.25, 0.75)
        
        # ===================================================================
        # TIER 3: NOT CALIBRATABLE (29 parameters)
        # Fixed at defaults - no tuning
        # ===================================================================
        
        # Demographics, personalities, trip purposes, etc. remain at config defaults
        # Phase 2.6 boolean flags (use_log_price, use_reference_prices, etc.) fixed at True
        # Phase 2.7 enable_seasonality_learning fixed based on data availability
        # See docs/PARAMETER_TIER_CLASSIFICATION.md for rationale
        
        # Enable Sprint 2 features (fixed)
        config.enable_nonlinear_utilities = True
        config.use_log_price = True
        config.use_reference_prices = True
        config.use_psychological_thresholds = True
        config.use_quadratic_quality = True
        config.enable_seasonality_learning = True
        config.seasonal_patterns_path = 'data/processed/seasonal_patterns/seasonal_patterns.pkl'
        
        return config
    
    def _compute_ks_complement(self, synth_dist: np.ndarray, target_dist: np.ndarray) -> float:
        """
        Compute KS complement score (1 - KS statistic)
        Higher is better (closer to 1.0 means better match)
        """
        if len(synth_dist) == 0 or len(target_dist) == 0:
            return 0.0
        
        ks_stat, _ = stats.ks_2samp(synth_dist, target_dist)
        return 1.0 - ks_stat
    
    def objective_function(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization
        
        Returns:
            Score to maximize (KS complement)
        """
        try:
            # Get suggested parameters
            config = self._suggest_parameters(trial)
            
            if self.synthetic_data_path is None:
                # Generate synthetic data with suggested parameters
                print(f"\n🔬 Trial {trial.number}: Generating synthetic data...")
                print(f"   📊 Scale: {config.n_customers} customers, {config.n_products} products, {config.simulation_weeks} weeks")
                
                try:
                    generator = EnhancedRetailSynthV4_1(config)
                    datasets = generator.generate_all_datasets()
                    
                    # Extract and merge transaction data
                    # transaction_items has: line_number, product_id, quantity, unit_price, line_total, transaction_id
                    # transactions has: transaction_id, customer_id, store_id, transaction_date, etc.
                    items_df = datasets['transaction_items']
                    trans_df = datasets['transactions']
                    
                    # Merge to get customer_id and transaction_date
                    synth_df = items_df.merge(trans_df[['transaction_id', 'customer_id', 'transaction_date', 'total_discount']], on='transaction_id', how='left')
                    
                    print(f"   ✅ Generated {len(synth_df):,} transaction items from {len(trans_df):,} transactions")
                    
                except Exception as gen_error:
                    print(f"   ❌ Generation failed: {gen_error}")
                    import traceback
                    traceback.print_exc()
                    return 0.0
            
            else:
                # Load pre-generated synthetic data (already merged)
                print(f"\n🔬 Trial {trial.number}: Loading pre-generated synthetic data...")
                synth_df = pd.read_csv(self.synthetic_data_path)
                print(f"   ✅ Loaded {len(synth_df):,} rows")

            if len(synth_df) == 0:
                print(f"   ❌ No transactions generated")
                return 0.0
            
            # Check if required columns exist
            required_cols = ['transaction_id', 'customer_id', 'line_total', 'quantity', 'total_discount']
            missing_cols = [col for col in required_cols if col not in synth_df.columns]
            if missing_cols:
                print(f"   ❌ Missing columns: {missing_cols}")
                print(f"   📋 Available columns: {list(synth_df.columns)}")
                return 0.0
            
            # Compute synthetic distributions
            print(f"   📊 Computing distributions...")
            synth_basket_sizes = synth_df.groupby('transaction_id').size().values
            synth_revenues = synth_df.groupby('transaction_id')['line_total'].sum().values
            
            # Visit frequency (NORMALIZED by time period)
            synth_df['transaction_date'] = pd.to_datetime(synth_df['transaction_date'])
            
            # CRITICAL FIX: Count distinct WEEKS visited, not transaction IDs
            # A customer can have multiple transactions per week but should only count as 1 visit
            synth_customer_visits = synth_df.groupby('customer_id')['transaction_date'].agg([('weeks_visited', lambda x: x.dt.isocalendar().week.nunique()),  # Distinct weeks
                                                                                            ('first_date', 'min'),
                                                                                            ('last_date', 'max')]).reset_index()
            
            # Calculate weeks span (for customers who don't visit every week)
            synth_customer_visits['weeks_span'] = ((synth_customer_visits['last_date'] - synth_customer_visits['first_date']).dt.days / 7 + 1)
            
            # Visit frequency = weeks visited / weeks span
            synth_visit_freq = (synth_customer_visits['weeks_visited'] / synth_customer_visits['weeks_span']).values
            
            # LOGGING: Debug visit frequency calculation (with CORRECTED values)
            if trial.number % 5 == 0:  # Every 5 trials
                print(f"\n   🔍 Visit Frequency Debug (Trial {trial.number}) - CORRECTED:")
                sample_customers = synth_customer_visits.head(10)
                for _, row in sample_customers.iterrows():
                    weeks_visited = row['weeks_visited']
                    weeks_span = row['weeks_span']
                    freq = weeks_visited / weeks_span
                    print(f"      Customer {row['customer_id']}: {weeks_visited} weeks / {weeks_span:.1f} span = {freq:.2f}")
                
                print(f"   📊 Visit Frequency Distribution (CORRECTED):")
                print(f"      Mean: {synth_visit_freq.mean():.3f}")
                print(f"      Median: {np.median(synth_visit_freq):.3f}")
                print(f"      Std: {synth_visit_freq.std():.3f}")
                print(f"      P10: {np.percentile(synth_visit_freq, 10):.3f}")
                print(f"      P90: {np.percentile(synth_visit_freq, 90):.3f}")
            
            # For backward compatibility with existing code that expects synth_customer_data
            synth_customer_data = synth_customer_visits.set_index('customer_id')
            synth_customer_data['transaction_id'] = synth_customer_visits['weeks_visited'].values
            synth_customer_data['transaction_date'] = synth_customer_visits['weeks_span'].values
            
            synth_quantities = synth_df['quantity'].values
            
            # Extract marketing signal strength from generator logs
            #Transaction-level (% of transactions with at least one promo)
            promo_transactions = synth_df[synth_df['total_discount'] > 0]['transaction_id'].nunique()
            total_transactions = synth_df['transaction_id'].nunique()
            synth_marketing_signal = promo_transactions / total_transactions

            print(f"   📊 Synthetic stats:")
            print(f"      • Transactions: {len(synth_basket_sizes):,}")
            print(f"      • Customers: {synth_df['customer_id'].nunique():,}")
            print(f"      • Avg basket size: {synth_basket_sizes.mean():.2f}")
            print(f"      • Avg revenue: ${synth_revenues.mean():.2f}")
            print(f"      • Avg visit frequency: {synth_visit_freq.mean():.2f}")
            print(f"      • Avg quantity: {synth_quantities.mean():.2f}")
            print(f"      • Marketing signal: {synth_marketing_signal:.3f}")
            
            
            #Print target distributions
            print(f"   📊 Target stats:")
            print(f"      • Basket size: {self.target_basket_size_dist.mean():.2f}")
            print(f"      • Revenue: ${self.target_revenue_dist.mean():.2f}")
            print(f"      • Visit frequency: {self.target_visit_freq_dist.mean():.2f}")
            print(f"      • Quantity: {self.target_quantity_mean:.2f}")
            print(f"      • Marketing signal: {self.target_marketing_signal:.3f}")

            # Compute KS complements
            ks_basket = self._compute_ks_complement(synth_basket_sizes, self.target_basket_size_dist)
            ks_revenue = self._compute_ks_complement(synth_revenues, self.target_revenue_dist)
            ks_visit_freq = self._compute_ks_complement(synth_visit_freq, self.target_visit_freq_dist)
            ks_quantity = self._compute_ks_complement(synth_quantities, self.target_quantity_dist)
            # Marketing signal as single value comparison (absolute difference)
            marketing_signal_error = abs(synth_marketing_signal - self.target_marketing_signal)
            ks_marketing_signal = max(0.0, 1.0 - marketing_signal_error) # Scale to 0-1 range

            # Compute objective based on optimization goal
            if self.objective == 'basket_size':
                score = ks_basket
            elif self.objective == 'revenue':
                score = ks_revenue
            elif self.objective == 'visit_frequency':
                score = ks_visit_freq
            elif self.objective == 'marketing_signal':
                score = ks_marketing_signal
            elif self.objective == 'combined' or self.objective == 'all':
                # PHASE 1 IMPROVEMENT: Harmonic mean + penalties for balanced optimization
                # Forces Optuna to optimize ALL metrics, not just 3/4
                
                # Core metrics with equal importance
                core_metrics = [ks_basket, ks_revenue, ks_visit_freq, ks_quantity]
                
                # 1. Harmonic mean (penalizes low outliers heavily)
                # If one metric is bad, harmonic mean drops significantly
                harmonic_mean = len(core_metrics) / sum(1.0 / max(m, 0.01) for m in core_metrics)
                
                # 2. Standard deviation penalty (reward consistency)
                std_dev = np.std(core_metrics)
                std_penalty = max(0.0, 1.0 - std_dev / 0.3)  # Penalize if std > 0.3
                
                # 3. Low metric penalty (extra penalty for any metric < 0.5)
                low_count = sum(1 for m in core_metrics if m < 0.5)
                low_penalty = low_count * 0.05
                
                # 4. Excellence bonus (bonus if all metrics > 0.7)
                excellence_bonus = 0.1 if all(m > 0.7 for m in core_metrics) else 0.0
                
                # 5. Marketing signal component (secondary importance)
                marketing_component = ks_marketing_signal * 0.1
                
                # Combine components
                score = (
                    0.65 * harmonic_mean +        # Main score (forces balance)
                    0.15 * std_penalty +          # Reward consistency
                    0.10 * marketing_component +  # Marketing signal
                    excellence_bonus -            # Bonus for all good
                    low_penalty                   # Penalty for any bad
                )
                
                # Ensure score is in [0, 1]
                score = max(0.0, min(1.0, score))
            else:
                score = ks_basket
            
            print(f"   📊 Scores: Basket={ks_basket:.3f}, Revenue={ks_revenue:.3f}, "
                  f"VisitFreq={ks_visit_freq:.3f}, Quantity={ks_quantity:.3f}", 
                  f"Marketing Signal={ks_marketing_signal:.3f}")
            print(f"   🎯 Combined Score: {score:.4f}")
            
            return score
            
        except Exception as e:
            print(f"   ❌ Error in trial: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def run_optimization(self, n_trials: int = 50, output_dir: str = 'outputs/tuning_results'):
        """
        Run Optuna optimization
        
        Args:
            n_trials: Number of trials to run
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        tier_str = '_'.join(map(str, self.tiers))
        print(f"\n🚀 Starting Optuna optimization")
        print(f"   • Objective: {self.objective}")
        print(f"   • Tiers: {tier_str}")
        print(f"   • Trials: {n_trials}")
        print(f"   • Output: {output_dir}")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        start_time = datetime.now()
        study.optimize(self.objective_function, n_trials=n_trials)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Save results
        print(f"\n✅ Optimization complete in {duration/60:.1f} minutes")
        print(f"   • Best score: {study.best_value:.4f}")
        
        # Save best parameters
        best_params_path = output_path / f'best_parameters_tier{tier_str}.json'
        with open(best_params_path, 'w') as f:
            json.dump({
                'best_score': study.best_value,
                'best_params': study.best_params,
                'tiers_tuned': self.tiers,
                'n_trials': n_trials,
                'objective': self.objective,
                'duration_minutes': duration / 60
            }, f, indent=2)
        
        print(f"   • Best parameters saved to: {best_params_path}")
        
        # Save optimization history
        history_df = study.trials_dataframe()
        history_path = output_path / f'optimization_history_tier{tier_str}.csv'
        history_df.to_csv(history_path, index=False)
        print(f"   • Optimization history saved to: {history_path}")
        
        # Print best parameters
        print(f"\n📋 Best Parameters:")
        for param, value in sorted(study.best_params.items()):
            print(f"   • {param}: {value:.4f}" if isinstance(value, float) else f"   • {param}: {value}")
        
        return study


def main():
    parser = argparse.ArgumentParser(description='Tune RetailSynth parameters using Optuna')
    parser.add_argument('--real-data', type=str, required=False, 
                       default='data/processed/dunnhumby_calibration.csv',
                       help='Path to real transaction data CSV')
    parser.add_argument('--n-trials', type=int, default=20,
                       help='Number of Optuna trials (default: 20)')
    parser.add_argument('--objective', type=str, default='combined',
                       choices=['basket_size', 'revenue', 'visit_frequency', 'marketing_signal', 'combined'],
                       help='Optimization objective (default: combined)')
    parser.add_argument('--tier', type=str, default='1',
                       help='Parameter tiers to tune: "1", "2", or "1,2" (default: 1)')
    parser.add_argument('--output', type=str, default='outputs/tuning_results',
                       help='Output directory for results')
    parser.add_argument('--synthetic-data', type=str, required=False, 
                       help='Path to synthetic transaction data CSV')
    
    args = parser.parse_args()
    
    # Parse tiers
    tiers = [int(t.strip()) for t in args.tier.split(',')]
    
    # Run tuning
    tuner = ParameterTuner(args.real_data, args.objective, tiers, args.synthetic_data)
    study = tuner.run_optimization(args.n_trials, args.output)
    
    print(f"\n🎉 Tuning complete! Apply best parameters to your config.py")


if __name__ == '__main__':
    main()