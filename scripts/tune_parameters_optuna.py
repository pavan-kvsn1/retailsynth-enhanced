"""
Automated Parameter Tuning with Optuna

This script uses Bayesian optimization to find optimal RetailSynth parameters
that best match real Dunnhumby data distributions.

**Parameter Tiers**:
- Tier 1 (15 params): Directly calibratable - direct observable effect
- Tier 2 (8 params): Indirectly calibratable - aggregate effect
- Tier 3 (29 params): Not calibratable - fixed at literature defaults

See docs/PARAMETER_TIER_CLASSIFICATION.md for full rationale.

Usage:
    # Tune Tier 1 only (recommended, fast)
    python scripts/tune_parameters_optuna.py \
        --real-data data/raw/dunnhumby/transaction_data.csv \
        --tier 1 \
        --n-trials 50 \
        --output outputs/tuning_tier1
    
    # Tune Tier 1 + 2 (slower, marginal improvement)
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
        
        tier_counts = {1: 15, 2: 8, 3: 29}
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
        
        customer_data = self.real_df.groupby('customer_id').agg({
            'transaction_date': lambda x: (x.max() - x.min()).days / 7 + 1,  # Active weeks
            'transaction_id': 'nunique'  # Total visits
        })
        
        # Visits per week (normalized)
        visit_freq_per_week = customer_data['transaction_id'] / customer_data['transaction_date']
        self.target_visit_freq_dist = visit_freq_per_week.values
        self.target_visit_freq_mean = visit_freq_per_week.mean()
        self.target_visit_freq_std = visit_freq_per_week.std()
        
        # Quantity distribution
        self.target_quantity_dist = self.real_df['quantity'].values
        self.target_quantity_mean = self.real_df['quantity'].mean()
        self.target_quantity_std = self.real_df['quantity'].std()
    
    def _suggest_parameters(self, trial: optuna.Trial) -> EnhancedRetailConfig:
        """
        Suggest parameter values for this trial
        
        Only tunes Tier 1 & 2 parameters. Tier 3 fixed at defaults.
        
        IMPORTANT: Uses SMALL SCALE for fast tuning (~2-3 min/trial)
        - 1,000 customers (vs 10,000+ in production)
        - 1,000 products (vs 15,000+ in production)
        - 4 weeks (vs 52+ in production)
        
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
        config.n_customers = 1000       # ~2-3 min generation time
        config.n_products = 1000        # Enough for distribution patterns
        config.n_stores = 5             # Reduced from 10
        config.simulation_weeks = 20    # Enough for behavioral patterns
        
        print(f"   🔧 Tuning scale: {config.n_customers} customers, {config.n_products} products, {config.simulation_weeks} weeks")
        
        # ===================================================================
        # TIER 1: DIRECTLY CALIBRATABLE (15 parameters)
        # ===================================================================
        
        if 1 in self.tiers:
            print(f"   🔧 Tuning Tier 1 parameters...")
            
            # 1. Visit Behavior (1 param)
            config.base_visit_probability = trial.suggest_float('base_visit_prob', 0.15, 0.50)
            
            # 2. Basket Size (1 param)
            config.basket_size_lambda = trial.suggest_float('basket_size_lambda', 1.0, 30.0)
            
            # 3. Quantity Distribution (3 params)
            config.quantity_mean = trial.suggest_float('quantity_mean', 1.2, 2.5)
            config.quantity_std = trial.suggest_float('quantity_std', 0.5, 1.5)
            config.quantity_max = trial.suggest_int('quantity_max', 5, 15)
            
            # 4. Temporal Dynamics (2 params)
            config.inventory_depletion_rate = trial.suggest_float('inventory_depletion_rate', 0.05, 0.20)
            config.replenishment_threshold = trial.suggest_float('replenishment_threshold', 0.2, 0.5)
            
            # 5. Basket Composition (3 params)
            config.complement_probability = trial.suggest_float('complement_probability', 0.2, 0.7)
            config.substitute_avoidance = trial.suggest_float('substitute_avoidance', 0.6, 0.95)
            config.category_diversity_weight = trial.suggest_float('category_diversity_weight', 0.1, 0.6)
            
            # 6. Purchase History Weights (5 params)
            config.loyalty_weight = trial.suggest_float('loyalty_weight', 0.1, 0.6)
            config.habit_weight = trial.suggest_float('habit_weight', 0.2, 0.7)
            config.inventory_weight = trial.suggest_float('inventory_weight', 0.3, 0.8)
            config.variety_weight = trial.suggest_float('variety_weight', 0.1, 0.5)
            config.price_memory_weight = trial.suggest_float('price_memory_weight', 0.05, 0.3)
        
        # ===================================================================
        # TIER 2: INDIRECTLY CALIBRATABLE (8 parameters)
        # ===================================================================
        
        if 2 in self.tiers:
            print(f"   🔧 Tuning Tier 2 parameters...")
            
            # 1. Promotion Response (3 params)
            config.promotion_sensitivity_mean = trial.suggest_float('promo_sensitivity_mean', 0.3, 0.7)
            config.promotion_sensitivity_std = trial.suggest_float('promo_sensitivity_std', 0.1, 0.3)
            config.promotion_quantity_boost = trial.suggest_float('promo_quantity_boost', 1.2, 2.0)
            
            # 2. Store Loyalty (4 params)
            config.store_loyalty_weight = trial.suggest_float('store_loyalty_weight', 0.4, 0.8)
            config.store_switching_probability = trial.suggest_float('store_switching_prob', 0.05, 0.30)
            config.distance_weight = trial.suggest_float('distance_weight', 0.2, 0.6)
            config.satisfaction_weight = trial.suggest_float('satisfaction_weight', 0.4, 0.8)
            
            # 3. Customer Drift (1 param)
            config.drift_rate = trial.suggest_float('drift_rate', 0.01, 0.15)
        
        # ===================================================================
        # TIER 3: NOT CALIBRATABLE (29 parameters)
        # Fixed at defaults - no tuning
        # ===================================================================
        
        # Demographics, personalities, trip purposes, etc. remain at config defaults
        # See docs/PARAMETER_TIER_CLASSIFICATION.md for rationale
        
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
                    synth_df = items_df.merge(
                        trans_df[['transaction_id', 'customer_id', 'transaction_date']],
                        on='transaction_id',
                        how='left'
                    )
                    
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
            required_cols = ['transaction_id', 'customer_id', 'line_total', 'quantity']
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
            synth_customer_data = synth_df.groupby('customer_id').agg({
                'transaction_date': lambda x: (x.max() - x.min()).days / 7 + 1,  # Active weeks
                'transaction_id': 'nunique'  # Total visits
            })
            synth_visit_freq = (synth_customer_data['transaction_id'] / synth_customer_data['transaction_date']).values
            
            synth_quantities = synth_df['quantity'].values
            
            print(f"   📊 Synthetic stats:")
            print(f"      • Transactions: {len(synth_basket_sizes):,}")
            print(f"      • Customers: {synth_df['customer_id'].nunique():,}")
            print(f"      • Avg basket size: {synth_basket_sizes.mean():.2f}")
            print(f"      • Avg revenue: ${synth_revenues.mean():.2f}")
            
            # Compute KS complements
            ks_basket = self._compute_ks_complement(synth_basket_sizes, self.target_basket_size_dist)
            ks_revenue = self._compute_ks_complement(synth_revenues, self.target_revenue_dist)
            ks_visit_freq = self._compute_ks_complement(synth_visit_freq, self.target_visit_freq_dist)
            ks_quantity = self._compute_ks_complement(synth_quantities, self.target_quantity_dist)
            
            #Print target distributions
            print(f"   📊 Target stats:")
            print(f"      • Basket size: {self.target_basket_size_dist.mean():.2f}")
            print(f"      • Revenue: ${self.target_revenue_dist.mean():.2f}")
            print(f"      • Visit frequency: {self.target_visit_freq_dist.mean():.2f}")
            print(f"      • Quantity: {self.target_quantity_dist.mean():.2f}")

            # Compute objective based on optimization goal
            if self.objective == 'basket_size':
                score = ks_basket
            elif self.objective == 'revenue':
                score = ks_revenue
            elif self.objective == 'visit_frequency':
                score = ks_visit_freq
            elif self.objective == 'combined':
                # Weighted average of all metrics
                score = (ks_basket * 0.3 + ks_revenue * 0.3 + ks_visit_freq * 0.3 + ks_quantity * 0.1)
            else:
                score = ks_basket
            
            print(f"   📊 Scores: Basket={ks_basket:.3f}, Revenue={ks_revenue:.3f}, "
                  f"VisitFreq={ks_visit_freq:.3f}, Quantity={ks_quantity:.3f}")
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
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of Optuna trials (default: 50)')
    parser.add_argument('--objective', type=str, default='combined',
                       choices=['basket_size', 'revenue', 'visit_frequency', 'combined'],
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
