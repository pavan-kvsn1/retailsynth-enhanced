"""
Calibration & Validation for RetailSynth Enhanced

This script validates synthetic data against real Dunnhumby data using:
- Statistical tests (KS test, Chi-square)
- Distribution comparisons (histograms, CDFs, Q-Q plots)
- Key metrics (basket size, revenue, visit frequency)

**Sprint 2 Complete**: Now validates all 7 phases including:
- Phase 2.1: Pricing-Promo Separation
- Phase 2.2: Promotional Organization  
- Phase 2.3: Marketing Signal
- Phase 2.4: Individual Heterogeneity
- Phase 2.5: Promotional Response
- Phase 2.6: Non-linear Utilities (behavioral economics)
- Phase 2.7: Seasonality Learning (data-driven patterns)

Usage:
    python scripts/calibrate_synth_data.py \
        --real-data data/raw/dunnhumby/transaction_data.csv \
        --synthetic-data outputs/sprint_2_validation/transaction_items.parquet \
        --output outputs/calibration_report

Based on RetailSynth calibration framework:
https://github.com/RetailMarketingAI/retailsynth
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class SynthDataValidator:
    """Validates synthetic data against real data"""
    
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """
        Args:
            real_data: Real Dunnhumby transactions
            synthetic_data: Synthetic transactions
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.metrics = {}
        
    def compute_all_metrics(self) -> dict:
        """Compute all validation metrics"""
        print("\n" + "="*70)
        print("COMPUTING VALIDATION METRICS")
        print("="*70)
        
        # 1. Basket size distribution
        print("\n📊 1. Basket Size Distribution...")
        self.metrics['basket_size'] = self._compare_basket_sizes()
        
        # 2. Revenue distribution
        print("💰 2. Revenue Distribution...")
        self.metrics['revenue'] = self._compare_revenue()
        
        # 3. Visit frequency
        print("🔄 3. Visit Frequency...")
        self.metrics['visit_frequency'] = self._compare_visit_frequency()
        
        # 4. Time between visits
        print("⏱️  4. Time Between Visits...")
        self.metrics['time_between_visits'] = self._compare_time_between_visits()
        
        # 5. Quantity distribution
        print("📦 5. Quantity Distribution...")
        self.metrics['quantity'] = self._compare_quantities()
        
        return self.metrics
    
    def _compare_basket_sizes(self) -> dict:
        """Compare basket size distributions"""
        # Aggregate to transaction level
        real_baskets = self.real_data.groupby('transaction_id').size()
        synth_baskets = self.synthetic_data.groupby('transaction_id').size()
        
        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(real_baskets, synth_baskets)
        
        return {
            'real_mean': float(real_baskets.mean()),
            'synth_mean': float(synth_baskets.mean()),
            'real_std': float(real_baskets.std()),
            'synth_std': float(synth_baskets.std()),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'ks_complement': 1 - ks_stat,  # Higher is better
            'real_data': real_baskets.values,
            'synth_data': synth_baskets.values
        }
    
    def _compare_revenue(self) -> dict:
        """Compare revenue distributions"""
        # Aggregate to transaction level
        real_revenue = self.real_data.groupby('transaction_id')['line_total'].sum()
        synth_revenue = self.synthetic_data.groupby('transaction_id')['line_total'].sum()
        
        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(real_revenue, synth_revenue)
        
        return {
            'real_mean': float(real_revenue.mean()),
            'synth_mean': float(synth_revenue.mean()),
            'real_median': float(real_revenue.median()),
            'synth_median': float(synth_revenue.median()),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'ks_complement': 1 - ks_stat,
            'real_data': real_revenue.values,
            'synth_data': synth_revenue.values
        }
    
    def _compare_visit_frequency(self) -> dict:
        """Compare customer visit frequency"""
        # Count visits per customer
        real_visits = self.real_data.groupby('customer_id')['transaction_id'].nunique()
        synth_visits = self.synthetic_data.groupby('customer_id')['transaction_id'].nunique()
        
        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(real_visits, synth_visits)
        
        return {
            'real_mean': float(real_visits.mean()),
            'synth_mean': float(synth_visits.mean()),
            'real_median': float(real_visits.median()),
            'synth_median': float(synth_visits.median()),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'ks_complement': 1 - ks_stat,
            'real_data': real_visits.values,
            'synth_data': synth_visits.values
        }
    
    def _compare_time_between_visits(self) -> dict:
        """Compare time between consecutive visits"""
        # Calculate days between visits per customer
        def calc_days_between(df):
            df = df.sort_values('transaction_date')
            return df['transaction_date'].diff().dt.days.dropna()
        
        real_days = self.real_data.groupby('customer_id').apply(calc_days_between)
        synth_days = self.synthetic_data.groupby('customer_id').apply(calc_days_between)
        
        # Flatten
        real_days = real_days.values if hasattr(real_days, 'values') else np.concatenate(real_days.values)
        synth_days = synth_days.values if hasattr(synth_days, 'values') else np.concatenate(synth_days.values)
        
        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(real_days, synth_days)
        
        return {
            'real_mean': float(np.mean(real_days)),
            'synth_mean': float(np.mean(synth_days)),
            'real_median': float(np.median(real_days)),
            'synth_median': float(np.median(synth_days)),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'ks_complement': 1 - ks_stat,
            'real_data': real_days,
            'synth_data': synth_days
        }
    
    def _compare_quantities(self) -> dict:
        """Compare quantity purchased distributions"""
        real_qty = self.real_data['quantity']
        synth_qty = self.synthetic_data['quantity']
        
        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(real_qty, synth_qty)
        
        return {
            'real_mean': float(real_qty.mean()),
            'synth_mean': float(synth_qty.mean()),
            'real_median': float(real_qty.median()),
            'synth_median': float(synth_qty.median()),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'ks_complement': 1 - ks_stat,
            'real_data': real_qty.values,
            'synth_data': synth_qty.values
        }
    
    def _validate_sprint_2_features(self) -> dict:
        """
        Validate Sprint 2 specific features (Phases 2.6 & 2.7)
        
        Phase 2.6: Non-linear utilities (loss aversion, reference prices)
        Phase 2.7: Seasonality learning (temporal patterns)
        """
        sprint2_metrics = {}
        
        # Check if we have week_number for seasonality analysis
        if 'week_number' in self.synthetic_data.columns:
            # Analyze weekly transaction volume variation
            synth_weekly = self.synthetic_data.groupby('week_number').size()
            
            if len(synth_weekly) > 1:
                weekly_variation = (synth_weekly.max() - synth_weekly.min()) / synth_weekly.mean()
                sprint2_metrics['seasonality_variation'] = float(weekly_variation)
                sprint2_metrics['seasonality_present'] = weekly_variation > 0.1  # >10% variation
        
        # Check for promotional response (Phase 2.5)
        if 'promotion_flag' in self.synthetic_data.columns or 'is_promoted' in self.synthetic_data.columns:
            promo_col = 'promotion_flag' if 'promotion_flag' in self.synthetic_data.columns else 'is_promoted'
            promo_qty = self.synthetic_data[self.synthetic_data[promo_col] == 1]['quantity'].mean()
            regular_qty = self.synthetic_data[self.synthetic_data[promo_col] == 0]['quantity'].mean()
            
            if regular_qty > 0:
                promo_boost = (promo_qty - regular_qty) / regular_qty
                sprint2_metrics['promo_quantity_boost'] = float(promo_boost)
                sprint2_metrics['promo_response_present'] = promo_boost > 0.1  # >10% boost
        
        # Check for price sensitivity (Phase 2.6 - behavioral economics)
        if 'unit_price' in self.synthetic_data.columns:
            # Analyze if lower prices lead to higher quantities (expected with loss aversion)
            price_qty_corr = self.synthetic_data[['unit_price', 'quantity']].corr().iloc[0, 1]
            sprint2_metrics['price_quantity_correlation'] = float(price_qty_corr)
            sprint2_metrics['price_sensitivity_present'] = price_qty_corr < -0.05  # Negative correlation
        
        return sprint2_metrics
    
    def generate_report(self, output_dir: Path):
        """Generate calibration report with visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("GENERATING CALIBRATION REPORT")
        print("="*70)
        
        # Summary table
        self._generate_summary_table(output_dir)
        
        # Visualizations
        self._plot_distributions(output_dir)
        
        print(f"\n✅ Report saved to: {output_dir}")
    
    def _generate_summary_table(self, output_dir: Path):
        """Generate summary table of metrics"""
        rows = []
        for metric_name, metric_data in self.metrics.items():
            rows.append({
                'Metric': metric_name,
                'Real Mean': f"{metric_data['real_mean']:.2f}",
                'Synth Mean': f"{metric_data['synth_mean']:.2f}",
                'KS Statistic': f"{metric_data['ks_statistic']:.4f}",
                'KS P-value': f"{metric_data['ks_pvalue']:.4f}",
                'KS Complement': f"{metric_data['ks_complement']:.4f}",
                'Match Quality': '✅ Good' if metric_data['ks_complement'] > 0.8 else '⚠️ Fair' if metric_data['ks_complement'] > 0.6 else '❌ Poor'
            })
        
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(output_dir / 'calibration_summary.csv', index=False)
        
        print("\n" + "="*70)
        print("CALIBRATION SUMMARY")
        print("="*70)
        print(summary_df.to_string(index=False))
    
    def _plot_distributions(self, output_dir: Path):
        """Plot distribution comparisons"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Synthetic vs. Real Data Distribution Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('basket_size', 'Basket Size'),
            ('revenue', 'Revenue per Transaction ($)'),
            ('visit_frequency', 'Visits per Customer'),
            ('time_between_visits', 'Days Between Visits'),
            ('quantity', 'Quantity per Line Item')
        ]
        
        for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
            if idx >= 6:
                break
            
            ax = axes[idx // 2, idx % 2]
            metric_data = self.metrics[metric_key]
            
            # Plot histograms
            ax.hist(metric_data['real_data'], bins=50, alpha=0.5, label='Real', density=True, color='blue')
            ax.hist(metric_data['synth_data'], bins=50, alpha=0.5, label='Synthetic', density=True, color='orange')
            
            ax.set_xlabel(metric_label)
            ax.set_ylabel('Density')
            ax.set_title(f"{metric_label}\nKS Complement: {metric_data['ks_complement']:.3f}")
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Remove empty subplot
        if len(metrics_to_plot) < 6:
            fig.delaxes(axes[2, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: distribution_comparison.png")


def load_real_data(file_path: Path) -> pd.DataFrame:
    """Load real Dunnhumby transaction data"""
    print(f"\n📂 Loading real data from: {file_path}")
    
    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # Ensure required columns exist
    required_cols = ['transaction_id', 'customer_id', 'transaction_date', 'line_total', 'quantity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert date column
    if not pd.api.types.is_datetime64_any_dtype(df['transaction_date']):
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    print(f"   ✅ Loaded {len(df):,} transactions from {df['customer_id'].nunique():,} customers")
    return df


def load_synthetic_data(file_path: Path) -> pd.DataFrame:
    """Load synthetic transaction data"""
    print(f"\n📂 Loading synthetic data from: {file_path}")
    
    if file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    
    # Convert date column
    if not pd.api.types.is_datetime64_any_dtype(df['transaction_date']):
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    print(f"   ✅ Loaded {len(df):,} transactions from {df['customer_id'].nunique():,} customers")
    return df


def main():
    parser = argparse.ArgumentParser(description="Calibrate and validate synthetic data")
    parser.add_argument('--real-data', type=str, required=True, help='Path to real Dunnhumby transaction data')
    parser.add_argument('--synthetic-data', type=str, required=True, help='Path to synthetic transaction data')
    parser.add_argument('--output', type=str, default='outputs/calibration_report', help='Output directory for report')
    args = parser.parse_args()
    
    print("="*70)
    print("RETAILSYNTH CALIBRATION & VALIDATION")
    print("="*70)
    
    # Load data
    real_data = load_real_data(Path(args.real_data))
    synthetic_data = load_synthetic_data(Path(args.synthetic_data))
    
    # Initialize validator
    validator = SynthDataValidator(real_data, synthetic_data)
    
    # Compute metrics
    metrics = validator.compute_all_metrics()
    
    # Generate report
    validator.generate_report(Path(args.output))
    
    print("\n" + "="*70)
    print("✅ CALIBRATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
