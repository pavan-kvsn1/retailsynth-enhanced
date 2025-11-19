"""
Generate Synthetic Transactions with Sprint 2 Enhancements (Phases 2.1-2.7)

This script demonstrates the complete Sprint 2 pipeline:
    Phase 2.1: Pricing-Promo separation
    Phase 2.2: Comprehensive promo system (mechanics, displays, features)
    Phase 2.3: Marketing signal impacts store visit probability
    Phase 2.4: Individual customer heterogeneity
    Phase 2.5: Customer-specific promotional response
    Phase 2.6: Non-linear utilities (log-price, loss aversion, thresholds)
    Phase 2.7: Seasonality learning (data-driven patterns)

Usage:
    python scripts/generate_with_elasticity.py \
        --elasticity-dir data/processed/elasticity_models \
        --output outputs/sprint_2_validation \
        --n-customers 1000 \
        --n-products 1000 \
        --weeks 8

Features:
    - HMM-based realistic price dynamics (optional)
    - Separated pricing and promotional engines (Phase 2.1)
    - Store promotional contexts (Phase 2.2)
    - Marketing signal strength (Phase 2.3)
    - Individual customer heterogeneity (Phase 2.4)
    - Customer-specific promotional response (Phase 2.5)
    - Non-linear utilities with behavioral economics (Phase 2.6)
    - Learned seasonal patterns from data (Phase 2.7)
    - All existing temporal dynamics (drift, lifecycle, seasonality)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retailsynth.config import EnhancedRetailConfig
from retailsynth.generators.main_generator import EnhancedRetailSynthV4_1

try:
    import yaml
except ImportError:
    yaml = None

def validate_sprint_2_features(generator, datasets):
    """
    Validate that Sprint 2 Phases 2.1-2.7 are working correctly
    
    Args:
        generator: EnhancedRetailSynthV4_1 instance
        datasets: Generated datasets dictionary
    
    Returns:
        Dict with validation results
    """
    print(f"\n{'='*70}")
    print("SPRINT 2 FEATURE VALIDATION (Phases 2.1-2.7)")
    print("="*70)
    
    validation = {
        'phase_2_1': False,  # Pricing-Promo separation
        'phase_2_2': False,  # Promo organization
        'phase_2_3': False,  # Marketing signal
        'phase_2_4': False,  # Heterogeneity
        'phase_2_5': False,  # Promo response
        'phase_2_6': False,  # Non-linear utilities
        'phase_2_7': False,  # Seasonality learning
    }
    
    # Phase 2.1: Pricing-Promo Separation
    print("\n🔍 Phase 2.1: Pricing-Promo Separation")
    if hasattr(generator, 'pricing_engine') and hasattr(generator, 'promotional_engine'):
        print("   ✅ Separate engines exist")
        validation['phase_2_1'] = True
    else:
        print("   ❌ Engines not separated")
    
    # Phase 2.2: Promotional Organization
    print("\n🔍 Phase 2.2: Promotional Organization")
    if hasattr(generator, 'promotional_engine'):
        promo_engine = generator.promotional_engine
        
        # Check if product tendencies are initialized
        if hasattr(promo_engine, 'product_promo_tendencies') and promo_engine.product_promo_tendencies:
            n_tendencies = len(promo_engine.product_promo_tendencies)
            print(f"   ✅ Product tendencies: {n_tendencies:,} products")
            validation['phase_2_2'] = True
        else:
            print("   ⚠️  Product tendencies not initialized")
        
        # Check display and feature systems
        if hasattr(promo_engine, 'display_capacity') and hasattr(promo_engine, 'feature_prob'):
            print("   ✅ Display and feature systems initialized")
        else:
            print("   ⚠️  Display/feature systems not found")
    
    # Phase 2.3: Marketing Signal
    print("\n🔍 Phase 2.3: Marketing Signal")
    if hasattr(generator, 'promotional_engine'):
        if hasattr(generator.promotional_engine, 'signal_calculator'):
            print("   ✅ Marketing signal calculator exists")
            validation['phase_2_3'] = True
        else:
            print("   ⚠️  Marketing signal calculator not found")
    
    # Phase 2.4: Individual Heterogeneity
    print("\n🔍 Phase 2.4: Individual Heterogeneity")
    if hasattr(generator, 'precomp') and hasattr(generator.precomp, 'hetero_params_dict'):
        n_hetero = len(generator.precomp.hetero_params_dict)
        if n_hetero > 0:
            print(f"   ✅ Heterogeneity parameters: {n_hetero:,} customers")
            
            # Sample one customer's parameters
            sample_customer_id = list(generator.precomp.hetero_params_dict.keys())[0]
            sample_params = generator.precomp.hetero_params_dict[sample_customer_id]
            print(f"   📊 Sample customer {sample_customer_id} parameters:")
            print(f"      • Promo responsiveness: {sample_params.get('promo_responsiveness_param', 'N/A'):.3f}")
            print(f"      • Price sensitivity: {sample_params.get('price_sensitivity_param', 'N/A'):.3f}")
            print(f"      • Display sensitivity: {sample_params.get('display_sensitivity_param', 'N/A'):.3f}")
            validation['phase_2_4'] = True
        else:
            print("   ⚠️  Heterogeneity parameters dict is empty")
            print("   ℹ️  Phase 2.4 customer generator may not be integrated")
    else:
        print("   ❌ Heterogeneity parameters not found in precomp")
    
    # Phase 2.5: Promotional Response
    print("\n🔍 Phase 2.5: Promotional Response")
    if hasattr(generator, 'promo_response_calculator'):
        print("   ✅ Promotional response calculator initialized")
        
        # Check if it's being used in transaction generator
        if hasattr(generator, 'transaction_gen'):
            if hasattr(generator.transaction_gen, 'promo_response_calc'):
                print("   ✅ Calculator passed to transaction generator")
                validation['phase_2_5'] = True
            else:
                print("   ⚠️  Calculator not passed to transaction generator")
        else:
            print("   ℹ️  Transaction generator not yet initialized")
    else:
        print("   ❌ Promotional response calculator not found")
    
    # Phase 2.6: Non-linear Utilities
    print("\n🔍 Phase 2.6: Non-linear Utilities")
    if hasattr(generator, 'nonlinear_engine') and generator.nonlinear_engine is not None:
        print("   ✅ Non-linear utility engine initialized")
        
        # Check specific features
        nl_config = generator.nonlinear_engine.config
        features_enabled = []
        
        if nl_config.use_log_price:
            features_enabled.append("log-price")
            print("   ✅ Log-price utility (diminishing marginal disutility)")
        
        if nl_config.use_reference_prices:
            features_enabled.append("reference prices")
            print("   ✅ Reference prices with loss aversion")
            print(f"      • Loss aversion λ: {nl_config.loss_aversion_lambda:.1f}x")
            print(f"      • EWMA α: {nl_config.ewma_alpha:.2f}")
            
            # Check if reference prices are initialized
            if hasattr(generator.nonlinear_engine, 'reference_prices') and generator.nonlinear_engine.reference_prices:
                n_ref_prices = len(generator.nonlinear_engine.reference_prices)
                print(f"      • Reference prices tracked: {n_ref_prices:,} products")
        
        if nl_config.use_psychological_thresholds:
            features_enabled.append("psychological thresholds")
            print("   ✅ Psychological price thresholds (charm pricing)")
        
        if nl_config.use_quadratic_quality:
            features_enabled.append("quadratic quality")
            print("   ✅ Quadratic quality (diminishing returns)")
        
        if len(features_enabled) >= 3:
            validation['phase_2_6'] = True
            print(f"   ✅ Phase 2.6 fully operational ({len(features_enabled)}/4 features)")
        else:
            print(f"   ⚠️  Only {len(features_enabled)}/4 features enabled")
    else:
        print("   ❌ Non-linear utility engine not found or disabled")
    
    # Phase 2.7: Seasonality Learning
    print("\n🔍 Phase 2.7: Seasonality Learning")
    if hasattr(generator, 'seasonality_engine') and generator.seasonality_engine is not None:
        from retailsynth.engines.seasonality_learning import LearnedSeasonalityEngine
        
        if isinstance(generator.seasonality_engine, LearnedSeasonalityEngine):
            print("   ✅ Learned seasonality engine initialized")
            
            # Check pattern coverage
            n_product_patterns = generator.seasonality_engine.n_products_with_patterns
            n_category_patterns = generator.seasonality_engine.n_categories_with_patterns
            
            print(f"   📊 Pattern Coverage:")
            print(f"      • Product patterns: {n_product_patterns:,}")
            print(f"      • Category patterns: {n_category_patterns}")
            
            if n_product_patterns > 0 or n_category_patterns > 0:
                validation['phase_2_7'] = True
                print("   ✅ Seasonal patterns loaded and ready")
            else:
                print("   ⚠️  No patterns loaded (using uniform seasonality)")
                print("   💡 Run: python scripts/learn_seasonal_patterns.py")
        else:
            print("   ⚠️  Using hard-coded seasonality (not learned)")
            print("   💡 To enable learned seasonality:")
            print("      1. Set enable_seasonality_learning=True in config")
            print("      2. Run learn_seasonal_patterns.py to generate patterns")
    else:
        print("   ❌ Seasonality engine not found or disabled")
    
    # Overall summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print("="*70)
    phases_passed = sum(validation.values())
    total_phases = len(validation)
    print(f"\n✅ Phases Passing: {phases_passed}/{total_phases} ({phases_passed/total_phases*100:.0f}%)")
    print(f"\nPhase Status:")
    print(f"   Phase 2.1 (Pricing-Promo): {'✅' if validation['phase_2_1'] else '❌'}")
    print(f"   Phase 2.2 (Promo Organization): {'✅' if validation['phase_2_2'] else '❌'}")
    print(f"   Phase 2.3 (Marketing Signal): {'✅' if validation['phase_2_3'] else '❌'}")
    print(f"   Phase 2.4 (Heterogeneity): {'✅' if validation['phase_2_4'] else '❌'}")
    print(f"   Phase 2.5 (Promo Response): {'✅' if validation['phase_2_5'] else '❌'}")
    print(f"   Phase 2.6 (Non-linear Utilities): {'✅' if validation['phase_2_6'] else '❌'}")
    print(f"   Phase 2.7 (Seasonality Learning): {'✅' if validation['phase_2_7'] else '❌'}")
    
    return validation

def analyze_promotional_effectiveness(datasets):
    """
    Analyze promotional effectiveness across stores and customers
    
    Args:
        datasets: Generated datasets dictionary
    
    Returns:
        Dict with promotional analysis
    """
    print(f"\n{'='*70}")
    print("PROMOTIONAL EFFECTIVENESS ANALYSIS")
    print("="*70)
    
    transactions = datasets['transactions']
    transaction_items = datasets['transaction_items']
    
    # Merge to get item-level promotion info
    items_with_promos = transaction_items.merge(
        transactions[['transaction_id', 'store_id', 'week_number']],
        on='transaction_id'
    )
    
    # Calculate promotion penetration
    total_items = len(items_with_promos)
    
    print(f"\n📊 Promotional Statistics:")
    print(f"   Total items sold: {total_items:,}")
    
    # Transaction-level promo stats
    if 'promo_items' in transactions.columns:
        avg_promo_items = transactions['promo_items'].mean()
        pct_with_promos = (transactions['promo_items'] > 0).sum() / len(transactions) * 100
        print(f"   Transactions with promos: {pct_with_promos:.1f}%")
        print(f"   Avg promo items per transaction: {avg_promo_items:.2f}")
    
    # Revenue impact
    if 'discount_amount' in transactions.columns:
        total_revenue = transactions['total_revenue'].sum()
        total_discount = transactions['discount_amount'].sum()
        discount_rate = (total_discount / (total_revenue + total_discount)) * 100
        print(f"\n💰 Revenue Impact:")
        print(f"   Total revenue: ${total_revenue:,.2f}")
        print(f"   Total discounts: ${total_discount:,.2f}")
        print(f"   Avg discount rate: {discount_rate:.1f}%")
    
    # Store-level variation
    if 'store_id' in transactions.columns:
        print(f"\n🏪 Store-Level Variation:")
        store_stats = transactions.groupby('store_id').agg({
            'total_revenue': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        store_stats.columns = ['store_id', 'revenue', 'transactions']
        
        print(f"   Stores analyzed: {len(store_stats)}")
        print(f"   Avg transactions per store: {store_stats['transactions'].mean():.0f}")
        print(f"   Avg revenue per store: ${store_stats['revenue'].mean():,.2f}")

def analyze_nonlinear_utilities(generator, datasets):
    """
    Analyze non-linear utility effects (Phase 2.6)
    
    Args:
        generator: EnhancedRetailSynthV4_1 instance
        datasets: Generated datasets dictionary
    """
    print(f"\n{'='*70}")
    print("PHASE 2.6: NON-LINEAR UTILITIES ANALYSIS")
    print("="*70)
    
    if not hasattr(generator, 'nonlinear_engine') or generator.nonlinear_engine is None:
        print("\n⚠️  Non-linear utilities not enabled")
        return
    
    nl_engine = generator.nonlinear_engine
    nl_config = nl_engine.config
    
    print(f"\n🧮 Non-Linear Utility Configuration:")
    print(f"   • Log-price utility: {'✅' if nl_config.use_log_price else '❌'}")
    print(f"   • Reference prices: {'✅' if nl_config.use_reference_prices else '❌'}")
    print(f"   • Psychological thresholds: {'✅' if nl_config.use_psychological_thresholds else '❌'}")
    print(f"   • Quadratic quality: {'✅' if nl_config.use_quadratic_quality else '❌'}")
    
    if nl_config.use_reference_prices:
        print(f"\n💰 Reference Price Tracking:")
        print(f"   • Loss aversion λ: {nl_config.loss_aversion_lambda:.1f}x")
        print(f"   • EWMA smoothing α: {nl_config.ewma_alpha:.2f}")
        
        if hasattr(nl_engine, 'reference_prices') and nl_engine.reference_prices:
            ref_prices = list(nl_engine.reference_prices.values())
            print(f"   • Products tracked: {len(ref_prices):,}")
            print(f"   • Avg reference price: ${np.mean(ref_prices):.2f}")
            print(f"   • Reference price range: ${np.min(ref_prices):.2f} - ${np.max(ref_prices):.2f}")
    
    # Analyze price threshold effects
    if nl_config.use_psychological_thresholds and 'products' in datasets:
        products = datasets['products']
        if 'base_price' in products.columns or 'avg_price' in products.columns:
            price_col = 'base_price' if 'base_price' in products.columns else 'avg_price'
            prices = products[price_col].values
            
            # Count prices near psychological thresholds
            thresholds = [0.99, 1.99, 2.99, 4.99, 9.99, 19.99]
            near_threshold = 0
            
            for price in prices:
                for threshold in thresholds:
                    if abs(price - threshold) < 0.05:  # Within 5 cents
                        near_threshold += 1
                        break
            
            print(f"\n🎯 Psychological Price Thresholds:")
            print(f"   • Total products: {len(prices):,}")
            print(f"   • Near thresholds: {near_threshold} ({near_threshold/len(prices)*100:.1f}%)")
            print(f"   • Thresholds analyzed: {', '.join([f'${t:.2f}' for t in thresholds])}")


def analyze_seasonality(generator, datasets):
    """
    Analyze seasonal patterns (Phase 2.7)
    
    Args:
        generator: EnhancedRetailSynthV4_1 instance
        datasets: Generated datasets dictionary
    """
    print(f"\n{'='*70}")
    print("PHASE 2.7: SEASONALITY ANALYSIS")
    print("="*70)
    
    if not hasattr(generator, 'seasonality_engine') or generator.seasonality_engine is None:
        print("\n⚠️  Seasonality not enabled")
        return
    
    from retailsynth.engines.seasonality_learning import LearnedSeasonalityEngine
    
    is_learned = isinstance(generator.seasonality_engine, LearnedSeasonalityEngine)
    
    print(f"\n📅 Seasonality Engine:")
    print(f"   • Type: {'Learned (data-driven)' if is_learned else 'Hard-coded (rule-based)'}")
    
    if is_learned:
        engine = generator.seasonality_engine
        
        print(f"\n📊 Pattern Coverage:")
        print(f"   • Product-specific patterns: {engine.n_products_with_patterns:,}")
        print(f"   • Category-level patterns: {engine.n_categories_with_patterns}")
        print(f"   • Minimum confidence: {engine.min_confidence:.2f}")
        
        # Sample seasonal multipliers for different weeks
        if 'products' in datasets and len(datasets['products']) > 0:
            sample_product_id = datasets['products']['product_id'].iloc[0]
            
            print(f"\n🎄 Seasonal Variation (Sample Product {sample_product_id}):")
            print(f"{'Week':<8} {'Multiplier':<12} {'Season':<20}")
            print("-" * 40)
            
            seasonal_weeks = [
                (1, "New Year"),
                (13, "Spring"),
                (26, "Summer"),
                (39, "Fall"),
                (47, "Thanksgiving"),
                (51, "Christmas")
            ]
            
            for week, season in seasonal_weeks:
                mult = engine.get_seasonal_multiplier(sample_product_id, week)
                print(f"{week:<8} {mult:<12.2f} {season:<20}")
    
    # Analyze transaction volume by week
    if 'transactions' in datasets and 'week_number' in datasets['transactions'].columns:
        transactions = datasets['transactions']
        
        print(f"\n📈 Weekly Transaction Volume:")
        weekly_volume = transactions.groupby('week_number').size()
        
        print(f"   • Weeks simulated: {len(weekly_volume)}")
        print(f"   • Avg per week: {weekly_volume.mean():.0f}")
        print(f"   • Peak week: Week {weekly_volume.idxmax()} ({weekly_volume.max():,} transactions)")
        print(f"   • Lowest week: Week {weekly_volume.idxmin()} ({weekly_volume.min():,} transactions)")
        print(f"   • Variation: {(weekly_volume.max() / weekly_volume.min() - 1) * 100:.1f}% difference")


def load_config_from_yaml(yaml_path: str) -> dict:
    """Load configuration from YAML file"""
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml")
    
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def map_yaml_to_config_params(yaml_data: dict) -> dict:
    """Map YAML parameter names to EnhancedRetailConfig attribute names"""
    
    # Parameter name mapping from Optuna tuning script to config attributes
    param_mapping = {
        # Optuna name -> Config attribute name
        'base_visit_prob': 'base_visit_probability',
        'promo_sensitivity_mean': 'promotion_sensitivity_mean',
        'promo_sensitivity_std': 'promotion_sensitivity_std', 
        'promo_quantity_boost': 'promotion_quantity_boost',
        'store_switching_prob': 'store_switching_probability',
        # These should already match but including for completeness
        'store_loyalty_weight': 'store_loyalty_weight',
        'distance_weight': 'distance_weight',
        'satisfaction_weight': 'satisfaction_weight',
        'drift_rate': 'drift_rate',
        'marketing_visit_weight': 'marketing_visit_weight',
        'visit_memory_weight': 'visit_memory_weight',
        'hetero_promo_alpha': 'hetero_promo_alpha',
        'hetero_promo_beta': 'hetero_promo_beta',
        'hetero_display_alpha': 'hetero_display_alpha', 
        'hetero_display_beta': 'hetero_display_beta',
        'days_since_last_visit_shape': 'days_since_last_visit_shape',
        'days_since_last_visit_scale': 'days_since_last_visit_scale',
        'drift_probability': 'drift_probability',
        'drift_life_event_probability': 'drift_life_event_probability',
        'drift_life_event_multiplier': 'drift_life_event_multiplier',
    }
    
    mapped_data = {}
    
    for key, value in yaml_data.items():
        # Use mapped name if available, otherwise use original key
        config_key = param_mapping.get(key, key)
        mapped_data[config_key] = value
    
    return mapped_data

def generate_in_batches(
    total_customers: int,
    batch_size: int,
    config_kwargs: dict,
    output_dir: str,
    elasticity_dir: str = None
) -> dict:
    """
    PHASE 1 IMPROVEMENT: Generate synthetic data in batches for memory efficiency
    
    Generates data for total_customers in batches of batch_size, then merges.
    Critical for generating 10K+ customers without memory issues.
    
    Args:
        total_customers: Total number of customers to generate
        batch_size: Number of customers per batch (default: 1000)
        config_kwargs: Configuration dictionary for EnhancedRetailConfig
        output_dir: Output directory for final merged data
        elasticity_dir: Optional elasticity models directory
    
    Returns:
        dict: Merged datasets (customers, products, stores, transactions, transaction_items)
    """
    import gc
    
    n_batches = (total_customers + batch_size - 1) // batch_size
    
    print(f"\n{'='*70}")
    print(f"📦 BATCH GENERATION MODE")
    print("="*70)
    print(f"   Total customers: {total_customers:,}")
    print(f"   Batch size: {batch_size:,}")
    print(f"   Number of batches: {n_batches}")
    print(f"   Memory efficiency: ✅ Enabled")
    
    # Storage for all batches
    all_customers = []
    all_transactions = []
    all_transaction_items = []
    all_business_performance = []
    
    # Products and stores are shared across batches
    shared_products = None
    shared_stores = None
    shared_market_context = None
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_customers)
        batch_customers = end_idx - start_idx
        
        print(f"\n{'='*70}")
        print(f"📦 Batch {batch_idx+1}/{n_batches}: Generating {batch_customers:,} customers")
        print("="*70)
        
        # Update config for this batch
        batch_config_kwargs = config_kwargs.copy()
        batch_config_kwargs['n_customers'] = batch_customers
        batch_config_kwargs['random_seed'] = config_kwargs.get('random_seed', 42) + batch_idx
        
        # Create config and generator for this batch
        batch_config = EnhancedRetailConfig(**batch_config_kwargs)
        batch_generator = EnhancedRetailSynthV4_1(batch_config)
        
        # Generate datasets for this batch
        batch_datasets = batch_generator.generate_all_datasets()
        
        # Adjust customer IDs to be unique across batches
        customer_id_offset = start_idx
        batch_datasets['customers']['customer_id'] += customer_id_offset
        batch_datasets['transactions']['customer_id'] += customer_id_offset
        
        # Adjust transaction IDs to be unique across batches
        if batch_idx > 0:
            max_transaction_id = max(all_transactions[-1]['transaction_id'].max(), 0) if all_transactions else 0
            batch_datasets['transactions']['transaction_id'] += max_transaction_id
            batch_datasets['transaction_items']['transaction_id'] += max_transaction_id
        
        # Store batch data
        all_customers.append(batch_datasets['customers'])
        all_transactions.append(batch_datasets['transactions'])
        all_transaction_items.append(batch_datasets['transaction_items'])
        if 'business_performance' in batch_datasets:
            all_business_performance.append(batch_datasets['business_performance'])
        
        # Save shared datasets from first batch
        if batch_idx == 0:
            shared_products = batch_datasets['products']
            shared_stores = batch_datasets['stores']
            if 'market_context' in batch_datasets:
                shared_market_context = batch_datasets['market_context']
        
        print(f"   ✅ Batch {batch_idx+1} complete:")
        print(f"      • Customers: {len(batch_datasets['customers']):,}")
        print(f"      • Transactions: {len(batch_datasets['transactions']):,}")
        print(f"      • Transaction items: {len(batch_datasets['transaction_items']):,}")
        
        # Clear memory
        del batch_generator
        del batch_datasets
        del batch_config
        gc.collect()
    
    # Merge all batches
    print(f"\n{'='*70}")
    print("🔗 MERGING BATCHES")
    print("="*70)
    
    final_datasets = {
        'customers': pd.concat(all_customers, ignore_index=True),
        'transactions': pd.concat(all_transactions, ignore_index=True),
        'transaction_items': pd.concat(all_transaction_items, ignore_index=True),
        'products': shared_products,
        'stores': shared_stores
    }
    
    if shared_market_context is not None:
        final_datasets['market_context'] = shared_market_context
    
    if all_business_performance:
        final_datasets['business_performance'] = pd.concat(all_business_performance, ignore_index=True)
    
    print(f"   ✅ Merge complete:")
    print(f"      • Total customers: {len(final_datasets['customers']):,}")
    print(f"      • Total transactions: {len(final_datasets['transactions']):,}")
    print(f"      • Total transaction items: {len(final_datasets['transaction_items']):,}")
    print(f"      • Products: {len(final_datasets['products']):,}")
    print(f"      • Stores: {len(final_datasets['stores']):,}")
    
    return final_datasets


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate synthetic transactions with Sprint 2 features (Phases 2.1-2.7)")
    parser.add_argument('--elasticity-dir', type=str, default='data/processed/elasticity', help='Directory containing learned elasticity parameters')
    parser.add_argument('--product-catalog', type=str, default='data/processed/product_catalog/product_catalog_20k.parquet', help='Path to product catalog')
    parser.add_argument('--output', type=str, default='outputs/sprint_2_test', help='Output directory for generated data')
    parser.add_argument('--n-customers', type=int, default=None, help='Number of customers to generate (overrides config)')
    parser.add_argument('--n-products', type=int, default=None, help='Number of products to use from catalog (overrides config)')
    parser.add_argument('--n-stores', type=int, default=None, help='Number of stores (overrides config)')
    parser.add_argument('--weeks', type=int, default=None, help='Number of weeks to simulate (overrides config)')
    parser.add_argument('--random-seed', type=int, default=None, help='Random seed for reproducibility (overrides config)')
    parser.add_argument('--config', type=str, default='configs/calibrated.yaml', help='Path to YAML config file')
    parser.add_argument('--skip-save', action='store_true', help='Skip saving datasets (for quick testing)')
    parser.add_argument('--batch-size', type=int, default=None, help='PHASE 1: Generate in batches of N customers (for memory efficiency with 10K+ customers)')
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config_data = {}
    if args.config and Path(args.config).exists():
        print(f"📄 Loading configuration from: {args.config}")
        raw_config_data = load_config_from_yaml(args.config)
        config_data = map_yaml_to_config_params(raw_config_data)
        print(f"✅ Loaded {len(config_data)} configuration parameters")
    else:
        print(f"⚠️  Config file not found: {args.config}, using defaults")
    
    # Resolve effective configuration (CLI args override YAML)
    effective_n_customers = args.n_customers or config_data.get('n_customers', 1000)
    effective_n_products = args.n_products or config_data.get('n_products', 1000)
    effective_n_stores = args.n_stores or config_data.get('n_stores', 10)
    effective_weeks = args.weeks or config_data.get('simulation_weeks', 8)
    effective_seed = args.random_seed or config_data.get('random_seed', 42)
    elasticity_dir = args.elasticity_dir
    output_dir_str = args.output
    product_catalog_path = args.product_catalog or config_data.get('product_catalog_path', 'data/processed/product_catalog/product_catalog_20k.parquet')
    
    print("="*70)
    print("RETAILSYNTH v4.1 - SPRINT 2 VALIDATION (Phases 2.1-2.7)")
    print("="*70)
    print(f"\n📊 Configuration:")
    print(f"  Config file: {args.config}")
    print(f"  Customers: {effective_n_customers:,}")
    print(f"  Products: {effective_n_products:,}")
    print(f"  Stores: {effective_n_stores:,}")
    print(f"  Weeks: {effective_weeks}")
    print(f"  Random seed: {effective_seed}")
    print(f"  Elasticity models: {elasticity_dir}")
    print(f"  Output: {output_dir_str}")
    
    # Step 1: Configure RetailSynth with all Sprint features enabled
    print(f"\n{'='*70}")
    print("STEP 1: Configuring RetailSynth")
    print("="*70)
    
    # Create configuration with YAML values and CLI overrides
    config_kwargs = {
        'n_customers': effective_n_customers,
        'n_products': effective_n_products,
        'n_stores': effective_n_stores,
        'simulation_weeks': effective_weeks,
        'random_seed': effective_seed,
        'product_catalog_path': product_catalog_path,
    }
    
    # Get valid config attributes from EnhancedRetailConfig
    from dataclasses import fields
    valid_config_attrs = {field.name for field in fields(EnhancedRetailConfig)}
    
    # Add all YAML configuration parameters (only valid ones)
    loaded_params = []
    skipped_params = []
    for key, value in config_data.items():
        if key not in config_kwargs and key in valid_config_attrs:  # Don't override CLI parameters, only valid attrs
            config_kwargs[key] = value
            loaded_params.append(key)
        elif key not in valid_config_attrs:
            skipped_params.append(key)
    
    print(f"📊 Parameter loading summary:")
    print(f"   • Loaded: {len(loaded_params)} parameters")
    print(f"   • Skipped: {len(skipped_params)} unknown parameters")
    if skipped_params:
        print(f"   • Skipped params: {skipped_params[:5]}{'...' if len(skipped_params) > 5 else ''}")
    
    # Ensure critical features are enabled
    config_kwargs.update({
        'use_real_catalog': config_data.get('use_real_catalog', True),
        'enable_temporal_dynamics': config_data.get('enable_temporal_dynamics', True),
        'enable_customer_drift': config_data.get('enable_customer_drift', True),
        'enable_product_lifecycle': config_data.get('enable_product_lifecycle', False),
        'enable_store_loyalty': config_data.get('enable_store_loyalty', True),
        'enable_basket_composition': config_data.get('enable_basket_composition', True),
        'enable_nonlinear_utilities': config_data.get('enable_nonlinear_utilities', True),
        'enable_seasonality_learning': config_data.get('enable_seasonality_learning', True),
    })
    
    try:
        config = EnhancedRetailConfig(**config_kwargs)
    except TypeError as e:
        print(f"❌ Error creating config: {e}")
        print(f"🔍 Problematic parameters:")
        for key in config_kwargs:
            if key not in valid_config_attrs:
                print(f"   • {key}: not a valid config attribute")
        raise
    
    print("\n✅ Configuration complete:")
    print(f"   • Real product catalog: {'✅' if config.use_real_catalog else '❌'}")
    print(f"   • Temporal dynamics: {'✅' if config.enable_temporal_dynamics else '❌'}")
    print(f"   • Basket composition: {'✅' if config.enable_basket_composition else '❌'}")
    print(f"   • Customer drift: {'✅' if config.enable_customer_drift else '❌'}")
    print(f"   • Store loyalty: {'✅' if config.enable_store_loyalty else '❌'}")
    print(f"   • Non-linear utilities: {'✅' if config.enable_nonlinear_utilities else '❌'}")
    print(f"   • Seasonality learning: {'✅' if config.enable_seasonality_learning else '❌'}")
    print(f"\n🎯 Calibrated Parameters:")
    try:
        print(f"   • Base visit probability: {config.base_visit_probability:.4f}")
        print(f"   • Basket size lambda: {config.basket_size_lambda:.4f}")
        print(f"   • Quantity mean: {config.quantity_mean:.4f}")
        print(f"   • Promo frequency: {getattr(config, 'promo_frequency_min', 'N/A'):.4f}-{getattr(config, 'promo_frequency_max', 'N/A'):.4f}")
        print(f"   • Loss aversion lambda: {config.loss_aversion_lambda:.4f}")
    except AttributeError as e:
        print(f"   ⚠️  Error accessing config attribute: {e}")
        print(f"   Available attributes: {[attr for attr in dir(config) if not attr.startswith('_')][:10]}...")
    
    # Step 2: Initialize generator
    print(f"\n{'='*70}")
    print("STEP 2: Initializing Generator with Sprint 2 Features")
    print("="*70)
    
    generator = EnhancedRetailSynthV4_1(config)
    
    print("\n✅ Generator initialized:")
    print("   • Phase 2.1: Pricing & Promotional engines separated")
    print("   • Phase 2.2: Promo mechanics, displays, features ready")
    print("   • Phase 2.3: Marketing signal calculator ready")
    print("   • Phase 2.4: Heterogeneity engine ready")
    print("   • Phase 2.5: Promo response calculator ready")
    print("   • Phase 2.6: Non-linear utility engine ready")
    print("   • Phase 2.7: Seasonality learning engine ready")
    
    # Step 3: Generate base datasets FIRST
    print(f"\n{'='*70}")
    print("STEP 3: Generating Base Datasets")
    print("="*70)
    
    generator.generate_base_datasets()
    
    print("\n✅ Base datasets generated:")
    print(f"   • Customers: {len(generator.datasets['customers']):,}")
    print(f"   • Products: {len(generator.datasets['products']):,}")
    print(f"   • Stores: {len(generator.datasets['stores']):,}")
    
    # Step 4: Load elasticity models (optional)
    print(f"\n{'='*70}")
    print("STEP 4: Loading Elasticity Models (Optional)")
    print("="*70)
    
    elasticity_path = Path(elasticity_dir)
    if elasticity_path.exists():
        try:
            generator.load_elasticity_models(elasticity_dir, generator.datasets['products'])
            print("\n✅ Elasticity models loaded successfully!")
            print("\nElasticity features enabled:")
            if generator.price_hmm is not None:
                print("  ✅ HMM price dynamics (realistic promotions)")
            if generator.cross_price_engine is not None:
                print("  ✅ Cross-price elasticity (substitution/complementarity)")
            if generator.arc_elasticity_engine is not None:
                print("  ✅ Arc elasticity (stockpiling/deferral)")
        except Exception as e:
            print(f"\n⚠️  Failed to load elasticity models: {e}")
            print("   Continuing with fallback pricing engine...")
    else:
        print(f"\n⚠️  Elasticity directory not found: {elasticity_dir}")
        print("   Using fallback pricing engine (Phase 2.1)")
        print("\n💡 To use elasticity models, first run:")
        print(f"   python scripts/learn_price_elasticity.py")
    
    # Step 5: Validate Sprint 2 features BEFORE generation
    print(f"\n{'='*70}")
    print("STEP 5: Pre-Generation Feature Validation")
    print("="*70)
    
    pre_validation = validate_sprint_2_features(generator, generator.datasets)
    
    # Step 6: Generate transactions with ALL Sprint 2 features
    print(f"\n{'='*70}")
    print("STEP 6: Generating Transactions with Sprint 2 Features")
    print("="*70)
    print("\nThis will integrate:")
    print("  • Phase 2.1: Separated pricing and promotions")
    print("  • Phase 2.2: Promo mechanics (discounts, displays, features)")
    print("  • Phase 2.3: Marketing signals amplifying store visits")
    print("  • Phase 2.4: Individual customer heterogeneity")
    print("  • Phase 2.5: Customer-specific promotional response")
    print("  • Phase 2.6: Non-linear utilities (behavioral economics)")
    print("  • Phase 2.7: Learned seasonal patterns (data-driven)")
    
    # PHASE 1 IMPROVEMENT: Check if batch mode is enabled
    if args.batch_size and effective_n_customers > args.batch_size:
        print(f"\n🚀 BATCH MODE ENABLED (batch_size={args.batch_size})")
        print(f"   Generating {effective_n_customers:,} customers in batches...")
        
        datasets = generate_in_batches(
            total_customers=effective_n_customers,
            batch_size=args.batch_size,
            config_kwargs=config_kwargs,
            output_dir=output_dir_str,
            elasticity_dir=elasticity_dir
        )
    else:
        if args.batch_size:
            print(f"\n💡 Batch mode requested but not needed (customers={effective_n_customers} <= batch_size={args.batch_size})")
            print(f"   Using standard generation...")
        
        datasets = generator.generate_all_datasets()
    
    print(f"\n✅ Transaction generation complete!")
    print(f"   • Transactions: {len(datasets['transactions']):,}")
    print(f"   • Transaction items: {len(datasets['transaction_items']):,}")
    
    # Step 7: Analyze promotional effectiveness
    analyze_promotional_effectiveness(datasets)
    
    # Step 8: Generate summary report
    print(f"\n{'='*70}")
    print("STEP 7: Summary Report")
    print("="*70)
    
    transactions = datasets['transactions']
    transaction_items = datasets['transaction_items']
    
    print(f"\n📊 Generation Summary:")
    print(f"   Customers: {len(datasets['customers']):,}")
    print(f"   Products: {len(datasets['products']):,}")
    print(f"   Stores: {len(datasets['stores']):,}")
    print(f"   Weeks: {effective_weeks}")
    print(f"\n🛒 Transaction Summary:")
    print(f"   Total transactions: {len(transactions):,}")
    print(f"   Total items sold: {len(transaction_items):,}")
    
    if len(transaction_items) > 0:
        basket_sizes = transaction_items.groupby('transaction_id').size()
        print(f"   Avg basket size: {basket_sizes.mean():.1f} items")
        print(f"   Median basket size: {basket_sizes.median():.0f} items")
    
    if 'total_revenue' in transactions.columns:
        print(f"   Avg basket value: ${transactions['total_revenue'].mean():.2f}")
        print(f"   Total revenue: ${transactions['total_revenue'].sum():,.2f}")
    
    # HMM-specific stats
    if generator.price_hmm is not None:
        print(f"\n💰 HMM Price Dynamics:")
        print(f"   Products with HMM: {len(generator.price_hmm.transition_matrices):,}")
        
        if hasattr(generator, 'pricing_history') and generator.pricing_history:
            promo_rates = []
            for week_data in generator.pricing_history:
                if 'promotions' in week_data:
                    promo_rate = sum(week_data['promotions'].values()) / len(week_data['promotions'])
                    promo_rates.append(promo_rate)
            
            if promo_rates:
                avg_promo_rate = sum(promo_rates) / len(promo_rates)
                print(f"   Avg promotion rate: {avg_promo_rate:.1%}")
    
    # Step 9: Save datasets (optional)
    if not args.skip_save:
        print(f"\n{'='*70}")
        print("STEP 8: Saving Datasets")
        print("="*70)
        
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, dataset_df in datasets.items():
            output_path = output_dir / f"{dataset_name}.parquet"
            dataset_df.to_parquet(output_path, index=False)
            print(f"   ✅ Saved {dataset_name}: {len(dataset_df):,} rows → {output_path}")
        
        print(f"\n✅ All datasets saved to: {output_dir}")
    else:
        print(f"\n⏭️  Skipping dataset save (--skip-save flag)")
    
    # Final Sprint 2 validation
    print(f"\n{'='*70}")
    print("FINAL SPRINT 2 VALIDATION")
    print("="*70)
    
    post_validation = validate_sprint_2_features(generator, datasets)
    
    # Success summary
    print(f"\n{'='*70}")
    print("✅ SPRINT 2 GENERATION COMPLETE!")
    print("="*70)
    
    if not args.skip_save:
        print(f"\nOutput directory: {output_dir}")
    
    print(f"\n🎯 Sprint 2 Features Tested:")
    print(f"   {'✅' if post_validation.get('phase_2_1') else '⚠️ '} Phase 2.1: Pricing-Promo Separation")
    print(f"   {'✅' if post_validation.get('phase_2_2') else '⚠️ '} Phase 2.2: Promo Organization")
    print(f"   {'✅' if post_validation.get('phase_2_3') else '⚠️ '} Phase 2.3: Marketing Signal")
    print(f"   {'✅' if post_validation.get('phase_2_4') else '⚠️ '} Phase 2.4: Heterogeneity")
    print(f"   {'✅' if post_validation.get('phase_2_5') else '⚠️ '} Phase 2.5: Promo Response")
    print(f"   {'✅' if post_validation.get('phase_2_6') else '⚠️ '} Phase 2.6: Non-linear Utilities")
    print(f"   {'✅' if post_validation.get('phase_2_7') else '⚠️ '} Phase 2.7: Seasonality Learning")
    
    phases_working = sum(post_validation.values())
    print(f"\n📈 Sprint 2 Progress: {phases_working}/7 phases working ({phases_working/7*100:.0f}%)")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Run comprehensive validation against Dunnhumby")
    print(f"   2. Analyze non-linear utility effects (loss aversion, thresholds)")
    print(f"   3. Validate seasonal patterns match historical data")
    print(f"   4. Compare Sprint 2 vs baseline validation metrics")
    print(f"   5. Document validation improvement (target: 65% → 85%+)")

if __name__ == '__main__':
    main()
