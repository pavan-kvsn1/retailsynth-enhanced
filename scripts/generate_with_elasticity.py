"""
Generate Synthetic Transactions with Elasticity Models (Sprint 2)

This script demonstrates how to use the learned elasticity models
(HMM, cross-price, arc) in transaction generation.

Usage:
    python scripts/generate_with_elasticity.py \
        --elasticity-dir data/processed/elasticity \
        --output outputs/synthetic_data_v2 \
        --n-customers 5000 \
        --n-products 1000 \
        --weeks 52

Features:
    - HMM-based realistic price dynamics
    - Cross-price elasticity effects on demand
    - Arc elasticity for stockpiling/deferral behavior
    - All existing temporal dynamics (drift, lifecycle, seasonality)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retailsynth.config import EnhancedRetailConfig
from retailsynth.generators.main_generator import EnhancedRetailSynthV4_1

try:
    import yaml
except ImportError:
    yaml = None

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate synthetic transactions with elasticity models")
    parser.add_argument('--elasticity-dir', type=str, default='data/processed/elasticity', help='Directory containing learned elasticity parameters')
    parser.add_argument('--product-catalog', type=str, default='data/processed/product_catalog/product_catalog_20k.parquet', help='Path to product catalog')
    parser.add_argument('--output', type=str, default='outputs/synthetic_data_with_elasticity', help='Output directory for generated data')
    parser.add_argument('--n-customers', type=int, default=5000, help='Number of customers to generate')
    parser.add_argument('--n-products', type=int, default=15000, help='Number of products to use from catalog')
    parser.add_argument('--weeks', type=int, default=104, help='Number of weeks to simulate')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config (see configs/generation_example.yaml)')
    args = parser.parse_args()
    
    # Resolve effective configuration from YAML (if provided) and CLI
    effective_n_customers = args.n_customers
    effective_n_products = args.n_products
    effective_weeks = args.weeks
    elasticity_dir = args.elasticity_dir
    output_dir_str = args.output
    product_catalog_path = args.product_catalog
    _yaml_config = None
    
    if args.config:
        if yaml is None:
            print("\n⚠️  PyYAML is not installed. Install with: pip install pyyaml")
            sys.exit(1)
        with open(args.config, 'r') as f:
            data = yaml.safe_load(f) or {}
        generator_settings = data.get('generator', {})
        config_settings = data.get('config', {k: v for k, v in data.items() if k != 'generator'})
        _yaml_config = config_settings
        
        # Apply YAML to effective values
        effective_n_customers = config_settings.get('n_customers', effective_n_customers)
        effective_n_products = config_settings.get('n_products', effective_n_products)
        effective_weeks = config_settings.get('simulation_weeks', effective_weeks)
        elasticity_dir = generator_settings.get('elasticity_dir', elasticity_dir)
        output_dir_str = generator_settings.get('output', output_dir_str)
        product_catalog_path = generator_settings.get('product_catalog', config_settings.get('product_catalog_path', product_catalog_path))
    
    print("="*70)
    print("RETAILSYNTH v4.1 WITH ELASTICITY MODELS (SPRINT 2)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Customers: {effective_n_customers:,}")
    print(f"  Products: {effective_n_products:,}")   
    print(f"  Weeks: {effective_weeks}")
    print(f"  Elasticity models: {elasticity_dir}")
    print(f"  Output: {output_dir_str}")
    
    # Step 1: Configure RetailSynth
    print(f"\n{'='*70}")
    print("STEP 1: Configuring RetailSynth")
    print("="*70)
    
    if _yaml_config is not None:
        # Build config from YAML, then enforce effective and script-level values
        config = EnhancedRetailConfig(**_yaml_config)
        config.n_customers = effective_n_customers
        config.n_products = effective_n_products
        config.simulation_weeks = effective_weeks
        config.random_seed = args.random_seed  # CLI still controls seed
        if product_catalog_path:
            config.product_catalog_path = product_catalog_path
    else:
        config = EnhancedRetailConfig(n_customers=effective_n_customers,
                                      n_products=effective_n_products,
                                      n_stores=10,
                                      simulation_weeks=effective_weeks,
                                      random_seed=args.random_seed,
                                      
                                      # Use real catalog
                                      use_real_catalog=True,
                                      product_catalog_path=product_catalog_path,
                                      
                                      # Enable temporal dynamics (but NOT product lifecycle)
                                      enable_temporal_dynamics=True,
                                      enable_customer_drift=True,
                                      enable_product_lifecycle=False,  # Disabled: causes issues with basket composition
                                      enable_store_loyalty=True,
                                      
                                      # Sprint 1.4: TEMPORARILY DISABLE basket composition for debugging
                                      enable_basket_composition=True,  # TODO: Re-enable after fixing 0 transactions issue
                                      
                                      # Region for seasonality
                                      region='US')
    
    # Step 2: Initialize generator
    print(f"\n{'='*70}")
    print("STEP 2: Initializing Generator")
    print("="*70)
    
    generator = EnhancedRetailSynthV4_1(config)
    
    # Step 3: Generate base datasets FIRST (customers, products, stores - NO transactions yet)
    print(f"\n{'='*70}")
    print("STEP 3: Generating Base Datasets (Customers, Products, Stores)")
    print("="*70)
    
    generator.generate_base_datasets()
    
    # Step 4: Load elasticity models AFTER products are generated (NEW - Sprint 2)
    print(f"\n{'='*70}")
    print("STEP 4: Loading Elasticity Models")
    print("="*70)
    
    elasticity_path = Path(elasticity_dir)
    if elasticity_path.exists():
        generator.load_elasticity_models(elasticity_dir, generator.datasets['products'])
        print("\n✅ Elasticity models loaded successfully!")
        print("\nElasticity features enabled:")
        print("  ✅ HMM price dynamics (realistic promotions)")
        print("  ✅ Cross-price elasticity (substitution/complementarity)")
        print("  ✅ Arc elasticity (stockpiling/deferral)")
    else:
        print(f"\n⚠️  Elasticity directory not found: {elasticity_dir}")
        print("   Falling back to simple pricing engine")
        print("\n💡 To use elasticity models, first run:")
        print(f"   python scripts/learn_price_elasticity.py")
    
    # Step 5: Complete dataset generation (transactions with elasticity models)
    print(f"\n{'='*70}")
    print("STEP 5: Generating Transactions with Elasticity")
    print("="*70)
    
    datasets = generator.generate_all_datasets()
    
    # Step 6: Save datasets
    print(f"\n{'='*70}")
    print("STEP 6: Saving Datasets")
    print("="*70)
    
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all datasets
    for dataset_name, dataset_df in datasets.items():
        output_path = output_dir / f"{dataset_name}.parquet"
        dataset_df.to_parquet(output_path, index=False)
        print(f"   ✅ Saved {dataset_name}: {len(dataset_df):,} rows → {output_path}")
    
    # Step 7: Generate summary report
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
    print(f"   Avg basket size: {transaction_items.groupby('transaction_id').size().mean():.1f} items")
    print(f"   Avg basket value: ${transactions['total_revenue'].mean():.2f}")
    print(f"   Total revenue: ${transactions['total_revenue'].sum():,.2f}")
    
    if generator.price_hmm is not None:
        print(f"\n💰 Price Dynamics (HMM):")
        # Calculate promotion statistics
        promo_weeks = []
        for week_data in generator.pricing_history:
            promo_rate = sum(week_data['promotions'].values()) / len(week_data['promotions'])
            promo_weeks.append(promo_rate)
        
        avg_promo_rate = sum(promo_weeks) / len(promo_weeks)
        print(f"   Avg promotion rate: {avg_promo_rate:.1%}")
        print(f"   Products with HMM: {len(generator.price_hmm.transition_matrices):,}")
    
    if generator.cross_price_engine is not None and generator.cross_price_engine.cross_elasticity_matrix is not None:
        print(f"\n🔗 Cross-Price Effects:")
        print(f"   Substitute pairs: {len(generator.cross_price_engine.substitute_groups):,}")
        print(f"   Complement pairs: {len(generator.cross_price_engine.complement_pairs):,}")
    
    print(f"\n{'='*70}")
    print("✅ GENERATION COMPLETE WITH ELASTICITY MODELS!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Validate against Dunnhumby benchmarks")
    print(f"  2. Analyze price elasticity effects")
    print(f"  3. Compare with/without elasticity models")


if __name__ == '__main__':
    main()
