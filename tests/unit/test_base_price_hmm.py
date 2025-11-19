#!/usr/bin/env python3
"""
Test Base Price HMM Learning and Sampling

This script demonstrates:
1. Loading Dunnhumby data
2. Learning Base Price HMM from non-promotional weeks
3. Sampling base prices
4. Validating learned parameters

Usage:
    python scripts/test_base_price_hmm.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retailsynth.engines.base_price_hmm import BasePriceHMM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("="*70)
    print("BASE PRICE HMM - TEST SCRIPT")
    print("="*70)
    
    # Step 1: Load data
    print("\n📂 Step 1: Loading Dunnhumby data...")
    
    # Load product catalog
    products_path = 'data/processed/product_catalog/product_catalog_20k.parquet'
    if not Path(products_path).exists():
        products_path = 'data/raw/dunnhumby/product.csv'
    
    products_df = pd.read_parquet(products_path) if products_path.endswith('.parquet') else pd.read_csv(products_path)
    logger.info(f"  Loaded {len(products_df):,} products")
    
    # Load transactions (sample for testing)
    transactions_path = 'data/raw/dunnhumby/transaction_data.csv'
    logger.info("  Loading transactions (this may take a moment)...")
    
    transactions_df = pd.read_csv(
        transactions_path,
        dtype={
            'household_key': 'int32',
            'BASKET_ID': 'int64',
            'PRODUCT_ID': 'int32',
            'QUANTITY': 'int16',
            'SALES_VALUE': 'float32',
            'STORE_ID': 'int16',
            'RETAIL_DISC': 'float32',
            'WEEK_NO': 'int16'
        },
        nrows=500000  # Sample for testing
    )
    logger.info(f"  Loaded {len(transactions_df):,} transactions")
    
    # Load causal data (optional)
    causal_path = 'data/raw/dunnhumby/causal_data.csv'
    causal_df = None
    if Path(causal_path).exists():
        causal_df = pd.read_csv(causal_path)
        logger.info(f"  Loaded {len(causal_df):,} promotional records")
    
    # Step 2: Initialize and learn Base Price HMM
    print("\n🧠 Step 2: Learning Base Price HMM...")
    
    base_price_hmm = BasePriceHMM(products_df, n_states=4)
    base_price_hmm.learn_from_data(
        transactions_df, 
        causal_df,
        min_observations=10
    )
    
    # Step 3: Get summary statistics
    print("\n📊 Step 3: Summary Statistics...")
    stats = base_price_hmm.get_summary_statistics()
    
    if stats:
        print(f"\n  Products with learned parameters: {stats['n_products_learned']:,}")
        print(f"  Diagonal strength (stickiness): {stats['diagonal_strength']:.3f}")
        
        print("\n  Average Transition Matrix:")
        avg_tm = stats['avg_transition_matrix']
        state_names = ['Low', 'Mid-L', 'Mid-H', 'High']
        print("         " + "  ".join(f"{s:>6s}" for s in state_names))
        for i, row in enumerate(avg_tm):
            print(f"  {state_names[i]:6s} " + "  ".join(f"{val:6.3f}" for val in row))
        
        print("\n  State Prevalence:")
        for state, prob in stats['state_prevalence'].items():
            state_name = stats['state_names'][state]
            print(f"    {state_name:15s}: {prob:.1%}")
    
    # Step 4: Sample base prices
    print("\n💰 Step 4: Sampling base prices...")
    
    # Get a few products to test
    test_products = list(base_price_hmm.current_states.keys())[:10]
    
    print(f"\n  Sampling prices for {len(test_products)} products over 5 weeks:")
    print("\n  Product ID | Week 1  | Week 2  | Week 3  | Week 4  | Week 5  | State Info")
    print("  " + "-"*80)
    
    for product_id in test_products:
        prices = []
        for week in range(5):
            # Sample price
            price_dict = base_price_hmm.sample_base_prices([product_id], week=week)
            prices.append(price_dict[product_id])
            
            # Transition to next week
            if week < 4:
                base_price_hmm.transition_states([product_id])
        
        # Get state info
        state_info = base_price_hmm.get_state_info(product_id)
        state_name = state_info.get('state_name', 'unknown')
        
        price_str = " | ".join(f"${p:6.2f}" for p in prices)
        print(f"  {product_id:10d} | {price_str} | {state_name}")
    
    # Step 5: Save parameters
    print("\n💾 Step 5: Saving parameters...")
    output_dir = Path('data/processed/base_price_hmm')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'base_price_hmm_params.pkl'
    base_price_hmm.save_parameters(output_file)
    
    # Step 6: Test loading
    print("\n🔄 Step 6: Testing parameter loading...")
    
    # Create new instance and load
    base_price_hmm_loaded = BasePriceHMM(products_df, n_states=4)
    base_price_hmm_loaded.load_parameters(output_file)
    
    # Verify
    print(f"  ✅ Loaded {len(base_price_hmm_loaded.transition_matrices):,} product parameters")
    
    # Step 7: Validation
    print("\n✅ Step 7: Validation...")
    
    # Check transition matrix properties
    if stats:
        diagonal = stats['diagonal_strength']
        if diagonal > 0.85:
            print(f"  ✅ Diagonal strength {diagonal:.3f} > 0.85 (prices are sticky)")
        else:
            print(f"  ⚠️  Diagonal strength {diagonal:.3f} < 0.85 (prices may be too volatile)")
        
        # Check state balance
        state_probs = list(stats['state_prevalence'].values())
        min_prob = min(state_probs)
        max_prob = max(state_probs)
        
        if max_prob / min_prob < 3.0:
            print(f"  ✅ States are reasonably balanced (ratio: {max_prob/min_prob:.2f})")
        else:
            print(f"  ⚠️  States are imbalanced (ratio: {max_prob/min_prob:.2f})")
    
    print("\n" + "="*70)
    print("✅ BASE PRICE HMM TEST COMPLETE!")
    print("="*70)
    print(f"\n📁 Parameters saved to: {output_file}")
    print("\n🚀 Next steps:")
    print("   1. Create Promotional HMM (promo_hmm.py)")
    print("   2. Integrate both into PromotionalEngine")
    print("   3. Test combined pricing system")


if __name__ == '__main__':
    main()
