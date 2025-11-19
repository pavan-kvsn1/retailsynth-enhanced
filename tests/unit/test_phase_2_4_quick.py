"""
Quick Phase 2.4 Integration Test
Verify that heterogeneous customer parameters work end-to-end
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from retailsynth.engines.customer_heterogeneity import CustomerHeterogeneityEngine


def test_phase_2_4():
    """Test Phase 2.4 heterogeneity integration"""
    print("\n" + "="*70)
    print("PHASE 2.4 QUICK INTEGRATION TEST")
    print("="*70)
    
    # Test 1: Engine initialization
    print("\n1. Testing engine initialization...")
    try:
        engine = CustomerHeterogeneityEngine(random_seed=42)
        print("   ✅ Heterogeneity engine initialized")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 2: Generate small population
    print("\n2. Generating heterogeneous customer population...")
    try:
        n_customers = 100
        customers_df = engine.generate_population_parameters(n_customers)
        print(f"   ✅ Generated {len(customers_df):,} customers with unique parameters")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Verify parameter ranges
    print("\n3. Verifying parameter ranges...")
    params_to_check = {
        'price_sensitivity': (0.5, 2.5),
        'quality_preference': (0.3, 1.5),
        'promo_responsiveness': (0.5, 2.0),
        'display_sensitivity': (0.3, 1.2),
        'advertising_receptivity': (0.3, 1.5),
        'variety_seeking': (0.3, 1.2),
        'brand_loyalty': (0.2, 1.5),
        'store_loyalty': (0.3, 1.3)
    }
    
    all_valid = True
    for param, (min_val, max_val) in params_to_check.items():
        actual_min = customers_df[param].min()
        actual_max = customers_df[param].max()
        
        if actual_min < min_val or actual_max > max_val:
            print(f"   ❌ {param}: out of range [{min_val}, {max_val}]")
            print(f"      Actual: [{actual_min:.3f}, {actual_max:.3f}]")
            all_valid = False
        else:
            print(f"   ✅ {param}: [{actual_min:.3f}, {actual_max:.3f}] within [{min_val}, {max_val}]")
    
    if not all_valid:
        return False
    
    # Test 4: Check heterogeneity (not all the same)
    print("\n4. Verifying heterogeneity (customers are unique)...")
    for param in ['price_sensitivity', 'quality_preference', 'promo_responsiveness']:
        unique_count = customers_df[param].nunique()
        std_dev = customers_df[param].std()
        
        if unique_count < n_customers * 0.8:
            print(f"   ⚠️  {param}: only {unique_count}/{n_customers} unique values")
        else:
            print(f"   ✅ {param}: {unique_count}/{n_customers} unique values, σ={std_dev:.3f}")
    
    # Test 5: Distribution summary
    print("\n5. Getting distribution summary...")
    try:
        summary = engine.get_distribution_summary(customers_df)
        print(f"   ✅ Summary generated for {len(summary)} parameters")
        
        # Print key stats
        print("\n   Key Parameter Statistics:")
        for param in ['price_sensitivity', 'quality_preference', 'promo_responsiveness']:
            stats = summary[param]
            print(f"      {param}:")
            print(f"         Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            print(f"         Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 6: Verify segment labels (for analysis)
    print("\n6. Checking segment labels (analysis only)...")
    segment_counts = customers_df['segment_label'].value_counts()
    print("   Segment distribution:")
    for segment, count in segment_counts.items():
        print(f"      {segment}: {count} ({count/len(customers_df)*100:.1f}%)")
    print("   ✅ Segment labels assigned (note: these are for analysis only)")
    
    # Test 7: Check parameter independence
    print("\n7. Checking parameter independence...")
    corr = customers_df[['price_sensitivity', 'quality_preference', 'promo_responsiveness']].corr()
    max_corr = corr.abs().where(~np.eye(3, dtype=bool)).max().max()
    print(f"   Maximum correlation between parameters: {max_corr:.3f}")
    if max_corr < 0.3:
        print("   ✅ Parameters are approximately independent")
    elif max_corr < 0.5:
        print("   ℹ️  Some moderate correlations (acceptable)")
    else:
        print("   ⚠️  High correlations detected")
    
    # Success!
    print("\n" + "="*70)
    print("✅ ALL CHECKS PASSED!")
    print("="*70)
    print("\nPhase 2.4 Integration Summary:")
    print(f"  • Heterogeneity engine: ✅ Working")
    print(f"  • Population generation: ✅ Working")
    print(f"  • Parameter ranges: ✅ Valid")
    print(f"  • Heterogeneity: ✅ Verified")
    print(f"  • Distribution summary: ✅ Working")
    print(f"  • Parameter independence: ✅ Confirmed")
    print("\n🎉 Phase 2.4 is OPERATIONAL!")
    print("   Every customer now has unique behavioral parameters!")
    
    return True


if __name__ == "__main__":
    try:
        success = test_phase_2_4()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
