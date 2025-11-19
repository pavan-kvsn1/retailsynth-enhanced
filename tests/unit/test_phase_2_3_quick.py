"""
Quick Phase 2.3 Integration Test
Verify that marketing signal calculation works end-to-end
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from retailsynth.engines.promotional_engine import PromotionalEngine
from retailsynth.engines.marketing_signal import MarketingSignalCalculator


def create_test_data():
    """Create minimal test data"""
    products = pd.DataFrame({
        'product_id': range(1, 51),
        'product_name': [f'Product {i}' for i in range(1, 51)],
        'category': ['Grocery'] * 50,
        'base_price': np.random.uniform(2, 20, 50)
    })
    
    stores = pd.DataFrame({
        'store_id': [1, 2],
        'store_name': ['Store 1', 'Store 2']
    })
    
    return products, stores


def test_phase_2_3():
    """Test Phase 2.3 marketing signal integration"""
    print("\n" + "="*70)
    print("PHASE 2.3 QUICK INTEGRATION TEST")
    print("="*70)
    
    # Create test data
    print("\n1. Creating test data...")
    products, stores = create_test_data()
    print(f"   ✅ Created {len(products)} products, {len(stores)} stores")
    
    # Initialize promotional engine
    print("\n2. Initializing promotional engine...")
    try:
        engine = PromotionalEngine(
            products_df=products,
            stores_df=stores,
            config=None
        )
        print("   ✅ Promotional engine initialized")
    except Exception as e:
        print(f"   ❌ FAILED to initialize engine: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check signal calculator exists
    print("\n3. Verifying signal calculator...")
    if not hasattr(engine, 'signal_calculator'):
        print("   ❌ Signal calculator not found!")
        return False
    if engine.signal_calculator is None:
        print("   ❌ Signal calculator is None!")
        return False
    print("   ✅ Signal calculator initialized")
    print(f"   Type: {type(engine.signal_calculator).__name__}")
    
    # Generate promotions for a store
    print("\n4. Generating store promotions...")
    product_ids = products['product_id'].values
    base_prices = products['base_price'].values
    
    try:
        promo_context = engine.generate_store_promotions(
            store_id=1,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        print(f"   ✅ Generated promotions for store 1, week 1")
    except Exception as e:
        print(f"   ❌ FAILED to generate promotions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check promotional context
    print("\n5. Validating promotional context...")
    print(f"   Promoted products: {len(promo_context.promoted_products)}")
    print(f"   Avg discount: {promo_context.avg_discount_depth:.1%}")
    print(f"   Deep discounts (>30%): {promo_context.n_deep_discounts}")
    print(f"   End caps: {len(promo_context.end_cap_products)}")
    print(f"   Feature displays: {len(promo_context.feature_display_products)}")
    print(f"   In-ad: {len(promo_context.in_ad_products)}")
    print(f"   Mailer: {len(promo_context.mailer_products)}")
    
    # Check marketing signal
    print("\n6. Checking marketing signal strength...")
    if not hasattr(promo_context, 'marketing_signal_strength'):
        print("   ❌ marketing_signal_strength not found in context!")
        return False
    
    signal = promo_context.marketing_signal_strength
    print(f"   ✅ Marketing signal strength: {signal:.3f}")
    
    # Validate signal range
    if not (0.0 <= signal <= 1.0):
        print(f"   ❌ Signal out of range [0, 1]: {signal}")
        return False
    print("   ✅ Signal in valid range [0.0, 1.0]")
    
    # Test signal breakdown
    print("\n7. Testing signal breakdown...")
    try:
        breakdown = engine.signal_calculator.get_signal_breakdown(promo_context)
        print(f"   Discount signal: {breakdown['discount_signal']:.3f}")
        print(f"   Display signal: {breakdown['display_signal']:.3f}")
        print(f"   Advertising signal: {breakdown['advertising_signal']:.3f}")
        print(f"   Total signal: {breakdown['total_signal']:.3f}")
        print("   ✅ Signal breakdown working")
    except Exception as e:
        print(f"   ❌ Signal breakdown failed: {e}")
        return False
    
    # Test visit probability boost
    print("\n8. Testing visit probability boost...")
    try:
        base_prob = 0.3
        boosted_prob = engine.signal_calculator.calculate_visit_probability_boost(
            signal, base_prob
        )
        boost_pct = ((boosted_prob - base_prob) / base_prob) * 100
        print(f"   Base probability: {base_prob:.1%}")
        print(f"   Boosted probability: {boosted_prob:.1%}")
        print(f"   Boost: +{boost_pct:.1f}%")
        print("   ✅ Visit probability boost working")
    except Exception as e:
        print(f"   ❌ Visit boost failed: {e}")
        return False
    
    # Test multiple stores
    print("\n9. Testing multi-store variation...")
    signals = []
    for store_id in [1, 2]:
        ctx = engine.generate_store_promotions(
            store_id=store_id,
            week_number=1,
            base_prices=base_prices,
            product_ids=product_ids
        )
        signals.append(ctx.marketing_signal_strength)
        print(f"   Store {store_id}: signal = {ctx.marketing_signal_strength:.3f}, "
              f"promos = {len(ctx.promoted_products)}, "
              f"avg discount = {ctx.avg_discount_depth:.1%}")
    
    print("   ✅ Multi-store signals generated")
    
    # Success!
    print("\n" + "="*70)
    print("✅ ALL CHECKS PASSED!")
    print("="*70)
    print("\nPhase 2.3 Integration Summary:")
    print(f"  • Marketing signal calculator: ✅ Working")
    print(f"  • Signal calculation: ✅ Working")
    print(f"  • Signal range validation: ✅ Passing")
    print(f"  • Signal breakdown: ✅ Working")
    print(f"  • Visit probability boost: ✅ Working")
    print(f"  • Multi-store support: ✅ Working")
    print("\n🎉 Phase 2.3 is FULLY INTEGRATED and OPERATIONAL!")
    
    return True


if __name__ == "__main__":
    try:
        success = test_phase_2_3()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
